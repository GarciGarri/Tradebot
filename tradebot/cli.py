"""Command line entry point: `tradebot <command>`. See docs/RUNBOOK.md."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone

from .config import LIVE_ACK_PHRASE, load_risk_limits, load_settings, timeframe_ms
from .types import Bar, Mode


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stdout)


def cmd_run(args) -> int:
    from .runtime.service import BotService
    settings, limits = load_settings(args.config), load_risk_limits()
    svc = BotService(settings, limits)
    svc.run_forever()
    return 0


def cmd_healthcheck(args) -> int:
    from .runtime.service import heartbeat_age_s
    settings = load_settings(args.config)
    age = heartbeat_age_s(settings.path(settings.state_dir))
    ok = age is not None and age <= settings.runtime.heartbeat_max_age_s
    print(f"heartbeat age: {'none' if age is None else f'{age:.0f}s'} -> {'ok' if ok else 'UNHEALTHY'}")
    return 0 if ok else 1


def _recorder(settings):
    from .storage.recorder import SqliteRecorder
    return SqliteRecorder(settings.path(settings.state_dir) / "tradebot.db")


def cmd_status(args) -> int:
    settings = load_settings(args.config)
    hb = settings.path(settings.state_dir) / "heartbeat.json"
    print(hb.read_text() if hb.exists() else "no heartbeat yet")
    rec = _recorder(settings)
    print(json.dumps({"kill": rec.get("kill", False), "pause_until": rec.get("pause_until", {}),
                      "daily_halt_day": rec.get("daily_halt_day", -1)}))
    for e in rec.query("SELECT ts, level, key, message FROM events ORDER BY id DESC LIMIT 10"):
        print(datetime.fromtimestamp(e["ts"] / 1000, timezone.utc).strftime("%m-%d %H:%M"), e["level"], e["key"], "-", e["message"])
    return 0


def cmd_kill(args) -> int:
    rec = _recorder(load_settings(args.config))
    rec.set("kill", True)
    rec.event(int(time.time() * 1000), "critical", "manual_kill", "kill switch set by operator")
    print("kill switch latched. A running bot sees it on its next risk tick (seconds), closes every position "
          "and opens no new risk until `tradebot resume` and a restart.")
    return 0


def cmd_resume(args) -> int:
    settings = load_settings(args.config)
    rec = _recorder(settings)
    rec.set("kill", False)
    rec.set("pause_until", {})
    if args.reset_peak:
        led = rec.get("paper_ledger")
        if led:
            eq = led["cash"]
            led["peak_equity"] = eq
            rec.set("paper_ledger", led)
        rec.set("live_peak_equity", 0)
    rec.event(int(time.time() * 1000), "warning", "manual_resume", f"resumed by operator (reset_peak={args.reset_peak})")
    print("kill latch and pauses cleared" + ("; drawdown reference reset" if args.reset_peak else "") +
          ". Restart the bot container for it to take effect.")
    return 0


def cmd_backtest(args) -> int:
    from .backtest.run import run
    from .data.feed import CcxtFeed
    from .policy.policy import PolicyStore
    logging.disable(logging.CRITICAL)
    settings, limits = load_settings(args.config), load_risk_limits()
    policy = PolicyStore(settings.path(settings.policies_dir)).active("paper")
    feed = None if args.offline else CcxtFeed(settings.exchange.id, settings.exchange.timeframe,
                                             timeframe_ms(settings.exchange.timeframe))
    out = run(settings=settings, limits=limits, policy=policy, symbols=args.symbols.split(","), start=args.start,
              end=args.end, train_days=args.train_days, test_days=args.test_days, spot_proxy=args.spot_proxy,
              workers=args.workers, label=args.label, out_dir=settings.path(settings.reports_dir) / "backtests", feed=feed)
    return 0 if out else 1


def _nightly_once(settings) -> dict:
    from .policy.policy import PolicyStore
    from .review.nightly import run_nightly
    from .setups.schema import load_setups
    rec = _recorder(settings)
    tf = timeframe_ms(settings.exchange.timeframe)
    since = int((datetime.now(timezone.utc) - timedelta(days=60)).timestamp() * 1000)
    rows = rec.query("SELECT * FROM decisions WHERE ts >= ? ORDER BY ts", (since,))
    trades = rec.query("SELECT * FROM trades WHERE exit_ts >= ? ORDER BY exit_ts", (since,))
    events = rec.query("SELECT * FROM events WHERE ts >= ? ORDER BY ts", (int(time.time() * 1000) - 86_400_000,))
    bars: dict[str, list[Bar]] = {}
    for b in rec.query("SELECT * FROM bars WHERE ts >= ? ORDER BY symbol, ts", (since,)):
        bars.setdefault(b["symbol"], []).append(Bar(b["ts"], b["open"], b["high"], b["low"], b["close"], b["volume"], tf))
    return run_nightly(rows=rows, trades=trades, bars=bars, fees=settings.exchange.fees,
                       store=PolicyStore(settings.path(settings.policies_dir)),
                       reports_dir=settings.path(settings.reports_dir), events=events,
                       setups=load_setups(settings.path(settings.setups_dir)), recorder=rec)


def cmd_nightly(args) -> int:
    from .alerts.base import Alerter, telegram_sender_from_env
    settings = load_settings(args.config)
    alerts = Alerter(0.0, telegram_sender_from_env())
    while True:
        if args.loop:
            now = datetime.now(timezone.utc)
            nxt = now.replace(hour=settings.runtime.nightly_utc_hour, minute=settings.runtime.nightly_utc_minute,
                              second=0, microsecond=0)
            if nxt <= now:
                nxt += timedelta(days=1)
            time.sleep(max(1.0, (nxt - now).total_seconds()))
        try:
            rep = _nightly_once(settings)
            s = rep["stats"]
            alerts.alert("info", "nightly", f"nightly {rep['date']}: {s['n_labeled']} labelled, Brier "
                         f"{s['brier_raw'] if s['brier_raw'] is None else round(s['brier_raw'], 4)}, trades "
                         f"{s['fills']['n_trades']} PnL {s['fills']['pnl']:+.2f}, candidate "
                         f"{rep['candidate_version']} promoted_to_paper={rep['promoted_to_paper']}", force=True)
            print(f"nightly review written for {rep['date']}")
        except Exception as exc:
            logging.exception("nightly failed")
            alerts.alert("error", "nightly_failed", f"nightly review failed: {type(exc).__name__}", force=True)
            if not args.loop:
                return 1
        if not args.loop:
            return 0


def cmd_regime(args) -> int:
    from .research.regime import run
    settings = load_settings(args.config)
    s = run(settings.path(settings.reports_dir) / "research")
    print(json.dumps({"regime": s["regime"], "gaps": s["gaps"]}, indent=1))
    return 0


def cmd_reflex_check(args) -> int:
    """One real REFLEX call on a live snapshot: exact tokens, latency, parsed output. Needs the API key."""
    from .data.feed import CcxtFeed, fill_gaps
    from .reflex import build_reflex
    from .reflex.base import ReflexRunner
    from .state.engine import MarketStore, StateEngine
    from .types import AccountState
    settings, limits = load_settings(args.config), load_risk_limits()
    tf = timeframe_ms(settings.exchange.timeframe)
    feed = CcxtFeed(settings.exchange.id, settings.exchange.timeframe, tf)
    store = MarketStore(tf)
    now = feed.server_time_ms()
    symbols = list(dict.fromkeys([args.symbol] + list(settings.context_symbols)))
    for s in symbols:
        bars, syn = fill_gaps(feed.closed_bars(s, limit=300, now_ms=now), tf)
        for b in bars:
            store.add_bar(s, b, synthetic=b.ts in syn)
    store.set_book(args.symbol, feed.book(args.symbol))
    engine = StateEngine(store, settings.guards, limits, tuple(settings.context_symbols))
    acct = AccountState(ts=now, cash=settings.paper_balance, equity=settings.paper_balance,
                        peak_equity=settings.paper_balance, day_start_equity=settings.paper_balance)
    inst = feed.instruments(settings.exchange.market_type, settings.exchange.quote)[args.symbol]
    snap = engine.snapshot(args.symbol, now, acct, contract_size=inst.contract_size)
    print("snapshot:", snap.text)
    print(f"status={snap.status.value} local token estimate={snap.tokens_est}")
    reflex = build_reflex(settings.reflex)
    if hasattr(reflex, "count_tokens"):
        print("exact snapshot tokens:", reflex.count_tokens(snap.text))
    runner = ReflexRunner(reflex, settings.reflex.deadline_ms)
    for i in range(args.n):
        res = runner.run(snap)
        print(i, type(res).__name__, getattr(res, "latency_ms", None), getattr(res, "output", getattr(res, "detail", "")))
    print("latency:", runner.latency_summary(), "usage:", getattr(reflex, "last_usage", None))
    runner.close()
    return 0


def cmd_preflight(args) -> int:
    """Read-only venue check with real keys. Places nothing: at most one order PREVIEW."""
    from .data.feed import make_ccxt
    import os
    settings = load_settings(args.config)
    key, secret = os.environ.get("EXCHANGE_API_KEY"), os.environ.get("EXCHANGE_API_SECRET")
    if not (key and secret):
        print("EXCHANGE_API_KEY / EXCHANGE_API_SECRET are not set: nothing to check")
        return 1
    ex = make_ccxt(settings.exchange.id, {"apiKey": key, "secret": secret})
    ex.load_markets()
    swaps = sorted(s for s, m in ex.markets.items() if m.get("type") == settings.exchange.market_type)
    print(f"perp-style markets visible to this account: {len(swaps)}: {swaps}")
    bal = ex.fetch_balance()
    print("balance currencies with funds:", {k: v for k, v in (bal.get('total') or {}).items() if v})
    try:
        print("open positions:", [(p["symbol"], p.get("contracts"), p.get("side")) for p in ex.fetch_positions(swaps[:20])])
    except Exception as exc:
        print("fetch_positions failed:", type(exc).__name__)
    if args.preview_symbol:
        try:
            r = ex.create_order(args.preview_symbol, "market", "buy", 1, None, {"preview": True})
            print("order PREVIEW (nothing placed):", {k: r.get(k) for k in ("status", "symbol", "amount", "cost", "fee")})
        except Exception as exc:
            print("order preview failed:", type(exc).__name__, str(exc)[:200])
    return 0


def cmd_policy(args) -> int:
    from .policy.policy import PolicyStore
    settings = load_settings(args.config)
    store = PolicyStore(settings.path(settings.policies_dir))
    if args.action == "show":
        print(json.dumps(store.pointer(), indent=1))
        print(store.active("paper").model_dump_json(indent=1))
    elif args.action == "rollback":
        print("rolled back", args.target, "to v", store.rollback(args.target, by=args.by or "operator", reason=args.reason))
    elif args.action == "promote-live":
        if not args.by:
            print("--by <your name> is required: a policy reaches live only with a named human approver")
            return 2
        store.promote(args.version, "live", by=args.by, reason=args.reason, human_approved=True)
        print(f"policy v{args.version} is now the live policy (approved by {args.by})")
    return 0


def cmd_instruments(args) -> int:
    from .backtest.baseline import save_instruments
    from .config import REPO_ROOT
    from .data.feed import CcxtFeed
    settings = load_settings(args.config)
    feed = CcxtFeed(settings.exchange.id, settings.exchange.timeframe, timeframe_ms(settings.exchange.timeframe))
    inst = feed.instruments(settings.exchange.market_type, settings.exchange.quote)
    save_instruments(REPO_ROOT / "config" / "instruments.yaml", inst, f"ccxt {settings.exchange.id} load_markets (public)")
    print(f"saved {len(inst)} instruments")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tradebot", description="24/7 crypto trading agent (paper by default)")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="start the 24/7 loop in the configured mode").set_defaults(fn=cmd_run)
    sub.add_parser("healthcheck", help="exit 0 if the heartbeat is fresh").set_defaults(fn=cmd_healthcheck)
    sub.add_parser("status", help="heartbeat, halts and recent events").set_defaults(fn=cmd_status)
    sub.add_parser("kill", help="latch the kill switch by hand").set_defaults(fn=cmd_kill)
    r = sub.add_parser("resume", help="clear the kill latch and pauses (human only)")
    r.add_argument("--reset-peak", action="store_true")
    r.set_defaults(fn=cmd_resume)
    b = sub.add_parser("backtest", help="walk-forward backtest of the code rules vs holding BTC")
    b.add_argument("--symbols", required=True)
    b.add_argument("--start", required=True)
    b.add_argument("--end", required=True)
    b.add_argument("--train-days", type=int, default=180)
    b.add_argument("--test-days", type=int, default=60)
    b.add_argument("--spot-proxy", action="store_true")
    b.add_argument("--offline", action="store_true", help="use only cached candles")
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--label", default="wf")
    b.set_defaults(fn=cmd_backtest)
    n = sub.add_parser("nightly", help="run the nightly review once, or forever with --loop")
    n.add_argument("--loop", action="store_true")
    n.set_defaults(fn=cmd_nightly)
    sub.add_parser("regime", help="pull the live market regime snapshot").set_defaults(fn=cmd_regime)
    rc = sub.add_parser("reflex-check", help="one real REFLEX call: tokens, latency, output")
    rc.add_argument("--symbol", default="BTC/USD:USD")
    rc.add_argument("-n", type=int, default=3)
    rc.set_defaults(fn=cmd_reflex_check)
    pf = sub.add_parser("preflight", help="read-only venue check with real keys (no orders)")
    pf.add_argument("--preview-symbol", default=None)
    pf.set_defaults(fn=cmd_preflight)
    po = sub.add_parser("policy", help="show, roll back or (human only) promote a policy to live")
    po.add_argument("action", choices=["show", "rollback", "promote-live"])
    po.add_argument("--target", choices=["paper", "live"], default="paper")
    po.add_argument("--version", type=int)
    po.add_argument("--by", default="")
    po.add_argument("--reason", default="manual")
    po.set_defaults(fn=cmd_policy)
    sub.add_parser("instruments", help="refresh config/instruments.yaml from the venue").set_defaults(fn=cmd_instruments)
    args = p.parse_args(argv)
    _setup_logging()
    if getattr(args, "cmd", "") == "run":
        s = load_settings(args.config)
        if s.mode is Mode.LIVE:
            print(f"LIVE mode requested. It only starts if TRADEBOT_LIVE_ACK={LIVE_ACK_PHRASE} and keys are set.")
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())

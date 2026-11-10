"""The 24/7 runtime: owns the clock and the feeds and drives the trading core.

Loop, every `risk_tick_s` seconds:
  1. read the exchange clock (skew is measured, never assumed);
  2. refresh order books and marks for every symbol in play;
  3. `core.risk_tick` - stops, targets, kill switch, daily halt. No model involved;
  4. if a candle has closed (plus a short settle delay): fetch the closed bars, funding and
     open interest, charge paper funding, then `core.on_candle`;
  5. write a heartbeat file. The container health check reads it.

Every network call is wrapped. A failing symbol is flagged in the store, which makes its
snapshot non-OK, which skips its candle. A failing loop iteration is logged and retried on
the next tick; the process is also restarted by Docker if it ever dies.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import time
from pathlib import Path

from ..alerts.base import Alerter, telegram_sender_from_env
from ..config import RiskLimits, Settings, timeframe_ms
from ..data.feed import CcxtFeed, fill_gaps
from ..execution.paper import PaperExchange
from ..policy.policy import PolicyStore
from ..reflex import build_reflex
from ..reflex.base import ReflexRunner
from ..setups.schema import load_setups
from ..state.engine import MarketStore, StateEngine
from ..storage.recorder import SqliteRecorder
from ..types import Mode
from .core import PriceRange, TradingCore

log = logging.getLogger("tradebot.service")


class BotService:
    def __init__(self, settings: Settings, limits: RiskLimits, feed: CcxtFeed | None = None, reflex=None,
                 recorder=None, env: dict[str, str] | None = None, clock=None):
        env = os.environ if env is None else env
        self._clock = clock or (lambda: int(time.time() * 1000))     # the LOCAL clock, compared with the venue's
        self.settings, self.limits = settings, limits
        self.tf_ms = timeframe_ms(settings.exchange.timeframe)
        self.state_dir = settings.path(settings.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.recorder = recorder or SqliteRecorder(self.state_dir / "tradebot.db")
        self.alerts = Alerter(settings.alerts.min_interval_s, telegram_sender_from_env(env))
        self.feed = feed or CcxtFeed(settings.exchange.id, settings.exchange.timeframe, self.tf_ms)

        setups = load_setups(settings.path(settings.setups_dir), only_active=True)
        venue = self.feed.instruments(settings.exchange.market_type, settings.exchange.quote)
        missing = [s.symbol for s in setups.values() if s.symbol not in venue]
        if missing:
            # an instrument this account cannot trade must never be traded "by proxy"
            raise RuntimeError(f"active setups reference instruments the venue does not list: {missing}")
        self.symbols = list(dict.fromkeys([s.symbol for s in setups.values()] +
                                          [c for c in settings.context_symbols if c in venue]))
        self.instruments = {s: venue[s] for s in self.symbols}
        target = "live" if settings.mode is Mode.LIVE else "paper"
        self.policy = PolicyStore(settings.path(settings.policies_dir)).active(target)

        self.store = MarketStore(self.tf_ms)
        self.engine = StateEngine(self.store, settings.guards, limits, tuple(settings.context_symbols),
                                  settings.reflex.snapshot_token_budget)
        now = self.feed.server_time_ms()
        if settings.mode is Mode.PAPER:
            self.exchange = PaperExchange(settings.paper_balance, now, self.instruments, settings.exchange.fees)
            saved = self.recorder.get("paper_ledger")
            if saved:
                self.exchange.ledger.restore(saved)       # the paper account survives restarts
        else:
            from ..execution.ccxt_adapter import CcxtExchange
            self.exchange = CcxtExchange(settings, self.instruments, self.recorder, env)
        self.reflex = reflex if reflex is not None else build_reflex(settings.reflex)
        self.runner = ReflexRunner(self.reflex, settings.reflex.deadline_ms)
        self.core = TradingCore(settings=settings, limits=limits, policy=self.policy, setups=setups,
                                instruments=self.instruments, engine=self.engine, runner=self.runner,
                                exchange=self.exchange, recorder=self.recorder, alerts=self.alerts)
        self.last_candle = 0
        self.skew_ms = 0
        self.errors: dict[str, int] = {}
        self._stop = False
        self._ticks = 0

    # ---- data -----------------------------------------------------------------
    def _load_bars(self, symbol: str, limit: int, now_ms: int) -> None:
        have = self.store.last_bar(symbol, now_ms)
        if have is not None:
            # after an outage, fetch enough bars to cover the hole: real data, not synthetic filler
            missing = int((now_ms - have.close_ts) // self.tf_ms) + 2
            limit = max(limit, min(300, missing))
        bars = self.feed.closed_bars(symbol, limit=limit, now_ms=now_ms)
        if have is not None:
            bars = [have] + [b for b in bars if b.ts > have.ts]     # bridge from the last stored bar
        filled, synthetic = fill_gaps(bars, self.tf_ms)
        for b in filled:
            if have is not None and b.ts <= have.ts:
                continue
            self.store.add_bar(symbol, b, synthetic=b.ts in synthetic)
            self.recorder.bar(symbol, b.ts, b.open, b.high, b.low, b.close, b.volume)

    def _guard(self, symbol: str, what: str, fn) -> bool:
        try:
            fn()
            self.errors.pop(f"{symbol}:{what}", None)
            return True
        except Exception as exc:
            k = f"{symbol}:{what}"
            self.errors[k] = self.errors.get(k, 0) + 1
            self.store.mark_error(symbol, f"{what}: {type(exc).__name__}", int(time.time() * 1000))
            log.warning("%s %s failed: %s", symbol, what, type(exc).__name__)
            if self.errors[k] in (3, 30):
                self.alerts.alert("error", f"feed:{k}", f"{symbol}: {what} failed {self.errors[k]} times in a row")
            return False

    def backfill(self) -> None:
        now = self.feed.server_time_ms()
        for s in self.symbols:
            self._guard(s, "bars", lambda s=s: self._load_bars(s, 300, now))
            if f"{s}:bars" not in self.errors:
                self.store.clear_error(s)

    def _refresh_books(self, now_ms: int) -> dict[str, PriceRange]:
        ranges: dict[str, PriceRange] = {}
        for s in self.symbols:
            def read(s=s):
                book = self.feed.book(s)
                self.store.set_book(s, book)
                if self.settings.mode is Mode.PAPER:
                    last = self.store.last_bar(s, now_ms)
                    snap_vol = None
                    if last is not None:
                        snap_vol = last.close * last.volume * self.instruments[s].contract_size or None
                    self.exchange.mark(s, book.mid)
                    self.exchange.set_quote(s, book.best_bid, book.best_ask, book.ts, snap_vol)
                ranges[s] = PriceRange(book.mid, book.mid, book.mid)
            if self._guard(s, "book", read) and f"{s}:bars" not in self.errors:
                self.store.clear_error(s)
        return ranges

    # ---- one iteration ----------------------------------------------------------
    def tick(self) -> None:
        local = self._clock()
        exch = self.feed.server_time_ms()
        self.skew_ms = local - exch
        now = exch
        ranges = self._refresh_books(now)
        self.core.risk_tick(now, ranges)
        self._ticks += 1
        if self._ticks % 15 == 0:
            self.core.reconcile(now)

        candle_close = (now // self.tf_ms) * self.tf_ms
        if candle_close > self.last_candle and now >= candle_close + int(self.settings.runtime.candle_settle_s * 1000):
            for s in self.symbols:
                self._guard(s, "bars", lambda s=s: self._load_bars(s, 5, now))

                def perp(s=s):
                    d = self.feed.perp_details(s)
                    if d["funding_rate"] is not None:
                        self.store.set_funding(s, d["funding_rate"], now)
                    if d["open_interest"] is not None:
                        self.store.set_open_interest(s, d["open_interest"], now)
                    if d["index_price"] is not None:
                        self.store.set_index(s, d["index_price"], now)
                self._guard(s, "perp_details", perp)
                if f"{s}:bars" not in self.errors and f"{s}:book" not in self.errors:
                    self.store.clear_error(s)          # a failed funding read alone must not block trading
            if self.settings.mode is Mode.PAPER:
                for s in list(self.exchange.ledger.positions):
                    f = self.store.funding(s, now)
                    if f is not None:
                        self.exchange.ledger.apply_funding(s, f[1])
                    else:
                        self.exchange.ledger.apply_funding(s, self.settings.exchange.fees.funding_bps_per_hour / 1e4,
                                                           pure_cost=True)
            # Decide at the real current time: bars are those closed by `candle_close`, the book is
            # the one just read. The second clock passed in is the local one, so the state engine
            # can refuse to act when the two clocks disagree.
            self.core.on_candle(now, exchange_time_ms=local)
            self.last_candle = candle_close
        if self.settings.mode is Mode.PAPER:
            self.recorder.set("paper_ledger", self.exchange.ledger.to_dict())
        self.heartbeat(now)

    def heartbeat(self, now_ms: int) -> dict:
        acct = self.exchange.account(now_ms)
        hb = {
            "ts": int(time.time() * 1000), "exchange_ts": now_ms, "skew_ms": self.skew_ms,
            "mode": self.settings.mode.value, "policy_version": self.policy.version,
            "reflex_backend": self.reflex.name, "equity": round(acct.equity, 2), "drawdown": round(acct.drawdown, 4),
            "day_pnl": round(acct.day_pnl_frac, 4), "leverage": round(acct.leverage, 3),
            "positions": {s: p.qty for s, p in acct.positions.items()}, "last_candle": self.last_candle,
            "kill": self.core.kill, "paused": {k: v for k, v in self.core.pause_until.items() if v > now_ms},
            "reflex": dict(self.runner.stats), "reflex_latency": self.runner.latency_summary(),
            "feed_errors": dict(self.errors), "symbols": self.symbols,
        }
        tmp = self.state_dir / "heartbeat.tmp"
        tmp.write_text(json.dumps(hb))
        tmp.replace(self.state_dir / "heartbeat.json")
        return hb

    # ---- forever ----------------------------------------------------------------
    def run_forever(self) -> None:
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "_stop", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "_stop", True))
        self.alerts.alert("info", "startup",
                          f"tradebot started: mode={self.settings.mode.value}, policy v{self.policy.version}, "
                          f"reflex={self.reflex.name}, symbols={len(self.symbols)}", force=True)
        self.backfill()
        self.last_candle = (self.feed.server_time_ms() // self.tf_ms) * self.tf_ms   # never act on a stale candle at boot
        while not self._stop:
            started = time.time()
            try:
                self.tick()
            except Exception as exc:    # keep the loop alive; Docker restarts us if the process dies
                log.exception("tick failed")
                self.alerts.alert("error", "tick_failed", f"runtime tick failed: {type(exc).__name__}")
            time.sleep(max(1.0, self.settings.runtime.risk_tick_s - (time.time() - started)))
        self.runner.close()
        self.alerts.alert("info", "shutdown", "tradebot stopped cleanly; open positions keep their recorded stops", force=True)


def heartbeat_age_s(state_dir: Path) -> float | None:
    path = state_dir / "heartbeat.json"
    if not path.exists():
        return None
    try:
        return time.time() - json.loads(path.read_text())["ts"] / 1000.0
    except Exception:
        return None

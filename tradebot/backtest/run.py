"""Walk-forward runner and report writer (used by `tradebot backtest`)."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from ..config import REPO_ROOT, FeeModel, RiskLimits, Settings
from ..policy.policy import Policy
from .baseline import load_instruments, load_market_data, technical_setups
from .walkforward import WalkForwardResult, default_grid, walk_forward


def _ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def summarize(r: WalkForwardResult) -> dict:
    return {
        "oos_start": _day(int(r.ts[0])) if len(r.ts) else None, "oos_end": _day(int(r.ts[-1])) if len(r.ts) else None,
        "strategy": r.metrics, "btc_hold": r.benchmark, "beats_btc_hold": r.beats_benchmark(),
        "kills": r.kills, "fees": r.fees, "funding": r.funding, "final_equity": float(r.equity[-1]) if len(r.equity) else None,
        "folds": len(r.folds), "folds_flat": sum(1 for f in r.folds if f.chosen == "flat"),
        "folds_positive": sum(1 for f in r.folds if f.test_return > 0),
        "folds_negative": sum(1 for f in r.folds if f.test_return < 0),
        "counters": r.counters,
        "choices": [f.chosen for f in r.folds],
        "fold_returns": [round(f.test_return, 5) for f in r.folds],
    }


def run(*, settings: Settings, limits: RiskLimits, policy: Policy, symbols: list[str], start: str, end: str,
        train_days: int, test_days: int, spot_proxy: bool, workers: int = 4, cache_dir: str | Path | None = None,
        fee_levels: tuple[float, ...] = (0.0005, 0.0002), label: str = "wf", out_dir: str | Path | None = None,
        feed=None, log=print) -> dict:
    cache_dir = Path(cache_dir or REPO_ROOT / "data_cache")
    instruments = load_instruments(REPO_ROOT / "config" / "instruments.yaml")
    start_ms, end_ms = _ms(start), _ms(end)
    t0 = time.time()
    data = load_market_data(settings=settings, limits=limits, instruments=instruments, symbols=symbols,
                            start_ms=start_ms, end_ms=end_ms, cache_dir=cache_dir, feed=feed, spot_proxy=spot_proxy)
    symbols = [s for s in symbols if s in data.bars]
    setups = technical_setups(symbols)
    bars_per_day = 86_400_000 // data.tf_ms
    grid = default_grid()
    log(f"[{label}] data: {len(symbols)} symbols, {sum(len(b) for b in data.bars.values())} bars, "
        f"{time.time() - t0:.0f}s; grid {len(grid)} candidates; workers {workers}")

    def progress(f):
        log(f"[{label}] fold {f.index:02d} test {_day(f.test_start)}..{_day(f.test_end)} chosen={f.chosen} "
            f"train_ret={f.train_return:+.2%} test_ret={f.test_return:+.2%} trades={f.test_trades} "
            f"equity={f.equity_end:.2f}{' KILLED' if f.killed else ''}")

    common = dict(data=data, limits=limits, base_policy=policy, setups=setups, grid=grid, start_ms=start_ms,
                  end_ms=end_ms, train_bars=train_days * bars_per_day, test_bars=test_days * bars_per_day,
                  cash=settings.paper_balance, benchmark_symbol=settings.context_symbols[0])
    base = walk_forward(settings=settings, workers=workers, progress=progress, **common)
    out = {
        "label": label, "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data": "Coinbase spot candles as a proxy for the perp contracts" if spot_proxy else "Coinbase perp-style futures candles",
        "symbols": symbols, "timeframe": settings.exchange.timeframe, "train_days": train_days, "test_days": test_days,
        "fees_assumed": settings.exchange.fees.model_dump(), "paper_balance": settings.paper_balance,
        "reflex": "rules (deterministic code, no model)", "base": summarize(base), "fee_sensitivity": {},
        "no_resume_after_kill": None, "peak_reset_each_fold": None,
    }
    choices = [f.chosen for f in base.folds]
    # Cost sensitivity: same per-fold selections, cheaper taker fees. This shows how much of the
    # result is fee drag; it does not re-select parameters, so it is not a new optimisation.
    for taker in fee_levels:
        fees = FeeModel(**{**settings.exchange.fees.model_dump(), "taker": taker})
        s2 = settings.model_copy(update={"exchange": settings.exchange.model_copy(update={"fees": fees})})
        r = walk_forward(settings=s2, fixed_choices=choices, **common)
        out["fee_sensitivity"][f"taker_{taker:.4%}"] = summarize(r)
        log(f"[{label}] taker {taker:.3%}: total_return={r.metrics['total_return']:+.2%} maxDD={r.metrics['max_drawdown']:.2%}")
    if base.kills:
        r = walk_forward(settings=settings, fixed_choices=choices, resume_after_kill=False, **common)
        out["no_resume_after_kill"] = summarize(r)
    # What the raw rule does when a human resets the drawdown reference at every fold boundary.
    # This is NOT the deployed behaviour (the deployed bot stops adding risk deep in drawdown);
    # it is reported so the risk overlay cannot hide a rule that keeps losing.
    r = walk_forward(settings=settings, fixed_choices=choices, reset_peak_each_fold=True, **common)
    out["peak_reset_each_fold"] = summarize(r)
    log(f"[{label}] peak reset each fold: total_return={r.metrics['total_return']:+.2%} "
        f"maxDD={r.metrics['max_drawdown']:.2%} trades={r.metrics['n_trades']} kills={r.kills}")
    if out_dir is not None:
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        (od / f"{label}.json").write_text(json.dumps(out, indent=1, default=float))
        (od / f"{label}_equity.csv").write_text(
            "ts,equity\n" + "\n".join(f"{int(t)},{e:.4f}" for t, e in zip(base.ts[::24], base.equity[::24])))
    b, h = out["base"]["strategy"], out["base"]["btc_hold"]
    log(f"[{label}] OOS {out['base']['oos_start']}..{out['base']['oos_end']}: strategy {b['total_return']:+.2%} "
        f"(maxDD {b['max_drawdown']:.2%}, Sharpe {b['sharpe']:.2f}, trades {b['n_trades']}) vs BTC hold "
        f"{h['total_return']:+.2%} (maxDD {h['max_drawdown']:.2%}, Sharpe {h['sharpe']:.2f}) -> "
        f"{'BEATS' if out['base']['beats_btc_hold'] else 'DOES NOT BEAT'} BTC hold")
    return out

"""Walk-forward evaluation: parameters are chosen on a training window and judged only on the
window that follows it. Only the stitched out-of-sample (OOS) curve is ever reported.

Per fold:
1. every candidate runs on the training window with a fresh account;
2. the best candidate by the training objective is selected. If no candidate made money in
   training (after costs) the fold trades nothing: a losing rule is never deployed;
3. the chosen candidate runs on the test window, starting from the equity, the equity peak
   and the kill-switch state left by the previous test window.

The benchmark is buying BTC at the first OOS bar and holding it, with entry costs.
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import asdict, dataclass, field

import numpy as np

from ..config import RiskLimits, Settings
from ..policy.policy import ExitCfg, Policy
from ..reflex.rules import RuleParams, RuleReflex
from ..setups.schema import SetupSchema
from .engine import MarketData, run_backtest
from .metrics import buy_and_hold, curve_metrics, trade_metrics

BARS_PER_YEAR = {3_600_000: 8760.0, 14_400_000: 2190.0, 86_400_000: 365.0}


@dataclass(frozen=True)
class Candidate:
    name: str
    rule: RuleParams
    exits: dict = field(default_factory=dict)     # overrides for Policy.exits

    def policy(self, base: Policy) -> Policy:
        exits = ExitCfg(**{**base.exits.model_dump(), **self.exits})
        return base.model_copy(update={"exits": exits, "rule_params": asdict(self.rule)})


def default_grid() -> list[Candidate]:
    out = []
    for thr in (0.35, 0.50):
        for stop in (2.0, 3.5):
            for tp in (1.5, 3.0):
                for short in (True, False):
                    out.append(Candidate(
                        name=f"thr{thr:g}_stop{stop:g}_tp{tp:g}_{'ls' if short else 'lo'}",
                        rule=RuleParams(score_threshold=thr, allow_short=short),
                        exits={"stop_atr_mult": stop, "take_profit_r": tp}))
    return out


def objective(total_return: float, max_drawdown: float, n_trades: int, min_trades: int) -> float:
    """Return per unit of drawdown, with a floor on the drawdown so tiny samples cannot win."""
    if n_trades < min_trades:
        return float("-inf")
    return total_return / max(max_drawdown, 0.03)


@dataclass
class Fold:
    index: int
    train_start: int
    test_start: int
    test_end: int
    chosen: str
    train_score: float
    train_return: float
    test_return: float
    test_trades: int
    equity_end: float
    killed: bool


@dataclass
class WalkForwardResult:
    folds: list[Fold]
    ts: np.ndarray
    equity: np.ndarray
    trades: list
    metrics: dict
    benchmark: dict
    kills: int
    fees: float
    funding: float
    counters: dict

    def beats_benchmark(self) -> bool:
        return self.metrics["total_return"] > self.benchmark["total_return"]


_CTX: dict = {}


def _train_one(args: tuple[int, int, int]) -> tuple[int, float, float, float, int]:
    idx, start, end = args
    c = _CTX
    cand: Candidate = c["grid"][idx]
    r = run_backtest(data=c["data"], settings=c["settings"], limits=c["limits"], policy=cand.policy(c["policy"]),
                     setups=c["setups"], reflex=RuleReflex(cand.rule), start_ms=start, end_ms=end, cash=c["cash"])
    cm = curve_metrics(r.ts, r.equity, c["bpy"])
    return idx, cm["total_return"], cm["max_drawdown"], objective(cm["total_return"], cm["max_drawdown"],
                                                                 len(r.trades), c["min_trades"]), len(r.trades)


def walk_forward(*, data: MarketData, settings: Settings, limits: RiskLimits, base_policy: Policy,
                 setups: dict[str, SetupSchema], grid: list[Candidate], start_ms: int, end_ms: int,
                 train_bars: int, test_bars: int, cash: float, benchmark_symbol: str, min_trades: int = 8,
                 resume_after_kill: bool = True, workers: int = 1, fixed_choices: list[str] | None = None,
                 reset_peak_each_fold: bool = False, progress=None) -> WalkForwardResult:
    """`fixed_choices` replays a previous run's per-fold selections (for cost sensitivity runs)
    instead of re-selecting; it must come from a run with the same folds."""
    tf = data.tf_ms
    bpy = BARS_PER_YEAR.get(tf, 8760.0)
    _CTX.update(data=data, settings=settings, limits=limits, policy=base_policy, setups=setups, grid=grid,
                cash=cash, bpy=bpy, min_trades=min_trades)
    by_name = {c.name: c for c in grid}
    pool = mp.get_context("fork").Pool(workers) if workers > 1 and fixed_choices is None else None

    folds: list[Fold] = []
    ts_all: list[np.ndarray] = []
    eq_all: list[np.ndarray] = []
    trades: list = []
    counters: dict[str, int] = {}
    equity, peak, killed, kills, fees, funding = cash, cash, False, 0, 0.0, 0.0
    test_start = (start_ms // tf) * tf + train_bars * tf
    i = 0
    try:
        while test_start < end_ms:
            train_start = test_start - train_bars * tf
            test_end = min(end_ms, test_start + test_bars * tf)
            if fixed_choices is not None:
                chosen = fixed_choices[i] if i < len(fixed_choices) else "flat"
                best = (0.0, 0.0)
            else:
                jobs = [(k, train_start, test_start) for k in range(len(grid))]
                results = pool.map(_train_one, jobs) if pool else [_train_one(j) for j in jobs]
                k_best, ret, _dd, score, _n = max(results, key=lambda r: (r[3], -r[0]))
                chosen = grid[k_best].name if (score > 0 and ret > 0) else "flat"
                best = (score, ret)

            if reset_peak_each_fold or (killed and resume_after_kill):
                # a human reviewed the drawdown at the fold boundary and resumed with a fresh peak
                killed, peak = False, equity
            if chosen == "flat" or killed:
                n = int((test_end - test_start) // tf)
                ts = test_start + tf * np.arange(0, n + 1, dtype=np.int64)
                eq = np.full(n + 1, equity)
                fold_ret, n_tr = 0.0, 0
            else:
                cand = by_name[chosen]
                r = run_backtest(data=data, settings=settings, limits=limits, policy=cand.policy(base_policy),
                                 setups=setups, reflex=RuleReflex(cand.rule), start_ms=test_start, end_ms=test_end,
                                 cash=equity, peak_equity=peak)
                ts, eq = r.ts, r.equity
                fold_ret, n_tr = float(eq[-1] / eq[0] - 1.0), len(r.trades)
                trades.extend(r.trades)
                fees += r.fees
                funding += r.funding
                for k, v in r.counters.items():
                    counters[k] = counters.get(k, 0) + v
                peak = max(peak, r.peak_equity)
                if r.killed:
                    killed = True
                    kills += 1
            equity = float(eq[-1])
            ts_all.append(ts if not ts_all else ts[1:])
            eq_all.append(eq if not eq_all else eq[1:])
            folds.append(Fold(i, train_start, test_start, test_end, chosen, best[0], best[1], fold_ret, n_tr,
                              equity, killed))
            if progress:
                progress(folds[-1])
            test_start = test_end
            i += 1
    finally:
        if pool:
            pool.close()
            pool.join()

    ts = np.concatenate(ts_all) if ts_all else np.array([], dtype=np.int64)
    eq = np.concatenate(eq_all) if eq_all else np.array([], dtype=float)
    m = {**curve_metrics(ts, eq, bpy), **trade_metrics(trades)}
    bench = _benchmark(data, benchmark_symbol, ts, settings, bpy)
    return WalkForwardResult(folds=folds, ts=ts, equity=eq, trades=trades, metrics=m, benchmark=bench, kills=kills,
                             fees=fees, funding=funding, counters=counters)


def _benchmark(data: MarketData, symbol: str, ts: np.ndarray, settings: Settings, bpy: float) -> dict:
    """Buy at the open of the first OOS bar, pay taker fee and slippage once, hold to the end."""
    if len(ts) < 2 or symbol not in data.bars:
        return {"total_return": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "calmar": 0.0}
    idx = data.index[symbol]
    series = data.bars[symbol]
    tf = data.tf_ms
    first = idx.get(int(ts[0]))
    if first is None:
        return {"total_return": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "calmar": 0.0}
    closes, stamps, last = [], [], series[first].open
    for t in ts[1:]:
        j = idx.get(int(t) - tf)
        if j is not None:
            last = series[j].close
        closes.append(last)
        stamps.append(int(t))
    fees = settings.exchange.fees
    return buy_and_hold(np.array(stamps, dtype=np.int64), np.array(closes, dtype=float), series[first].open,
                        fees.taker + fees.slippage_bps / 1e4, bpy)

"""Performance metrics. Everything is computed from an equity curve sampled once per bar."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

YEAR_MS = 365.25 * 86_400_000


@dataclass(frozen=True)
class Metrics:
    start_ts: int
    end_ts: int
    total_return: float
    cagr: float
    max_drawdown: float
    sharpe: float
    calmar: float
    n_trades: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    profit_factor: float = 0.0
    fees: float = 0.0
    funding: float = 0.0
    exposure: float = 0.0     # share of bars with an open position

    def as_dict(self) -> dict:
        return asdict(self)


def curve_metrics(ts: np.ndarray, equity: np.ndarray, bars_per_year: float) -> dict:
    if len(equity) < 2 or equity[0] <= 0:
        return {"total_return": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "calmar": 0.0}
    total = float(equity[-1] / equity[0] - 1.0)
    years = max(1e-9, (int(ts[-1]) - int(ts[0])) / YEAR_MS)
    cagr = (float(equity[-1] / equity[0]) ** (1.0 / years) - 1.0) if equity[-1] > 0 else -1.0
    peak = np.maximum.accumulate(equity)
    mdd = float(np.max(1.0 - equity / peak))
    rets = np.diff(equity) / equity[:-1]
    sd = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
    sharpe = float(np.mean(rets) / sd * math.sqrt(bars_per_year)) if sd > 0 else 0.0
    calmar = cagr / mdd if mdd > 0 else 0.0
    return {"total_return": total, "cagr": cagr, "max_drawdown": mdd, "sharpe": sharpe, "calmar": calmar}


def trade_metrics(trades: list) -> dict:
    if not trades:
        return {"n_trades": 0, "win_rate": 0.0, "avg_r": 0.0, "profit_factor": 0.0}
    pnl = [t.pnl for t in trades]
    wins = [p for p in pnl if p > 0]
    losses = [-p for p in pnl if p < 0]
    return {
        "n_trades": len(trades),
        "win_rate": len(wins) / len(trades),
        "avg_r": float(np.mean([t.r_multiple for t in trades])),
        "profit_factor": (sum(wins) / sum(losses)) if losses else float("inf") if wins else 0.0,
    }


def buy_and_hold(ts: np.ndarray, closes: np.ndarray, entry_price: float, cost_frac: float,
                 bars_per_year: float) -> dict:
    """Hold one unit bought at `entry_price`, paying `cost_frac` once on entry. No leverage."""
    equity = closes / (entry_price * (1.0 + cost_frac))
    equity = np.concatenate([[1.0], equity])
    ts = np.concatenate([[ts[0] - (ts[1] - ts[0]) if len(ts) > 1 else ts[0]], ts])
    return curve_metrics(ts, equity, bars_per_year)

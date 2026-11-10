"""Event-driven backtest that reuses the live trading core unchanged.

Timeline for the bar that closes at time T (bar k):

1. marks move to bar k's close; the execution quote becomes bar k+1's OPEN;
2. `risk_tick` applies bar k's low/high to every open position (stops first), then the
   drawdown kill switch and the daily loss halt;
3. one hour of funding is charged;
4. `on_candle(T)` builds snapshots from bars closed at or before T and may place orders, which
   fill at bar k+1's open plus slippage and fees.

So a decision made on bar k's close can never be filled at a price from inside bar k, and the
snapshot can never contain bar k+1. Fees, slippage, impact and funding are always on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..alerts.base import Alerter
from ..config import RiskLimits, Settings, timeframe_ms
from ..execution.ledger import ClosedTrade
from ..execution.paper import PaperExchange
from ..policy.policy import Policy
from ..reflex.base import Reflex, ReflexRunner
from ..runtime.core import PriceRange, TradingCore
from ..setups.schema import SetupSchema
from ..state.engine import MarketStore, StateEngine
from ..storage.recorder import MemoryRecorder
from ..types import Bar, Instrument
from .cutoff import assert_backtest_allowed
from .metrics import Metrics, curve_metrics, trade_metrics

HALF_SPREAD_BPS = 1.5   # assumed half spread around the next open in backtests


class SilentAlerter(Alerter):
    """Counts alerts without logging: a backtest raises thousands of them."""

    def __init__(self):
        super().__init__(min_interval_s=0.0)
        self.counts: dict[str, int] = {}

    def alert(self, level: str, key: str, message: str, force: bool = False) -> bool:
        k = key.split(":")[0]
        self.counts[k] = self.counts.get(k, 0) + 1
        return True


@dataclass
class MarketData:
    """Static bar history shared by many backtest runs (feature cache included)."""

    store: MarketStore
    engine: StateEngine
    instruments: dict[str, Instrument]
    tf_ms: int
    index: dict[str, dict[int, int]] = field(default_factory=dict)      # symbol -> open ts -> position
    bars: dict[str, list[Bar]] = field(default_factory=dict)
    dollar_vol: dict[str, np.ndarray] = field(default_factory=dict)     # rolling 24-bar mean $ volume

    @classmethod
    def build(cls, bars: dict[str, list[Bar]], synthetic: dict[str, set[int]], instruments: dict[str, Instrument],
              settings: Settings, limits: RiskLimits) -> "MarketData":
        tf = timeframe_ms(settings.exchange.timeframe)
        store = MarketStore(tf)
        md = cls(store=store, engine=None, instruments=instruments, tf_ms=tf)  # type: ignore[arg-type]
        for sym, series in bars.items():
            syn = synthetic.get(sym, set())
            for b in series:
                store.add_bar(sym, b, synthetic=b.ts in syn)
            md.bars[sym] = series
            md.index[sym] = {b.ts: i for i, b in enumerate(series)}
            size = instruments[sym].contract_size if sym in instruments else 1.0
            dv = np.array([b.close * b.volume * size for b in series], dtype=float)
            csum = np.concatenate([[0.0], np.cumsum(dv)])
            n = np.arange(1, len(dv) + 1)
            w = np.minimum(n, 24)
            md.dollar_vol[sym] = (csum[n] - csum[n - w]) / w
        guards = settings.guards.model_copy(update={"require_book": False})
        md.engine = StateEngine(store, guards, limits, tuple(settings.context_symbols),
                                settings.reflex.snapshot_token_budget)
        return md


@dataclass
class BacktestResult:
    ts: np.ndarray
    equity: np.ndarray
    trades: list[ClosedTrade]
    counters: dict[str, int]
    alerts: dict[str, int]
    fees: float
    funding: float
    killed: bool
    final_cash: float
    peak_equity: float
    exposure: float
    decisions: list[dict] = field(default_factory=list)

    def metrics(self, bars_per_year: float) -> Metrics:
        cm = curve_metrics(self.ts, self.equity, bars_per_year)
        tm = trade_metrics(self.trades)
        return Metrics(start_ts=int(self.ts[0]) if len(self.ts) else 0, end_ts=int(self.ts[-1]) if len(self.ts) else 0,
                       fees=self.fees, funding=self.funding, exposure=self.exposure, **cm, **tm)


def run_backtest(*, data: MarketData, settings: Settings, limits: RiskLimits, policy: Policy,
                 setups: dict[str, SetupSchema], reflex: Reflex, start_ms: int, end_ms: int, cash: float,
                 peak_equity: float | None = None, kill_latched: bool = False, keep_decisions: bool = False,
                 reflex_model: str | None = None, flatten_at_end: bool = True) -> BacktestResult:
    assert_backtest_allowed(reflex.name, reflex_model, start_ms)
    tf = data.tf_ms
    start_ms = (start_ms // tf) * tf
    fees = settings.exchange.fees
    recorder = MemoryRecorder(keep_decisions=keep_decisions)
    if kill_latched:
        recorder.set("kill", True)
    alerts = SilentAlerter()
    exchange = PaperExchange(cash, start_ms, data.instruments, fees)
    if peak_equity is not None:
        exchange.ledger.peak_equity = max(peak_equity, cash)
    runner = ReflexRunner(reflex, settings.reflex.deadline_ms, inline=True)
    core = TradingCore(settings=settings, limits=limits, policy=policy, setups=setups, instruments=data.instruments,
                       engine=data.engine, runner=runner, exchange=exchange, recorder=recorder, alerts=alerts)
    symbols = [s for s in core.by_symbol if s in data.bars]
    hs = HALF_SPREAD_BPS / 1e4
    ts_out: list[int] = [start_ms]
    eq_out: list[float] = [cash]
    exposed = 0
    steps = 0

    t = start_ms + tf
    while t <= end_ms:
        ranges: dict[str, PriceRange] = {}
        for sym in symbols:
            idx = data.index[sym].get(t - tf)
            if idx is None:
                continue
            series = data.bars[sym]
            bar = series[idx]
            nxt = series[idx + 1].open if idx + 1 < len(series) and series[idx + 1].ts == t else bar.close
            exchange.mark(sym, bar.close)
            exchange.set_quote(sym, nxt * (1 - hs), nxt * (1 + hs), t, float(data.dollar_vol[sym][idx]) or None)
            ranges[sym] = PriceRange(bar.open, bar.low, bar.high)
        if ranges:
            core.risk_tick(t, ranges)
            for sym in list(exchange.ledger.positions):
                f = data.store.funding(sym, t)
                if f is not None:
                    exchange.ledger.apply_funding(sym, f[1])
                else:
                    exchange.ledger.apply_funding(sym, fees.funding_bps_per_hour / 1e4, pure_cost=True)
            if not (core.kill and not exchange.ledger.positions):
                core.on_candle(t)
            steps += 1
            if exchange.ledger.positions:
                exposed += 1
        acct = exchange.account(t)
        ts_out.append(t)
        eq_out.append(acct.equity)
        t += tf

    if flatten_at_end and exchange.ledger.positions:
        core.flatten(end_ms, "backtest_end")
        eq_out[-1] = exchange.account(end_ms).equity

    return BacktestResult(
        ts=np.array(ts_out, dtype=np.int64), equity=np.array(eq_out, dtype=float), trades=list(exchange.ledger.closed),
        counters=dict(core.counters), alerts=dict(alerts.counts), fees=exchange.ledger.fees_total,
        funding=exchange.ledger.funding_total, killed=core.kill, final_cash=exchange.ledger.cash,
        peak_equity=exchange.ledger.peak_equity, exposure=exposed / steps if steps else 0.0,
        decisions=recorder.decisions)

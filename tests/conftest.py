"""Shared test builders. Everything is synthetic and offline: no network, no keys."""
from __future__ import annotations

import math
from datetime import date

import pytest

from tradebot.alerts.base import Alerter
from tradebot.config import DataGuards, RiskLimits, Settings
from tradebot.execution.paper import PaperExchange
from tradebot.policy.policy import Policy
from tradebot.reflex.base import ReflexRunner
from tradebot.reflex.schema import ReflexOutput
from tradebot.runtime.core import TradingCore
from tradebot.setups.schema import Invalidation, SetupReflexSpec, SetupSchema
from tradebot.state.engine import MarketStore, StateEngine
from tradebot.storage.recorder import MemoryRecorder
from tradebot.types import Bar, Book, Direction, Instrument, Regime, RiskState

TF = 3_600_000
T0 = 1_750_000_000_000 // TF * TF          # an aligned hour in 2025
SYM = "BTC/USD:USD"
ETH = "ETH/USD:USD"


def make_bars(n: int, start: int = T0, price: float = 100.0, drift: float = 0.0, wiggle: float = 0.002,
              volume: float = 5e7) -> list[Bar]:
    """n contiguous hourly bars. `drift` is the per-bar return, `wiggle` a deterministic oscillation."""
    out, p = [], price
    for i in range(n):
        o = p
        c = o * (1 + drift + wiggle * math.sin(i / 3.0))
        hi, lo = max(o, c) * 1.001, min(o, c) * 0.999
        out.append(Bar(start + i * TF, o, hi, lo, c, volume, TF))
        p = c
    return out


def inst(symbol: str = SYM, contract_size: float = 0.01, tick: float = 0.01) -> Instrument:
    return Instrument(symbol=symbol, base=symbol.split("/")[0], exchange_id=f"X-{symbol}", contract_size=contract_size,
                      price_tick=tick, qty_step=1.0, min_qty=1.0, max_qty=1e9)


def out(direction="long", conf=0.9, quality=3, regime="trending", risk="safe", toxic=False) -> ReflexOutput:
    return ReflexOutput(regime=Regime(regime), direction=Direction(direction), direction_confidence=conf,
                        toxic_flow=toxic, setup_quality=quality, risk_state=RiskState(risk))


def make_setup(symbol: str = SYM, bias: str = "neutral", status: str = "active", **kw) -> SetupSchema:
    b = Direction(bias)
    inv = Invalidation(price_below=kw.pop("price_below", 1.0 if b is Direction.LONG else None),
                       price_above=kw.pop("price_above", 1e12 if b is Direction.SHORT else None))
    return SetupSchema(
        id=kw.pop("id", f"t-{symbol.split('/')[0].lower()}"), symbol=symbol, base=symbol.split("/")[0], status=status,
        bias=b, quality_prior=kw.pop("quality_prior", 2), thesis="test", bear_case="test", invalidation=inv,
        reflex=kw.pop("reflex", SetupReflexSpec(allowed_directions=[Direction.LONG, Direction.SHORT])),
        created=kw.pop("created", date(2020, 1, 1)), expires=kw.pop("expires", date(2100, 1, 1)), **kw)


@pytest.fixture
def limits() -> RiskLimits:
    return RiskLimits()


@pytest.fixture
def settings() -> Settings:
    return Settings(guards=DataGuards(require_book=False), context_symbols=[SYM, ETH])


@pytest.fixture
def policy() -> Policy:
    return Policy(version=1)


def fresh_book(mid: float, ts: int, spread_bps: float = 1.0, size: float = 50.0) -> Book:
    h = mid * spread_bps / 2e4
    return Book(ts=ts, bids=((mid - h, size), (mid - 2 * h, size)), asks=((mid + h, size), (mid + 2 * h, size)))


class Rig:
    """A complete core wired to a paper exchange and synthetic data."""

    def __init__(self, settings: Settings, limits: RiskLimits, policy: Policy, reflex, n_bars: int = 260,
                 cash: float = 100_000.0, drift: float = 0.0, setups=None, recorder=None, deadline_ms: int = 2000,
                 inline: bool = True, contract_size: float = 0.01):
        self.settings, self.limits, self.policy = settings, limits, policy
        self.instruments = {SYM: inst(SYM, contract_size), ETH: inst(ETH, 0.1)}
        self.store = MarketStore(TF)
        self.bars = {SYM: make_bars(n_bars, drift=drift), ETH: make_bars(n_bars, price=50.0, drift=drift)}
        for s, bs in self.bars.items():
            self.store.add_bars(s, bs)
        self.engine = StateEngine(self.store, settings.guards, limits, (SYM, ETH))
        self.now = T0 + n_bars * TF                       # close time of the last bar
        self.recorder = recorder or MemoryRecorder()
        self.alerts = Alerter(min_interval_s=0.0)
        self.exchange = PaperExchange(cash, self.now, self.instruments, settings.exchange.fees)
        self.runner = ReflexRunner(reflex, deadline_ms, inline=inline)
        self.setups = setups if setups is not None else {"t-btc": make_setup(SYM)}
        self.core = TradingCore(settings=settings, limits=limits, policy=policy, setups=self.setups,
                                instruments=self.instruments, engine=self.engine, runner=self.runner,
                                exchange=self.exchange, recorder=self.recorder, alerts=self.alerts)
        self.quote(self.price)

    @property
    def price(self) -> float:
        return self.bars[SYM][-1].close

    def quote(self, px: float, symbol: str = SYM) -> None:
        self.exchange.mark(symbol, px)
        self.exchange.set_quote(symbol, px * 0.9999, px * 1.0001, self.now, bar_dollar_volume=5e7)

    def add_bar(self, close: float, symbol: str = SYM, high: float | None = None, low: float | None = None) -> Bar:
        last = self.bars[symbol][-1]
        b = Bar(last.ts + TF, last.close, high or max(last.close, close), low or min(last.close, close), close,
                5e7, TF)
        self.bars[symbol].append(b)
        self.store.add_bar(symbol, b)
        return b

    def step(self, close: float | None = None) -> None:
        """Advance one hour on every symbol, keep prices flat unless told otherwise, run tick + candle."""
        from tradebot.runtime.core import PriceRange
        ranges = {}
        for s in self.bars:
            c = close if (close is not None and s == SYM) else self.bars[s][-1].close
            b = self.add_bar(c, s)
            ranges[s] = PriceRange(b.open, b.low, b.high)
        self.now += TF
        for s in self.bars:
            self.quote(self.bars[s][-1].close, s)
        self.core.risk_tick(self.now, ranges)
        self.core.on_candle(self.now)


@pytest.fixture
def rig_factory(settings, limits, policy):
    def build(reflex, **kw):
        return Rig(kw.pop("settings", settings), kw.pop("limits", limits), kw.pop("policy", policy), reflex, **kw)
    return build

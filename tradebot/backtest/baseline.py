"""Data preparation and the neutral technical baseline used for code-rule backtests.

Research theses cannot be backtested honestly: a thesis written today uses today's
information, so replaying it on the past is look-ahead by construction. What CAN be tested
on history is the machinery (state, gate, sizing, risk, costs) driven by code rules with a
neutral setup per symbol. That is what this module builds.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import yaml

from ..config import RiskLimits, Settings, timeframe_ms
from ..data.feed import CcxtFeed
from ..data.history import load_history
from ..setups.schema import Invalidation, SetupReflexSpec, SetupSchema
from ..types import Bar, Direction, Instrument, Regime
from .engine import MarketData


def load_instruments(path: str | Path) -> dict[str, Instrument]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return {sym: Instrument(symbol=sym, **spec) for sym, spec in raw["instruments"].items()}


def save_instruments(path: str | Path, instruments: dict[str, Instrument], source: str) -> None:
    body = {
        "source": source,
        "fetched": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "instruments": {
            s: {"base": i.base, "exchange_id": i.exchange_id, "contract_size": i.contract_size,
                "price_tick": i.price_tick, "qty_step": i.qty_step, "min_qty": i.min_qty, "max_qty": i.max_qty}
            for s, i in sorted(instruments.items())},
    }
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(body, fh, sort_keys=False)


def technical_setups(symbols: list[str], allow_short: bool = True) -> dict[str, SetupSchema]:
    """One neutral, thesis-free setup per symbol. Not a research finalist."""
    dirs = [Direction.LONG, Direction.SHORT] if allow_short else [Direction.LONG]
    out = {}
    for sym in symbols:
        base = sym.split("/")[0]
        sid = f"tech-{base.lower()}"
        out[sid] = SetupSchema(
            id=sid, symbol=sym, base=base, status="active", bias=Direction.NEUTRAL, quality_prior=0,
            thesis="Neutral technical baseline: no fundamental thesis.", bear_case="No edge is assumed.",
            edge_type="technical", invalidation=Invalidation(),
            reflex=SetupReflexSpec(allowed_directions=dirs, allowed_regimes=[Regime.TRENDING, Regime.MEAN_REVERTING,
                                                                             Regime.HIGH_VOL]),
            created=date(2000, 1, 1), expires=date(2100, 1, 1))
    return out


def spot_symbol(perp_symbol: str) -> str:
    return perp_symbol.split(":")[0]


def load_market_data(*, settings: Settings, limits: RiskLimits, instruments: dict[str, Instrument],
                     symbols: list[str], start_ms: int, end_ms: int, cache_dir: str | Path,
                     feed: CcxtFeed | None = None, spot_proxy: bool = False) -> MarketData:
    """Load bars for `symbols` (plus the context symbols) into a reusable MarketData.

    `spot_proxy=True` loads the spot pair's history under the perp's symbol and contract size.
    It exists because the perp-style contracts only list from mid-2025; spot goes back years.
    Spot volume (base units) is converted to contracts so dollar-volume maths stays correct.
    """
    tf = timeframe_ms(settings.exchange.timeframe)
    wanted = list(dict.fromkeys(list(symbols) + list(settings.context_symbols)))
    bars: dict[str, list[Bar]] = {}
    synthetic: dict[str, set[int]] = {}
    for sym in wanted:
        src = spot_symbol(sym) if spot_proxy else sym
        series, syn = load_history(feed, cache_dir, settings.exchange.id, src, settings.exchange.timeframe, tf,
                                   start_ms, end_ms)
        if not series:
            continue
        if spot_proxy and sym in instruments:
            size = instruments[sym].contract_size
            series = [Bar(b.ts, b.open, b.high, b.low, b.close, b.volume / size, tf) for b in series]
        bars[sym] = series
        synthetic[sym] = syn
    usable = {s: i for s, i in instruments.items() if s in bars}
    return MarketData.build(bars, synthetic, usable, settings, limits)

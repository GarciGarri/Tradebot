"""Position sizing: fractional Kelly from a calibrated probability, capped by hard limits.

For a trade that loses `loss` per dollar of notional when stopped and wins `win` when the
target is hit (both net of round-trip costs), the payoff ratio is b = win / loss and full
Kelly risks f* = p - (1 - p) / b of equity. The code uses at most a quarter of that, and
never more than the per-trade risk limit. Quantities are rounded DOWN to whole contracts, so
the realised risk can only be smaller than the budget.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import ABS_MAX_KELLY_FRACTION, FeeModel, RiskLimits
from ..types import Instrument


@dataclass(frozen=True)
class SizeResult:
    qty: float
    p: float
    payoff_ratio: float
    kelly_full: float
    risk_fraction: float          # share of equity at risk after all caps and rounding
    notional: float
    reasons: tuple[str, ...]      # caps that bound the size, or why it is zero


def round_trip_cost_frac(fees: FeeModel, notional: float, bar_dollar_volume: float | None,
                         hold_bars: int = 24) -> float:
    """Expected cost of entering and leaving, as a fraction of notional."""
    impact = 0.0
    if bar_dollar_volume and bar_dollar_volume > 0 and notional > 0:
        impact = fees.impact_coeff * math.sqrt(notional / bar_dollar_volume)
    per_side = fees.taker + fees.slippage_bps / 1e4 + impact
    funding = fees.funding_bps_per_hour / 1e4 * hold_bars
    return 2.0 * per_side + funding


def kelly_full(p: float, b: float) -> float:
    if b <= 0:
        return -1.0
    return p - (1.0 - p) / b


def size_position(*, p: float, stop_frac: float, take_profit_r: float, equity: float, price: float,
                  instrument: Instrument, limits: RiskLimits, fees: FeeModel, kelly_fraction: float,
                  gross_notional: float, bar_dollar_volume: float | None, hold_bars: int = 24,
                  risk_scale: float = 1.0) -> SizeResult:
    """`risk_scale` (0-1] can only shrink the per-trade risk budget, never enlarge it."""
    reasons: list[str] = []
    risk_scale = min(1.0, max(0.0, risk_scale))
    kf = min(kelly_fraction, ABS_MAX_KELLY_FRACTION)
    if equity <= 0 or price <= 0 or stop_frac <= 0:
        return SizeResult(0.0, p, 0.0, 0.0, 0.0, 0.0, ("bad_inputs",))
    unit = instrument.contract_size * price                       # notional of one contract
    cost = round_trip_cost_frac(fees, unit, bar_dollar_volume, hold_bars)
    win = take_profit_r * stop_frac - cost
    loss = stop_frac + cost
    if win <= 0:
        return SizeResult(0.0, p, 0.0, 0.0, 0.0, 0.0, ("costs_exceed_target",))
    b = win / loss
    f_star = kelly_full(p, b)
    if f_star <= 0:
        return SizeResult(0.0, p, b, f_star, 0.0, 0.0, ("no_edge_after_costs",))
    risk_frac = kf * f_star
    budget = limits.risk_per_trade * risk_scale
    if risk_scale < 1.0:
        reasons.append("drawdown_caution")
    if risk_frac > budget:
        risk_frac = budget
        reasons.append("capped_risk_per_trade")
    notional = risk_frac * equity / loss
    room = limits.max_leverage * equity - gross_notional
    if notional > room:
        notional = max(0.0, room)
        reasons.append("capped_leverage")
    if bar_dollar_volume is not None:
        liq = limits.max_participation * bar_dollar_volume
        if notional > liq:
            notional = liq
            reasons.append("capped_liquidity")
    qty = instrument.round_qty_down(notional / unit)
    qty = min(qty, instrument.max_qty)
    if qty <= 0:
        reasons.append("below_one_contract")
        return SizeResult(0.0, p, b, f_star, 0.0, 0.0, tuple(reasons))
    final_notional = qty * unit
    return SizeResult(qty, p, b, f_star, final_notional * loss / equity, final_notional, tuple(reasons))

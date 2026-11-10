"""The hard risk layer.

Nothing in this file calls a model, reads a model output, or can be configured by one.
`check_order` runs immediately before every order, and `account_action` runs on every tick,
whether or not the REFLEX answered. Stopping unsafe risk therefore never waits on a model.

Limits come from `RiskLimits` (config/risk.yaml, human-edited, ceiling-validated).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from ..config import FeeModel, RiskLimits
from ..types import AccountState, Instrument, OrderIntent, Side


class AccountAction(str, Enum):
    NONE = "none"
    DAILY_HALT = "daily_halt"   # flatten and stop new risk until the next UTC day
    KILL = "kill"               # flatten, latch, require a human to resume


@dataclass(frozen=True)
class Halts:
    """Flags owned by the runtime, all set by code."""

    kill: bool = False          # latched kill switch
    daily_halt: bool = False    # daily loss limit hit today
    paused: bool = False        # REFLEX confidence floor / crisis pause
    reconcile_block: bool = False  # local and exchange state disagree, or an order is unresolved


@dataclass(frozen=True)
class RiskVerdict:
    ok: bool
    violations: tuple[str, ...]


class RiskManager:
    def __init__(self, limits: RiskLimits, fees: FeeModel):
        self.limits = limits
        self.fees = fees

    # ---- account level ------------------------------------------------------
    def account_action(self, account: AccountState) -> AccountAction:
        if account.equity <= 0 or account.drawdown >= self.limits.kill_switch_drawdown:
            return AccountAction.KILL
        if -account.day_pnl_frac >= self.limits.max_daily_loss:
            return AccountAction.DAILY_HALT
        return AccountAction.NONE

    # ---- order level --------------------------------------------------------
    def worst_case_loss(self, intent: OrderIntent, instrument: Instrument) -> float:
        """Loss if the stop is hit, plus fees and fixed slippage on both sides."""
        if intent.stop_price is None or intent.ref_price is None:
            return math.inf
        notional = instrument.notional(intent.qty, intent.ref_price)
        stop_loss = intent.qty * instrument.contract_size * abs(intent.ref_price - intent.stop_price)
        costs = notional * 2.0 * (self.fees.taker + self.fees.slippage_bps / 1e4)
        return stop_loss + costs

    def check_order(self, intent: OrderIntent, account: AccountState, instrument: Instrument,
                    mark_price: float | None, bar_dollar_volume: float | None, data_ok: bool,
                    halts: Halts) -> RiskVerdict:
        v: list[str] = []
        lim = self.limits

        if not (intent.qty > 0 and math.isfinite(intent.qty)):
            v.append("qty_not_positive")
        elif abs(intent.qty / instrument.qty_step - round(intent.qty / instrument.qty_step)) > 1e-9:
            v.append("qty_not_on_step")
        if intent.symbol != instrument.symbol:
            v.append("instrument_mismatch")

        pos = account.positions.get(intent.symbol)

        if intent.reduce_only:
            # Closing risk is always allowed, as long as it really is closing risk.
            if pos is None:
                v.append("reduce_only_without_position")
            else:
                if intent.side is pos.side:
                    v.append("reduce_only_wrong_side")
                if intent.qty > abs(pos.qty) + 1e-9:
                    v.append("reduce_only_exceeds_position")
            return RiskVerdict(not v, tuple(v))

        # ---- everything below adds risk ----
        if halts.kill:
            v.append("kill_switch_active")
        if halts.daily_halt:
            v.append("daily_loss_halt")
        if halts.paused:
            v.append("new_risk_paused")
        if halts.reconcile_block:
            v.append("reconciliation_pending")
        if not data_ok:
            v.append("market_data_not_ok")
        if account.equity <= 0:
            v.append("no_equity")
            return RiskVerdict(False, tuple(v))
        if account.drawdown >= lim.kill_switch_drawdown:
            v.append("drawdown_at_kill_level")
        if -account.day_pnl_frac >= lim.max_daily_loss:
            v.append("daily_loss_limit_reached")

        if intent.qty < instrument.min_qty:
            v.append("below_min_qty")
        if intent.qty > instrument.max_qty:
            v.append("above_max_qty")

        if pos is not None:
            v.append("position_already_open")          # one position per symbol, no pyramiding
        elif len(account.positions) >= lim.max_open_positions:
            v.append("max_open_positions")

        ref = intent.ref_price
        if ref is None or not math.isfinite(ref) or ref <= 0:
            v.append("no_reference_price")
        elif mark_price is None or not math.isfinite(mark_price) or mark_price <= 0:
            v.append("no_mark_price")
        elif abs(ref / mark_price - 1.0) > lim.max_price_deviation:
            v.append("reference_price_far_from_mark")

        if intent.stop_price is None:
            v.append("entry_without_stop")
        elif ref is not None and ref > 0:
            wrong_side = intent.stop_price >= ref if intent.side is Side.BUY else intent.stop_price <= ref
            if wrong_side:
                v.append("stop_on_wrong_side")

        if ref is not None and ref > 0 and math.isfinite(ref) and intent.qty > 0:
            notional = instrument.notional(intent.qty, ref)
            loss = self.worst_case_loss(intent, instrument)
            if loss > lim.risk_per_trade * account.equity * (1 + 1e-9):
                v.append("risk_per_trade_exceeded")
            if (account.gross_notional + notional) > lim.max_leverage * account.equity * (1 + 1e-9):
                v.append("max_leverage_exceeded")
            if bar_dollar_volume is None or bar_dollar_volume <= 0:
                v.append("unknown_liquidity")
            elif notional > lim.max_participation * bar_dollar_volume * (1 + 1e-9):
                v.append("liquidity_cap_exceeded")

        return RiskVerdict(not v, tuple(v))

"""Hard risk layer: every limit, and proof that nothing model-shaped can reach it."""
from __future__ import annotations

import ast
import inspect

import pytest
from pydantic import ValidationError

import tradebot.risk.manager as risk_module
from tradebot.config import FeeModel, RiskLimits, Settings, live_trading_approved
from tradebot.risk.manager import AccountAction, Halts, RiskManager
from tradebot.types import AccountState, Mode, OrderIntent, Position, Side

from .conftest import ETH, SYM, inst

LIM = RiskLimits()
RM = RiskManager(LIM, FeeModel())
I = inst(SYM, 0.01)          # at price 100: one contract = $1 notional
NO = Halts()


def acct(equity=10_000.0, peak=None, day=None, positions=None, gross=0.0) -> AccountState:
    return AccountState(ts=0, cash=equity, equity=equity, peak_equity=peak or equity, day_start_equity=day or equity,
                        positions=positions or {}, gross_notional=gross)


def entry(qty=1000.0, side=Side.BUY, ref=100.0, stop=98.0, **kw) -> OrderIntent:
    return OrderIntent(client_id="e1", symbol=SYM, side=side, qty=qty, reduce_only=False, reason="entry", ts=0,
                       stop_price=stop, ref_price=ref, **kw)


def check(intent, account=None, mark=100.0, vol=1e9, data_ok=True, halts=NO, instrument=I):
    return RM.check_order(intent, account or acct(), instrument, mark, vol, data_ok, halts)


def pos(symbol=SYM, qty=10.0) -> Position:
    return Position(symbol, qty, 100.0, 98.0, None, 0, "s")


def test_a_sane_entry_passes():
    v = check(entry())        # $1,000 notional, $20 stop loss + $2.6 costs on $10k equity
    assert v.ok, v.violations


def test_risk_per_trade_limit():
    # 9,000 contracts = $9,000 notional, 2% stop -> $180 + costs > 2% of $10k
    v = check(entry(qty=9000.0))
    assert not v.ok and "risk_per_trade_exceeded" in v.violations
    assert RM.worst_case_loss(entry(qty=9000.0), I) == pytest.approx(9000 * 0.01 * 2 + 9000 * 2 * (0.001 + 0.0003))


def test_leverage_limit():
    v = check(entry(qty=5000.0, stop=99.8), account=acct(gross=19_000.0))
    assert "max_leverage_exceeded" in v.violations


def test_max_open_positions_and_no_pyramiding():
    three = {f"S{i}": pos(f"S{i}") for i in range(3)}
    assert "max_open_positions" in check(entry(), account=acct(positions=three)).violations
    assert "position_already_open" in check(entry(), account=acct(positions={SYM: pos()})).violations


def test_entry_needs_a_stop_on_the_right_side():
    assert "entry_without_stop" in check(entry(stop=None)).violations
    assert "stop_on_wrong_side" in check(entry(stop=101.0)).violations
    assert "stop_on_wrong_side" in check(entry(side=Side.SELL, stop=99.0)).violations
    assert check(entry(side=Side.SELL, stop=102.0)).ok


@pytest.mark.parametrize("halts,code", [(Halts(kill=True), "kill_switch_active"),
                                        (Halts(daily_halt=True), "daily_loss_halt"),
                                        (Halts(paused=True), "new_risk_paused"),
                                        (Halts(reconcile_block=True), "reconciliation_pending")])
def test_halts_block_new_risk(halts, code):
    assert code in check(entry(), halts=halts).violations


def test_bad_data_blocks_new_risk():
    assert "market_data_not_ok" in check(entry(), data_ok=False).violations
    assert "no_mark_price" in check(entry(), mark=None).violations
    assert "reference_price_far_from_mark" in check(entry(), mark=110.0).violations
    assert "unknown_liquidity" in check(entry(), vol=None).violations
    assert "liquidity_cap_exceeded" in check(entry(), vol=10_000.0).violations   # 2% of $10k = $200 < $1,000


def test_quantity_sanity():
    assert "qty_not_positive" in check(entry(qty=0.0)).violations
    assert "qty_not_on_step" in check(entry(qty=1.5)).violations
    from tradebot.types import Instrument
    capped = Instrument(symbol=SYM, base="BTC", exchange_id="X", contract_size=0.01, price_tick=0.01, max_qty=5000.0)
    assert "above_max_qty" in check(entry(qty=6000.0, stop=99.9), account=acct(equity=1e6), instrument=capped).violations
    assert "instrument_mismatch" in check(entry(), instrument=inst(ETH, 0.1)).violations


def test_drawdown_and_daily_loss_block_even_before_the_latch_is_set():
    assert "drawdown_at_kill_level" in check(entry(qty=100.0), account=acct(equity=8500.0, peak=10_000.0)).violations
    assert "daily_loss_limit_reached" in check(entry(qty=100.0), account=acct(equity=9500.0, day=10_000.0)).violations


def test_reducing_risk_is_always_allowed_but_must_really_reduce():
    a = acct(positions={SYM: pos(qty=10.0)})
    close = OrderIntent("c1", SYM, Side.SELL, 10.0, True, "stop", 0)
    everything_halted = Halts(kill=True, daily_halt=True, paused=True, reconcile_block=True)
    assert RM.check_order(close, a, I, None, None, False, everything_halted).ok
    assert "reduce_only_wrong_side" in check(OrderIntent("c2", SYM, Side.BUY, 10.0, True, "x", 0), account=a).violations
    assert "reduce_only_exceeds_position" in check(OrderIntent("c3", SYM, Side.SELL, 11.0, True, "x", 0), account=a).violations
    assert "reduce_only_without_position" in check(close, account=acct()).violations


def test_account_actions():
    assert RM.account_action(acct()) is AccountAction.NONE
    assert RM.account_action(acct(equity=9499.0, day=10_000.0)) is AccountAction.DAILY_HALT
    assert RM.account_action(acct(equity=8500.0, peak=10_000.0, day=8500.0)) is AccountAction.KILL
    assert RM.account_action(acct(equity=8501.0, peak=10_000.0, day=8501.0)) is AccountAction.NONE
    assert RM.account_action(acct(equity=0.0, peak=10_000.0)) is AccountAction.KILL


def test_limits_have_hard_ceilings_and_reject_unknown_keys():
    for bad in (dict(risk_per_trade=0.06), dict(max_daily_loss=0.2), dict(max_leverage=5.0),
                dict(kill_switch_drawdown=0.5), dict(max_open_positions=50), dict(risk_per_trade=0.0),
                dict(max_daily_loss=0.2, kill_switch_drawdown=0.15), dict(model_override=True)):
        with pytest.raises(ValidationError):
            RiskLimits(**bad)
    with pytest.raises(ValidationError):                       # frozen: nothing can mutate limits at runtime
        LIM.max_leverage = 10.0


def test_risk_layer_has_no_path_to_a_model():
    """The risk module must not import the reflex, the policy, or any network/model library."""
    tree = ast.parse(inspect.getsource(risk_module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert not [m for m in imported if any(x in m for x in ("reflex", "policy", "anthropic", "ccxt", "urllib"))]
    params = inspect.signature(RiskManager.check_order).parameters
    assert not [p for p in params if "decision" in p or "reflex" in p or "model" in p or "confidence" in p]


def test_live_gate_needs_mode_ack_and_keys():
    live = Settings(mode=Mode.LIVE)
    full = {"TRADEBOT_LIVE_ACK": "I-APPROVE-LIVE-TRADING", "EXCHANGE_API_KEY": "k", "EXCHANGE_API_SECRET": "s"}
    assert live_trading_approved(live, full)
    assert not live_trading_approved(Settings(mode=Mode.PAPER), full)
    assert not live_trading_approved(Settings(mode=Mode.PREVIEW), full)
    for missing in full:
        assert not live_trading_approved(live, {k: v for k, v in full.items() if k != missing})
    assert not live_trading_approved(live, {**full, "TRADEBOT_LIVE_ACK": "yes"})

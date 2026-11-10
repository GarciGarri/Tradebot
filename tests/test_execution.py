"""Paper exchange, ledger, order manager: fees, slippage, failed orders, reconciliation."""
from __future__ import annotations

import pytest

from tradebot.alerts.base import Alerter, telegram_sender_from_env
from tradebot.config import FeeModel, RiskLimits
from tradebot.execution.orders import OrderManager
from tradebot.execution.paper import PaperExchange
from tradebot.risk.manager import Halts, RiskManager
from tradebot.storage.recorder import MemoryRecorder, SqliteRecorder
from tradebot.types import OrderIntent, OrderStatus, Side

from .conftest import ETH, SYM, inst

FEES = FeeModel(taker=0.001, slippage_bps=5.0, impact_coeff=0.0, funding_bps_per_hour=1.0)
INSTS = {SYM: inst(SYM, 1.0), ETH: inst(ETH, 1.0)}


def venue(cash=10_000.0) -> PaperExchange:
    ex = PaperExchange(cash, 0, INSTS, FEES)
    ex.set_quote(SYM, 99.9, 100.1, 0, bar_dollar_volume=1e9)
    ex.mark(SYM, 100.0)
    return ex


def buy(cid="e1", qty=10.0, stop=98.0, **kw) -> OrderIntent:
    return OrderIntent(cid, SYM, Side.BUY, qty, False, "entry", 0, setup_id="s", stop_price=stop, ref_price=100.0, **kw)


def sell_close(cid="c1", qty=10.0, reason="stop") -> OrderIntent:
    return OrderIntent(cid, SYM, Side.SELL, qty, True, reason, 0)


def manager(ex, attempts=3):
    rec, al = MemoryRecorder(), Alerter(0.0)
    return OrderManager(ex, RiskManager(RiskLimits(), FEES), rec, al, resolve_attempts=attempts), rec, al


def run(om, ex, intent, **kw):
    return om.execute(intent, ex.account(0), INSTS[intent.symbol], 100.0, 1e9, True, kw.pop("halts", Halts()), **kw)


# ---- fills, fees, slippage ------------------------------------------------------------------
def test_buy_fills_at_ask_plus_slippage_and_pays_fee():
    ex = venue()
    res = ex.submit(buy())
    f = res.fills[0]
    assert res.status is OrderStatus.FILLED
    assert f.price == pytest.approx(100.1 * 1.0005)
    assert f.fee == pytest.approx(10 * f.price * 0.001)
    assert ex.ledger.cash == pytest.approx(10_000 - f.fee)


def test_round_trip_at_flat_price_loses_exactly_the_costs():
    ex = venue()
    ex.submit(buy())
    ex.submit(sell_close())
    entry, exit_ = 100.1 * 1.0005, 99.9 * 0.9995
    expected = 10 * (exit_ - entry) - 10 * entry * 0.001 - 10 * exit_ * 0.001
    assert ex.ledger.cash - 10_000 == pytest.approx(expected)
    assert expected < 0 and not ex.ledger.positions
    t = ex.ledger.closed[0]
    assert t.pnl == pytest.approx(expected) and t.fees == pytest.approx(10 * entry * 0.001 + 10 * exit_ * 0.001)
    assert t.r_multiple == pytest.approx(expected / (10 * (entry - 98.0)))


def test_market_impact_grows_with_order_size():
    fees = FeeModel(taker=0.0, slippage_bps=0.0, impact_coeff=0.1, funding_bps_per_hour=0.0)
    ex = PaperExchange(1e9, 0, INSTS, fees)
    ex.set_quote(SYM, 100.0, 100.0, 0, bar_dollar_volume=1_000_000.0)
    small = ex.submit(buy("a", qty=1.0)).fills[0].price
    ex.submit(sell_close("a2", qty=1.0))
    large = ex.submit(buy("b", qty=1000.0)).fills[0].price
    assert large > small > 100.0
    assert large == pytest.approx(100.0 * (1 + 0.1 * (100_000 / 1_000_000) ** 0.5))


def test_short_pnl_and_funding():
    ex = venue()
    ex.submit(OrderIntent("s1", SYM, Side.SELL, 10.0, False, "entry", 0, stop_price=102.0, ref_price=100.0))
    ex.mark(SYM, 90.0)
    st = ex.account(0)
    assert st.equity - st.cash == pytest.approx(10 * (99.9 * 0.9995 - 90.0))
    cash = ex.ledger.cash
    paid = ex.ledger.apply_funding(SYM, 0.0001)                  # positive rate: shorts RECEIVE
    assert paid == pytest.approx(-10 * 90.0 * 0.0001) and ex.ledger.cash == pytest.approx(cash - paid)
    paid2 = ex.ledger.apply_funding(SYM, 0.0001, pure_cost=True)  # conservative mode always charges
    assert paid2 > 0


def test_equity_peak_drawdown_and_day_roll():
    ex = venue()
    ex.submit(buy(qty=50.0))
    ex.mark(SYM, 110.0)
    a = ex.account(1000)
    assert a.equity > 10_000 and a.peak_equity == a.equity and a.drawdown == 0
    ex.mark(SYM, 95.0)
    b = ex.account(2000)
    assert b.peak_equity == a.equity and b.drawdown == pytest.approx(1 - b.equity / a.equity)
    assert b.day_pnl_frac == pytest.approx(b.equity / 10_000 - 1)
    c = ex.account(86_400_000 + 1)                                # next UTC day
    assert c.day_start_equity == pytest.approx(c.equity) and c.day_pnl_frac == 0
    assert c.leverage == pytest.approx(50 * 95.0 / c.equity)


def test_protective_fill_at_trigger_price():
    ex = venue()
    ex.submit(buy())
    r = ex.submit(sell_close(), at_price=98.0)
    assert r.fills[0].price == pytest.approx(98.0 * 0.9995)       # stop: trigger minus slippage
    ex.submit(buy("e2"))
    r2 = ex.submit(sell_close("c2", reason="take_profit"), at_price=104.0, passive=True)
    assert r2.fills[0].price == pytest.approx(104.0)               # resting target: no slippage


# ---- failures -------------------------------------------------------------------------------
def test_submit_is_idempotent_on_client_id():
    ex = venue()
    a, b = ex.submit(buy()), ex.submit(buy())
    assert a == b and ex.ledger.positions[SYM].qty == 10.0


def test_rejected_order_changes_nothing():
    ex = venue()
    ex.inject_failure("reject")
    om, rec, _ = manager(ex)
    res = run(om, ex, buy())
    assert res.status is OrderStatus.REJECTED and not ex.ledger.positions and ex.ledger.cash == 10_000
    assert rec.orders[-1]["status"] == "rejected"


def test_timeout_after_fill_is_resolved_by_asking_the_venue_not_by_resending():
    ex = venue()
    ex.inject_failure("timeout_filled")
    om, rec, _ = manager(ex)
    res = run(om, ex, buy())
    assert res.status is OrderStatus.FILLED and ex.ledger.positions[SYM].qty == 10.0   # exactly one fill
    assert len(rec.fills) == 1 and not om.blocked
    assert run(om, ex, buy()).status is OrderStatus.FILLED and ex.ledger.positions[SYM].qty == 10.0


def test_timeout_without_fill_is_reported_as_not_received():
    ex = venue()
    ex.inject_failure("timeout_unfilled")
    om, _, _ = manager(ex)
    res = run(om, ex, buy())
    assert res.status is OrderStatus.REJECTED and "never received" in res.message and not ex.ledger.positions


def test_unresolvable_order_blocks_new_risk_until_resolved():
    ex = venue()
    om, _, alerts = manager(ex)
    real_fetch = ex.fetch_order
    ex.fetch_order = lambda cid: (_ for _ in ()).throw(ConnectionError("down"))      # venue unreachable
    ex.inject_failure("timeout_filled")
    res = run(om, ex, buy())
    assert res.status is OrderStatus.UNKNOWN and om.blocked
    assert alerts.sent[-1][1] == "order_unresolved"
    blocked = om.execute(OrderIntent("e2", ETH, Side.BUY, 1.0, False, "entry", 0, stop_price=98.0, ref_price=100.0),
                         ex.account(0), INSTS[ETH], 100.0, 1e9, True, Halts())
    assert blocked.status is OrderStatus.REJECTED and "reconciliation_pending" in blocked.message
    assert run(om, ex, sell_close()).status is OrderStatus.FILLED      # reducing risk is never blocked
    ex.fetch_order = real_fetch
    om.retry_unresolved()
    assert not om.blocked


def test_partial_fill_is_booked_at_the_filled_size():
    ex = venue()
    ex.inject_failure("partial")
    om, _, _ = manager(ex)
    res = run(om, ex, buy(qty=10.0))
    assert res.status is OrderStatus.PARTIAL and res.filled_qty == 5.0 and ex.ledger.positions[SYM].qty == 5.0


def test_risk_rejected_order_never_reaches_the_venue():
    ex = venue()
    om, rec, _ = manager(ex)
    res = run(om, ex, buy(stop=None))
    assert res.status is OrderStatus.REJECTED and res.message.startswith("risk:") and ex.submit_count == 0
    assert rec.orders[-1]["status"] == "risk_rejected"
    assert run(om, ex, buy("e9"), halts=Halts(kill=True)).status is OrderStatus.REJECTED and ex.submit_count == 0


def test_reconcile_detects_position_mismatch():
    ex = venue()
    om, _, _ = manager(ex)
    run(om, ex, buy())
    assert om.reconcile({SYM: 10.0}, [SYM, ETH]) == []
    assert om.reconcile({SYM: 7.0}, [SYM]) == [f"{SYM}: venue 10 vs local 7"]
    assert om.reconcile({}, [SYM]) == [f"{SYM}: venue 10 vs local 0"]


def test_pyramiding_fill_is_refused_by_the_ledger():
    ex = venue()
    ex.submit(buy())
    with pytest.raises(RuntimeError):
        ex.submit(buy("e2"))


# ---- recorder / alerts ---------------------------------------------------------------------
def test_sqlite_recorder_persists_state(tmp_path):
    path = tmp_path / "s.db"
    r = SqliteRecorder(path)
    r.set("kill", True)
    r.set("pause_until", {"*": 5})
    did = r.decision({"ts": 1, "symbol": SYM, "status": "decision", "confidence": 0.9})
    r.event(1, "critical", "kill_switch", "x")
    r.close()
    r2 = SqliteRecorder(path)
    assert r2.get("kill") is True and r2.get("pause_until") == {"*": 5} and r2.get("missing", 7) == 7
    assert r2.query("SELECT confidence FROM decisions WHERE id=?", (did,))[0]["confidence"] == 0.9
    assert r2.query("SELECT key FROM events")[0]["key"] == "kill_switch"


def test_alerts_are_rate_limited_and_never_raise():
    clock = [0.0]
    boom = lambda text: (_ for _ in ()).throw(RuntimeError("telegram down"))      # noqa: E731
    a = Alerter(min_interval_s=300, sender=boom, clock=lambda: clock[0])
    assert a.alert("warning", "k", "one") is True
    assert a.alert("warning", "k", "two") is False                # suppressed
    assert a.alert("critical", "k", "three", force=True) is True  # forced through
    clock[0] = 301
    assert a.alert("warning", "k", "four") is True
    assert telegram_sender_from_env({}) is None
    assert callable(telegram_sender_from_env({"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": "1"}))

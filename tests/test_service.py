"""The 24/7 service loop against a fake venue feed: restarts, feed errors, manual kill, heartbeat."""
from __future__ import annotations

import json
import time

import pytest
import yaml

from tradebot.config import DataGuards, RiskLimits, RuntimeCfg, Settings
from tradebot.execution.ccxt_adapter import CcxtExchange, LiveTradingNotApproved
from tradebot.policy.policy import Policy, PolicyStore
from tradebot.reflex.mock import MockReflex
from tradebot.runtime.service import BotService, heartbeat_age_s
from tradebot.storage.recorder import MemoryRecorder, SqliteRecorder
from tradebot.types import Bar, Book, Mode, OrderIntent, OrderStatus, Side

from .conftest import ETH, SYM, TF, inst, make_setup, out


class FakeFeed:
    def __init__(self, n=260, price=100.0):
        self.tf_ms = TF
        self.now = (int(time.time() * 1000) // TF) * TF + 60_000       # one minute into the current hour
        self.price = {SYM: price, ETH: 50.0}
        self.n = n
        self.fail_book = set()
        self.calls = {"bars": 0, "book": 0}

    def server_time_ms(self):
        return self.now

    def instruments(self, market_type="swap", quote="USD"):
        return {SYM: inst(SYM, 0.01), ETH: inst(ETH, 0.1)}

    def closed_bars(self, symbol, limit=300, since_ms=None, now_ms=None):
        self.calls["bars"] += 1
        now_ms = now_ms or self.now
        last_open = (now_ms // TF) * TF - TF
        p = self.price[symbol]
        k = min(limit, self.n)
        return [Bar(last_open - (k - 1 - i) * TF, p, p * 1.001, p * 0.999, p, 5e7, TF) for i in range(k)]

    def book(self, symbol, depth=10):
        self.calls["book"] += 1
        if symbol in self.fail_book:
            raise ConnectionError("venue down")
        p = self.price[symbol]
        return Book(self.now - 200, ((p * 0.9999, 100.0),), ((p * 1.0001, 100.0),))

    def perp_details(self, symbol):
        return {"funding_rate": 0.00001, "open_interest": 1000.0, "index_price": self.price[symbol], "funding_time": None}

    def next_hour(self):
        self.now = (self.now // TF) * TF + TF + 10_000                 # ten seconds after the next close


@pytest.fixture
def env(tmp_path):
    setups = tmp_path / "setups"
    setups.mkdir()
    s = make_setup(SYM)
    (setups / "btc.yaml").write_text(yaml.safe_dump(json.loads(s.model_dump_json())))
    pol = PolicyStore(tmp_path / "policies")
    pol.save_new(Policy(version=1))
    pol.promote(1, "paper", by="bootstrap", reason="init")
    settings = Settings(guards=DataGuards(require_book=True), context_symbols=[SYM, ETH], paper_balance=100_000.0,
                        state_dir=str(tmp_path / "state"), setups_dir=str(setups), policies_dir=str(tmp_path / "policies"),
                        reports_dir=str(tmp_path / "reports"), runtime=RuntimeCfg(candle_settle_s=5))
    return settings, tmp_path


def service(settings, feed, reflex, recorder=None, skew_ms=0):
    svc = BotService(settings, RiskLimits(), feed=feed, reflex=reflex, recorder=recorder, env={},
                     clock=lambda: feed.now + skew_ms)
    svc.runner.inline = True
    svc.backfill()
    svc.last_candle = (feed.now // TF) * TF
    return svc


def test_tick_trades_on_candle_close_and_writes_heartbeat(env):
    settings, tmp = env
    feed = FakeFeed()
    svc = service(settings, feed, MockReflex([out("long", 0.9, 3)]))
    svc.tick()
    assert not svc.exchange.ledger.positions                # same candle as boot: nothing to decide yet
    feed.next_hour()
    svc.tick()
    pos = svc.exchange.ledger.positions[SYM]
    assert pos.qty > 0 and pos.entry_price == pytest.approx(100.0 * 1.0001 * 1.0003, rel=1e-3)   # bought at the ask
    hb = json.loads((tmp / "state" / "heartbeat.json").read_text())
    assert hb["mode"] == "paper" and hb["positions"] == {SYM: pos.qty} and hb["reflex"]["ok"] == 1
    assert heartbeat_age_s(tmp / "state") < 5
    row = svc.recorder.query("SELECT snapshot, action FROM decisions ORDER BY id DESC LIMIT 1")[0]
    assert row["action"] == "enter" and "spr=" in row["snapshot"] and "fnd=0.1" in row["snapshot"]


def test_paper_account_and_halts_survive_a_restart(env):
    settings, tmp = env
    feed = FakeFeed()
    svc = service(settings, feed, MockReflex([out("long", 0.9, 3)]))
    feed.next_hour()
    svc.tick()
    qty, cash = svc.exchange.ledger.positions[SYM].qty, svc.exchange.ledger.cash
    stop = svc.exchange.ledger.positions[SYM].stop_price
    svc.recorder.close()
    again = service(settings, feed, MockReflex([out("neutral", 0.5, 0)]))        # a fresh process
    p = again.exchange.ledger.positions[SYM]
    assert (p.qty, p.stop_price) == (qty, stop) and again.exchange.ledger.cash == pytest.approx(cash)
    feed.price[SYM] = stop * 0.995                                               # price falls through the stop
    again.tick()
    assert SYM not in again.exchange.ledger.positions                            # the restored stop still works


def test_feed_failure_blocks_entries_and_alerts_but_loop_survives(env):
    settings, _ = env
    feed = FakeFeed()
    svc = service(settings, feed, MockReflex([out("long", 0.9, 3)]))
    feed.fail_book.add(SYM)
    for _ in range(3):
        feed.next_hour()
        svc.tick()                                             # must not raise
    assert not svc.exchange.ledger.positions
    assert svc.errors[f"{SYM}:book"] == 3
    reasons = [r["skip_reason"] for r in svc.recorder.query("SELECT skip_reason FROM decisions")]
    assert reasons and set(reasons) <= {"exchange_error", "no_book"}
    assert any(k.startswith("feed:") for _, k, _ in svc.alerts.sent)
    feed.fail_book.clear()
    feed.next_hour()
    svc.tick()
    assert SYM in svc.exchange.ledger.positions                # recovers on its own


def test_operator_kill_is_seen_by_the_running_bot(env):
    settings, _ = env
    feed = FakeFeed()
    svc = service(settings, feed, MockReflex([out("long", 0.9, 3)]))
    feed.next_hour()
    svc.tick()
    assert SYM in svc.exchange.ledger.positions
    SqliteRecorder(svc.state_dir / "tradebot.db").set("kill", True)      # `tradebot kill` from another process
    svc.tick()
    assert svc.core.kill and not svc.exchange.ledger.positions
    assert svc.exchange.ledger.closed[-1].reason == "manual_kill"
    feed.next_hour()
    svc.tick()
    assert not svc.exchange.ledger.positions


def test_clock_skew_between_host_and_venue_blocks_decisions(env):
    settings, _ = env
    feed = FakeFeed()
    svc = service(settings, feed, MockReflex([out("long", 0.9, 3)]), skew_ms=5000)   # host clock 5 s off
    feed.next_hour()
    svc.tick()
    assert not svc.exchange.ledger.positions and svc.skew_ms == 5000
    assert svc.recorder.query("SELECT skip_reason FROM decisions")[0]["skip_reason"] == "clock_skew"


def test_setup_for_unlisted_instrument_refuses_to_start(env):
    settings, tmp = env
    s = make_setup("DOESNOTEXIST/USD:USD", id="ghost")
    (tmp / "setups" / "ghost.yaml").write_text(yaml.safe_dump(json.loads(s.model_dump_json())))
    with pytest.raises(RuntimeError, match="does not list"):
        BotService(settings, RiskLimits(), feed=FakeFeed(), reflex=MockReflex([out()]), env={})


# ---- ccxt adapter: gates only (it has never touched a real account) ---------------------------
class FakeVenue:
    def __init__(self):
        self.calls = []

    def create_order(self, symbol, type_, side, amount, price=None, params=None):
        self.calls.append(("create_order", symbol, side, amount, dict(params or {})))
        return {"id": "o1", "status": "closed", "filled": amount, "average": 100.0, "fee": {"cost": 0.1}, "timestamp": 1}

    def close_position(self, symbol, side=None, params=None):
        self.calls.append(("close_position", symbol, dict(params or {})))
        return {"id": "o2", "status": "closed", "filled": params["size"], "average": 101.0, "fee": {"cost": 0.1}, "timestamp": 2}

    def fetch_balance(self):
        return {"total": {"USD": 1000.0}}

    def fetch_positions(self, symbols=None):
        return [{"symbol": SYM, "contracts": 2.0, "side": "long", "entryPrice": 100.0, "markPrice": 101.0, "unrealizedPnl": 0.02}]


ENTRY = OrderIntent("e:BTC:1", SYM, Side.BUY, 2.0, False, "entry", 1, setup_id="s", stop_price=98.0, ref_price=100.0)


def test_live_adapter_refuses_without_the_full_gate():
    insts = {SYM: inst(SYM, 0.01)}
    with pytest.raises(LiveTradingNotApproved):
        CcxtExchange(Settings(mode=Mode.LIVE), insts, MemoryRecorder(), env={}, client=FakeVenue())
    with pytest.raises(LiveTradingNotApproved):
        CcxtExchange(Settings(mode=Mode.LIVE), insts, MemoryRecorder(),
                     env={"EXCHANGE_API_KEY": "k", "EXCHANGE_API_SECRET": "s"}, client=FakeVenue())
    with pytest.raises(LiveTradingNotApproved):
        CcxtExchange(Settings(mode=Mode.PREVIEW), insts, MemoryRecorder(), env={})          # no keys
    with pytest.raises(ValueError):
        CcxtExchange(Settings(mode=Mode.PAPER), insts, MemoryRecorder(), env={}, client=FakeVenue())


def test_preview_mode_never_places_an_order():
    venue = FakeVenue()
    ex = CcxtExchange(Settings(mode=Mode.PREVIEW), {SYM: inst(SYM, 0.01)}, MemoryRecorder(), env={}, client=venue)
    res = ex.submit(ENTRY)
    assert res.status is OrderStatus.REJECTED and "preview only" in res.message and not res.fills
    assert venue.calls == [("create_order", SYM, "buy", 2.0, {"preview": True})]


def test_live_mode_passes_our_client_id_and_keeps_the_stop_locally():
    venue, rec = FakeVenue(), MemoryRecorder()
    env_ = {"TRADEBOT_LIVE_ACK": "I-APPROVE-LIVE-TRADING", "EXCHANGE_API_KEY": "k", "EXCHANGE_API_SECRET": "s"}
    ex = CcxtExchange(Settings(mode=Mode.LIVE), {SYM: inst(SYM, 0.01)}, rec, env=env_, client=venue, poll_s=0)
    res = ex.submit(ENTRY)
    assert res.status is OrderStatus.FILLED and res.fills[0].price == 100.0
    assert venue.calls[0][4] == {"client_order_id": "e:BTC:1"}
    acct = ex.account(5)
    assert acct.positions[SYM].qty == 2.0 and acct.positions[SYM].stop_price == 98.0 and acct.equity == pytest.approx(1000.02)
    close = OrderIntent("c:BTC:2:stop", SYM, Side.SELL, 2.0, True, "stop", 2)
    assert ex.submit(close).status is OrderStatus.FILLED
    assert venue.calls[-1] == ("close_position", SYM, {"clientOrderId": "c:BTC:2:stop", "size": 2.0})
    assert rec.get("live_positions") == {}

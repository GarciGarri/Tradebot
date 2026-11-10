"""REFLEX contract, deadline handling and backends."""
from __future__ import annotations

import inspect
import re
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tradebot.config import ReflexCfg
from tradebot.reflex import build_reflex
from tradebot.reflex.base import ReflexRunner
from tradebot.reflex.haiku import HaikuReflex
from tradebot.reflex.jev import JevReflex
from tradebot.reflex.mock import MockReflex, ReplayReflex
from tradebot.reflex.prompt import SYSTEM_PROMPT
from tradebot.reflex.rules import RuleParams, RuleReflex
from tradebot.reflex.schema import ReflexDecision, ReflexError, ReflexOutput, ReflexSkip, ReflexUnavailable
from tradebot.state.engine import Snapshot, SnapshotStatus
from tradebot.types import Direction, Regime, RiskState

from .conftest import SYM, out


def snap(ok=True, **fields) -> Snapshot:
    base = {"ema": 1.2, "px50": 2.0, "r24": 3.0, "r72": 6.0, "rvr": 1.0, "b24": 2.0, "bema": 0.8, "vz": 0.5,
            "rng": 0.7, "bias": 1, "inv": 10.0, "ddu": 0.0, "dayu": 0.0, "levu": 0.0, "posu": 0.0}
    base.update(fields)
    return Snapshot(ts=1, symbol=SYM, status=SnapshotStatus.OK if ok else SnapshotStatus.STALE,
                    issues=() if ok else ("stale",), fields=base, text="k=1", tokens_est=3, setup_id="s")


# ---- schema ---------------------------------------------------------------------------
def test_schema_enforces_types_and_ranges():
    good = dict(regime="trending", direction="long", direction_confidence=0.8, toxic_flow=None, setup_quality=2,
                risk_state="safe")
    assert ReflexOutput(**good).toxic_flow is None
    for bad in (dict(direction_confidence=1.2), dict(setup_quality=4), dict(regime="bullish"),
                dict(risk_state="ok"), dict(direction="buy"), dict(setup_quality=1.5)):
        with pytest.raises(ValidationError):
            ReflexOutput(**{**good, **bad})
    with pytest.raises(ValidationError):                 # the model cannot smuggle in sizes or orders
        ReflexOutput(**good, position_size=10)
    missing = dict(good)
    del missing["toxic_flow"]
    with pytest.raises(ValidationError):                 # nullable, but it must be stated
        ReflexOutput(**missing)


def test_schema_has_exactly_the_judged_fields():
    assert set(ReflexOutput.model_fields) == {"regime", "direction", "direction_confidence", "toxic_flow",
                                              "setup_quality", "risk_state"}


# ---- runner ---------------------------------------------------------------------------
def test_runner_returns_decision_with_latency():
    r = ReflexRunner(MockReflex([out()]), 1000, inline=True)
    d = r.run(snap())
    assert isinstance(d, ReflexDecision) and d.backend == "mock" and d.latency_ms >= 0 and r.stats["ok"] == 1


def test_runner_skips_candle_when_decision_misses_deadline():
    slow = MockReflex([out()], delay_s=0.5)
    r = ReflexRunner(slow, 80)                      # threaded path: the call is abandoned
    d = r.run(snap())
    assert isinstance(d, ReflexSkip) and d.reason == "deadline" and 60 <= d.latency_ms < 400
    r.close()
    r2 = ReflexRunner(MockReflex([out()], delay_s=0.05), 10, inline=True)   # inline path: late answer discarded
    assert r2.run(snap()).reason == "deadline"


def test_runner_skips_on_error_and_on_invalid_output():
    assert ReflexRunner(MockReflex([out()], fail=True), 1000, inline=True).run(snap()).reason == "error"
    bad = MockReflex([{"regime": "trending", "direction": "long", "direction_confidence": 7, "toxic_flow": None,
                       "setup_quality": 2, "risk_state": "safe"}])
    assert ReflexRunner(bad, 1000, inline=True).run(snap()).reason == "invalid"


def test_runner_never_calls_the_model_on_a_bad_snapshot():
    m = MockReflex([out()])
    d = ReflexRunner(m, 1000, inline=True).run(snap(ok=False))
    assert isinstance(d, ReflexSkip) and d.reason == "snapshot_not_ok" and m.calls == 0


# ---- Haiku backend (wiring only: the live call needs a key and is checked by `tradebot reflex-check`) ----
class _FakeMessages:
    def __init__(self, resp):
        self.resp, self.kwargs = resp, None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return self.resp


def _fake_client(parsed=None, stop="end_turn"):
    msgs = _FakeMessages(SimpleNamespace(parsed_output=parsed, stop_reason=stop, usage=SimpleNamespace(input_tokens=1)))
    return SimpleNamespace(messages=msgs), msgs


def test_haiku_request_shape():
    client, msgs = _fake_client(out())
    h = HaikuReflex(ReflexCfg(), client=client)
    s = snap()
    assert h.decide(s) == out()
    k = msgs.kwargs
    assert k["model"] == "claude-haiku-5-5"
    assert k["output_config"] == {"effort": "low"}
    assert k["output_format"] is ReflexOutput
    assert k["messages"] == [{"role": "user", "content": s.text}]       # the snapshot is all it sees
    assert k["system"][0]["text"] == SYSTEM_PROMPT and k["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "temperature" not in k and "tools" not in k
    assert "thinking" not in k
    client2, msgs2 = _fake_client(out())
    HaikuReflex(ReflexCfg(thinking="disabled"), client=client2).decide(s)
    assert msgs2.kwargs["thinking"] == {"type": "disabled"}


@pytest.mark.parametrize("parsed,stop", [(None, "end_turn"), (out(), "refusal"), (out(), "max_tokens")])
def test_haiku_unusable_answers_raise(parsed, stop):
    client, _ = _fake_client(parsed, stop)
    with pytest.raises(ReflexError):
        HaikuReflex(ReflexCfg(), client=client).decide(snap())


def test_haiku_without_key_is_unavailable(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(ReflexUnavailable):
        HaikuReflex(ReflexCfg())
    with pytest.raises(ReflexUnavailable):
        build_reflex(ReflexCfg(backend="haiku"))


def test_jev_backend_fails_loudly_instead_of_guessing():
    with pytest.raises(ReflexUnavailable, match="not implemented"):
        JevReflex(ReflexCfg(backend="jev"))
    with pytest.raises(ReflexUnavailable):
        build_reflex(ReflexCfg(backend="jev"))


def test_system_prompt_is_static():
    assert not re.search(r"20\d\d-\d\d", SYSTEM_PROMPT)       # no dates: cacheable and time-neutral
    assert "never place orders" in SYSTEM_PROMPT


# ---- rule backend ----------------------------------------------------------------------
def test_rules_are_deterministic_and_valid():
    r = RuleReflex()
    a, b = r.decide(snap()), r.decide(snap())
    assert a == b and isinstance(a, ReflexOutput)
    assert a.direction is Direction.LONG and a.regime is Regime.TRENDING and a.setup_quality == 3
    assert 0.5 < a.direction_confidence <= 0.95


def test_rules_short_neutral_crisis_and_invalidated():
    r = RuleReflex()
    s = r.decide(snap(ema=-1.2, px50=-2.0, r24=-3.0, r72=-6.0, b24=-2.0, bema=-0.8, bias=-1, rng=0.3))
    assert s.direction is Direction.SHORT
    assert r.decide(snap(ema=0.01, px50=0.0, r24=0.0, r72=0.0)).direction is Direction.NEUTRAL
    assert r.decide(snap(b24=-12.0, rvr=2.5)).regime is Regime.CRISIS
    assert r.decide(snap(inv=-1.0)).setup_quality == 0
    assert r.decide(snap(bias=-1)).setup_quality <= 1          # fighting the thesis is never a good setup
    assert RuleReflex(RuleParams(allow_short=False)).decide(
        snap(ema=-1.2, px50=-2.0, r24=-3.0, r72=-6.0)).direction is Direction.NEUTRAL


def test_rules_risk_state_and_toxic_flow():
    r = RuleReflex()
    assert r.decide(snap(dayu=0.75)).risk_state is RiskState.NEAR_LIMIT
    assert r.decide(snap(ddu=0.8)).risk_state is RiskState.SAFE          # drawdown only blocks at 0.9
    assert r.decide(snap(ddu=0.95)).risk_state is RiskState.REDUCE
    assert r.decide(snap(posu=1.0)).risk_state is RiskState.NEAR_LIMIT
    assert r.decide(snap()).toxic_flow is None
    assert r.decide(snap(spr=30.0, imb=0.0)).toxic_flow is True
    assert r.decide(snap(spr=1.0, imb=0.1)).toxic_flow is False


def test_rules_module_makes_no_network_or_model_calls():
    src = inspect.getsource(inspect.getmodule(RuleReflex))
    assert "anthropic" not in src and "urllib" not in src and "ccxt" not in src


def test_replay_reflex_skips_unknown_candles():
    rp = ReplayReflex({(SYM, 1): out()})
    assert rp.decide(snap()) == out()
    other = Snapshot(ts=2, symbol=SYM, status=SnapshotStatus.OK, issues=(), fields={}, text="", tokens_est=0)
    assert ReflexRunner(rp, 1000, inline=True).run(other).reason == "error"

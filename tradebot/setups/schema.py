"""Research setups compiled into a typed, machine-checkable schema.

A setup is the BRAIN's thesis about one asset. The prose (thesis, bear case, catalysts) is
for humans. The typed parts are enforced by code on every candle:

* `reflex`       - what the REFLEX output must look like before this setup may trade.
                   A setup can be stricter than the active policy, never looser.
* `invalidation` - price levels that kill the thesis. Checked in code, no model involved.
* `expires`      - a setup that was not reviewed in time stops opening trades by itself.
"""
from __future__ import annotations

from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..state.engine import SetupView
from ..types import Direction, Regime, RiskState


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str
    accessed: date
    claim: str


class Catalyst(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str
    status: Literal["confirmed", "speculative"]
    when: Optional[date] = None
    source: str = ""


class Invalidation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    price_below: Optional[float] = None      # a long thesis dies below this
    price_above: Optional[float] = None      # a short thesis dies above this
    conditions: list[str] = []               # non-price conditions, checked in the nightly review


class SetupReflexSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    allowed_directions: list[Direction]
    allowed_regimes: list[Regime] = [Regime.TRENDING, Regime.MEAN_REVERTING]
    min_setup_quality: int = Field(2, ge=0, le=3)
    min_direction_confidence: float = Field(0.80, ge=0.5, le=1.0)
    required_risk_state: RiskState = RiskState.SAFE


class SetupExits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stop_atr_mult: Optional[float] = Field(None, gt=0)
    take_profit_r: Optional[float] = Field(None, gt=0)
    max_hold_bars: Optional[int] = Field(None, gt=0)


class SetupSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    symbol: str                      # ccxt unified symbol on the trading venue
    base: str
    status: Literal["active", "watch", "retired"] = "watch"
    bias: Direction
    quality_prior: int = Field(ge=0, le=3)
    thesis: str
    bear_case: str
    valuation: str = ""              # numbers behind "disconnected from fundamentals", with dates
    value_accrual: str = ""          # does value reach the token, and how
    edge_type: Literal["organic", "incentive_driven", "mixed", "technical"] = "organic"
    catalysts: list[Catalyst] = []
    invalidation: Invalidation
    reflex: SetupReflexSpec
    exits: SetupExits = SetupExits()
    created: date
    expires: date
    sources: list[Source] = []

    @model_validator(mode="after")
    def _consistent(self) -> "SetupSchema":
        if self.bias is Direction.LONG and self.invalidation.price_below is None:
            raise ValueError(f"{self.id}: a long setup needs invalidation.price_below")
        if self.bias is Direction.SHORT and self.invalidation.price_above is None:
            raise ValueError(f"{self.id}: a short setup needs invalidation.price_above")
        if self.bias is not Direction.NEUTRAL and self.bias not in self.reflex.allowed_directions:
            raise ValueError(f"{self.id}: bias must be one of reflex.allowed_directions")
        if self.expires <= self.created:
            raise ValueError(f"{self.id}: expires must be after created")
        return self

    # ---- code-enforced checks ----------------------------------------------
    def expired(self, now_ms: int) -> bool:
        end = datetime.combine(self.expires, time.max, tzinfo=timezone.utc)
        return now_ms > int(end.timestamp() * 1000)

    def invalidated(self, price: float) -> bool:
        inv = self.invalidation
        if self.bias is Direction.LONG and inv.price_below is not None:
            return price <= inv.price_below
        if self.bias is Direction.SHORT and inv.price_above is not None:
            return price >= inv.price_above
        return False

    def tradable(self, now_ms: int, price: float) -> tuple[bool, str]:
        if self.status != "active":
            return False, f"setup_status_{self.status}"
        if self.expired(now_ms):
            return False, "setup_expired"
        if self.invalidated(price):
            return False, "setup_invalidated"
        return True, ""

    def view(self, now_ms: int) -> SetupView:
        bias = 1 if self.bias is Direction.LONG else -1 if self.bias is Direction.SHORT else 0
        level = self.invalidation.price_below if bias > 0 else self.invalidation.price_above if bias < 0 else None
        nxt = None
        for c in self.catalysts:
            if c.status == "confirmed" and c.when is not None:
                ts = int(datetime.combine(c.when, time.min, tzinfo=timezone.utc).timestamp() * 1000)
                if ts >= now_ms and (nxt is None or ts < nxt):
                    nxt = ts
        return SetupView(setup_id=self.id, bias=bias, quality_prior=self.quality_prior,
                         invalidation_price=level, next_catalyst_ts=nxt)


def load_setups(directory: str | Path, only_active: bool = False) -> dict[str, SetupSchema]:
    """Load every *.yaml setup in a directory, keyed by setup id. One bad file fails loudly."""
    out: dict[str, SetupSchema] = {}
    for path in sorted(Path(directory).glob("*.yaml")):
        with open(path, "r", encoding="utf-8") as fh:
            setup = SetupSchema(**yaml.safe_load(fh))
        if setup.id in out:
            raise ValueError(f"duplicate setup id {setup.id!r} in {path}")
        if only_active and setup.status != "active":
            continue
        out[setup.id] = setup
    return out

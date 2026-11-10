"""The typed REFLEX contract.

`ReflexOutput` is the only thing a model is allowed to produce. It contains judgements,
never orders, sizes, prices or limits. Code decides what (if anything) to do with it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from ..types import Direction, Regime, RiskState


class ReflexOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    regime: Regime
    direction: Direction
    direction_confidence: float = Field(ge=0.0, le=1.0)
    toxic_flow: Optional[bool]          # required but nullable: null = cannot judge
    setup_quality: int = Field(ge=0, le=3)
    risk_state: RiskState


@dataclass(frozen=True)
class ReflexDecision:
    output: ReflexOutput
    ts: int                 # decision time of the snapshot it was made from
    symbol: str
    setup_id: str
    backend: str
    latency_ms: float


@dataclass(frozen=True)
class ReflexSkip:
    """The candle is skipped: no decision arrived in time, or it was unusable."""

    reason: str             # snapshot_not_ok | deadline | error | invalid
    ts: int
    symbol: str
    latency_ms: float = 0.0
    detail: str = ""


class ReflexError(RuntimeError):
    """The backend answered but the answer cannot be used."""


class ReflexUnavailable(RuntimeError):
    """The backend cannot be used at all (missing key, missing integration)."""

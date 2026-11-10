"""Typed configuration.

Three separate things live here and are deliberately not mixed:

* `RiskLimits`  - hard limits. Only a human edits them (config/risk.yaml). No model, no
  nightly job, and no policy file can change them. Values are validated against absolute
  ceilings compiled into this file.
* `Settings`    - venue, universe, data guards, runtime plumbing.
* Policy        - tunable trading thresholds, versioned (see tradebot/policy/policy.py).

Secrets are read from environment variables only and are never part of these models.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .types import Mode

REPO_ROOT = Path(__file__).resolve().parent.parent

# Absolute ceilings. Raising a limit past these needs a code change and a review,
# not a config edit. They exist so a typo cannot turn 2% into 20%.
ABS_MAX_RISK_PER_TRADE = 0.05
ABS_MAX_DAILY_LOSS = 0.10
ABS_MAX_OPEN_POSITIONS = 10
ABS_MAX_LEVERAGE = 3.0
ABS_MAX_KILL_DRAWDOWN = 0.25
ABS_MAX_KELLY_FRACTION = 0.25  # quarter Kelly is the most the sizing code will ever use

LIVE_ACK_PHRASE = "I-APPROVE-LIVE-TRADING"


class RiskLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    risk_per_trade: float = Field(0.02, gt=0, le=ABS_MAX_RISK_PER_TRADE)
    max_daily_loss: float = Field(0.05, gt=0, le=ABS_MAX_DAILY_LOSS)
    max_open_positions: int = Field(3, ge=1, le=ABS_MAX_OPEN_POSITIONS)
    max_leverage: float = Field(2.0, gt=0, le=ABS_MAX_LEVERAGE)
    kill_switch_drawdown: float = Field(0.15, gt=0, le=ABS_MAX_KILL_DRAWDOWN)
    # order sanity
    max_price_deviation: float = Field(0.02, gt=0, le=0.10)   # ref price vs current mid
    max_participation: float = Field(0.02, gt=0, le=0.25)     # share of recent bar $ volume
    # share of a limit at which code reports near_limit / reduce
    near_limit_frac: float = Field(0.70, gt=0, lt=1)
    reduce_frac: float = Field(0.90, gt=0, le=1)
    # once drawdown passes near_limit_frac of the kill level, risk per trade is multiplied by this
    drawdown_caution_scale: float = Field(0.5, gt=0, le=1)

    @model_validator(mode="after")
    def _ordering(self) -> "RiskLimits":
        if self.near_limit_frac >= self.reduce_frac:
            raise ValueError("near_limit_frac must be below reduce_frac")
        if self.max_daily_loss >= self.kill_switch_drawdown:
            raise ValueError("max_daily_loss must be below kill_switch_drawdown")
        return self


class DataGuards(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_bars: int = Field(200, ge=60)
    max_bar_lag_s: float = Field(90.0, gt=0)       # how late the newest closed bar may arrive
    max_book_age_s: float = Field(20.0, gt=0)
    max_clock_skew_ms: int = Field(2000, gt=0)
    require_book: bool = True                      # live/paper need a fresh book before entries
    max_spread_bps: float = Field(25.0, gt=0)      # wider than this: no entries


class FeeModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # ESTIMATES until the first authenticated fee-schedule read. Conservative on purpose.
    taker: float = Field(0.0010, ge=0)             # 0.10% per side
    maker: float = Field(0.0005, ge=0)
    slippage_bps: float = Field(3.0, ge=0)         # fixed part, each side
    impact_coeff: float = Field(0.10, ge=0)        # extra bps = coeff*1e4*sqrt(notional/bar$vol)
    funding_bps_per_hour: float = Field(0.10, ge=0)  # used only when no real funding is known


class ExchangeCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = "coinbase"
    market_type: str = "swap"
    quote: str = "USD"
    timeframe: str = "1h"
    fees: FeeModel = FeeModel()


class ReflexCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: str = "haiku"                 # haiku | jev | rules | mock
    model: str = "claude-haiku-5-5"
    effort: str = "low"
    thinking: str = "adaptive"             # adaptive | disabled
    deadline_ms: int = Field(8000, gt=0)   # a decision later than this is discarded
    max_tokens: int = Field(1024, gt=0)
    snapshot_token_budget: int = Field(400, gt=0)


class AlertCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_interval_s: float = 300.0          # rate limit per alert key


class RuntimeCfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    risk_tick_s: float = Field(20.0, gt=0)
    candle_settle_s: float = Field(5.0, ge=0)      # wait after candle close before reading it
    heartbeat_max_age_s: float = Field(180.0, gt=0)
    health_port: int = 8080
    nightly_utc_hour: int = Field(0, ge=0, le=23)
    nightly_utc_minute: int = Field(20, ge=0, le=59)


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Mode = Mode.PAPER
    paper_balance: float = Field(1000.0, gt=0)
    exchange: ExchangeCfg = ExchangeCfg()
    guards: DataGuards = DataGuards()
    reflex: ReflexCfg = ReflexCfg()
    alerts: AlertCfg = AlertCfg()
    runtime: RuntimeCfg = RuntimeCfg()
    context_symbols: list[str] = ["BTC/USD:USD", "ETH/USD:USD"]
    off_limits_rule: str = "outside_top_50"
    state_dir: str = "state"
    reports_dir: str = "reports"
    setups_dir: str = "config/setups"
    policies_dir: str = "config/policies"

    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else REPO_ROOT / p


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_settings(path: str | os.PathLike[str] | None = None) -> Settings:
    p = Path(path or os.environ.get("TRADEBOT_CONFIG") or REPO_ROOT / "config" / "default.yaml")
    return Settings(**_read_yaml(p))


def load_risk_limits(path: str | os.PathLike[str] | None = None) -> RiskLimits:
    p = Path(path or os.environ.get("TRADEBOT_RISK_CONFIG") or REPO_ROOT / "config" / "risk.yaml")
    return RiskLimits(**_read_yaml(p))


def live_trading_approved(settings: Settings, env: dict[str, str] | None = None) -> bool:
    """The live gate. All three must hold; any one missing means no real order can be sent.

    1. config mode is `live`
    2. the operator exported TRADEBOT_LIVE_ACK with the exact phrase
    3. exchange credentials exist in the environment
    """
    env = os.environ if env is None else env
    return (
        settings.mode is Mode.LIVE
        and env.get("TRADEBOT_LIVE_ACK", "") == LIVE_ACK_PHRASE
        and bool(env.get("EXCHANGE_API_KEY"))
        and bool(env.get("EXCHANGE_API_SECRET"))
    )


TIMEFRAME_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


def timeframe_ms(tf: str) -> int:
    try:
        return TIMEFRAME_MS[tf]
    except KeyError as exc:
        raise ValueError(f"unsupported timeframe {tf!r}") from exc

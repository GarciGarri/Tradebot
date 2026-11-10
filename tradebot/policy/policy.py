"""Versioned trading policy.

A policy holds the tunable thresholds: the gate, the pause rule, sizing, calibration, exits
and the code-rule parameters. Hard risk limits are NOT here and cannot be put here.

Every change is a new numbered file. `ACTIVE.json` points at the version used for paper and
(separately) for live. The nightly job may write candidates and move the PAPER pointer after
its gates pass. Moving the LIVE pointer always requires a named human approver, and the
previous pointer is kept so a rollback is one command.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..config import ABS_MAX_KELLY_FRACTION

# Bounds inside which the nightly tuner may move thresholds. Outside them: human only.
TUNE_MIN_CONFIDENCE = (0.65, 0.95)
TUNE_MIN_QUALITY = (2, 3)


class GateCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    min_setup_quality: int = Field(2, ge=1, le=3)
    min_direction_confidence: float = Field(0.80, ge=0.5, lt=1.0)   # strict ">" in the gate
    block_on_toxic_flow: bool = True


class PauseCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    confidence_floor: float = Field(0.60, ge=0.0, le=1.0)
    cooldown_bars: int = Field(24, ge=1)


class CalibrationBin(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    lo: float
    hi: float
    p: float = Field(ge=0.0, le=1.0)
    n: int = 0


class CalibrationCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    method: Literal["shrink", "bins"] = "shrink"
    shrink: float = Field(0.5, ge=0.0, le=1.0)      # p = 0.5 + shrink * (confidence - 0.5)
    bins: list[CalibrationBin] = []


class SizingCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kelly_fraction: float = Field(0.25, gt=0.0, le=ABS_MAX_KELLY_FRACTION)


class ExitCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    stop_atr_mult: float = Field(2.0, gt=0)
    min_stop_pct: float = Field(0.8, gt=0)
    max_stop_pct: float = Field(6.0, gt=0)
    take_profit_r: float = Field(2.0, gt=0)
    max_hold_bars: int = Field(72, gt=0)
    flip_exit_confidence: float = Field(0.70, ge=0.5, le=1.0)  # opposite signal this strong closes


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: int = Field(ge=1)
    parent: Optional[int] = None
    created: str = ""
    note: str = ""
    prompt_version: str = "reflex-v1"
    gate: GateCfg = GateCfg()
    pause: PauseCfg = PauseCfg()
    sizing: SizingCfg = SizingCfg()
    calibration: CalibrationCfg = CalibrationCfg()
    exits: ExitCfg = ExitCfg()
    rule_params: dict = {}          # parameters for the deterministic RuleReflex, if used


class LivePromotionRefused(RuntimeError):
    pass


class PolicyStore:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pointer_path = self.dir / "ACTIVE.json"

    # ---- files -------------------------------------------------------------
    def _path(self, version: int) -> Path:
        return self.dir / f"policy_v{version:04d}.yaml"

    def versions(self) -> list[int]:
        return sorted(int(p.stem.split("_v")[1]) for p in self.dir.glob("policy_v*.yaml"))

    def load(self, version: int) -> Policy:
        with open(self._path(version), "r", encoding="utf-8") as fh:
            return Policy(**yaml.safe_load(fh))

    def save_new(self, policy: Policy) -> Policy:
        """Persist a policy under the next free version number. Existing files are never overwritten."""
        nxt = (self.versions() or [0])[-1] + 1
        pol = policy.model_copy(update={"version": nxt, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        path = self._path(nxt)
        if path.exists():
            raise FileExistsError(path)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(json.loads(pol.model_dump_json()), fh, sort_keys=False)
        return pol

    # ---- pointers ----------------------------------------------------------
    def pointer(self) -> dict:
        if not self.pointer_path.exists():
            return {"paper": None, "live": None, "history": []}
        with open(self.pointer_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write_pointer(self, ptr: dict) -> None:
        tmp = self.pointer_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(ptr, fh, indent=2)
        tmp.replace(self.pointer_path)

    def active(self, target: Literal["paper", "live"] = "paper") -> Policy:
        v = self.pointer().get(target)
        if v is None:
            raise LookupError(f"no {target} policy is active")
        return self.load(int(v))

    def promote(self, version: int, target: Literal["paper", "live"], by: str, reason: str,
                human_approved: bool = False) -> None:
        self.load(version)  # must exist and validate
        if target == "live" and not (human_approved and by.strip() and not by.startswith("auto")):
            raise LivePromotionRefused(
                "a policy reaches live only with human_approved=True and a named human approver")
        ptr = self.pointer()
        ptr.setdefault("history", []).append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "target": target,
            "from": ptr.get(target), "to": version, "by": by, "reason": reason,
        })
        ptr[target] = version
        self._write_pointer(ptr)

    def rollback(self, target: Literal["paper", "live"], by: str, reason: str) -> int:
        """Return the pointer to the version it had before the last promotion of `target`."""
        ptr = self.pointer()
        last = next((h for h in reversed(ptr.get("history", [])) if h["target"] == target and h["from"] is not None), None)
        if last is None:
            raise LookupError(f"no earlier {target} policy to roll back to")
        prev = int(last["from"])
        self.load(prev)
        ptr["history"].append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "target": target,
            "from": ptr.get(target), "to": prev, "by": by, "reason": f"rollback: {reason}",
        })
        ptr[target] = prev
        self._write_pointer(ptr)
        return prev

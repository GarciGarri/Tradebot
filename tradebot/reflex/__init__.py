from __future__ import annotations

from ..config import ReflexCfg
from .base import Reflex, ReflexRunner
from .schema import ReflexDecision, ReflexError, ReflexOutput, ReflexSkip, ReflexUnavailable

__all__ = ["Reflex", "ReflexRunner", "ReflexDecision", "ReflexError", "ReflexOutput", "ReflexSkip",
           "ReflexUnavailable", "build_reflex"]


def build_reflex(cfg: ReflexCfg) -> Reflex:
    """Backend factory. Jev is preferred when available; it is not, so Haiku is the default."""
    if cfg.backend == "haiku":
        from .haiku import HaikuReflex
        return HaikuReflex(cfg)
    if cfg.backend == "jev":
        from .jev import JevReflex
        return JevReflex(cfg)
    if cfg.backend == "rules":
        from .rules import RuleReflex
        return RuleReflex()
    raise ValueError(f"unknown reflex backend {cfg.backend!r}")

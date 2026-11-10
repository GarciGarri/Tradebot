"""REFLEX backend slot for Jev (TypeSafe).

Deliberately not implemented. No Jev API key and no Jev API documentation were provided, and
this codebase does not guess at an API. The class exists so the backend can be selected in
config and fails loudly, at start-up, with the exact missing piece.

To add it: implement `decide(snapshot) -> ReflexOutput` against the real Jev API, send
`snapshot.text` only, validate the answer with `ReflexOutput.model_validate`, and add a
contract test next to tests/test_reflex.py.
"""
from __future__ import annotations

from ..state.engine import Snapshot
from .schema import ReflexOutput, ReflexUnavailable


class JevReflex:
    name = "jev"

    def __init__(self, *_args, **_kwargs):
        raise ReflexUnavailable(
            "Jev backend is not implemented: no Jev API documentation or key was provided. "
            "Use backend 'haiku' (Claude Haiku 5.5) or supply the Jev docs."
        )

    def decide(self, snapshot: Snapshot) -> ReflexOutput:  # pragma: no cover - unreachable
        raise ReflexUnavailable("Jev backend is not implemented")

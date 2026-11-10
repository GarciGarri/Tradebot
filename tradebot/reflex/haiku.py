"""REFLEX backend: Claude Haiku 5.5, low effort, structured outputs.

The request is one static cached system prompt plus the snapshot line. The SDK validates the
answer against `ReflexOutput`; the runner validates it again. There are no retries: a
decision that misses the candle deadline is worthless, so the HTTP timeout sits just under
the deadline and `max_retries` is 0.
"""
from __future__ import annotations

import os

from ..config import ReflexCfg
from ..state.engine import Snapshot
from .prompt import SYSTEM_PROMPT
from .schema import ReflexError, ReflexOutput, ReflexUnavailable


class HaikuReflex:
    name = "haiku"

    def __init__(self, cfg: ReflexCfg, client=None):
        self.cfg = cfg
        self.last_usage = None
        if client is not None:
            self._client = client
            self._errors = ()
            return
        import anthropic  # imported lazily so backtests and tests need no SDK or key

        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise ReflexUnavailable(
                "Haiku REFLEX needs ANTHROPIC_API_KEY in the environment (.env on the server). "
                "It is not set, so the model backend cannot be used."
            )
        timeout_s = max(1.0, cfg.deadline_ms / 1000.0 - 0.25)
        self._client = anthropic.Anthropic(timeout=timeout_s, max_retries=0)
        self._errors = (anthropic.APITimeoutError, anthropic.RateLimitError,
                        anthropic.APIStatusError, anthropic.APIConnectionError)

    def _request(self, snapshot_text: str) -> dict:
        req = dict(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": snapshot_text}],
            output_config={"effort": self.cfg.effort},
            output_format=ReflexOutput,
        )
        if self.cfg.thinking == "disabled":
            req["thinking"] = {"type": "disabled"}
        return req

    def decide(self, snapshot: Snapshot) -> ReflexOutput:
        try:
            resp = self._client.messages.parse(**self._request(snapshot.text))
        except self._errors as exc:
            raise ReflexError(f"{type(exc).__name__}: {exc}") from exc
        self.last_usage = getattr(resp, "usage", None)
        stop = getattr(resp, "stop_reason", None)
        if stop in ("refusal", "max_tokens"):
            raise ReflexError(f"unusable stop_reason: {stop}")
        out = getattr(resp, "parsed_output", None)
        if out is None:
            raise ReflexError("response carried no parsed output")
        return out

    def count_tokens(self, snapshot_text: str) -> int:
        """Exact input size of a snapshot line with the real tokenizer (needs a key)."""
        r = self._client.messages.count_tokens(
            model=self.cfg.model, messages=[{"role": "user", "content": snapshot_text}])
        return int(r.input_tokens)

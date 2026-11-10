"""Snapshot serialisation and a conservative token estimate.

The REFLEX sees exactly one thing: the text produced by `render`. The budget (400 tokens)
is enforced with a deliberately pessimistic local estimator so the limit holds without a
network call. `tradebot reflex-check` measures the exact count against the real tokenizer
once an API key exists.
"""
from __future__ import annotations

import math
import re
from typing import Mapping

_TOKEN_RE = re.compile(r"[A-Za-z_]+|\d+|\s+|.", re.S)
SAFETY = 1.15


def estimate_tokens(text: str) -> int:
    """Upper-leaning estimate: letters ~4 chars/token, digits 2 per token, each symbol 1."""
    n = 0.0
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok.isspace():
            continue  # whitespace merges into the following token
        if tok[0].isalpha() or tok[0] == "_":
            n += math.ceil(len(tok) / 4)
        elif tok[0].isdigit():
            n += math.ceil(len(tok) / 2)
        else:
            n += 1
    return int(math.ceil(n * SAFETY))


def fmt(value: object) -> str:
    """Compact, fixed-precision formatting. None becomes `na` so the model sees the gap."""
    if value is None:
        return "na"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "na"
        a = abs(value)
        if a >= 1000:
            s = f"{value:.0f}"
        elif a >= 100:
            s = f"{value:.1f}"
        elif a >= 1:
            s = f"{value:.2f}"
        else:
            s = f"{value:.3f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return "0" if s in ("-0", "") else s
    return str(value)


def render(fields: Mapping[str, object]) -> str:
    return " ".join(f"{k}={fmt(v)}" for k, v in fields.items())

"""Pure feature math. Inputs are arrays of CLOSED bars, oldest first. No I/O, no clock."""
from __future__ import annotations

import math

import numpy as np

from ..types import Book


def pct_return(closes: np.ndarray, lookback: int) -> float | None:
    if len(closes) <= lookback or closes[-1 - lookback] <= 0:
        return None
    return float(closes[-1] / closes[-1 - lookback] - 1.0) * 100.0


def realized_vol(closes: np.ndarray, window: int) -> float | None:
    """Std of 1-bar log returns over `window` bars, in percent per bar."""
    if len(closes) < window + 1:
        return None
    seg = closes[-(window + 1):]
    if np.any(seg <= 0):
        return None
    lr = np.diff(np.log(seg))
    return float(np.std(lr, ddof=1)) * 100.0


def atr_pct(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> float | None:
    if len(close) < window + 1 or close[-1] <= 0:
        return None
    h, l, c_prev = high[-window:], low[-window:], close[-window - 1:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    return float(np.mean(tr) / close[-1]) * 100.0


def ema(values: np.ndarray, span: int) -> float | None:
    if len(values) < span:
        return None
    alpha = 2.0 / (span + 1.0)
    # seed with the simple mean of the first `span` points, then roll forward
    e = float(np.mean(values[:span]))
    for v in values[span:]:
        e = alpha * float(v) + (1.0 - alpha) * e
    return e


def range_position(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> float | None:
    if len(close) < window:
        return None
    hi, lo = float(np.max(high[-window:])), float(np.min(low[-window:]))
    if hi <= lo:
        return 0.5
    return float((close[-1] - lo) / (hi - lo))


def below_high_pct(high: np.ndarray, close: np.ndarray, window: int) -> float | None:
    if len(close) < window:
        return None
    hi = float(np.max(high[-window:]))
    if hi <= 0:
        return None
    return float(1.0 - close[-1] / hi) * 100.0


def zscore_last(values: np.ndarray, window: int) -> float | None:
    if len(values) < window + 1:
        return None
    ref = values[-(window + 1):-1]
    sd = float(np.std(ref, ddof=1))
    if sd <= 0 or not math.isfinite(sd):
        return 0.0
    return float((values[-1] - np.mean(ref)) / sd)


def book_features(book: Book, levels: int = 5) -> dict[str, float]:
    bids, asks = book.bids[:levels], book.asks[:levels]
    bid_sz = sum(s for _, s in bids)
    ask_sz = sum(s for _, s in asks)
    mid = book.mid
    spread_bps = (book.best_ask - book.best_bid) / mid * 1e4 if mid > 0 else float("nan")
    tot = bid_sz + ask_sz
    imbalance = (bid_sz - ask_sz) / tot if tot > 0 else 0.0
    return {"mid": mid, "spread_bps": spread_bps, "imbalance": imbalance, "bid_sz": bid_sz, "ask_sz": ask_sz}

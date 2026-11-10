"""Historical candles with an on-disk CSV cache (data_cache/, not committed)."""
from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

from ..types import Bar
from .feed import CcxtFeed, fill_gaps

log = logging.getLogger("tradebot.history")


def _cache_path(cache_dir: Path, exchange_id: str, symbol: str, timeframe: str) -> Path:
    safe = symbol.replace("/", "_").replace(":", "_")
    return cache_dir / f"{exchange_id}_{safe}_{timeframe}.csv"


def read_csv(path: Path, tf_ms: int) -> list[Bar]:
    if not path.exists():
        return []
    with open(path, "r", newline="") as fh:
        return [Bar(int(r["ts"]), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]),
                    float(r["volume"]), tf_ms) for r in csv.DictReader(fh)]


def write_csv(path: Path, bars: list[Bar]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ts", "open", "high", "low", "close", "volume"])
        for b in bars:
            w.writerow([b.ts, b.open, b.high, b.low, b.close, b.volume])


def fetch_range(feed: CcxtFeed, symbol: str, start_ms: int, end_ms: int, page: int = 300,
                pause_s: float = 0.12, max_empty_pages: int = 3) -> list[Bar]:
    """Page forward from start to end. Only bars already closed at `end_ms` are kept."""
    out: dict[int, Bar] = {}
    cursor = start_ms
    empty = 0
    while cursor < end_ms:
        bars = feed.closed_bars(symbol, limit=page, since_ms=cursor, now_ms=end_ms)
        fresh = [b for b in bars if b.ts >= cursor]
        if not fresh:
            empty += 1
            if empty >= max_empty_pages and out:
                break
            cursor += page * feed.tf_ms       # nothing here (before listing, or a long gap): step on
            continue
        empty = 0
        for b in fresh:
            out[b.ts] = b
        cursor = max(b.ts for b in fresh) + feed.tf_ms
        time.sleep(pause_s)
    return [out[t] for t in sorted(out) if out[t].close_ts <= end_ms]


def load_history(feed: CcxtFeed | None, cache_dir: str | Path, exchange_id: str, symbol: str, timeframe: str,
                 tf_ms: int, start_ms: int, end_ms: int, refresh: bool = False) -> tuple[list[Bar], set[int]]:
    """Return gap-filled bars for [start, end) plus the set of synthetic timestamps."""
    path = _cache_path(Path(cache_dir), exchange_id, symbol, timeframe)
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() and not refresh else {}
    cached = [] if refresh else read_csv(path, tf_ms)
    have_from = cached[0].ts if cached else None
    have_to = cached[-1].ts + tf_ms if cached else None
    # `searched_from` remembers how far back we already looked, so a symbol that simply was
    # not listed earlier is not re-queried on every run.
    searched_from = meta.get("searched_from", have_from)
    need_head = have_from is None or searched_from is None or start_ms < searched_from - tf_ms
    need_tail = have_to is None or end_ms > have_to
    if (need_head or need_tail) and feed is not None:
        merged = {b.ts: b for b in cached}
        if need_head:
            for b in fetch_range(feed, symbol, start_ms, have_from if have_from else end_ms):
                merged[b.ts] = b
            meta["searched_from"] = start_ms if searched_from is None else min(start_ms, searched_from)
        if need_tail and have_to is not None:
            for b in fetch_range(feed, symbol, have_to, end_ms):
                merged[b.ts] = b
        cached = [merged[t] for t in sorted(merged)]
        write_csv(path, cached)
        meta_path.write_text(json.dumps(meta))
        log.info("cached %d bars for %s", len(cached), symbol)
    window = [b for b in cached if start_ms <= b.ts and b.close_ts <= end_ms]
    return fill_gaps(window, tf_ms)

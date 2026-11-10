"""Market data through ccxt public endpoints (no credentials needed).

Verified against Coinbase on 2026-11-10 with ccxt 4.5.92:
* `fetch_ohlcv` returns the bar that is still forming as its last element, so every bar whose
  close time is after the exchange clock is dropped here;
* empty hours are omitted by the venue, so interior gaps are filled with flat zero-volume
  bars and flagged as synthetic (the state engine refuses a window with too many of them);
* funding, open interest and index price come from the single-product endpoint.
"""
from __future__ import annotations

import logging
import os
import time

from ..types import Bar, Book, Instrument

log = logging.getLogger("tradebot.feed")


def make_ccxt(exchange_id: str, credentials: dict | None = None):
    import ccxt

    opts = {"enableRateLimit": True, "timeout": 15000}
    # ccxt ignores HTTPS_PROXY unless told to trust the environment
    if os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"):
        opts["requests_trust_env"] = True
    if credentials:
        opts.update(credentials)
    return getattr(ccxt, exchange_id)(opts)


def fill_gaps(bars: list[Bar], tf_ms: int) -> tuple[list[Bar], set[int]]:
    """Insert flat zero-volume bars for interior gaps. Returns (bars, synthetic open timestamps)."""
    if not bars:
        return [], set()
    out: list[Bar] = [bars[0]]
    synthetic: set[int] = set()
    for b in bars[1:]:
        prev = out[-1]
        if b.ts <= prev.ts:
            continue
        t = prev.ts + tf_ms
        while t < b.ts:
            out.append(Bar(t, prev.close, prev.close, prev.close, prev.close, 0.0, tf_ms))
            synthetic.add(t)
            t += tf_ms
        out.append(b)
    return out, synthetic


class CcxtFeed:
    def __init__(self, exchange_id: str, timeframe: str, tf_ms: int, client=None):
        self.ex = client if client is not None else make_ccxt(exchange_id)
        self.timeframe = timeframe
        self.tf_ms = tf_ms
        self._markets_loaded = False

    def _load(self) -> None:
        if not self._markets_loaded:
            self.ex.load_markets()
            self._markets_loaded = True

    def server_time_ms(self) -> int:
        try:
            t = self.ex.fetch_time()
            if t:
                return int(t)
        except Exception as exc:
            log.warning("fetch_time failed: %s", type(exc).__name__)
        return int(time.time() * 1000)

    def instruments(self, market_type: str = "swap", quote: str = "USD") -> dict[str, Instrument]:
        self._load()
        out: dict[str, Instrument] = {}
        for sym, m in self.ex.markets.items():
            if m.get("type") != market_type or m.get("quote") != quote or not m.get("active", True):
                continue
            lim = (m.get("limits") or {}).get("amount") or {}
            prec = m.get("precision") or {}
            out[sym] = Instrument(
                symbol=sym, base=m.get("base") or sym.split("/")[0], exchange_id=str(m.get("id")),
                contract_size=float(m.get("contractSize") or 1.0), price_tick=float(prec.get("price") or 0.0),
                qty_step=float(prec.get("amount") or 1.0), min_qty=float(lim.get("min") or 1.0),
                max_qty=float(lim.get("max") or 1e9))
        return out

    def closed_bars(self, symbol: str, limit: int = 300, since_ms: int | None = None,
                    now_ms: int | None = None) -> list[Bar]:
        """Closed bars only, oldest first. The forming bar is never returned."""
        now_ms = self.server_time_ms() if now_ms is None else now_ms
        raw = self.ex.fetch_ohlcv(symbol, self.timeframe, since=since_ms, limit=limit)
        bars = [Bar(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5] or 0.0), self.tf_ms)
                for r in raw if r and r[0] is not None and r[4] is not None]
        bars.sort(key=lambda b: b.ts)
        return [b for b in bars if b.close_ts <= now_ms]

    def book(self, symbol: str, depth: int = 10) -> Book:
        ob = self.ex.fetch_order_book(symbol, limit=depth)
        ts = ob.get("timestamp") or self.server_time_ms()
        return Book(ts=int(ts), bids=tuple((float(p), float(s)) for p, s in ob["bids"][:depth]),
                    asks=tuple((float(p), float(s)) for p, s in ob["asks"][:depth]))

    def perp_details(self, symbol: str) -> dict:
        """Funding (per hour), open interest and index price, with the venue's own timestamps."""
        self._load()
        market = self.ex.market(symbol)
        raw = self.ex.v3PublicGetBrokerageMarketProductsProductId({"product_id": market["id"]})
        f = raw.get("future_product_details") or {}

        def num(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return None

        return {"funding_rate": num(f.get("funding_rate")), "open_interest": num(f.get("open_interest")),
                "index_price": num(f.get("index_price")), "funding_time": f.get("funding_time")}

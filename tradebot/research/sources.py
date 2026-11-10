"""Thin, dated readers for public market-data sources used by the research layer.

Every reader returns plain data plus the UTC time it was fetched. A reader that fails returns
an `unavailable` record with the reason: the caller must report the gap, never fill it.
Hosts used: api.coingecko.com, api.exchange.coinbase.com, api.coinbase.com, api.llama.fi,
stablecoins.llama.fi, api.hyperliquid.xyz, www.okx.com, api.alternative.me, fred.stlouisfed.org.
"""
from __future__ import annotations

import csv
import io
import json
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable

UA = {"User-Agent": "tradebot-research/0.1", "Accept": "application/json"}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def http_json(url: str, body: dict | None = None, timeout: float = 40.0, retries: int = 2) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    headers = dict(UA)
    if body is not None:
        headers["Content-Type"] = "application/json"
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except Exception as exc:  # network, HTTP or decode error
            last = exc
            time.sleep(2 * (attempt + 1))
    raise last  # type: ignore[misc]


def http_text(url: str, timeout: float = 40.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA["User-Agent"]})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def guarded(name: str, url: str, fn: Callable[[], Any]) -> dict:
    """Run one reader. Success and failure share a shape so gaps stay visible downstream."""
    try:
        return {"source": name, "url": url, "fetched": utc_now(), "ok": True, "data": fn()}
    except Exception as exc:
        return {"source": name, "url": url, "fetched": utc_now(), "ok": False,
                "error": f"{type(exc).__name__}: {str(exc)[:200]}", "data": None}


# ---- CoinGecko ---------------------------------------------------------------------
CG = "https://api.coingecko.com/api/v3"


def coingecko_global() -> dict:
    url = f"{CG}/global"
    return guarded("coingecko_global", url, lambda: http_json(url)["data"])


def coingecko_markets(n: int = 100) -> dict:
    url = (f"{CG}/coins/markets?vs_currency=usd&order=market_cap_desc&per_page={n}&page=1"
           "&price_change_percentage=24h,7d,30d,200d,1y")
    return guarded("coingecko_markets", url, lambda: http_json(url))


# ---- Coinbase ------------------------------------------------------------------------
def coinbase_daily(product: str, days: int = 300) -> dict:
    url = f"https://api.exchange.coinbase.com/products/{product}/candles?granularity=86400"

    def read():
        rows = http_json(url)                       # [time, low, high, open, close, volume], newest first
        rows = sorted(rows, key=lambda r: r[0])[-days:]
        return [{"ts": int(r[0]), "low": r[1], "high": r[2], "open": r[3], "close": r[4], "volume": r[5]} for r in rows]

    return guarded(f"coinbase_daily_{product}", url, read)


def coinbase_perps() -> dict:
    url = "https://api.coinbase.com/api/v3/brokerage/market/products?product_type=FUTURE&limit=250"

    def read():
        out = []
        for p in http_json(url).get("products", []):
            f = p.get("future_product_details") or {}
            if not f.get("funding_interval"):
                continue                            # dated futures have no funding
            out.append({
                "product_id": p["product_id"], "name": f.get("group_short_description"),
                "base": f.get("contract_root_unit"), "price": _f(p.get("price")),
                "contract_size": _f(f.get("contract_size")), "volume_24h_usd": _f(p.get("approximate_quote_24h_volume")),
                "open_interest_contracts": _f(f.get("open_interest")), "funding_rate_hourly": _f(f.get("funding_rate")),
                "funding_time": f.get("funding_time"), "index_price": _f(f.get("index_price")),
            })
        return out

    return guarded("coinbase_perps", url, read)


# ---- DefiLlama -----------------------------------------------------------------------
def llama_stablecoin_history() -> dict:
    url = "https://stablecoins.llama.fi/stablecoincharts/all"

    def read():
        rows = http_json(url)
        return [{"ts": int(r["date"]), "usd": float(r["totalCirculatingUSD"]["peggedUSD"])} for r in rows[-400:]]

    return guarded("llama_stablecoin_history", url, read)


def llama_overview(kind: str, data_type: str | None = None) -> dict:
    """kind: fees | dexs | derivatives. data_type e.g. dailyRevenue for fees."""
    url = f"https://api.llama.fi/overview/{kind}?excludeTotalDataChart=true&excludeTotalDataChartBreakdown=true"
    if data_type:
        url += f"&dataType={data_type}"

    def read():
        raw = http_json(url, timeout=90)
        keep = ("name", "displayName", "slug", "category", "chains", "total24h", "total7d", "total30d", "total1y",
                "change_1m", "change_7d", "parentProtocol")
        return {"total24h": raw.get("total24h"), "total7d": raw.get("total7d"), "total30d": raw.get("total30d"),
                "change_1m": raw.get("change_1m"),
                "protocols": [{k: p.get(k) for k in keep} for p in raw.get("protocols", [])]}

    name = f"llama_{kind}" + (f"_{data_type}" if data_type else "")
    return guarded(name, url, read)


def llama_chains() -> dict:
    url = "https://api.llama.fi/v2/chains"
    return guarded("llama_chains", url, lambda: http_json(url))


def llama_protocols() -> dict:
    url = "https://api.llama.fi/protocols"

    def read():
        keep = ("name", "slug", "symbol", "category", "tvl", "change_1d", "change_7d", "change_1m", "mcap", "gecko_id")
        return [{k: p.get(k) for k in keep} for p in http_json(url, timeout=90)]

    return guarded("llama_protocols", url, read)


# ---- Hyperliquid / OKX: cross-venue funding and open interest ---------------------------
def hyperliquid_ctx() -> dict:
    url = "https://api.hyperliquid.xyz/info"

    def read():
        meta, ctxs = http_json(url, body={"type": "metaAndAssetCtxs"})
        out = []
        for u, c in zip(meta["universe"], ctxs):
            out.append({"coin": u["name"], "funding_hourly": _f(c.get("funding")), "mark": _f(c.get("markPx")),
                        "open_interest_coins": _f(c.get("openInterest")), "volume_24h_usd": _f(c.get("dayNtlVlm")),
                        "premium": _f(c.get("premium"))})
        return out

    return guarded("hyperliquid_ctx", url, read)


def okx_funding(inst: str) -> dict:
    url = f"https://www.okx.com/api/v5/public/funding-rate?instId={inst}"
    return guarded(f"okx_funding_{inst}", url, lambda: http_json(url)["data"][0])


def okx_open_interest_history(ccy: str, period: str = "1D") -> dict:
    url = f"https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume?ccy={ccy}&period={period}"

    def read():   # rows: [ts, open interest usd, volume usd], newest first
        rows = http_json(url)["data"]
        return [{"ts": int(r[0]) // 1000, "oi_usd": float(r[1]), "vol_usd": float(r[2])} for r in rows][::-1]

    return guarded(f"okx_oi_{ccy}", url, read)


# ---- Sentiment / macro -------------------------------------------------------------------
def fear_greed(days: int = 35) -> dict:
    url = f"https://api.alternative.me/fng/?limit={days}"
    return guarded("fear_greed", url, lambda: [{"ts": int(d["timestamp"]), "value": int(d["value"]),
                                                "label": d["value_classification"]} for d in http_json(url)["data"]])


def fred_series(series_id: str, start: str) -> dict:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"

    def read():
        rows = []
        for r in csv.reader(io.StringIO(http_text(url))):
            if len(r) == 2 and r[0][:2] in ("19", "20"):
                try:
                    rows.append({"date": r[0], "value": float(r[1])})
                except ValueError:
                    continue                      # FRED writes "." for missing days
        return rows

    return guarded(f"fred_{series_id}", url, read)


def _f(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

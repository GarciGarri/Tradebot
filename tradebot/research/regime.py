"""Market regime read from live, dated data.

`collect()` pulls every source once and `derive()` turns it into a small set of numbers and
a regime label. Nothing is estimated: a metric whose source failed is reported as None and
listed under `gaps`. The label is a plain rule on the numbers, so it can be checked by hand.
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import sources as S

SECTORS = {
    "majors": ["btc", "eth"],
    "l1": ["sol", "bnb", "trx", "ada", "avax", "sui", "near", "hbar", "ton", "dot", "icp"],
    "payments": ["xrp", "xlm", "ltc", "bch"],
    "defi": ["link", "uni", "aave", "ena", "ondo", "morpho", "sky", "hype", "aster", "pump"],
    "privacy": ["zec", "xmr"],
    "memes": ["doge", "shib", "pepe"],
    "exchange": ["bnb", "leo", "okb", "cro", "bgb", "wbt", "htx"],
    "gold": ["paxg", "xaut"],
}


def collect() -> dict:
    start = (datetime.now(timezone.utc) - timedelta(days=200)).strftime("%Y-%m-%d")
    raw = {
        "global": S.coingecko_global(),
        "markets": S.coingecko_markets(100),
        "btc_daily": S.coinbase_daily("BTC-USD"),
        "eth_daily": S.coinbase_daily("ETH-USD"),
        "perps": S.coinbase_perps(),
        "stables": S.llama_stablecoin_history(),
        "hyperliquid": S.hyperliquid_ctx(),
        "okx_btc_funding": S.okx_funding("BTC-USDT-SWAP"),
        "okx_eth_funding": S.okx_funding("ETH-USDT-SWAP"),
        "okx_btc_oi": S.okx_open_interest_history("BTC"),
        "okx_eth_oi": S.okx_open_interest_history("ETH"),
        "fear_greed": S.fear_greed(),
    }
    for sid in ("DGS10", "DGS2", "DFF", "DTWEXBGS", "SP500", "VIXCLS", "BAMLH0A0HYM2"):
        raw[f"fred_{sid}"] = S.fred_series(sid, start)
    return raw


def _trend(daily: list[dict]) -> dict:
    closes = [d["close"] for d in daily]
    out: dict = {"last": closes[-1], "asof": datetime.fromtimestamp(daily[-1]["ts"], timezone.utc).strftime("%Y-%m-%d")}
    for n in (50, 200):
        out[f"sma{n}"] = statistics.fmean(closes[-n:]) if len(closes) >= n else None
        out[f"vs_sma{n}_pct"] = (closes[-1] / out[f"sma{n}"] - 1) * 100 if out[f"sma{n}"] else None
    for n in (7, 30, 90):
        out[f"ret_{n}d_pct"] = (closes[-1] / closes[-1 - n] - 1) * 100 if len(closes) > n else None
    rets = [closes[i] / closes[i - 1] - 1 for i in range(len(closes) - 30, len(closes))]
    out["rv_30d_annual_pct"] = statistics.pstdev(rets) * (365 ** 0.5) * 100
    hi = max(d["high"] for d in daily[-90:])
    out["below_90d_high_pct"] = (1 - closes[-1] / hi) * 100
    return out


def _chg(series: list[dict], key: str, days: int) -> float | None:
    if len(series) <= days or not series[-1 - days][key]:
        return None
    return (series[-1][key] / series[-1 - days][key] - 1) * 100


def derive(raw: dict) -> dict:
    gaps = [f"{k}: {v['error']}" for k, v in raw.items() if not v["ok"]]
    d: dict = {"generated": S.utc_now(), "gaps": gaps, "sources": {k: {"url": v["url"], "fetched": v["fetched"],
                                                                      "ok": v["ok"]} for k, v in raw.items()}}
    if raw["global"]["ok"]:
        g = raw["global"]["data"]
        d["market"] = {"total_mcap_usd": g["total_market_cap"]["usd"], "mcap_chg_24h_pct": g["market_cap_change_percentage_24h_usd"],
                       "btc_dominance_pct": g["market_cap_percentage"].get("btc"),
                       "eth_dominance_pct": g["market_cap_percentage"].get("eth"),
                       "stable_dominance_pct": sum(g["market_cap_percentage"].get(k, 0) for k in ("usdt", "usdc"))}
    for k in ("btc", "eth"):
        if raw[f"{k}_daily"]["ok"] and len(raw[f"{k}_daily"]["data"]) > 200:
            d[f"{k}_trend"] = _trend(raw[f"{k}_daily"]["data"])
    if raw["btc_daily"]["ok"] and raw["eth_daily"]["ok"]:
        b, e = raw["btc_daily"]["data"], raw["eth_daily"]["data"]
        ratio = [{"v": y["close"] / x["close"]} for x, y in zip(b[-91:], e[-91:])]
        d["eth_btc"] = {"last": ratio[-1]["v"], "chg_30d_pct": _chg(ratio, "v", 30), "chg_90d_pct": _chg(ratio, "v", 90)}
    if raw["stables"]["ok"]:
        s = raw["stables"]["data"]
        d["stablecoins"] = {"total_usd": s[-1]["usd"], "asof": datetime.fromtimestamp(s[-1]["ts"], timezone.utc).strftime("%Y-%m-%d"),
                            "chg_7d_pct": _chg(s, "usd", 7), "chg_30d_pct": _chg(s, "usd", 30), "chg_90d_pct": _chg(s, "usd", 90)}
    if raw["perps"]["ok"]:
        rows = [p for p in raw["perps"]["data"] if p["funding_rate_hourly"] is not None]
        fr = [p["funding_rate_hourly"] for p in rows]
        d["coinbase_perps"] = {
            "n": len(rows), "median_funding_hourly_bps": statistics.median(fr) * 1e4 if fr else None,
            "median_funding_annual_pct": statistics.median(fr) * 24 * 365 * 100 if fr else None,
            "share_negative": sum(1 for x in fr if x < 0) / len(fr) if fr else None,
            "by_base": {p["base"]: {"funding_hourly_bps": p["funding_rate_hourly"] * 1e4,
                                    "funding_annual_pct": p["funding_rate_hourly"] * 24 * 365 * 100,
                                    "oi_usd": (p["open_interest_contracts"] or 0) * (p["contract_size"] or 0) * (p["price"] or 0),
                                    "volume_24h_usd": p["volume_24h_usd"], "price": p["price"],
                                    "contract_usd": (p["contract_size"] or 0) * (p["price"] or 0)} for p in rows}}
    if raw["hyperliquid"]["ok"]:
        rows = [r for r in raw["hyperliquid"]["data"] if r["volume_24h_usd"] and r["volume_24h_usd"] > 5e6]
        fr = [r["funding_hourly"] for r in rows if r["funding_hourly"] is not None]
        oi = sum((r["open_interest_coins"] or 0) * (r["mark"] or 0) for r in rows)
        pick = {r["coin"]: r for r in raw["hyperliquid"]["data"]}
        d["hyperliquid"] = {"n_liquid": len(rows), "total_oi_usd": oi,
                            "median_funding_annual_pct": statistics.median(fr) * 24 * 365 * 100 if fr else None,
                            "btc_funding_annual_pct": (pick["BTC"]["funding_hourly"] * 24 * 365 * 100) if "BTC" in pick else None,
                            "eth_funding_annual_pct": (pick["ETH"]["funding_hourly"] * 24 * 365 * 100) if "ETH" in pick else None,
                            "by_coin": {c: {"funding_annual_pct": (r["funding_hourly"] or 0) * 24 * 365 * 100,
                                            "oi_usd": (r["open_interest_coins"] or 0) * (r["mark"] or 0),
                                            "volume_24h_usd": r["volume_24h_usd"]} for c, r in pick.items()
                                        if r["volume_24h_usd"] and r["volume_24h_usd"] > 5e6}}
    for k in ("btc", "eth"):
        f = raw[f"okx_{k}_funding"]
        if f["ok"]:
            d.setdefault("okx", {})[f"{k}_funding_8h_pct"] = float(f["data"]["fundingRate"]) * 100
        o = raw[f"okx_{k}_oi"]
        if o["ok"] and len(o["data"]) > 31:
            d.setdefault("okx", {})[f"{k}_oi_usd"] = o["data"][-1]["oi_usd"]
            d["okx"][f"{k}_oi_chg_7d_pct"] = _chg(o["data"], "oi_usd", 7)
            d["okx"][f"{k}_oi_chg_30d_pct"] = _chg(o["data"], "oi_usd", 30)
    if raw["fear_greed"]["ok"]:
        fg = raw["fear_greed"]["data"]
        d["fear_greed"] = {"value": fg[0]["value"], "label": fg[0]["label"],
                           "avg_30d": statistics.fmean(x["value"] for x in fg[:30])}
    macro = {}
    for sid in ("DGS10", "DGS2", "DFF", "DTWEXBGS", "SP500", "VIXCLS", "BAMLH0A0HYM2"):
        r = raw[f"fred_{sid}"]
        if r["ok"] and len(r["data"]) > 25:
            s = r["data"]
            macro[sid] = {"last": s[-1]["value"], "asof": s[-1]["date"], "chg_1m": s[-1]["value"] - s[-22]["value"],
                          "chg_1m_pct": (s[-1]["value"] / s[-22]["value"] - 1) * 100 if s[-22]["value"] else None,
                          "chg_3m_pct": (s[-1]["value"] / s[-64]["value"] - 1) * 100 if len(s) > 64 and s[-64]["value"] else None}
    d["macro"] = macro
    if raw["markets"]["ok"]:
        by = {c["symbol"].lower(): c for c in raw["markets"]["data"]}
        rot = {}
        for sector, syms in SECTORS.items():
            have = [by[s] for s in syms if s in by]
            if not have:
                continue
            rot[sector] = {f"median_{h}_pct": statistics.median(c.get(f"price_change_percentage_{h}_in_currency") or 0 for c in have)
                           for h in ("7d", "30d", "200d")}
            rot[sector]["n"] = len(have)
        d["rotation"] = rot
    d["regime"] = classify(d)
    return d


def classify(d: dict) -> dict:
    """Transparent rule set. Each vote is listed so the label can be audited."""
    votes: list[tuple[str, int, str]] = []

    def vote(name: str, value, bull, bear, text: str) -> None:
        if value is None:
            return
        votes.append((name, 1 if bull(value) else -1 if bear(value) else 0, text.format(value)))

    bt, et = d.get("btc_trend", {}), d.get("eth_trend", {})
    vote("btc_vs_sma200", bt.get("vs_sma200_pct"), lambda v: v > 0, lambda v: v < 0, "BTC {:+.1f}% vs 200d SMA")
    vote("btc_vs_sma50", bt.get("vs_sma50_pct"), lambda v: v > 0, lambda v: v < 0, "BTC {:+.1f}% vs 50d SMA")
    vote("btc_ret_90d", bt.get("ret_90d_pct"), lambda v: v > 10, lambda v: v < -10, "BTC {:+.1f}% over 90d")
    vote("eth_vs_sma200", et.get("vs_sma200_pct"), lambda v: v > 0, lambda v: v < 0, "ETH {:+.1f}% vs 200d SMA")
    vote("stables_30d", d.get("stablecoins", {}).get("chg_30d_pct"), lambda v: v > 1, lambda v: v < -0.5,
         "stablecoin supply {:+.2f}% over 30d")
    vote("funding", d.get("coinbase_perps", {}).get("median_funding_annual_pct"), lambda v: v > 5, lambda v: v < 0,
         "median perp funding {:+.1f}% annualised")
    vote("oi_30d", d.get("okx", {}).get("btc_oi_chg_30d_pct"), lambda v: v > 10, lambda v: v < -10,
         "BTC open interest {:+.1f}% over 30d")
    vote("dxy_1m", d.get("macro", {}).get("DTWEXBGS", {}).get("chg_1m_pct"), lambda v: v < -1, lambda v: v > 1,
         "broad dollar index {:+.2f}% over 1m")
    vote("vix", d.get("macro", {}).get("VIXCLS", {}).get("last"), lambda v: v < 16, lambda v: v > 25, "VIX {:.1f}")
    score = sum(v for _, v, _ in votes)
    n = len(votes)
    rv = bt.get("rv_30d_annual_pct")
    if n == 0:
        label = "unknown"
    elif score <= -max(3, n // 2):
        label = "risk_off_downtrend"
    elif score >= max(3, n // 2):
        label = "risk_on_uptrend"
    else:
        label = "mixed_range"
    if rv is not None and rv > 80:
        label += "_high_vol"
    return {"label": label, "score": score, "n_votes": n, "votes": [{"name": a, "vote": b, "evidence": c} for a, b, c in votes]}


def run(out_dir: str | Path, raw_dir: str | Path | None = None) -> dict:
    raw = collect()
    summary = derive(raw)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    day = summary["generated"][:10]
    (out / f"regime_{day}.json").write_text(json.dumps(summary, indent=1))
    if raw_dir is not None:
        rd = Path(raw_dir)
        rd.mkdir(parents=True, exist_ok=True)
        (rd / f"regime_raw_{day}.json").write_text(json.dumps(raw))
    return summary

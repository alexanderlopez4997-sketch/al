#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market-dashboard widgets for the web terminal: benchmark strip, signal
summary, sector rotation, top movers, data-freshness chips and the inline
sparkline SVG they share.

Ported from the Quantitative Terminal (Meridian repo's views/market.py). The
per-ticker aggregates take the rows that web_server._watchlist() already
builds (so a ticker's move/score has one source of truth) rather than
re-fetching, and nothing here imports web_server (it imports us).
"""
import glob
import os
import time

import quant_engine as qe
import quant_gui as g

# Static ticker -> sector heuristic, enough for a visual grouping (it is not a
# risk gate). Unknown tickers land in "Other".
SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", "META": "Technology",
    "NVDA": "Technology", "AMZN": "Technology",
    "AMD": "Semiconductors", "AVGO": "Semiconductors", "QCOM": "Semiconductors", "INTC": "Semiconductors",
    "MU": "Semiconductors", "ARM": "Semiconductors", "MRVL": "Semiconductors",
    "PLTR": "Growth/Fintech", "COIN": "Growth/Fintech", "SOFI": "Growth/Fintech", "XYZ": "Growth/Fintech",
    "PYPL": "Growth/Fintech", "HOOD": "Growth/Fintech", "UPST": "Growth/Fintech",
    "LLY": "Biotech/Pharma", "JNJ": "Biotech/Pharma", "ABBV": "Biotech/Pharma", "PFE": "Biotech/Pharma",
    "MRK": "Biotech/Pharma", "AMGN": "Biotech/Pharma",
    "TSLA": "Retail/Consumer", "NKE": "Retail/Consumer", "SBUX": "Retail/Consumer", "MCD": "Retail/Consumer",
    "CMG": "Retail/Consumer", "HD": "Retail/Consumer", "WMT": "Retail/Consumer",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "EOG": "Energy", "MPC": "Energy",
    "JPM": "Finance", "BAC": "Finance", "WFC": "Finance", "GS": "Finance", "MS": "Finance", "BLK": "Finance",
    "GME": "Volatile/Memes", "AMC": "Volatile/Memes", "MARA": "Volatile/Memes", "RIOT": "Volatile/Memes",
    "MSTR": "Volatile/Memes",
    "F": "Auto/EV", "GM": "Auto/EV", "RIVN": "Auto/EV", "LCID": "Auto/EV", "NIO": "Auto/EV",
    "SPY": "ETFs", "QQQ": "ETFs", "IWM": "ETFs", "DIA": "ETFs", "ARKK": "ETFs", "XLK": "ETFs", "XLC": "ETFs",
    "NFLX": "Growth", "ADBE": "Growth", "CRWD": "Growth", "SHOP": "Growth", "U": "Growth",
    "DDOG": "Growth", "NET": "Growth",
}

MARKET_INDICES = [("SPY", "S&P 500"), ("QQQ", "Nasdaq 100"), ("DIA", "Dow 30"), ("IWM", "Russell 2000")]

_UP, _DOWN = "#4ADE80", "#FF6B5E"


def price_spark_svg(closes, w=110, h=30, stroke=2, up=None):
    """Minimal inline price sparkline (no axes), colored by overall direction
    unless `up` is given so it can match the % change printed beside it."""
    vals = [float(v) for v in closes if v == v]
    if len(vals) < 2:
        return ""
    vmin, vmax = min(vals), max(vals)
    rng = (vmax - vmin) or (abs(vmax) or 1.0)
    step = w / (len(vals) - 1)
    pts = " ".join(f"{i * step:.1f},{h - (v - vmin) / rng * (h - stroke) - stroke / 2:.1f}"
                   for i, v in enumerate(vals))
    if up is None:
        up = vals[-1] >= vals[0]
    color = _UP if up else _DOWN
    return (f'<svg viewBox="0 0 {w} {h}" style="width:100%;height:{h}px;display:block" preserveAspectRatio="none">'
            f'<polygon points="0,{h} {pts} {w},{h}" fill="{color}" fill-opacity=".12" stroke="none"/>'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{stroke}" '
            f'stroke-linejoin="round" stroke-linecap="round"/></svg>')


def market_overview(demo):
    """SPY / QQQ / DIA / IWM last price, day change and a 60-day sparkline."""
    tickers = [t for t, _ in MARKET_INDICES]
    data = ({t: qe.demo_data(t) for t in tickers} if demo
            else g.fetch_many_concurrent(tickers, "6mo", "1d"))
    out = []
    for t, name in MARKET_INDICES:
        df = data.get(t)
        if df is None or len(df) < 30:
            continue
        c = df["Close"].astype(float)
        last, prev = float(c.iloc[-1]), float(c.iloc[-2])
        chg = round((last / prev - 1) * 100, 2)
        out.append({"ticker": t, "name": name, "last": round(last, 2), "chg": chg,
                    "spark": price_spark_svg(c.tail(60).values, w=140, h=36, up=chg >= 0)})
    return {"indices": out}


def sector_rotation(rows):
    """Per-sector average move, momentum (mean score) and winner/loser counts,
    best sector first. `rows` are web_server._watchlist() rows."""
    groups = {}
    for r in rows:
        groups.setdefault(SECTOR_MAP.get(r["ticker"], "Other"), []).append(r)
    sectors = []
    for name, rs in groups.items():
        sectors.append({
            "name": name,
            "avg_chg": round(sum(r["chg"] for r in rs) / len(rs), 2),
            "momentum": round(sum(r["score"] for r in rs) / len(rs)),
            "winners": sum(1 for r in rs if r["chg"] > 0),
            "losers": sum(1 for r in rs if r["chg"] < 0),
            "count": len(rs),
            "tickers": ",".join(r["ticker"] for r in sorted(rs, key=lambda x: -x["chg"])),
        })
    sectors.sort(key=lambda s: -s["avg_chg"])
    return {"sectors": sectors}


def top_movers(rows, n=3):
    """The n biggest gainers and the n biggest losers by day change."""
    by_chg = sorted(rows, key=lambda r: -r["chg"])
    pick = lambda rs: [{"ticker": r["ticker"], "chg": r["chg"], "last": r["last"],
                        "score": r["score"], "verdict": r["verdict"]} for r in rs]
    return {"gainers": pick([r for r in by_chg[:n] if r["chg"] > 0]),
            "losers": pick([r for r in by_chg[::-1][:n] if r["chg"] < 0])}


def signal_summary(rows):
    """Buy / hold / avoid split of the watchlist verdicts plus the mean score."""
    count = lambda word: sum(1 for r in rows if word in r["verdict"].lower())
    total = len(rows)
    return {"buy": count("buy"), "hold": count("hold"), "avoid": count("avoid"), "total": total,
            "avg_score": round(sum(r["score"] for r in rows) / total, 1) if total else 0}


# (label, glob under ~/.meridian_cache, warn after N hours, stale after N hours)
_CACHE_DIR = os.path.expanduser("~/.meridian_cache")
HEALTH_CHECKS = [
    ("EDGAR Index", "edgar_ciks.json", 24 * 14, 24 * 45),
    ("EDGAR Log", "edgar.log", 24, 24 * 7),
    ("Verdict Journal", "verdict_journal.json", 24 * 4, 24 * 14),
    ("Edge Tracker", "meridian_cache.db", 24, 24 * 7),
]


def system_health(base=None, now=None, checks=None):
    """How fresh each local data source is. status: ok / warn / stale / missing."""
    base = base or _CACHE_DIR
    now = time.time() if now is None else now
    out = []
    for label, pattern, warn_h, stale_h in (checks or HEALTH_CHECKS):
        matches = glob.glob(os.path.join(base, pattern))
        if not matches:
            out.append({"label": label, "age_hours": None, "status": "missing"})
            continue
        age_h = (now - max(os.path.getmtime(p) for p in matches)) / 3600.0
        status = "stale" if age_h >= stale_h else "warn" if age_h >= warn_h else "ok"
        out.append({"label": label, "age_hours": round(age_h, 1), "status": status})
    return {"checks": out}

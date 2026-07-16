#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fundamental (value/quality) filter for Meridian's Screener.

Layers a fundamentals gate on top of the technical screen: P/E, Debt/Equity,
revenue growth, current ratio.

SOURCES (tried in order):
  1. Finnhub /stock/metric — PRIMARY. One call per ticker, free tier 60/min,
     and it reuses the Finnhub key the app already uses for analyst recs, so no
     extra setup. This is what makes a full-universe scan practical.
  2. Alpha Vantage — FALLBACK. Two calls/ticker. Free tier is 25/DAY; a premium
     key lifts that to 75/min (set _MIN_INTERVAL accordingly). Used when Finnhub
     lacks a metric. Key via env ALPHA_VANTAGE_KEY.

Results are cached on disk (fundamentals change quarterly) so repeat scans are
free. Demo mode uses deterministic synthetic fundamentals — no key, no network.
"""
import hashlib
import json
import os
import time

CACHE_DIR = os.path.expanduser("~/.meridian_cache")
CACHE_TTL = 24 * 3600          # fundamentals change quarterly; a day is plenty fresh
AV_BASE = "https://www.alphavantage.co/query"
_MIN_INTERVAL = 0.85           # seconds between AV calls (premium 75/min ≈ 0.8s)
_last_call = [0.0]


def _cache_path(ticker):
    return os.path.join(CACHE_DIR, f"fund_{ticker.upper()}.json")


def _read_cache(ticker):
    try:
        with open(_cache_path(ticker)) as f:
            obj = json.load(f)
        if time.time() - obj.get("_ts", 0) < CACHE_TTL:
            return obj.get("data")
    except Exception:
        pass
    return None


def _write_cache(ticker, data):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(_cache_path(ticker), "w") as f:
            json.dump({"_ts": time.time(), "data": data}, f)
    except Exception:
        pass


def _throttle():
    """Space live calls out to stay under Alpha Vantage's 5-per-minute limit."""
    wait = _MIN_INTERVAL - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()


_last_finnhub = [0.0]
FINNHUB_MIN_INTERVAL = 1.05          # 60 calls/min free tier -> ~1/sec


def _av_get(params, timeout=15):
    import urllib.request
    import urllib.parse
    _throttle()
    url = AV_BASE + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _finnhub_get(url, timeout=10, throttle=True):
    import urllib.request
    if throttle:                                  # serial 1/sec; skipped for concurrent batches
        wait = FINNHUB_MIN_INTERVAL - (time.time() - _last_finnhub[0])
        if wait > 0:
            time.sleep(wait)
        _last_finnhub[0] = time.time()
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _num(x):
    """Coerce API values (which may be strings, 'None', '-', or '') to float/None."""
    try:
        if x in (None, "None", "-", ""):
            return None
        return float(x)
    except (ValueError, TypeError):
        return None


def _pick(m, *keys):
    for k in keys:
        v = _num(m.get(k))
        if v is not None:
            return v
    return None


def _from_finnhub(ticker, key, throttle=True):
    """Fundamentals from Finnhub /stock/metric — ONE call, ~all metrics ready-made.
    Free tier 60/min. Reuses the key the app already uses for analyst recs."""
    data = _finnhub_get(f"https://finnhub.io/api/v1/stock/metric"
                        f"?symbol={ticker}&metric=all&token={key}", throttle=throttle)
    m = data.get("metric") if isinstance(data, dict) else None
    if not m:
        return None
    out = {"pe": _pick(m, "peTTM", "peBasicExclExtraTTM"),
           "de": _pick(m, "totalDebt/totalEquityQuarterly", "totalDebt/totalEquityAnnual",
                       "longTermDebt/equityQuarterly"),
           "growth": _pick(m, "revenueGrowthTTMYoy", "revenueGrowthQuarterlyYoy"),
           "current_ratio": _pick(m, "currentRatioQuarterly", "currentRatioAnnual"),
           "name": ticker, "sector": ""}
    if all(out[k] is None for k in ("pe", "de", "growth", "current_ratio")):
        return None
    return out


def _from_alpha_vantage(ticker, key):
    """Fundamentals from Alpha Vantage (OVERVIEW + BALANCE_SHEET). Two calls;
    free tier only 25/DAY. Kept as a fallback when Finnhub has no data."""
    out = {"pe": None, "de": None, "growth": None, "current_ratio": None,
           "name": ticker, "sector": ""}
    ov = _av_get({"function": "OVERVIEW", "symbol": ticker, "apikey": key})
    if isinstance(ov, dict) and ov.get("Symbol"):
        out["pe"] = _num(ov.get("PERatio"))
        g = _num(ov.get("QuarterlyRevenueGrowthYOY"))         # fraction, e.g. 0.15
        out["growth"] = g * 100 if g is not None else None
        out["name"] = ov.get("Name") or ticker
        out["sector"] = ov.get("Sector") or ""
    bs = _av_get({"function": "BALANCE_SHEET", "symbol": ticker, "apikey": key})
    reports = bs.get("quarterlyReports") if isinstance(bs, dict) else None
    if reports:
        r0 = reports[0]
        debt = _num(r0.get("shortLongTermDebtTotal"))
        if debt is None:
            debt = _num(r0.get("totalLiabilities"))
        eq = _num(r0.get("totalShareholderEquity"))
        ca = _num(r0.get("totalCurrentAssets"))
        cl = _num(r0.get("totalCurrentLiabilities"))
        if debt is not None and eq not in (None, 0):
            out["de"] = debt / eq
        if ca is not None and cl not in (None, 0):
            out["current_ratio"] = ca / cl
    if all(out[k] is None for k in ("pe", "de", "growth", "current_ratio")):
        return None
    return out


def fetch_fundamentals(ticker, finnhub_key=None, av_key=None, use_cache=True, throttle=True):
    """Return {pe, de, growth, current_ratio, name, sector} for a ticker, or None.
    Tries Finnhub first (1 call, 60/min, reuses the app's key), falls back to
    Alpha Vantage (2 calls). Cached on disk. `throttle=False` skips the serial
    1/sec pacing — used by fetch_fundamentals_batch for concurrent bursts."""
    if use_cache:
        cached = _read_cache(ticker)
        if cached is not None:
            return cached
    out = None
    for src, key in (("finnhub", finnhub_key), ("av", av_key)):
        if not key:
            continue
        try:
            out = (_from_finnhub(ticker, key, throttle=throttle) if src == "finnhub"
                   else _from_alpha_vantage(ticker, key))
        except Exception:
            out = None
        if out:
            break
    if out and use_cache:
        _write_cache(ticker, out)
    return out


def fetch_fundamentals_batch(tickers, finnhub_key=None, av_key=None, workers=8):
    """Fetch fundamentals for many tickers CONCURRENTLY → {ticker: fund|None}.
    Finnhub's limit is 60 per MINUTE (not 1/sec), so a burst of up to ~60 names
    lands within budget and finishes in a couple of seconds instead of a minute
    of serial waits. Cached names cost nothing. Fails open per-ticker."""
    from concurrent.futures import ThreadPoolExecutor
    out = {}
    def one(t):
        try:
            out[t] = fetch_fundamentals(t, finnhub_key, av_key, throttle=False)
        except Exception:
            out[t] = None
    if tickers:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, tickers))
    return out


def demo_fundamentals(ticker):
    """Deterministic synthetic fundamentals for offline/demo mode (no API)."""
    h = int(hashlib.md5(ticker.upper().encode()).hexdigest(), 16)
    return {"pe": round(8 + (h % 600) / 10.0, 1),                 # 8 .. 68
            "de": round(((h >> 8) % 300) / 100.0, 2),            # 0 .. 3
            "growth": round(((h >> 16) % 800) / 10.0 - 20, 1),   # -20 .. +60
            "current_ratio": round(0.5 + ((h >> 24) % 400) / 100.0, 2),  # 0.5 .. 4.5
            "name": ticker, "sector": "Demo"}


def passes_fundamental_filter(fund, max_pe=None, max_de=None,
                              min_growth=None, min_current=None):
    """True if `fund` clears every ACTIVE constraint. A constraint only applies
    when both the threshold is set and that metric is present — a missing metric
    won't fail a filter, but no data at all (fund is None) does not pass."""
    if not fund:
        return False
    if max_pe and fund.get("pe") is not None and fund["pe"] > max_pe:
        return False
    if max_de is not None and fund.get("de") is not None and fund["de"] > max_de:
        return False
    if min_growth is not None and fund.get("growth") is not None and fund["growth"] < min_growth:
        return False
    if min_current is not None and fund.get("current_ratio") is not None and fund["current_ratio"] < min_current:
        return False
    return True


def fmt_fund(fund):
    """Compact one-line summary for the Screener row, or '' if none."""
    if not fund:
        return ""
    pe = f"{fund['pe']:.0f}" if fund.get("pe") is not None else "—"
    de = f"{fund['de']:.1f}" if fund.get("de") is not None else "—"
    g = f"{fund['growth']:+.0f}%" if fund.get("growth") is not None else "—"
    return f"P/E {pe} · D/E {de} · G {g}"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep company research for Meridian's Analyze view.

fundamental_engine.py answers "does this pass a quick P/E · D/E · growth ·
current-ratio gate" for the Screener, across a whole watchlist at once. This
module answers a different question — "what does this company actually do,
and is the earnings quality real" — for the ONE ticker a user is looking at
right now, so it affords a couple of extra API calls that would not scale
across a screener batch.

SOURCE: Alpha Vantage OVERVIEW + CASH_FLOW (free tier: 25 requests/DAY total
across the whole app — this module is the most AV-hungry consumer of that
budget, so results are cached hard: a week, not a day). Key via env
ALPHA_VANTAGE_KEY (same key fundamental_engine.py falls back to).

Deliberately NOT included: peer/sector-average multiples and multi-year
trend lines for margins or share count. Alpha Vantage's free tier has no
historical-fundamentals endpoint, and Finnhub's free tier caps out at the
same TTM snapshot fundamental_engine.py already uses — faking a trend from
a single data point would be worse than not showing one. What IS shown is
real: this quarter's numbers, and links to the primary filings for anyone
who wants the multi-year picture themselves.
"""
import hashlib
import json
import os
import time

CACHE_DIR = os.path.expanduser("~/.meridian_cache")
CACHE_TTL = 7 * 24 * 3600      # AV free tier is 25 req/day total; cache hard
AV_BASE = "https://www.alphavantage.co/query"
_MIN_INTERVAL = 0.85
_last_call = [0.0]


def _cache_path(ticker):
    return os.path.join(CACHE_DIR, f"research_{ticker.upper()}.json")


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
    wait = _MIN_INTERVAL - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()


def _av_get(params, timeout=15):
    import urllib.request
    import urllib.parse
    _throttle()
    url = AV_BASE + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _num(x):
    try:
        if x in (None, "None", "-", ""):
            return None
        return float(x)
    except (ValueError, TypeError):
        return None


def fetch_company_research(ticker, av_key, use_cache=True):
    """Business overview + valuation + quality + ownership for `ticker`, or
    None if no key / lookup failed. Shape: {
      name, description, sector, industry, exchange, employees,
      market_cap, pe, forward_pe, peg, ps, pb, ev_revenue, ev_ebitda,
      profit_margin, operating_margin, roe, roa,
      revenue_ttm, net_income_ttm, fcf_ttm, fcf_ni_gap_pct,
      dividend_yield, beta, week52_high, week52_low, analyst_target }
    Any individual field may be None where the source didn't report it —
    fails open per-field rather than discarding the whole result."""
    if not av_key:
        return None
    if use_cache:
        cached = _read_cache(ticker)
        if cached is not None:
            return cached
    ov = _av_get({"function": "OVERVIEW", "symbol": ticker, "apikey": av_key})
    if not (isinstance(ov, dict) and ov.get("Symbol")):
        return None
    revenue_ttm = _num(ov.get("RevenueTTM"))
    profit_margin = _num(ov.get("ProfitMargin"))
    net_income_ttm = (revenue_ttm * profit_margin
                       if revenue_ttm is not None and profit_margin is not None else None)

    fcf_ttm = None
    try:
        cf = _av_get({"function": "CASH_FLOW", "symbol": ticker, "apikey": av_key})
        reports = cf.get("annualReports") if isinstance(cf, dict) else None
        if reports:
            r0 = reports[0]
            ocf = _num(r0.get("operatingCashflow"))
            capex = _num(r0.get("capitalExpenditures"))
            if ocf is not None and capex is not None:
                fcf_ttm = ocf - abs(capex)
    except Exception:
        pass

    fcf_ni_gap_pct = None
    if fcf_ttm is not None and net_income_ttm not in (None, 0):
        fcf_ni_gap_pct = (fcf_ttm - net_income_ttm) / abs(net_income_ttm) * 100

    out = {
        "name": ov.get("Name") or ticker,
        "description": ov.get("Description") or "",
        "sector": ov.get("Sector") or "",
        "industry": ov.get("Industry") or "",
        "exchange": ov.get("Exchange") or "",
        "employees": _num(ov.get("FullTimeEmployees")),
        "market_cap": _num(ov.get("MarketCapitalization")),
        "pe": _num(ov.get("TrailingPE")) or _num(ov.get("PERatio")),
        "forward_pe": _num(ov.get("ForwardPE")),
        "peg": _num(ov.get("PEGRatio")),
        "ps": _num(ov.get("PriceToSalesRatioTTM")),
        "pb": _num(ov.get("PriceToBookRatio")),
        "ev_revenue": _num(ov.get("EVToRevenue")),
        "ev_ebitda": _num(ov.get("EVToEBITDA")),
        "profit_margin": profit_margin * 100 if profit_margin is not None else None,
        "operating_margin": (_num(ov.get("OperatingMarginTTM")) * 100
                              if _num(ov.get("OperatingMarginTTM")) is not None else None),
        "roe": _num(ov.get("ReturnOnEquityTTM")) * 100 if _num(ov.get("ReturnOnEquityTTM")) is not None else None,
        "roa": _num(ov.get("ReturnOnAssetsTTM")) * 100 if _num(ov.get("ReturnOnAssetsTTM")) is not None else None,
        "revenue_ttm": revenue_ttm,
        "net_income_ttm": net_income_ttm,
        "fcf_ttm": fcf_ttm,
        "fcf_ni_gap_pct": fcf_ni_gap_pct,
        "dividend_yield": _num(ov.get("DividendYield")) * 100 if _num(ov.get("DividendYield")) is not None else None,
        "beta": _num(ov.get("Beta")),
        "week52_high": _num(ov.get("52WeekHigh")),
        "week52_low": _num(ov.get("52WeekLow")),
        "analyst_target": _num(ov.get("AnalystTargetPrice")),
    }
    if use_cache:
        _write_cache(ticker, out)
    return out


def demo_company_research(ticker):
    """Deterministic synthetic research data for offline/demo mode — same
    approach as fundamental_engine.demo_fundamentals and quant_engine.demo_data."""
    h = int(hashlib.md5(ticker.upper().encode()).hexdigest(), 16)
    revenue_ttm = 5e9 + (h % 900) * 1e7
    profit_margin = 0.02 + ((h >> 4) % 300) / 1000.0     # 2% .. 32%
    net_income_ttm = revenue_ttm * profit_margin
    fcf_ttm = net_income_ttm * (0.6 + ((h >> 12) % 90) / 100.0)   # 0.6x .. 1.5x NI
    return {
        "name": f"{ticker} Corp (demo)",
        "description": ("[DEMO] Synthetic company overview — no live description available "
                         "without ALPHA_VANTAGE_KEY. In live mode this is the company's own "
                         "business summary."),
        "sector": "Demo Sector", "industry": "Demo Industry", "exchange": "DEMO",
        "employees": 1000 + (h % 50000),
        "market_cap": revenue_ttm * (2 + (h % 5)),
        "pe": round(8 + (h % 500) / 10.0, 1), "forward_pe": round(8 + ((h >> 8) % 400) / 10.0, 1),
        "peg": round(0.5 + ((h >> 16) % 250) / 100.0, 2),
        "ps": round(1 + (h % 150) / 10.0, 1), "pb": round(1 + ((h >> 4) % 200) / 10.0, 1),
        "ev_revenue": round(1 + ((h >> 8) % 120) / 10.0, 1), "ev_ebitda": round(5 + ((h >> 12) % 300) / 10.0, 1),
        "profit_margin": round(profit_margin * 100, 1),
        "operating_margin": round(profit_margin * 100 * 1.2, 1),
        "roe": round(5 + ((h >> 20) % 400) / 10.0, 1), "roa": round(2 + ((h >> 24) % 200) / 10.0, 1),
        "revenue_ttm": revenue_ttm, "net_income_ttm": net_income_ttm, "fcf_ttm": fcf_ttm,
        "fcf_ni_gap_pct": round((fcf_ttm - net_income_ttm) / abs(net_income_ttm) * 100, 1) if net_income_ttm else None,
        "dividend_yield": round(((h >> 28) % 40) / 10.0, 2), "beta": round(0.5 + (h % 200) / 100.0, 2),
        "week52_high": None, "week52_low": None, "analyst_target": None,
    }

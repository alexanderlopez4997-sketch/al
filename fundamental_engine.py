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
from datetime import date

CACHE_DIR = os.path.expanduser("~/.meridian_cache")
CACHE_TTL = 24 * 3600          # fundamentals change quarterly; a day is plenty fresh
AV_BASE = "https://www.alphavantage.co/query"
_MIN_INTERVAL = 0.85           # seconds between AV calls (premium 75/min ≈ 0.8s)
_last_call = [0.0]


def _cache_path(ticker, prefix="fund"):
    return os.path.join(CACHE_DIR, f"{prefix}_{ticker.upper()}.json")


def _read_cache(ticker, prefix="fund"):
    try:
        with open(_cache_path(ticker, prefix)) as f:
            obj = json.load(f)
        if time.time() - obj.get("_ts", 0) < CACHE_TTL:
            return obj.get("data")
    except Exception:
        pass
    return None


def _write_cache(ticker, data, prefix="fund"):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(_cache_path(ticker, prefix), "w") as f:
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


# ============================================================ QUARTERLY EARNINGS (SEC XBRL) ===
# Reads the latest 10-Q numbers straight from SEC's company-facts XBRL feed
# (no API key, no third-party package). Fetching reuses edgar.py's rate-limited,
# TTL-cached client; the parsing below is pure so it can be tested offline.
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_REVENUE_CONCEPTS = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                     "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet")
_NET_INCOME_CONCEPTS = ("NetIncomeLoss", "ProfitLoss")
_OP_INCOME_CONCEPTS = ("OperatingIncomeLoss",)
_QUARTER_DAYS = (70, 105)        # a single fiscal quarter; 10-Qs also carry 6/9-month YTD rows
_NINE_MONTH_DAYS = (250, 290)    # Q1-Q3 year-to-date row in the Q3 10-Q
_FISCAL_YEAR_DAYS = (350, 380)   # 52/53-week and calendar fiscal years
EARNINGS_CACHE_PREFIX = "q10"


def _days(start, end):
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def _latest_by(rows, keyfn, form_prefix, day_range):
    """{key: (val, filed, row)} for rows of the given form whose period length is in
    day_range; when a period was reported in several filings the latest filing wins
    (picks up restatements)."""
    out = {}
    for r in rows:
        try:
            if not str(r.get("form", "")).startswith(form_prefix) or r.get("val") is None:
                continue
            if not day_range[0] <= _days(r["start"], r["end"]) <= day_range[1]:
                continue
            k = keyfn(r)
            if k not in out or r.get("filed", "") >= out[k][1]:
                out[k] = (float(r["val"]), r.get("filed", ""), r)
        except (KeyError, ValueError, TypeError):
            continue
    return out


def _quarterly_series(facts, concepts):
    """{period_end: (value, period_start, derived)} of single quarters under the concept
    with the most recent quarter. 10-Qs only report Q1-Q3, so each fiscal Q4 is derived
    as the 10-K full-year value minus the nine-month YTD in that year's Q3 10-Q
    (`derived=True`); without the YTD row there is no Q4 and the series stops at Q3."""
    best = {}
    gaap = (facts or {}).get("facts", {}).get("us-gaap", {})
    for concept in concepts:
        rows = gaap.get(concept, {}).get("units", {}).get("USD", [])
        cur = {end: (v, r["start"], False) for end, (v, _, r) in
               _latest_by(rows, lambda r: r["end"], "10-Q", _QUARTER_DAYS).items()}
        ytd9 = _latest_by(rows, lambda r: r["start"], "10-Q", _NINE_MONTH_DAYS)
        for (start, end), (annual, _, _) in _latest_by(
                rows, lambda r: (r["start"], r["end"]), "10-K", _FISCAL_YEAR_DAYS).items():
            nine = ytd9.get(start)
            # the YTD must end ~one quarter before the fiscal year does, else it's another year's row
            if end in cur or not nine or not 60 <= _days(nine[2]["end"], end) <= 120:
                continue
            cur[end] = (annual - nine[0], start, True)
        # companies switch revenue tags over time: prefer the tag with the freshest quarter
        if cur and (not best or max(cur) > max(best)):
            best = cur
    return best


def _nearest(series, end, target_days, tol):
    """Value of the quarter ending ~target_days before `end` (within ±tol days), else None."""
    hits = [(abs(_days(e, end) - target_days), v[0]) for e, v in series.items()
            if abs(_days(e, end) - target_days) <= tol]
    return min(hits)[1] if hits else None


def _pct_change(new, old):
    """Relative change that stays meaningful across a negative base (loss -> smaller loss is +)."""
    if new is None or old in (None, 0):
        return None
    return (new - old) / abs(old)


def _clip(x, lim=1.0):
    return max(-lim, min(lim, x))


def earnings_from_facts(facts):
    """Pure transform: SEC company-facts JSON -> quarterly-earnings dict. Returns
    {"status": "no_data", "signal": 0.0, ...} when nothing usable."""
    rev = _quarterly_series(facts, _REVENUE_CONCEPTS)
    ni = _quarterly_series(facts, _NET_INCOME_CONCEPTS)
    op = _quarterly_series(facts, _OP_INCOME_CONCEPTS)
    ends = list(rev) or list(ni)
    if not ends:
        return {"status": "no_data", "signal": 0.0, "detail": "no 10-Q quarterly facts"}
    anchor = max(ends)

    def val(series):
        return series[anchor][0] if anchor in series else None

    revenue, net_income, op_income = val(rev), val(ni), val(op)
    # Prior-year quarter is ~364 days back and prior quarter ~91; both exist for every
    # quarter (including a derived Q4) as long as the earlier filings are in the facts feed.
    prev_rev = _nearest(rev, anchor, 364, 20)
    rev_yoy = _pct_change(revenue, prev_rev)
    rev_qoq = _pct_change(revenue, _nearest(rev, anchor, 91, 20))
    ni_yoy = _pct_change(net_income, _nearest(ni, anchor, 364, 20))
    margin = op_income / revenue if op_income is not None and revenue else None
    prev_op = _nearest(op, anchor, 364, 20)
    prev_margin = prev_op / prev_rev if prev_op is not None and prev_rev else None
    margin_chg = (margin - prev_margin) * 100 if margin is not None and prev_margin is not None else None

    # Signal in [-1, 1]: revenue growth saturates at +/-30% YoY, net income at +/-50%,
    # operating margin change at +/-5pp. Missing components are dropped, weights renormalised.
    parts = [(0.4, rev_yoy / 0.30 if rev_yoy is not None else None),
             (0.3, ni_yoy / 0.50 if ni_yoy is not None else None),
             (0.3, margin_chg / 5.0 if margin_chg is not None else None)]
    live = [(w, _clip(x)) for w, x in parts if x is not None]
    signal = round(sum(w * x for w, x in live) / sum(w for w, _ in live), 3) if live else 0.0

    def pct(x):
        return f"{x * 100:+.1f}%" if x is not None else "n/a"
    rev_s = f"${revenue:,.0f}" if revenue is not None else "n/a"
    margin_s = f"{margin * 100:.1f}%" if margin is not None else "n/a"
    chg_s = f" ({margin_chg:+.1f}pp YoY)" if margin_chg is not None else ""
    detail = (f"Quarter ended {anchor}: revenue {rev_s} (YoY {pct(rev_yoy)}, QoQ {pct(rev_qoq)}); "
              f"net income YoY {pct(ni_yoy)}; op margin {margin_s}{chg_s}")
    return {"status": "success" if live else "no_data", "signal": signal, "period_end": anchor,
            "revenue": revenue, "net_income": net_income, "operating_income": op_income,
            "revenue_yoy": rev_yoy, "revenue_qoq": rev_qoq, "net_income_yoy": ni_yoy,
            "operating_margin": margin, "margin_change_pp": margin_chg,
            "derived_q4": any(series[anchor][2] for series in (rev, ni, op) if anchor in series),
            "detail": detail}


def analyze_quarterly_earnings(ticker: str, use_cache: bool = True) -> dict:
    """Latest quarter's performance from SEC XBRL company facts: revenue growth (YoY/QoQ),
    net income shift and operating-margin trend, folded into a bounded `signal` in [-1, 1].
    The quarter is the newest 10-Q quarter, or the fiscal Q4 derived from the 10-K once that
    is filed (`derived_q4`). Fails soft (Meridian rule): always returns a dict with
    `status` and `signal`, never raises."""
    try:
        ticker = ticker.upper()
        if use_cache:
            cached = _read_cache(ticker, EARNINGS_CACHE_PREFIX)
            if cached is not None:
                return cached
        import edgar                      # repo's SEC client (rate-limited + cached), not PyPI edgartools
        cik = edgar._load_ciks().get(ticker)
        if not cik:
            return {"status": "no_data", "signal": 0.0, "detail": f"no SEC CIK for {ticker}"}
        facts = json.loads(edgar._get(COMPANY_FACTS_URL.format(cik=cik), timeout=30))
        out = earnings_from_facts(facts)
        out["ticker"] = ticker
        if use_cache and out["status"] == "success":
            _write_cache(ticker, out, EARNINGS_CACHE_PREFIX)
        return out
    except Exception as e:
        return {"status": "error", "signal": 0.0, "detail": str(e)}

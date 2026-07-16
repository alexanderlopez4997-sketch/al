#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEC EDGAR filing monitor for Meridian — the HONEST version of "institutional intent".

You cannot read who is buying/selling from the anonymous tape. But when an insider
or a company acts, they are LEGALLY REQUIRED to disclose it to the SEC — and those
filings are public and timestamped to the second. So instead of guessing intent
from prints, this reads the actual disclosures that move stocks after-hours:

  Form 4        an insider (CEO/CFO/director) reports a buy or sell — exact shares/price
  8-K           a material event (earnings, offering, M&A, exec change)
  S-3 / 424B    a securities offering — i.e. DILUTION, the classic after-hours dumper
  SC 13D/13G    a fund disclosing a >5% stake

Filings carry an `acceptanceDateTime`; anything accepted outside 9:30–16:00 ET is an
after-hours filing — precisely what gaps a stock overnight. Paired with the
After-Hours price move, this answers "why is it moving?" with a sourced document
instead of a fabricated "institutional distribution" narrative.

No API key. SEC requires a declarative User-Agent and allows ~10 req/s.
"""
import json
import os
import time
import urllib.request
from datetime import date, datetime, timedelta

SEC_UA = {"User-Agent": "Meridian Research meridian-app contact@example.com"}
_CIK_CACHE = os.path.expanduser("~/.meridian_cache/edgar_ciks.json")
_cik_map = {}

# form -> (plain-English meaning, directional bias: -1 bearish / 0 neutral / +1 bullish)
MATERIAL = {
    "8-K": ("material event (8-K)", 0),
    "4": ("insider trade (Form 4)", 0),
    "S-3": ("shelf registration — potential dilution", -1),
    "S-3ASR": ("shelf registration — potential dilution", -1),
    "424B5": ("securities offering — dilution", -1),
    "424B4": ("securities offering — dilution", -1),
    "424B3": ("securities offering — dilution", -1),
    "SC 13D": ("activist >5% stake", 1),
    "SC 13D/A": ("activist stake change", 0),
    "SC 13G": ("passive >5% stake", 0),
    "10-Q": ("quarterly report", 0),
    "10-K": ("annual report", 0),
}


def _get(url, timeout=15):
    with urllib.request.urlopen(urllib.request.Request(url, headers=SEC_UA), timeout=timeout) as r:
        return r.read().decode()


def _load_ciks():
    """Ticker -> zero-padded CIK, cached on disk for 30 days."""
    global _cik_map
    if _cik_map:
        return _cik_map
    try:
        if os.path.exists(_CIK_CACHE) and time.time() - os.path.getmtime(_CIK_CACHE) < 30 * 86400:
            with open(_CIK_CACHE) as f:
                _cik_map = json.load(f)
                return _cik_map
    except Exception:
        pass
    try:
        m = json.loads(_get("https://www.sec.gov/files/company_tickers.json"))
        _cik_map = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in m.values()}
        os.makedirs(os.path.dirname(_CIK_CACHE), exist_ok=True)
        with open(_CIK_CACHE, "w") as f:
            json.dump(_cik_map, f)
    except Exception:
        _cik_map = {}
    return _cik_map


def _after_hours(iso):
    """True if accepted outside regular hours (9:30–16:00 ET) or on a weekend."""
    try:
        from zoneinfo import ZoneInfo
        et = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(ZoneInfo("America/New_York"))
        mins = et.hour * 60 + et.minute
        return et.weekday() >= 5 or not (570 <= mins < 960)
    except Exception:
        return False


def recent_filings(ticker, days=4, timeout=15):
    """Material SEC filings for `ticker` in the last `days`, newest first. Each:
    {form, note, bias, date, accepted, after_hours, url}. [] on any failure."""
    cik = _load_ciks().get(ticker.upper())
    if not cik:
        return []
    try:
        d = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik}.json", timeout))
    except Exception:
        return []
    rec = d.get("filings", {}).get("recent", {})
    forms = rec.get("form", []); dates = rec.get("filingDate", [])
    acc = rec.get("accessionNumber", []); acct = rec.get("acceptanceDateTime", [])
    docs = rec.get("primaryDocument", [])
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    out = []
    for i in range(len(forms)):
        f = forms[i]
        if f not in MATERIAL or dates[i] < cutoff:
            continue
        note, bias = MATERIAL[f]
        a = acc[i].replace("-", "") if i < len(acc) else ""
        doc = docs[i] if i < len(docs) else ""
        url = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{a}/{doc}"
               if a and doc else f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}")
        out.append({"form": f, "note": note, "bias": bias, "date": dates[i],
                    "accepted": acct[i] if i < len(acct) else "",
                    "after_hours": _after_hours(acct[i]) if i < len(acct) else False,
                    "url": url})
    return out

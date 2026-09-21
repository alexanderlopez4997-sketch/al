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
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta

SEC_UA = {"User-Agent": "Meridian Research meridian-app contact@example.com"}
_CIK_CACHE = os.path.expanduser("~/.meridian_cache/edgar_ciks.json")
_cik_map = {}

DILUTIVE_FORMS = {"S-3", "S-3ASR", "424B5", "424B4", "424B3"}

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

# 8-K item numbers that historically drive the biggest gaps (exec shifts,
# bankruptcy, M&A, delisting) — used to tag the note, not to guess direction.
EXEC_ITEMS = {
    "5.02": "exec/director change",
    "1.03": "bankruptcy",
    "2.01": "acquisition/disposition",
    "3.01": "delisting notice",
}

C_SUITE_RE = re.compile(r"\b(chief|ceo|cfo|coo|cto|president|chairman)\b", re.I)

# Form 4 transaction codes that are genuine open-market sentiment (not a
# grant/exercise/gift/tax-withholding, which say nothing about conviction).
_OPEN_MARKET = {"P": 1, "S": -1}


def _get(url, timeout=15):
    with urllib.request.urlopen(urllib.request.Request(url, headers=SEC_UA), timeout=timeout) as r:
        return r.read().decode()


def _get_bytes(url, timeout=15):
    with urllib.request.urlopen(urllib.request.Request(url, headers=SEC_UA), timeout=timeout) as r:
        return r.read()


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


def _parse_form4(url, timeout=15):
    """Fetch + parse a Form 4 ownership XML into the reporting owner and their
    genuine open-market activity. Returns {owner, title, is_csuite, buy_usd,
    sell_usd} or None on any failure. Grants, option exercises, tax
    withholding and gifts are excluded — they're not a conviction signal,
    only actual open-market P(urchase)/S(ale) transactions are counted."""
    try:
        root = ET.fromstring(_get_bytes(url, timeout))
    except Exception:
        return None
    owner_el = root.find("reportingOwner")
    if owner_el is None:
        return None
    name = (owner_el.findtext("reportingOwnerId/rptOwnerName") or "").strip()
    rel = owner_el.find("reportingOwnerRelationship")
    title = (rel.findtext("officerTitle") or "").strip() if rel is not None else ""
    buy_usd = sell_usd = 0.0
    for tx in root.findall(".//nonDerivativeTransaction"):
        code = (tx.findtext("transactionCoding/transactionCode") or "").strip()
        if code not in _OPEN_MARKET:
            continue
        try:
            shares = float(tx.findtext("transactionAmounts/transactionShares/value") or 0)
            price = float(tx.findtext("transactionAmounts/transactionPricePerShare/value") or 0)
        except ValueError:
            continue
        if code == "P":
            buy_usd += shares * price
        else:
            sell_usd += shares * price
    return {"owner": name, "title": title, "is_csuite": bool(C_SUITE_RE.search(title)),
            "buy_usd": buy_usd, "sell_usd": sell_usd}


def _flag_insider_flood(filings):
    """Upgrade Form 4 notes in-place when >=2 distinct insiders transact the
    same way in the window: a flood of open-market buys is a strong bullish
    tell, while multiple C-suite officers selling at once can signal
    overhead resistance rather than routine diversification."""
    form4 = [f for f in filings if f["form"] == "4" and f.get("_owner")]
    buyers = {f["_owner"] for f in form4 if f["bias"] > 0}
    csuite_sellers = {f["_owner"] for f in form4 if f["bias"] < 0 and f["_csuite"]}
    if len(buyers) >= 2:
        for f in form4:
            if f["bias"] > 0:
                f["note"] = f"insider buying flood (Form 4) — {len(buyers)} insiders buying"
    if len(csuite_sellers) >= 2:
        for f in form4:
            if f["bias"] < 0 and f["_csuite"]:
                f["note"] = (f"C-suite selling flood (Form 4) — {len(csuite_sellers)} officers "
                             "selling, overhead-resistance risk")


PRIMARY_FORMS = ("10-K", "10-Q", "DEF 14A")


def primary_filings(ticker, timeout=15):
    """Latest 10-K, 10-Q and proxy (DEF 14A) for `ticker`, regardless of age —
    the source documents for research (full business/risk-factor read, executive
    pay and insider ownership), as opposed to recent_filings()'s few-day window
    for what just moved the stock. {} on any failure; missing forms are omitted."""
    cik = _load_ciks().get(ticker.upper())
    if not cik:
        return {}
    try:
        d = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik}.json", timeout))
    except Exception:
        return {}
    rec = d.get("filings", {}).get("recent", {})
    forms = rec.get("form", []); dates = rec.get("filingDate", [])
    acc = rec.get("accessionNumber", []); docs = rec.get("primaryDocument", [])
    out = {}
    for i in range(len(forms)):
        f = forms[i]
        if f not in PRIMARY_FORMS or f in out:
            continue
        a = acc[i].replace("-", "") if i < len(acc) else ""
        doc = docs[i] if i < len(docs) else ""
        if not (a and doc):
            continue
        out[f] = {"date": dates[i],
                   "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{a}/{doc}"}
    return out


def recent_filings(ticker, days=4, timeout=15):
    """Material SEC filings for `ticker` in the last `days`, newest first. Each:
    {form, note, bias, date, accepted, after_hours, url, vol_mult}. [] on any failure.

    Form 4s are parsed for the actual transaction so `bias` reflects real
    insider sentiment (open-market buy = bullish, sell = bearish) rather than
    just "a Form 4 was filed"; a flood of buying insiders or multiple
    C-suite officers selling at once is called out in `note` — see
    _flag_insider_flood. 8-Ks filed outside market hours get `vol_mult` > 1:
    they're exactly the after-hours-drop events that explain (and should
    raise confidence in) an otherwise-unexplained overnight gap."""
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
    docs = rec.get("primaryDocument", []); items = rec.get("items", [])
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
        ah = _after_hours(acct[i]) if i < len(acct) else False
        owner = csuite = None
        if f == "4" and a and doc:
            detail = _parse_form4(url, timeout)
            if detail:
                owner, csuite = detail["owner"], detail["is_csuite"]
                if detail["buy_usd"] != detail["sell_usd"] and (detail["buy_usd"] or detail["sell_usd"]):
                    bias = 1 if detail["buy_usd"] > detail["sell_usd"] else -1
                    usd = detail["buy_usd"] if bias > 0 else detail["sell_usd"]
                    who = detail["title"] or "insider"
                    note = f"insider {'buy' if bias > 0 else 'sell'} (Form 4) — {who} ${usd/1e6:.1f}M"
        elif f == "8-K" and i < len(items) and items[i]:
            tag = next((EXEC_ITEMS[c] for c in (x.strip() for x in items[i].split(",")) if c in EXEC_ITEMS), None)
            if tag:
                note = f"material event (8-K) — {tag}"
        out.append({"form": f, "note": note, "bias": bias, "date": dates[i],
                    "accepted": acct[i] if i < len(acct) else "", "after_hours": ah, "url": url,
                    "vol_mult": 1.5 if (f == "8-K" and ah) else 1.0,
                    "_owner": owner, "_csuite": csuite})
    _flag_insider_flood(out)
    for e in out:
        del e["_owner"], e["_csuite"]
    return out

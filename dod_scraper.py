#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DoD Daily Contract Scraper — real-time federal contract awards from defense.gov.

The Department of Defense publishes contract awards daily around 5pm ET via:
  https://www.defense.gov/News/Releases/?Category=Contracts

This module scrapes that feed to catch awards before they're priced in — a $100M
contract award on a Monday after-hours can move a small-cap 15%+ by Tuesday open.

Each award is parsed for:
  - Contractor name (mapped to ticker via fuzzy match or manual mapping)
  - Award value in USD
  - Award date (ISO 8601, ~5pm ET)
  - Service branch and description

Feeds into contracts.py scoring pipeline for signal generation.

Network:
  - No API key required (public HTML page, standard User-Agent)
  - Rate limit: ~1 req/min (DoD is relaxed; we fetch once daily)
  - Timezone: all times are ET (Eastern Time)

Example:
  today_awards = dod_daily_awards(html_or_url)  # fetch or pass cached HTML
  for award in today_awards:
      sig = dod_award_signal(award, ticker="RTX", market_cap=65e9)
"""

import re
import logging
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone, timedelta
from html.parser import HTMLParser

import pandas as pd
import numpy as np

logger = logging.getLogger("dod_scraper")

DOD_CONTRACTS_URL = "https://www.defense.gov/News/Releases/?Category=Contracts"
DOD_UA = {"User-Agent": "Meridian Research meridian-app contact@example.com"}

# Contractor name -> ticker. Keys are matched on normalised word tokens (see
# _contractor_to_ticker), so write them without legal suffixes (Inc/Corp/Co/LLC).
# A value of None means "known, but no tradeable US ticker" (private, foreign, or a
# joint venture) — it is an explicit answer, and wins over any shorter alias that
# would otherwise match (e.g. "Bell Boeing" must not fall through to Boeing).
CONTRACTOR_TICKER_MAP = {
    # --- Aerospace & defense primes
    "Lockheed Martin": "LMT",
    "Sikorsky": "LMT",
    "Terran Orbital": "LMT",
    "Northrop Grumman": "NOC",
    "Orbital ATK": "NOC",
    "General Dynamics": "GD",
    "GDIT": "GD",
    "Electric Boat": "GD",
    "Bath Iron Works": "GD",
    "Gulfstream Aerospace": "GD",
    "NASSCO": "GD",
    "National Steel and Shipbuilding": "GD",
    "BlueHalo": "GD",
    "Raytheon": "RTX",
    "Raytheon Technologies": "RTX",
    "RTX": "RTX",
    "Collins Aerospace": "RTX",
    "Rockwell Collins": "RTX",
    "Pratt & Whitney": "RTX",
    "Boeing": "BA",
    "Insitu": "BA",
    "Spirit AeroSystems": "BA",        # acquired by Boeing 2025-12-08 (SPR delisted)
    "Short Brothers": "BA",
    "L3Harris": "LHX",
    "L3 Harris": "LHX",
    "L3 Technologies": "LHX",
    "L3": "LHX",                       # _extract_contractor strips " Technologies"
    "Aerojet Rocketdyne": "LHX",
    "Huntington Ingalls": "HII",
    "Newport News Shipbuilding": "HII",
    "Ingalls Shipbuilding": "HII",
    "Textron": "TXT",
    "Bell Textron": "TXT",
    "Bell Helicopter": "TXT",
    "Cessna": "TXT",
    "Beechcraft": "TXT",
    "AAI": "TXT",                      # AAI Corp is a Textron Systems company
    "General Electric": "GE",
    "GE Aerospace": "GE",
    "GE Aviation": "GE",
    "GE Vernova": "GEV",
    "Honeywell": "HON",
    "Leonardo DRS": "DRS",
    "BWX Technologies": "BWXT",
    "BWXT": "BWXT",
    "BWX": "BWXT",                     # _extract_contractor strips " Technologies"
    "Babcock & Wilcox Nuclear Operations": "BWXT",
    "Babcock & Wilcox": "BW",
    "Oshkosh": "OSK",
    "AeroVironment": "AVAV",
    "Kratos": "KTOS",
    "Mercury Systems": "MRCY",
    "Curtiss-Wright": "CW",
    "HEICO": "HEI",
    "TransDigm": "TDG",
    "Ducommun": "DCO",
    "Hexcel": "HXL",
    "Woodward": "WWD",
    "Howmet": "HWM",
    "Astronics": "ATRO",
    "National Presto": "NPK",
    "Smith & Wesson": "SWBI",
    "Sturm, Ruger": "RGR",
    "Olin": "OLN",
    "Cadre Holdings": "CDRE",
    "Axon": "AXON",
    # --- Space
    "Rocket Lab": "RKLB",
    "Redwire": "RDW",
    "Viasat": "VSAT",
    "Iridium": "IRDM",
    # --- Services / IT integrators
    "Leidos": "LDOS",
    "Science Applications International": "SAIC",
    "SAIC": "SAIC",
    "CACI": "CACI",
    "Booz Allen Hamilton": "BAH",
    "KBR": "KBR",
    "Parsons": "PSN",
    "Fluor": "FLR",
    "Jacobs": "J",
    "AECOM": "ACM",
    "Tetra Tech": "TTEK",
    "Amentum": "AMTM",
    "V2X": "VVX",
    "Vectrus": "VVX",
    "Maximus": "MMS",
    "ICF": "ICFI",
    "Unisys": "UIS",
    "Telos": "TLS",
    "Accenture Federal Services": "ACN",
    # --- Big tech / telecom
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Google": "GOOGL",
    "Alphabet": "GOOGL",
    "Oracle": "ORCL",
    "International Business Machines": "IBM",
    "IBM": "IBM",
    "Dell": "DELL",
    "Cisco Systems": "CSCO",
    "Palantir": "PLTR",
    "Verizon": "VZ",
    "AT&T": "T",
    "Lumen": "LUMN",
    # --- Health / logistics (TRICARE, medical supply)
    "Humana Military": "HUM",
    "Health Net Federal Services": "CNC",
    "UnitedHealth": "UNH",
    "Optum": "UNH",
    "Cardinal Health": "CAH",
    "McKesson": "MCK",
    "Cencora": "COR",
    "AmerisourceBergen": "COR",
    # --- Known, but no tradeable US ticker (private / foreign / joint venture)
    "General Atomics": None,
    "Sierra Nevada": None,
    "Anduril": None,
    "SpaceX": None,
    "Peraton": None,
    "ManTech": None,
    "Triumph Group": None,             # taken private 2025-07 (TGI delisted)
    "BAE Systems": None,               # UK-listed
    "Bell Boeing": None,               # Textron/Boeing joint venture
    "United Launch Alliance": None,    # Lockheed/Boeing joint venture
    "Javelin Joint Venture": None,     # Raytheon/Lockheed joint venture
}

# Legal-form tokens carry no identity: "The Boeing Co." == "Boeing".
_LEGAL_TOKENS = frozenset({"the", "inc", "incorporated", "corp", "corporation", "co",
                           "company", "llc", "lp", "llp", "ltd", "limited", "plc"})


def _name_tokens(name):
    """'Pratt & Whitney Corp.' -> ['pratt', 'and', 'whitney']. Lowercased, punctuation
    folded to word breaks ('&' -> 'and'), legal-form tokens dropped."""
    s = str(name).lower().replace("&", " and ")
    s = re.sub(r"['\u2019.]", "", s)           # "L3Harris." / "Ruger's" -> no stray breaks
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return [t for t in s.split() if t not in _LEGAL_TOKENS]


def _lookup(contractor_name):
    """-> (matched, ticker). `matched` is True for ANY alias hit, including the
    explicit-None ones (private/foreign/JV), so callers can tell "known, no
    ticker" from "never heard of it".

    Matching is on whole word tokens against CONTRACTOR_TICKER_MAP, never raw
    substrings (so "Saxon Industries" can't hit "Axon", nor "Hawaiian ..." hit "AAI").
    Rules, in priority order:
      1. An alias at the START of the name beats one found later in it.
      2. Multi-word aliases ("Lockheed Martin", "General Dynamics") also match
         mid-name ("... a Lockheed Martin Co."); one-word aliases must lead the
         name — a lone common word buried mid-name is too weak to trust.
      3. Longer alias beats shorter ("Bell Boeing" -> None beats "Boeing" -> BA;
         "General Dynamics" never collides with "General Electric").
    A None-valued alias is a definite answer (private/foreign/JV) and is returned
    as None rather than falling through to a shorter alias."""
    if not contractor_name:
        return False, None
    toks = _name_tokens(contractor_name)
    if not toks:
        return False, None
    best = None                                  # (rank, ticker)
    for alias, ticker in CONTRACTOR_TICKER_MAP.items():
        a = _name_tokens(alias)
        n = len(a)
        if not n or n > len(toks):
            continue
        for i in range(len(toks) - n + 1):
            if toks[i:i + n] != a:
                continue
            if i > 0 and n == 1:
                continue                         # rule 2
            rank = (i == 0, n, -i)
            if best is None or rank > best[0]:
                best = (rank, ticker)
    return (True, best[1]) if best else (False, None)


def _contractor_to_ticker(contractor_name):
    """Map a contractor name to a ticker, or None. See _lookup() for the rules."""
    return _lookup(contractor_name)[1]


def unmapped_contractors(awards, min_value=50e6):
    """Contractors with a big award but no ticker — the to-do list for growing
    CONTRACTOR_TICKER_MAP. Returns [(contractor, total_value_usd)], biggest first.
    Firms deliberately mapped to None (private/foreign/JV) are not listed — map a
    private contractor to None to silence it."""
    totals = {}
    for a in awards or []:
        name = a.get("contractor") or "Unknown"
        if a.get("ticker") or name.lower() == "unknown" or _lookup(name)[0]:
            continue
        totals[name] = totals.get(name, 0.0) + float(a.get("value_usd") or 0)
    return sorted(((n, v) for n, v in totals.items() if v >= min_value), key=lambda x: -x[1])


class DODReleaseParser(HTMLParser):
    """Parse DoD contract release HTML, extracting award details."""

    def __init__(self):
        super().__init__()
        self.in_article = False
        self.in_title = False
        self.in_body = False
        self.current_title = ""
        self.current_body = ""
        self.current_date = ""
        self.articles = []

    def handle_starttag(self, tag, attrs):
        if tag == "article":
            self.in_article = True
        elif tag == "h2" and self.in_article:
            self.in_title = True
        elif tag == "p" and self.in_article:
            self.in_body = True
        # Extract date from time tag
        elif tag == "time" and self.in_article:
            for k, v in attrs:
                if k == "datetime":
                    self.current_date = v

    def handle_endtag(self, tag):
        if tag == "article":
            if self.current_title and self.current_body:
                self.articles.append({
                    "title": self.current_title.strip(),
                    "body": self.current_body.strip(),
                    "date": self.current_date,
                })
            self.in_article = False
            self.current_title = ""
            self.current_body = ""
            self.current_date = ""
        elif tag == "h2":
            self.in_title = False
        elif tag == "p":
            self.in_body = False

    def handle_data(self, data):
        if self.in_title:
            self.current_title += data
        elif self.in_body:
            self.current_body += data


def _fetch_dod_html(timeout=15):
    """Fetch DoD contracts page HTML. Returns HTML text or None on failure."""
    try:
        req = urllib.request.Request(DOD_CONTRACTS_URL, headers=DOD_UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode('utf-8')
    except urllib.error.URLError as e:
        logger.warning(f"Failed to fetch DoD page: {e.reason}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error fetching DoD page: {e}")
        return None


def _is_contract_release(title, body):
    """Heuristic: is this a contract award announcement (not a personnel change, etc)?

    Look for keywords that indicate contract value and contractor."""
    keywords = [
        "contract", "award", "obligat", "modif", "$", "million", "billion",
        "sole source", "competitive", "service", "system"
    ]
    text = (title + " " + body).lower()
    return sum(1 for kw in keywords if kw in text) >= 2


def _extract_award_value(text):
    """Extract USD amount from contract announcement text.

    Looks for patterns like:
      - $123.45 million
      - $1.2 billion
      - $12,345,678
    Returns amount in USD (float) or None."""
    # Try: $X.X million/billion/thousand
    for pattern in [
        r'\$\s*([\d,]+\.?\d*)\s*billion',
        r'\$\s*([\d,]+\.?\d*)\s*million',
        r'\$\s*([\d,]+\.?\d*)\s*thousand',
        r'\$\s*([\d,]+(?:,\d{3})*\.?\d*)\b',  # $X,XXX or $X.XX
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount_str = match.group(1).replace(',', '')
            try:
                amount = float(amount_str)
                # Scale by unit
                if 'billion' in match.group(0).lower():
                    amount *= 1e9
                elif 'million' in match.group(0).lower():
                    amount *= 1e6
                elif 'thousand' in match.group(0).lower():
                    amount *= 1e3
                return amount
            except ValueError:
                continue
    return None


def _extract_contractor(text):
    """Extract primary contractor name from announcement.

    Looks for patterns like:
      - "Contractor: Lockheed Martin..."
      - "awarded to Lockheed Martin..."
      - "...Lockheed Martin Corporation..."
    Returns contractor name or None."""
    # First-line heuristic: major contractors usually in headline or first sentence
    lines = text.split('\n')
    full_text = ' '.join(lines[:3])  # first ~3 lines

    # Try: "contractor: NAME" or "awarded to NAME" or named entity in first sentence
    for pattern in [
        r'(?:contractor|prime|awardee):\s*([^,;.\n]+)',
        r'awarded to\s+([^,;.\n]+)',
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Corporation|Corp|Inc|Company|Ltd|Technologies)))\b',
    ]:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            # Clean up common suffixes
            for suffix in [" corporation", " corp", " inc", " company", " ltd", " technologies"]:
                if name.lower().endswith(suffix):
                    name = name[:-len(suffix)].strip()
            if name and len(name) > 2:  # ignore tiny matches
                return name
    return None


def dod_daily_awards(html=None, url=None, timeout=15):
    """Fetch and parse DoD daily contract awards.

    Args:
      html: pre-fetched HTML (if None, fetches from url/default)
      url: custom DoD URL (default: defense.gov/News/Releases)
      timeout: HTTP timeout in seconds

    Returns:
      list of {
        "title": str,           # release headline
        "contractor": str,      # primary contractor name
        "ticker": str or None,  # mapped ticker (if known)
        "value_usd": float,     # award value in dollars
        "date": str,            # ISO 8601 date (ET)
        "description": str,     # full announcement text
      }
    """
    if html is None:
        html = _fetch_dod_html(timeout)

    if not html:
        logger.warning("No HTML to parse; returning empty list")
        return []

    parser = DODReleaseParser()
    try:
        parser.feed(html)
    except Exception as e:
        logger.warning(f"HTML parsing error: {e}")
        return []

    awards = []
    for article in parser.articles:
        if not _is_contract_release(article["title"], article["body"]):
            continue

        value = _extract_award_value(article["body"])
        if not value or value < 1e6:  # ignore sub-$1M (noise)
            continue

        contractor = _extract_contractor(article["body"])
        ticker = _contractor_to_ticker(contractor) if contractor else None

        awards.append({
            "title": article["title"],
            "contractor": contractor or "Unknown",
            "ticker": ticker,
            "value_usd": value,
            "date": article["date"][:10],  # ISO date only
            "description": article["body"][:500],  # first 500 chars
        })

    for name, total in unmapped_contractors(awards, min_value=100e6):
        logger.info("unmapped DoD contractor %r ($%.0fM) — add to CONTRACTOR_TICKER_MAP "
                    "(or map to None if private)", name, total / 1e6)
    return awards


def dod_award_signal(award, ticker=None, market_cap=None, ttm_revenue=None):
    """Score a single DoD award against market fundamentals.

    Similar to contracts.contract_signal(), but for a single award.

    Args:
      award: dict from dod_daily_awards() with value_usd, date, etc.
      ticker: override ticker from award (award["ticker"] if None)
      market_cap: market cap in USD (preferred)
      ttm_revenue: TTM revenue in USD (fallback)

    Returns:
      {signal, confidence, detail} or None if no market data
    """
    if not award or award.get("value_usd", 0) <= 0:
        return None

    tk = ticker or award.get("ticker")
    fundamentals = market_cap or ttm_revenue
    if fundamentals is None or fundamentals <= 0:
        return None

    ratio = award["value_usd"] / fundamentals
    ratio = float(np.clip(ratio, -1, 1))

    if abs(ratio) < 0.001:  # sub-0.1% is noise
        return None

    # Confidence: award value + freshness (DoD awards are TODAY, so max decay)
    val_conf = min(1.0, award["value_usd"] / 1e8)  # $100M = full confidence
    recency = 1.0  # Today's award = max recency (no decay yet)

    return {
        "signal": ratio,
        "confidence": val_conf * recency,
        "detail": (f"${award['value_usd']/1e6:.0f}M DoD award to {award.get('contractor', '?')} "
                   f"· {award.get('date', '?')}"),
    }


def dod_bulk_score(awards, fundamentals_fn=None):
    """Score all DoD awards from today, grouped by ticker.

    Args:
      awards: list from dod_daily_awards()
      fundamentals_fn: callable(ticker) -> {market_cap, ttm_revenue}

    Returns:
      {ticker: {signal, confidence, detail}}
    """
    by_ticker = {}
    for award in awards:
        if not award.get("ticker"):
            continue

        tk = award["ticker"]
        mcap = ttm_rev = None
        if fundamentals_fn:
            try:
                fund = fundamentals_fn(tk)
                if fund:
                    mcap = fund.get("market_cap")
                    ttm_rev = fund.get("ttm_revenue")
            except Exception:
                pass

        sig = dod_award_signal(award, ticker=tk, market_cap=mcap, ttm_revenue=ttm_rev)
        if sig:
            by_ticker[tk] = sig

    return by_ticker


def annotate_awards(awards, market_cap_fn):
    """Attach `signal`/`confidence` to each award in place, using the
    contractor's market cap. `market_cap_fn(ticker)` is called once per ticker.
    Awards with no ticker, no market cap, or a sub-noise ratio keep
    signal/confidence = None so callers can show "n/a" instead of a fake 0.

    Returns `awards` for chaining."""
    caps = {}
    for award in awards:
        award.setdefault("signal", None)
        award.setdefault("confidence", None)
        tk = award.get("ticker")
        if not tk:
            continue
        if tk not in caps:
            try:
                caps[tk] = market_cap_fn(tk)
            except Exception:
                caps[tk] = None
        sig = dod_award_signal(award, ticker=tk, market_cap=caps[tk])
        if sig:
            award["signal"], award["confidence"] = sig["signal"], sig["confidence"]
    return awards


def dod_signal_for_ticker(awards, ticker, market_cap):
    """Collapse today's awards for one ticker into a single {signal, confidence,
    detail} (mean when there are several), or None if nothing scoreable."""
    if not market_cap or market_cap <= 0:
        return None
    sigs = [s for s in (dod_award_signal(a, ticker=ticker, market_cap=market_cap)
                        for a in awards if a.get("ticker") == ticker) if s]
    if not sigs:
        return None
    if len(sigs) == 1:
        return sigs[0]
    return {"signal": sum(s["signal"] for s in sigs) / len(sigs),
            "confidence": sum(s["confidence"] for s in sigs) / len(sigs),
            "detail": f"{len(sigs)} DoD awards"}


def schedule_dod_scraper(hour=17, minute=0):
    """Return a cron expression for scraping DoD awards daily at 5pm ET.

    Args:
      hour: hour in ET (0-23, default 17 = 5pm)
      minute: minute (0-59, default 0)

    Returns:
      cron expression string (e.g., "0 17 * * 1-5" for 5pm weekdays)
    """
    # Run 5pm ET, Monday-Friday (1-5 = Mon-Fri in cron)
    return f"{minute} {hour} * * 1-5"


if __name__ == "__main__":
    # Example: parse sample HTML or fetch live
    import sys

    if len(sys.argv) > 1:
        # Read HTML from file if provided
        with open(sys.argv[1]) as f:
            html = f.read()
    else:
        print("Fetching live DoD contract awards...")
        html = None  # will fetch

    awards = dod_daily_awards(html)
    if awards:
        print(f"Found {len(awards)} contract awards:")
        for a in awards[:5]:
            print(f"  {a['date']} | ${a['value_usd']/1e6:.0f}M | {a['contractor']} ({a.get('ticker', '?')})")
    else:
        print("No awards found (page structure may have changed)")

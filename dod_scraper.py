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

# The department was renamed (Department of War); the daily "Contracts for <date>" page
# now lives at war.gov/News/Contracts. The old defense.gov listing is kept as a fallback.
DOD_CONTRACTS_URLS = [
    "https://www.war.gov/News/Contracts/",
    "https://www.defense.gov/News/Releases/?Category=Contracts",
]
DOD_CONTRACTS_URL = DOD_CONTRACTS_URLS[0]      # back-compat alias
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
    """Parse DoD contract release HTML, extracting award details.

    Paragraphs inside an <article> are kept separate (one per line, line-wraps inside a
    paragraph collapsed to spaces) — a day's
    release is dozens of one-paragraph awards, and gluing them together made the
    extractor see a single blob. Every <p> on the page is also collected in
    `loose_paragraphs`, as a fallback for markup that doesn't use <article>."""

    def __init__(self):
        super().__init__()
        self.in_article = False
        self.in_title = False
        self.in_body = False
        self.in_p = False
        self.current_title = ""
        self.current_body = ""
        self.current_date = ""
        self.current_p = ""
        self.body_par = ""
        self.articles = []
        self.loose_paragraphs = []
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "article":
            self.in_article = True
        elif tag == "h2" and self.in_article:
            self.in_title = True
        elif tag == "p":
            self.in_p = True
            self.current_p = ""
            self.body_par = ""
            if self.in_article:
                self.in_body = True
        elif tag == "br":
            if self.in_body:
                self._flush_body_par()
            if self.in_p:
                self.current_p += " "
        elif tag == "a":
            for k, v in attrs:
                if k == "href" and v:
                    self.hrefs.append(v)
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
            if self.in_body:
                self._flush_body_par()
            par = " ".join(self.current_p.split())
            if par:
                self.loose_paragraphs.append(par)
            self.in_p = False
            self.in_body = False

    def _flush_body_par(self):
        """End the current paragraph: collapse its source line-wraps to single spaces and
        add it to the article body on its own line."""
        par = " ".join(self.body_par.split())
        if par:
            self.current_body += par + "\n"
        self.body_par = ""

    def handle_data(self, data):
        if self.in_title:
            self.current_title += data
        elif self.in_body:
            self.body_par += data
        if self.in_p:
            self.current_p += data


def _fetch_url(url, timeout=15):
    """GET `url` as text, or None on failure."""
    try:
        req = urllib.request.Request(url, headers=DOD_UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode('utf-8', errors='replace')
    except urllib.error.URLError as e:
        logger.warning(f"Failed to fetch DoD page {url}: {getattr(e, 'reason', e)}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error fetching DoD page {url}: {e}")
        return None


def _fetch_dod_html(timeout=15, url=None):
    """Fetch the DoD contracts page. Tries `url` (or each of DOD_CONTRACTS_URLS in
    order) and returns the first page that looks like it has content
    (contains an <article> or a <p>), else the first non-empty page, else None."""
    first = None
    for u in ([url] if url else DOD_CONTRACTS_URLS):
        html = _fetch_url(u, timeout)
        if not html:
            continue
        if re.search(r"<article|<p[\s>]", html, re.I):
            return html
        first = first or html
    return first


# Article pages for a given day. UNVERIFIED against live markup (war.gov was not
# reachable when this was written) — used only when the listing page itself has no
# award text, and it fails soft.
_ARTICLE_HREF_RE = re.compile(r"/News/(?:Contracts/Contract|Releases/Release)/Article/\d+", re.I)


def _latest_article_url(hrefs, base="https://www.war.gov"):
    for h in hrefs:
        if _ARTICLE_HREF_RE.search(h):
            return h if h.startswith("http") else base + (h if h.startswith("/") else "/" + h)
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


_VALUE_RE = re.compile(r'\$\s*(\d[\d,]*(?:\.\d+)?)\s*(billion|million|thousand)?', re.I)
_UNIT = {"billion": 1e9, "million": 1e6, "thousand": 1e3}

# How much of a CEILING (IDIQ / multiple-award / blanket-purchase / "not-to-exceed")
# counts toward the signal when no funds are reported obligated at award. A ceiling is
# the most the government MAY order over the contract's life; the firm money at award is
# usually a token minimum, so counting it at face value would let a $4B shared vehicle
# outweigh a $200M definite contract. A judgment call, not backtested — tune freely.
CEILING_WEIGHT = 0.10

# Counted award value as a fraction of the contractor's market cap that earns a FULL +1.0
# signal; smaller awards scale linearly below it. 0.1% of market cap = full signal (so a
# $65M award to a $65B company, or $1B to a $1T one, maxes out). Set to 1.0 to get the old
# behavior: signal = the raw award/market-cap ratio, the same convention as
# contracts.contract_signal() (the Quiver gov-contracts signal).
DOD_FULL_SIGNAL_RATIO = 0.001


def _first_amount(text):
    """(usd, start, end) of the first dollar amount in `text`, or None."""
    for m in _VALUE_RE.finditer(text or ""):
        try:
            amount = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        return amount * _UNIT.get((m.group(2) or "").lower(), 1.0), m.start(), m.end()
    return None


def _extract_award_value(text):
    """First dollar amount in `text`, in USD (float), or None.

    Handles "$1.2 billion", "$125.5 million", "$199,303,678". Takes the first amount in
    reading order (a paragraph's headline value comes before any cumulative total or
    obligated-funds figure) — not the first unit that happens to appear anywhere."""
    hit = _first_amount(text)
    return hit[0] if hit else None


_CEILING_RE = re.compile(
    r"indefinite[- ]delivery|\bidiq\b|multiple[- ]award|\bceiling\b|not[- ]to[- ]exceed|"
    r"potential value|blanket purchase agreement|basic ordering agreement", re.I)
_OBLIGATED_AT_AWARD_RE = re.compile(r"obligat\w*[^.]{0,40}\b(?:at|upon)\b[^.]{0,20}\baward\b", re.I)
_NO_FUNDS_RE = re.compile(r"\bno\b[^.]{0,60}\b(?:funds|money|dollars)\b|\bnone of\b", re.I)


def _classify_value(par, amt_start, amt_end):
    """'definite' or 'ceiling' for the headline amount at par[amt_start:amt_end].

    DoD announces two very different things under one "$X":
      definite  a firm award, a modification, or a task/delivery order — real, committed work
      ceiling   the maximum an IDIQ / multiple-award / blanket-purchase vehicle may reach
                over its life; actual money arrives later as orders
    Reads the contract-type phrase right around the amount, not the whole paragraph, so
    work placed UNDER an IDIQ still counts as definite:
      "$30M modification to a previously awarded IDIQ"        -> definite
      "$30M firm-fixed-price task order against an IDIQ"      -> definite
      "$90M multiple-award task order contract"               -> ceiling (that's the vehicle)
      "$4B indefinite-delivery/indefinite-quantity contract"  -> ceiling
      "... a maximum ceiling of $500M"                        -> ceiling"""
    before = par[max(0, amt_start - 40):amt_start].lower()
    after = par[amt_end:amt_end + 120].lower()
    if re.search(r"\bmodification\b|\boption\b", after[:70]):
        return "definite"
    to = re.search(r"\b(?:task|delivery) order\b", after[:80])
    if to:
        return "ceiling" if _CEILING_RE.search(after[:to.start()]) else "definite"
    if _CEILING_RE.search(before) or _CEILING_RE.search(after):
        return "ceiling"
    return "definite"


def _obligated_at_award(par):
    """Dollars the paragraph says are obligated AT TIME OF AWARD (summed over its amounts),
    0.0 if it says no funds are, or None if it says nothing (or defers funding to later
    orders: 'obligated as task orders are issued' is not money at award)."""
    total, seen = 0.0, False
    for sent in re.split(r"(?<=[a-z0-9\)])\.\s+(?=[A-Z])", par):
        if not _OBLIGATED_AT_AWARD_RE.search(sent):
            continue
        if _NO_FUNDS_RE.search(sent):
            return 0.0
        for m in _VALUE_RE.finditer(sent):
            seen = True
            total += float(m.group(1).replace(",", "")) * _UNIT.get((m.group(2) or "").lower(), 1.0)
    return total if seen else None


_IDIQ_RE = re.compile(r"\bidiq\b|indefinite[- ]delivery|multiple[- ]award|blanket purchase agreement|"
                      r"basic ordering agreement", re.I)
_CEILING_WORD_RE = re.compile(r"\bceiling\b|maximum (?:potential )?value|potential value|not[- ]to[- ]exceed|\bnte\b", re.I)
_MOD_RE = re.compile(r"\bmodification\b|\bmod\s*p\d|\bamendment\b", re.I)


def parse_contract_financial_obligations(description_text, headline_value, n_awardees=1):
    """Separate what DoD ANNOUNCED from what is actually worth counting.

    DoD headlines are often IDIQ / multiple-award ceilings (the most the government may
    order over years), not cash. This reads the award paragraph and returns the
    "realized economic value" a signal should use:

      funds obligated at award stated   -> that amount (capped at the headline), any contract type
      ceiling vehicle, nothing stated   -> headline x CEILING_WEIGHT (default 10%)
      ceiling vehicle, "no funds at award" -> 0
      firm award / modification / task order, nothing stated -> the full headline

    A modification or task order UNDER an IDIQ is real money and is not discounted — the
    vehicle is judged by the contract-type phrase at the headline amount, not by any
    IDIQ wording elsewhere in the paragraph. `n_awardees` shares the paragraph's
    obligated total evenly across joint awardees (headline_value is per awardee).

    Returns {headline_value, obligated_value, realized_economic_value, vehicle_weight,
    is_idiq, is_ceiling, is_modification, kind, detail}. `vehicle_weight` (IDIQ 0.25,
    modification 0.80, else 1.0) is informational — the discount is already in
    realized_economic_value, so multiplying by it again would double-count."""
    text = " ".join((description_text or "").split())
    headline = float(headline_value or 0.0)

    # locate the headline amount in the text (first one equal to it, else the first)
    amt = None
    for m in _VALUE_RE.finditer(text):
        v = float(m.group(1).replace(",", "")) * _UNIT.get((m.group(2) or "").lower(), 1.0)
        if amt is None:
            amt = m
        if abs(v - headline) <= max(1.0, headline * 1e-9) or abs(v / max(n_awardees, 1) - headline) <= 1.0:
            amt = m
            break
    kind = _classify_value(text, amt.start(), amt.end()) if amt else "definite"

    raw_obl = _obligated_at_award(text)
    obligated = None if raw_obl is None else min(raw_obl / max(n_awardees, 1), headline)

    if obligated is not None and obligated > 0:
        realized, detail = obligated, f"${obligated:,.0f} obligated at award (headline ${headline:,.0f})"
    elif kind == "ceiling" and obligated == 0:
        realized, detail = 0.0, f"Ceiling vehicle, no funds obligated at award (headline ${headline:,.0f})"
    elif kind == "ceiling":
        realized = headline * CEILING_WEIGHT
        detail = (f"Ceiling vehicle: headline ${headline:,.0f} discounted to "
                  f"${realized:,.0f} (no funds reported at award)")
    else:
        realized = headline
        detail = f"Firm award: ${headline:,.0f}"

    is_mod = bool(_MOD_RE.search(text))
    return {
        "headline_value": headline,
        "obligated_value": obligated,
        "realized_economic_value": realized,
        "vehicle_weight": 0.25 if kind == "ceiling" else (0.80 if is_mod else 1.0),
        "is_idiq": bool(_IDIQ_RE.search(text)),
        "is_ceiling": bool(_CEILING_WORD_RE.search(text)) or kind == "ceiling",
        "is_modification": is_mod,
        "kind": kind,
        "detail": detail,
    }


def award_counted_usd(award):
    """The dollar value an award contributes to a signal: `effective_value_usd` if the
    scraper set it (ceilings discounted / replaced by obligated funds), else the face
    `value_usd` (demo data, legacy-path awards)."""
    v = award.get("effective_value_usd")
    return award.get("value_usd", 0) if v is None else v


# "<awardees> is/was/are/has been/have been (each) awarded|selected ..." — the verb phrase
# that separates the awardee list from the rest of an award paragraph. A bare
# "awarded" is allowed ("Boeing, Seattle, Washington awarded $125M ...").
_AWARD_VERB_RE = re.compile(
    r"(?:\b(?:is|was|are|were|has|have)\b(?:\s+(?:been|being|each))*\s+)?\b(?:awarded|selected)\b",
    re.I)
_SUFFIX_TOKENS = frozenset({"inc", "incorporated", "corp", "corporation", "co", "company",
                            "llc", "lp", "llp", "ltd", "limited", "plc"})
_AWARDEE_SPLIT_RE = re.compile(r";\s*(?:and\s+)?|,\s+and\s+(?=[A-Z0-9])")


def _has_legal_suffix(seg):
    toks = re.sub(r"[^a-z0-9 ]+", " ", seg.lower()).split()
    return bool(toks) and toks[-1] in _SUFFIX_TOKENS


def _awardee_name(chunk):
    """'Lockheed Martin Corp., Fort Worth, Texas (FA8611-26-C-0001)' -> 'Lockheed Martin Corp.'

    The name is the text before the first comma, except when a name itself contains
    one: "Sturm, Ruger & Co. Inc., Newport, New Hampshire" (first segment has no legal
    suffix, the second does -> join them)."""
    chunk = re.sub(r"\([^)]*\)", " ", chunk)                 # contract numbers
    chunk = re.sub(r"^\s*[*\s]*(?:and\s+)?", "", chunk, flags=re.I)
    segs = [x.strip() for x in chunk.split(",") if x.strip()]
    if not segs:
        return None
    name = segs[0]
    if len(segs) >= 2 and not _has_legal_suffix(segs[0]) and _has_legal_suffix(segs[1]) \
            and len(segs[0].split()) == 1:
        name = f"{segs[0]}, {segs[1]}"
    name = " ".join(name.split())
    if not name or not name[0].isalnum() or len(name) > 120 or len(name) < 2:
        return None
    return name


def _parse_award_paragraph(par):
    """One award paragraph -> {awardees:[names], each:bool, value:float, text:str,
    kind:'definite'|'ceiling', obligated:float|None} or None.

    Works on the DoD release shape, where each award is one paragraph:
      "<Name>, <City>, <State>, is awarded a $<amount> ... contract ..."
      "<A>, <City>, <State>; and <B>, <City>, <State>, are each awarded ..."   (multi-award)
    Returns None for anything that isn't an award sentence (section headings such as
    "ARMY", footnotes, boilerplate) or that has no dollar amount after the verb."""
    par = " ".join(par.split())
    m = _AWARD_VERB_RE.search(par)
    if not m:
        return None
    if re.match(r"\s+to\b", par[m.end():], re.I):
        return None   # "... was awarded to <NAME>": awardee follows the verb; legacy path handles it
    lead = par[:m.start()].strip(" ,;")
    if not (3 <= len(lead) <= 600) or not lead.lstrip("* ")[:1].isalnum():
        return None
    tail = par[m.end():]
    hit = _first_amount(tail)
    if not hit:
        return None
    value = hit[0]
    kind = _classify_value(tail, hit[1], hit[2])
    obligated = _obligated_at_award(par)
    names = []
    for chunk in _AWARDEE_SPLIT_RE.split(lead):
        n = _awardee_name(chunk)
        if n and n.lower() not in {x.lower() for x in names}:
            names.append(n)
    if not names:
        return None
    return {"awardees": names, "each": "each" in m.group(0).lower(), "value": value, "text": par,
            "kind": kind, "obligated": obligated}


def _extract_contractor_legacy(text):
    """Old heuristics, kept as a fallback for text that isn't in the DoD award-sentence
    shape: "contractor: NAME", "awarded to NAME", or a "<Words> Corp/Inc/..." phrase."""
    lines = text.split('\n')
    full_text = ' '.join(lines[:3])  # first ~3 lines
    for pattern in [
        r'(?:contractor|prime|awardee):\s*([^,;.\n]+)',
        r'awarded to\s+([^,;.\n]+)',
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Corporation|Corp|Inc|Company|Ltd|Technologies)))\b',
    ]:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            for suffix in [" corporation", " corp", " inc", " company", " ltd", " technologies"]:
                if name.lower().endswith(suffix):
                    name = name[:-len(suffix)].strip()
            if name and len(name) > 2:  # ignore tiny matches
                return name
    return None


def _extract_contractor(text):
    """Primary contractor name from announcement text, or None.

    First choice: the awardee list of the first award sentence ("<Name>, <City>,
    <State>, is awarded ..."), which handles divisions, subsidiaries, and joint awards.
    Fallback: the legacy keyword/suffix heuristics."""
    for par in re.split(r"\n+", text or ""):
        parsed = _parse_award_paragraph(par)
        if parsed:
            return parsed["awardees"][0]
    return _extract_contractor_legacy(text or "")


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def _date_from_title(title):
    """'Contracts for Oct. 6, 2026' / 'Contracts for Sept. 30, 2026' -> '2026-10-06', else None."""
    m = re.search(r"([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})", title or "")
    if not m or m.group(1)[:3].lower() not in _MONTHS:
        return None
    try:
        return datetime(int(m.group(3)), _MONTHS[m.group(1)[:3].lower()], int(m.group(2))).date().isoformat()
    except ValueError:
        return None


def _awards_from_text_blocks(blocks, title, date):
    """Per-paragraph awards from a list of text blocks (each may hold several
    newline-separated paragraphs). One entry per awardee; a multi-award paragraph
    splits its value evenly across awardees unless it says "each awarded"."""
    out = []
    for block in blocks:
        for par in re.split(r"\n+", block):
            parsed = _parse_award_paragraph(par)
            if not parsed:
                continue
            names, total = parsed["awardees"], parsed["value"]
            per = total if (parsed["each"] or len(names) == 1) else total / len(names)
            if per < 1e6:  # ignore sub-$1M (noise)
                continue
            fin = parse_contract_financial_obligations(parsed["text"], per, n_awardees=len(names))
            effective, obl, kind = fin["realized_economic_value"], fin["obligated_value"], fin["kind"]
            for name in names:
                out.append({
                    "title": title,
                    "contractor": name,
                    "ticker": _contractor_to_ticker(name),
                    "value_usd": per,                # face value (per awardee)
                    "value_total_usd": total,
                    "value_kind": kind,
                    "obligated_usd": obl,
                    "effective_value_usd": effective,  # what a signal counts
                    "vehicle_weight": fin["vehicle_weight"],  # confidence multiplier (IDIQ 0.25, mod 0.80)
                    "n_awardees": len(names),
                    "date": date,
                    "description": parsed["text"][:500],
                })
    return out


def _parse_awards_html(html):
    """HTML -> list of award dicts (see dod_daily_awards). Per-paragraph extraction
    from <article> bodies first, then from any <p> on the page, then the legacy
    one-award-per-article path for articles that aren't in the award-sentence shape."""
    parser = DODReleaseParser()
    try:
        parser.feed(html)
    except Exception as e:
        logger.warning(f"HTML parsing error: {e}")
        return []

    awards = []
    for article in parser.articles:
        date = (article["date"][:10] if article["date"] else None) \
            or _date_from_title(article["title"]) or datetime.now(timezone.utc).date().isoformat()
        found = _awards_from_text_blocks([article["body"]], article["title"], date)
        if found:
            awards.extend(found)
            continue
        # Legacy path: one award per article (headline + free text, no award sentence)
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
            "date": date,
            "description": article["body"][:500],  # first 500 chars
        })

    if not awards and parser.loose_paragraphs:
        # A short paragraph that is just a date line ("Contracts for Oct. 6, 2026") dates the
        # page; a long award paragraph's own dates ("completed by Oct. 2031") must not.
        head = next((h for h in parser.loose_paragraphs if len(h) < 60 and _date_from_title(h)), "")
        date = _date_from_title(head) or datetime.now(timezone.utc).date().isoformat()
        awards = _awards_from_text_blocks(parser.loose_paragraphs, "Contracts", date)
    return awards


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
    fetched = html is None
    if fetched:
        html = _fetch_dod_html(timeout, url=url)

    if not html:
        logger.warning("No HTML to parse; returning empty list")
        return []

    awards = _parse_awards_html(html)
    if not awards and fetched:
        # The listing page may only carry headlines; follow the newest day's article page.
        parser = DODReleaseParser()
        try:
            parser.feed(html)
        except Exception:
            parser = None
        art = _latest_article_url(parser.hrefs) if parser else None
        if art:
            page = _fetch_url(art, timeout)
            awards = _parse_awards_html(page) if page else []

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
    counted = award_counted_usd(award) if award else 0
    if not award or counted <= 0:
        return None

    tk = ticker or award.get("ticker")
    fundamentals = market_cap or ttm_revenue
    if fundamentals is None or fundamentals <= 0:
        return None

    # Signal: counted value vs. market cap, saturating at DOD_FULL_SIGNAL_RATIO.
    signal = float(np.clip(counted / fundamentals / DOD_FULL_SIGNAL_RATIO, -1, 1))

    # Confidence: counted value ($100M = full) x contract-vehicle weight (IDIQ 0.25,
    # modification 0.80, else 1.0) x freshness (DoD awards are TODAY, so no decay yet).
    val_conf = min(1.0, counted / 1e8)
    recency = 1.0
    confidence = val_conf * float(award.get("vehicle_weight", 1.0)) * recency

    face = award.get("value_usd", counted)
    if award.get("value_kind") == "ceiling":
        basis = (f"${counted/1e6:.0f}M obligated of a ${face/1e6:.0f}M ceiling"
                 if award.get("obligated_usd") is not None
                 else f"${counted/1e6:.0f}M counted of a ${face/1e6:.0f}M ceiling (no funds reported at award)")
        detail = f"{basis} · DoD award to {award.get('contractor', '?')} · {award.get('date', '?')}"
    else:
        detail = (f"${counted/1e6:.0f}M DoD award to {award.get('contractor', '?')} "
                  f"· {award.get('date', '?')}")
    return {"signal": signal, "confidence": float(confidence), "detail": detail}


def enhanced_dod_award_signal(award_record, market_cap):
    """(signal, confidence) for one award, using its realized economic value rather than
    the raw ceiling. (0.0, 0.0) when there is no market cap or nothing counts.

    Awards from dod_daily_awards() already carry their parsed breakdown, which is used as
    is — re-parsing `description` would lose the obligated-funds clause (it is cut to 500
    characters) and the joint-awardee split. A bare record with only a description and
    headline value is parsed here instead. dod_award_signal() is the same scoring with a
    {signal, confidence, detail} result (what alt_data_tilt consumes)."""
    if not market_cap or market_cap <= 0:
        return 0.0, 0.0
    rec = dict(award_record or {})
    if rec.get("effective_value_usd") is None:
        fin = parse_contract_financial_obligations(rec.get("description", ""), rec.get("value_usd", 0.0))
        rec.update(effective_value_usd=fin["realized_economic_value"], vehicle_weight=fin["vehicle_weight"],
                   value_kind=fin["kind"], obligated_usd=fin["obligated_value"])
    sig = dod_award_signal(rec, market_cap=market_cap)
    return (sig["signal"], sig["confidence"]) if sig else (0.0, 0.0)


def _combined_award(awards, ticker):
    """All of `ticker`'s awards collapsed into one pseudo-award whose value is the SUM of
    their counted values (same-day awards stack: $100M + $60M is a $160M catalyst, not two
    averaged ones; ceilings already discounted). None if nothing counts."""
    mine = [a for a in awards if a.get("ticker") == ticker and award_counted_usd(a) > 0]
    if not mine:
        return None
    counted = sum(award_counted_usd(a) for a in mine)
    return {"ticker": ticker, "contractor": mine[0].get("contractor"), "date": mine[0].get("date"),
            "value_usd": counted,
            "face_usd": sum(a.get("value_usd", 0) for a in mine), "n": len(mine),
            # a ticker's confidence weight follows where its counted dollars came from
            "vehicle_weight": sum(award_counted_usd(a) * a.get("vehicle_weight", 1.0) for a in mine) / counted}


def _combined_signal(combo, **fundamentals):
    sig = dod_award_signal(combo, ticker=combo["ticker"], **fundamentals)
    if sig and combo["n"] > 1:
        sig["detail"] = (f"{combo['n']} DoD awards, ${combo['value_usd']/1e6:.0f}M counted"
                         + (f" of ${combo['face_usd']/1e6:.0f}M face" if combo["face_usd"] > combo["value_usd"] * 1.01 else ""))
    return sig


def dod_bulk_score(awards, fundamentals_fn=None):
    """Score all DoD awards from today, grouped by ticker (a ticker's awards are summed).

    Args:
      awards: list from dod_daily_awards()
      fundamentals_fn: callable(ticker) -> {market_cap, ttm_revenue}

    Returns:
      {ticker: {signal, confidence, detail}}
    """
    by_ticker = {}
    for tk in sorted({a["ticker"] for a in awards if a.get("ticker")}):
        mcap = ttm_rev = None
        if fundamentals_fn:
            try:
                fund = fundamentals_fn(tk)
                if fund:
                    mcap = fund.get("market_cap")
                    ttm_rev = fund.get("ttm_revenue")
            except Exception:
                pass
        combo = _combined_award(awards, tk)
        sig = _combined_signal(combo, market_cap=mcap, ttm_revenue=ttm_rev) if combo else None
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
    detail} (award values summed, then scored against market cap), or None if
    nothing scoreable."""
    if not market_cap or market_cap <= 0:
        return None
    combo = _combined_award(awards, ticker)
    return _combined_signal(combo, market_cap=market_cap) if combo else None


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

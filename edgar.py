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

Two things go a level deeper than the submissions-index metadata above:

  Form 4 XML   the actual transaction table is parsed (not just "a Form 4 was
               filed") so open-market buys/sells (code P/S) can be told apart
               from grants, option exercises and tax withholding (A/M/F/D).
               recent_filings() parses each Form 4's OWN transactions (via
               _parse_form4) to set a real per-filing bias/note and flag
               C-suite/flood clustering (_flag_insider_flood); for a
               role-weighted, 10b5-1-discounted AGGREGATE across a rolling
               24-72h window with exponential time decay, see
               form4_insider_bias() further down.
  8-K items    the Item codes (e.g. 5.02 exec change, 2.02 results) drive a
               volatility multiplier, and the filing's session (pre-market /
               after-hours) drives a gap-risk multiplier — see recent_filings().

No API key. SEC requires a declarative User-Agent and allows ~10 req/s — every
outbound request funnels through _get()/_get_bytes() below, which enforce
that ceiling with a token bucket and dedupe repeat fetches with an in-memory
TTL cache (see RATE LIMITING & CACHING further down).
"""
import collections
import json
import os
import re
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone

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


# =============================================================== RATE LIMITING & CACHING ===
# This module is called from several ThreadPoolExecutor workers at once
# (recent_filings() + form4_insider_bias() for N tickers, each pulling a
# submissions index plus one XML fetch per Form 4 found) — with no shared
# throttle, that fans out to way more than 10 concurrent requests to SEC
# well before any single caller's own rate limiting could kick in. Both
# problems are solved once, centrally, in _get_bytes() — every fetch in this
# module (including the pre-existing _get_bytes() Form 4 path) goes through it.

# A token bucket, not a fixed-window counter: a fixed window (e.g. "reset a
# counter to 0 every second") allows a full quota at the END of one window
# immediately followed by a full quota at the START of the next — up to 2x
# the intended rate in the 1-second window straddling the boundary. A token
# bucket has no such edge: over ANY window of length T, the number of
# requests it can release is bounded by `capacity + rate * T` (start with a
# full bucket, then accrue at `rate` for the rest of the window). Solving
# that for T=1s with real margin under SEC's ~10 req/s ceiling:
#   capacity=2.0, rate=7.0  ->  worst case in any 1s window = 2 + 7*1 = 9 req/s
# comfortably, provably under 10 — not just "usually" under it.
RATE_LIMIT_PER_SEC = 7.0
RATE_LIMIT_BURST = 2.0


class _TokenBucket:
    """Thread-safe, blocking token bucket. acquire() sleeps (never busy-polls
    past its own lock) until a token is available, then takes it."""

    def __init__(self, rate, capacity):
        self.rate = float(rate)
        self.capacity = float(capacity)
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self.rate
            time.sleep(wait)


_RATE_LIMITER = _TokenBucket(RATE_LIMIT_PER_SEC, RATE_LIMIT_BURST)

# In-memory response cache, keyed by URL, storing raw bytes so _get() (text)
# and _get_bytes() (bytes) share one entry — a Form 4 XML fetched via one
# path warms the cache for the other rather than double-fetching the same
# accession. TTL differs by what the URL actually is: a submissions index
# (".../submissions/CIK*.json") is the "recent filings" list and changes
# intraday as new filings land, so it gets a short TTL; an individual filed
# document (".../Archives/edgar/data/...") is immutable the moment SEC
# accepts it — an accession number is never revised in place — so it's
# cached far longer. No persistence: cleared on process restart, like the
# ticker->CIK map's own in-memory tier.
HTTP_CACHE_MAXSIZE = 1024
SUBMISSIONS_CACHE_TTL = 60.0        # seconds — "recent" filings can change any minute
DOCUMENT_CACHE_TTL = 6 * 3600.0     # seconds — a filed document never changes


class _TTLCache:
    """Thread-safe in-memory cache: per-entry TTL, LRU eviction past maxsize."""

    _MISS = object()

    def __init__(self, maxsize):
        self._maxsize = maxsize
        self._data = {}                     # key -> (expires_at or None, value)
        self._order = collections.OrderedDict()  # key -> None, insertion/access order
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return self._MISS
            expires_at, value = entry
            if expires_at is not None and time.monotonic() >= expires_at:
                del self._data[key]
                self._order.pop(key, None)
                return self._MISS
            self._order.move_to_end(key)
            return value

    def set(self, key, value, ttl):
        with self._lock:
            self._data[key] = ((time.monotonic() + ttl) if ttl is not None else None, value)
            self._order[key] = None
            self._order.move_to_end(key)
            while len(self._order) > self._maxsize:
                oldest, _ = self._order.popitem(last=False)
                self._data.pop(oldest, None)

    def clear(self):
        with self._lock:
            self._data.clear()
            self._order.clear()


_HTTP_CACHE = _TTLCache(HTTP_CACHE_MAXSIZE)


def _cache_ttl_for(url):
    if "/submissions/" in url:
        return SUBMISSIONS_CACHE_TTL
    if "/Archives/edgar/data/" in url:
        return DOCUMENT_CACHE_TTL
    return SUBMISSIONS_CACHE_TTL  # conservative default (e.g. company_tickers.json)


def clear_http_cache():
    """Drop every cached SEC response. Callers rarely need this — a stale
    submissions-index entry self-expires within SUBMISSIONS_CACHE_TTL — but
    it's here for tests and for forcing a fresh read on demand."""
    _HTTP_CACHE.clear()


def _get_bytes(url, timeout=15):
    cached = _HTTP_CACHE.get(url)
    if cached is not _TTLCache._MISS:
        return cached
    _RATE_LIMITER.acquire()
    with urllib.request.urlopen(urllib.request.Request(url, headers=SEC_UA), timeout=timeout) as r:
        data = r.read()
    _HTTP_CACHE.set(url, data, _cache_ttl_for(url))
    return data


def _get(url, timeout=15):
    return _get_bytes(url, timeout).decode()


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


def _session(iso):
    """Classify an acceptanceDateTime into the session it landed in:
    'pre_market' (06:00–09:25 ET), 'regular' (09:30–16:00 ET),
    'after_hours' (16:01–18:00 ET), or 'overnight' (everything else,
    including weekends). Unknown/unparseable input -> 'overnight' (neutral
    default; never raises)."""
    try:
        from zoneinfo import ZoneInfo
        et = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(ZoneInfo("America/New_York"))
        if et.weekday() >= 5:
            return "overnight"
        mins = et.hour * 60 + et.minute
        if 360 <= mins <= 565:
            return "pre_market"
        if 570 <= mins < 960:
            return "regular"
        if 961 <= mins <= 1080:
            return "after_hours"
        return "overnight"
    except Exception:
        return "overnight"


# Session -> morning gap-risk multiplier. Pre-market filings land into the
# thinnest order book of the day (before the 9:30 open ever prices them in),
# so they get the largest multiplier; after-hours is thinner than regular
# hours but has a full evening for the book to digest the news.
SESSION_GAP_MULTIPLIER = {"pre_market": 1.5, "after_hours": 1.2, "regular": 1.0, "overnight": 1.1}

# 8-K Item codes whose disclosures historically carry the most next-session
# volatility: entry into a material agreement, results of operations,
# non-reliance on previously issued financials (restatement), and officer/
# director changes. Everything else (general corporate items, Reg FD, etc.)
# gets the low multiplier. An Item code this dict has never heard of still
# resolves safely to "low" — no KeyError, no crash.
HIGH_IMPACT_8K_ITEMS = {"1.01", "2.02", "4.02", "5.02"}
ITEM_VOLATILITY_MULTIPLIER = {"high": 1.6, "low": 1.0}


def _8k_impact(items):
    """8-K Item codes (e.g. ["5.02", "9.01"]) -> (impact, volatility_multiplier).
    An unrecognized or empty item list is treated as low-impact — the honest
    neutral default rather than a guess."""
    impact = "high" if any(it in HIGH_IMPACT_8K_ITEMS for it in (items or [])) else "low"
    return impact, ITEM_VOLATILITY_MULTIPLIER[impact]


def _parse_form4(url, timeout=15):
    """Fetch + parse a Form 4 ownership XML into the reporting owner and their
    genuine open-market activity. Returns {owner, title, is_csuite, is_director,
    is_officer, is_ten_pct_owner, buy_usd, sell_usd} or None on any failure.
    Grants, option exercises, tax withholding and gifts are excluded — they're
    not a conviction signal, only actual open-market P(urchase)/S(ale)
    transactions are counted."""
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
    is_director = (rel.findtext("isDirector") or "0").strip() in ("1", "true", "True") if rel is not None else False
    is_officer = (rel.findtext("isOfficer") or "0").strip() in ("1", "true", "True") if rel is not None else False
    is_ten_pct = (rel.findtext("isTenPercentOwner") or "0").strip() in ("1", "true", "True") if rel is not None else False
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
            "is_director": is_director, "is_officer": is_officer, "is_ten_pct_owner": is_ten_pct,
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
    {form, note, bias, date, accepted, after_hours, url}, plus for 8-K filings:
    {items, impact, volatility_multiplier, session, gap_multiplier, vol_mult}.
    [] on any failure (network, missing ticker, malformed response).

    Form 4s are parsed for their OWN transactions (_parse_form4) so `bias`
    reflects real insider sentiment (open-market buy = bullish, sell =
    bearish) rather than just "a Form 4 was filed"; a flood of buying
    insiders or multiple C-suite officers selling at once is called out in
    `note` (_flag_insider_flood). Each Form 4 entry also carries `buy_usd`,
    `sell_usd`, `title` (reporting person's role), `owner_name`, and
    `is_director`/`is_officer`/`is_ten_pct_owner` as plain fields — not just
    baked into `note` — for callers (e.g. a UI badge, or signal_score() below)
    that want the structured amount/role rather than a parsed string. For a
    role-weighted, 10b5-1-discounted AGGREGATE across a rolling 24-72h
    window with exponential time decay (rather than one filing's own
    transactions), see form4_insider_bias().

    8-K entries get both a precise per-Item volatility multiplier
    (`volatility_multiplier`, from the exact Item codes) and a session-based
    gap-risk multiplier (`gap_multiplier`, pre-market > after-hours > regular)
    — `vol_mult` is their product, for callers that just want one number."""
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
    docs = rec.get("primaryDocument", []); items_col = rec.get("items", [])
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
        accepted = acct[i] if i < len(acct) else ""
        ah = _after_hours(accepted) if accepted else False
        owner = csuite = None
        extra = {}
        if f == "4" and a and doc:
            detail = _parse_form4(url, timeout)
            if detail:
                owner, csuite = detail["owner"], detail["is_csuite"]
                extra = {"buy_usd": detail["buy_usd"], "sell_usd": detail["sell_usd"],
                         "title": detail["title"], "owner_name": detail["owner"],
                         "is_director": detail["is_director"], "is_officer": detail["is_officer"],
                         "is_ten_pct_owner": detail["is_ten_pct_owner"]}
                if detail["buy_usd"] != detail["sell_usd"] and (detail["buy_usd"] or detail["sell_usd"]):
                    bias = 1 if detail["buy_usd"] > detail["sell_usd"] else -1
                    usd = detail["buy_usd"] if bias > 0 else detail["sell_usd"]
                    who = detail["title"] or "insider"
                    note = f"insider {'buy' if bias > 0 else 'sell'} (Form 4) — {who} ${usd/1e6:.1f}M"
        elif f == "8-K":
            try:
                raw_items = items_col[i] if i < len(items_col) else ""
                item_codes = [it.strip() for it in (raw_items or "").split(",") if it.strip()]
            except Exception:
                item_codes = []
            tag = next((EXEC_ITEMS[c] for c in item_codes if c in EXEC_ITEMS), None)
            if tag:
                note = f"material event (8-K) — {tag}"
            impact, item_mult = _8k_impact(item_codes)
            session = _session(accepted) if accepted else "overnight"
            gap_mult = SESSION_GAP_MULTIPLIER.get(session, 1.0)
            extra = {"items": item_codes, "impact": impact, "volatility_multiplier": item_mult,
                     "session": session, "gap_multiplier": gap_mult,
                     "vol_mult": round(item_mult * gap_mult, 3)}
        entry = {"form": f, "note": note, "bias": bias, "date": dates[i],
                 "accepted": accepted, "after_hours": ah, "url": url,
                 "_owner": owner, "_csuite": csuite}
        entry.update(extra)
        out.append(entry)
    _flag_insider_flood(out)
    for e in out:
        del e["_owner"], e["_csuite"]
    return out


# Max magnitude signal_score() can produce: bias in {-1, 0, 1} times the
# widest role weight (CEO_CFO_WEIGHT below, once defined).
SIGNAL_SCORE_BOUND = 2.0


def signal_score(row):
    """Composite, sign-directional strength for one recent_filings() row —
    `bias` scaled by the row's own weight, so magnitude is comparable across
    row types: bigger |signal_score| = a stronger, more-conviction-weighted
    signal, same sign convention as `bias` (positive = bullish).

      Form 4  -> role weight from _role_weight(): CEO/CFO 2.0x, any other
                 officer or 10%+ owner 1.5x, plain director 1.0x.
      8-K     -> item/session volatility multiplier (volatility_multiplier *
                 gap_multiplier); `bias` is always 0 for 8-K (this app never
                 guesses an 8-K's direction), so this is honestly 0 too —
                 an 8-K is an attention flag, not a directional signal.
      other   -> bias alone (weight 1.0) — a dilution filing or an activist
                 stake disclosure doesn't carry a role or item code to weight by.

    Missing/unrecognized fields fall back to neutral (bias 0 or weight 1.0),
    never raise. Rounded to 3 places and clamped to ±SIGNAL_SCORE_BOUND as a
    defensive bound, not because either weighting scheme can exceed it today."""
    bias = row.get("bias") or 0
    form = row.get("form")
    if form == "4":
        weight = _role_weight(row)
    elif form == "8-K":
        weight = row.get("volatility_multiplier", 1.0) * row.get("gap_multiplier", 1.0)
    else:
        weight = 1.0
    return round(max(-SIGNAL_SCORE_BOUND, min(SIGNAL_SCORE_BOUND, bias * weight)), 3)


# ===================================================================== Form 4 XML =====
# recent_filings() above parses each Form 4's OWN transactions in isolation
# (via _parse_form4) to set that ONE filing's bias/note. It can't tell an
# opportunistic $2M open-market buy from a routine RSU vest across MULTIPLE
# filings in a rolling window with role weighting and time decay, because
# that needs its own aggregation pass — that's what form4_insider_bias() below
# does, fetching and parsing each filing's XML directly (parse_form4_xml).

# Transaction codes: only open-market purchases/sales say anything about
# conviction. Grants/awards (A) and option exercises (M) are compensation,
# not a decision to spend cash; tax-withholding "sales" (F) and gifts (D)
# aren't market transactions at all. All are ignored, never counted either way.
BULLISH_CODES = {"P"}   # open-market purchase
BEARISH_CODES = {"S"}   # open-market sale
IGNORED_CODES = {"A", "M", "F", "D", "G"}

# Reporting-person role -> weight, a 3-tier scheme: the two roles with full
# P&L/operational visibility (CEO, CFO) spending their own cash carry the
# top multiplier; any other officer (COO, President, General Counsel, ...)
# or a 10%+ beneficial owner gets a mid-tier bump (economically motivated,
# even without a C-suite title); a plain director is the baseline.
CEO_CFO_WEIGHT = 2.0
OFFICER_WEIGHT = 1.5
DIRECTOR_WEIGHT = 1.0
TEN_PCT_OWNER_WEIGHT = OFFICER_WEIGHT
DEFAULT_ROLE_WEIGHT = DIRECTOR_WEIGHT

# Substring match on the free-text officerTitle, not an exact-title lookup —
# SEC filings spell the same role inconsistently ("Chief Executive Officer"
# vs "President and Chief Executive Officer" vs "Chief Executive Officer &
# President"), so a plain lookup would silently miss real CEOs/CFOs.
_CEO_CFO_TITLE_MARKERS = ("chief executive officer", " ceo", "ceo ",
                          "chief financial officer", " cfo", "cfo ")

# A transaction flagged as executed under a Rule 10b5-1 trading plan was
# scheduled in advance, often months earlier — it is compliance housekeeping,
# not a fresh opportunistic decision. It's heavily discounted rather than
# dropped outright, since a 10b5-1 sale can still be adopted/amended in ways
# that carry some information.
PLAN_10B5_1_DISCOUNT = 0.2


def _local(tag):
    """Strip an optional XML namespace off an element tag."""
    return tag.rsplit("}", 1)[-1] if tag else tag


def _iter_local(elem, name):
    """Yield every descendant of `elem` whose local tag name is `name`."""
    for child in elem.iter():
        if _local(child.tag) == name:
            yield child


def _child_text(elem, name, default=None):
    """First descendant's text named `name`, unwrapping SEC's <tag><value>x</value></tag>
    convention. `default` (never raises) if absent or empty."""
    e = next(_iter_local(elem, name), None)
    if e is None:
        return default
    v = e.findtext("value")
    if v is None:
        v = e.text
    v = (v or "").strip()
    return v if v else default


def _role_weight(tx):
    """Reporting-person role on one parsed transaction -> a scoring weight:
    CEO/CFO = 2.0x, any other officer or 10%+ owner = 1.5x, director = 1.0x."""
    title = (tx.get("title") or "").lower()
    if any(marker in title for marker in _CEO_CFO_TITLE_MARKERS):
        return CEO_CFO_WEIGHT
    if tx.get("is_officer") or tx.get("is_ten_pct_owner"):
        return OFFICER_WEIGHT
    return DIRECTOR_WEIGHT


def parse_form4_xml(xml_text):
    """Parse one Form 4 XML document -> list of open-market P/S transaction
    dicts: {code, shares, price, usd, date, is_10b5_1, owner, title,
    is_director, is_officer, is_ten_pct_owner}. Grants/exercises/tax-withholding
    (A/M/F/D/G) are dropped here so callers never have to re-filter. Returns
    [] on any parse failure — malformed or unexpected XML never raises."""
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []

    try:
        footnotes = {}
        for fn in _iter_local(root, "footnote"):
            fid = fn.get("id")
            if fid:
                footnotes[fid] = (fn.text or "").strip()

        owners = []
        for owner_el in _iter_local(root, "reportingOwner"):
            rel = next(_iter_local(owner_el, "reportingOwnerRelationship"), None)
            name = _child_text(owner_el, "rptOwnerName", "") or ""
            is_director = str(_child_text(rel, "isDirector", "0") if rel is not None else "0") in ("1", "true", "True")
            is_officer = str(_child_text(rel, "isOfficer", "0") if rel is not None else "0") in ("1", "true", "True")
            is_ten_pct = str(_child_text(rel, "isTenPercentOwner", "0") if rel is not None else "0") in ("1", "true", "True")
            title = _child_text(rel, "officerTitle", "") if rel is not None else ""
            owners.append({"name": name, "title": title or "", "is_director": is_director,
                            "is_officer": is_officer, "is_ten_pct_owner": is_ten_pct})
        # Form 4s are almost always single-owner; joint filings are rare enough
        # that attributing every transaction to the first listed owner is an
        # acceptable simplification rather than a source of silent wrong answers.
        owner = owners[0] if owners else {"name": "", "title": "", "is_director": False,
                                           "is_officer": False, "is_ten_pct_owner": False}

        out = []
        for tx in _iter_local(root, "nonDerivativeTransaction"):
            code = (_child_text(tx, "transactionCode", "") or "").strip().upper()
            if code not in BULLISH_CODES and code not in BEARISH_CODES:
                continue
            try:
                shares = float(_child_text(tx, "transactionShares", "0") or 0)
            except (TypeError, ValueError):
                shares = 0.0
            try:
                price = float(_child_text(tx, "transactionPricePerShare", "0") or 0)
            except (TypeError, ValueError):
                price = 0.0
            fn_ids = [e.get("id") for e in _iter_local(tx, "footnoteId") if e.get("id")]
            fn_text = " ".join(footnotes.get(i, "") for i in fn_ids).lower()
            is_10b5_1 = "10b5-1" in fn_text or "10b5(1)" in fn_text
            if not is_10b5_1:
                # Newer Form 4 schemas (2023 rule 33-11138) carry an explicit
                # per-transaction 10b5-1 flag instead of/alongside a footnote.
                for e in tx.iter():
                    if "10b5" in _local(e.tag).lower():
                        v = (e.findtext("value") or e.text or "").strip().lower()
                        if v in ("1", "true"):
                            is_10b5_1 = True
                            break
            out.append({
                "code": code, "shares": shares, "price": price, "usd": shares * price,
                "date": _child_text(tx, "transactionDate", "") or "",
                "is_10b5_1": is_10b5_1,
                "owner": owner["name"], "title": owner["title"],
                "is_director": owner["is_director"], "is_officer": owner["is_officer"],
                "is_ten_pct_owner": owner["is_ten_pct_owner"],
            })
        return out
    except Exception:
        return []


def _form4_filing_index(ticker, lookback_hours=72, timeout=15):
    """Form 4 filings for `ticker` accepted within `lookback_hours` -> list of
    {accepted, when (aware datetime), url}, newest first. [] on any failure."""
    cik = _load_ciks().get(ticker.upper())
    if not cik:
        return []
    try:
        d = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik}.json", timeout))
    except Exception:
        return []
    rec = d.get("filings", {}).get("recent", {})
    forms = rec.get("form", []); acct = rec.get("acceptanceDateTime", [])
    acc = rec.get("accessionNumber", []); docs = rec.get("primaryDocument", [])
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    out = []
    for i in range(len(forms)):
        if forms[i] != "4":
            continue
        ts = acct[i] if i < len(acct) else ""
        if not ts:
            continue
        try:
            when = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if when < cutoff:
            continue
        a = acc[i].replace("-", "") if i < len(acc) else ""
        doc = docs[i] if i < len(docs) else ""
        if not (a and doc):
            continue
        out.append({"accepted": ts, "when": when,
                    "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{a}/{doc}"})
    return sorted(out, key=lambda f: f["when"], reverse=True)


def form4_insider_bias(ticker, lookback_hours=72, half_life_hours=36, timeout=15, max_filings=15):
    """Net insider bias over a rolling `lookback_hours` window (24-72h is the
    intended range), from PARSED Form 4 XML rather than submissions metadata.

    Only open-market P/S transactions count (see parse_form4_xml); each is
    weighted by the reporting person's role (_role_weight) and discounted if
    filed under a Rule 10b5-1 plan (PLAN_10B5_1_DISCOUNT), then exponentially
    decayed by age (half-life `half_life_hours`) so a stale cluster fades out
    rather than permanently biasing the score.

    Returns {signal -1..+1, confidence 0..1, buy_usd, sell_usd, n_buys,
    n_sells, transactions, detail} or None if there's no qualifying activity,
    the ticker isn't found, or any fetch/parse step fails — this NEVER raises,
    so a flaky SEC response degrades to "no insider signal", not a crash."""
    try:
        filings = _form4_filing_index(ticker, lookback_hours, timeout)
    except Exception:
        return None
    if not filings:
        return None

    now = datetime.now(timezone.utc)
    weighted_buy = weighted_sell = 0.0
    buy_usd = sell_usd = 0.0
    n_buys = n_sells = 0
    transactions = []
    for f in filings[:max_filings]:
        try:
            xml_text = _get(f["url"], timeout)
        except Exception:
            continue
        txs = parse_form4_xml(xml_text)
        if not txs:
            continue
        age_h = max(0.0, (now - f["when"]).total_seconds() / 3600.0)
        decay = 0.5 ** (age_h / half_life_hours) if half_life_hours > 0 else 1.0
        for tx in txs:
            if tx["usd"] <= 0:
                continue
            w = _role_weight(tx) * decay
            if tx["is_10b5_1"]:
                w *= PLAN_10B5_1_DISCOUNT
            weighted = tx["usd"] * w
            if tx["code"] in BULLISH_CODES:
                weighted_buy += weighted; buy_usd += tx["usd"]; n_buys += 1
            elif tx["code"] in BEARISH_CODES:
                weighted_sell += weighted; sell_usd += tx["usd"]; n_sells += 1
            transactions.append({**tx, "weight": round(w, 3), "accepted": f["accepted"], "url": f["url"]})

    tot = weighted_buy + weighted_sell
    if tot <= 0:
        return None
    signal = max(-1.0, min(1.0, (weighted_buy - weighted_sell) / tot))
    confidence = min(1.0, (buy_usd + sell_usd) / 2_000_000.0)   # ~$2M raw flow = full confidence
    plan_n = sum(1 for t in transactions if t["is_10b5_1"])
    detail = (f"${buy_usd/1e6:.1f}M bought / ${sell_usd/1e6:.1f}M sold "
              f"({n_buys}B/{n_sells}S, {lookback_hours:.0f}h window, role/decay-weighted)"
              + (f" · {plan_n} 10b5-1" if plan_n else ""))
    return {"signal": float(signal), "confidence": float(confidence), "buy_usd": buy_usd,
            "sell_usd": sell_usd, "n_buys": n_buys, "n_sells": n_sells,
            "transactions": transactions, "detail": detail}

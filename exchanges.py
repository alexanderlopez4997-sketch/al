#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exchanges.py — stock exchange / TRF / SIP reference data.

Stores the Massive/Polygon-compatible `/v3/reference/exchanges` table:
each row maps a single-letter tape participant_id (as seen on SIP trade
and quote messages, e.g. sale_conditions.py's per-tape codes) to the
reporting venue's name, MIC, and operating MIC.

A bundled snapshot ships offline; fetch_all_exchanges() can refresh it
from the live API, following the same pagination pattern as
sale_conditions.fetch_all_conditions().
"""
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

MASSIVE_DEFAULT_KEY = os.environ.get("MASSIVE_KEY", "")
MASSIVE_API_BASE = os.environ.get("MASSIVE_API_BASE", "https://api.massive.com")


@dataclass
class Exchange:
    """One exchange / TRF / SIP reference row."""
    id: int
    type: str
    asset_class: str
    locale: str
    name: str
    operating_mic: str
    mic: Optional[str] = None
    acronym: Optional[str] = None
    participant_id: Optional[str] = None
    url: Optional[str] = None


def parse_exchanges(raw_results: List[dict]) -> List[Exchange]:
    """Turn the API's `results` list into Exchange objects."""
    out = []
    for r in raw_results:
        out.append(Exchange(
            id=r["id"],
            type=r.get("type", ""),
            asset_class=r.get("asset_class", "stocks"),
            locale=r.get("locale", "us"),
            name=r["name"],
            operating_mic=r.get("operating_mic", ""),
            mic=r.get("mic"),
            acronym=r.get("acronym"),
            participant_id=r.get("participant_id"),
            url=r.get("url"),
        ))
    return out


def build_participant_index(exchanges: List[Exchange]) -> Dict[str, Exchange]:
    """Index by single-letter tape participant_id (the code seen on SIP messages)."""
    return {e.participant_id: e for e in exchanges if e.participant_id}


def build_mic_index(exchanges: List[Exchange]) -> Dict[str, Exchange]:
    """Index by MIC code."""
    return {e.mic: e for e in exchanges if e.mic}


def build_id_index(exchanges: List[Exchange]) -> Dict[int, Exchange]:
    """Index by numeric id — the form real-time trade messages carry in
    their `exchange` field (mirrors sale_conditions.build_id_index)."""
    return {e.id: e for e in exchanges}


def get_exchange_by_id(exchange_id: int, index: Optional[Dict[int, Exchange]] = None) -> Optional[Exchange]:
    """Look up the Exchange for a single numeric exchange id from a trade message."""
    if index is None:
        index = DEFAULT_ID_INDEX
    return index.get(exchange_id)


def fetch_all_exchanges(
    api_key=None, asset_class="stocks", base_url=None, timeout=10, max_pages=50
):
    """Page through the live /v3/reference/exchanges endpoint.

    Returns the full List[Exchange], or None if no key is configured or
    the request fails (mirrors sale_conditions.fetch_all_conditions).
    """
    api_key = api_key or MASSIVE_DEFAULT_KEY
    if not api_key:
        return None
    base_url = base_url or MASSIVE_API_BASE
    url = f"{base_url}/v3/reference/exchanges?asset_class={asset_class}&apiKey={api_key}"
    results = []
    try:
        for _ in range(max_pages):
            with urllib.request.urlopen(url, timeout=timeout) as r:
                payload = json.loads(r.read())
            results.extend(payload.get("results", []))
            next_url = payload.get("next_url")
            if not next_url:
                break
            sep = "&" if "?" in next_url else "?"
            url = f"{next_url}{sep}apiKey={api_key}"
        return parse_exchanges(results)
    except (urllib.error.URLError, ValueError, KeyError, TimeoutError):
        return None


def get_exchange(participant_id: str, index: Optional[Dict[str, Exchange]] = None) -> Optional[Exchange]:
    """Look up the reporting Exchange for a single tape participant_id."""
    if index is None:
        index = DEFAULT_PARTICIPANT_INDEX
    return index.get(participant_id)


# ------------------------------------------------------- bundled snapshot ---
_BUNDLED_REFERENCE_RAW = [
    {"id": 1, "type": "exchange", "asset_class": "stocks", "locale": "us", "name": "NYSE American, LLC",
     "acronym": "AMEX", "mic": "XASE", "operating_mic": "XNYS", "participant_id": "A", "url": "https://www.nyse.com/markets/nyse-american"},
    {"id": 2, "type": "exchange", "asset_class": "stocks", "locale": "us", "name": "Nasdaq Texas, Inc.",
     "mic": "XBOS", "operating_mic": "XNAS", "participant_id": "B", "url": "https://www.nasdaq.com/solutions/nasdaq-bx-stock-market"},
    {"id": 3, "type": "exchange", "asset_class": "stocks", "locale": "us", "name": "NYSE National, Inc.",
     "acronym": "NSX", "mic": "XCIS", "operating_mic": "XNYS", "participant_id": "C", "url": "https://www.nyse.com/markets/nyse-national"},
    {"id": 4, "type": "TRF", "asset_class": "stocks", "locale": "us", "name": "FINRA Alternative Display Facility",
     "mic": "XADF", "operating_mic": "FINR", "participant_id": "D", "url": "https://www.finra.org"},
    {"id": 5, "type": "SIP", "asset_class": "stocks", "locale": "us", "name": "Unlisted Trading Privileges",
     "operating_mic": "XNAS", "participant_id": "E", "url": "https://www.utpplan.com"},
    {"id": 6, "type": "TRF", "asset_class": "stocks", "locale": "us", "name": "International Securities Exchange, LLC - Stocks",
     "mic": "XISE", "operating_mic": "XNAS", "participant_id": "I", "url": "https://nasdaq.com/solutions/nasdaq-ise"},
    {"id": 7, "type": "exchange", "asset_class": "stocks", "locale": "us", "name": "Cboe EDGA",
     "mic": "EDGA", "operating_mic": "XCBO", "participant_id": "J", "url": "https://www.cboe.com/us/equities"},
    {"id": 8, "type": "exchange", "asset_class": "stocks", "locale": "us", "name": "Cboe EDGX",
     "mic": "EDGX", "operating_mic": "XCBO", "participant_id": "K", "url": "https://www.cboe.com/us/equities"},
    {"id": 9, "type": "exchange", "asset_class": "stocks", "locale": "us", "name": "NYSE Texas, Inc.",
     "mic": "XCHI", "operating_mic": "XNYS", "participant_id": "M", "url": "https://www.nyse.com/markets/nyse-texas"},
    {"id": 10, "type": "exchange", "asset_class": "stocks", "locale": "us", "name": "New York Stock Exchange",
     "mic": "XNYS", "operating_mic": "XNYS", "participant_id": "N", "url": "https://www.nyse.com"},
    {"id": 11, "type": "exchange", "asset_class": "stocks", "locale": "us", "name": "NYSE Arca, Inc.",
     "mic": "ARCX", "operating_mic": "XNYS", "participant_id": "P", "url": "https://www.nyse.com/markets/nyse-arca"},
    {"id": 12, "type": "exchange", "asset_class": "stocks", "locale": "us", "name": "Nasdaq",
     "mic": "XNAS", "operating_mic": "XNAS", "participant_id": "T", "url": "https://www.nasdaq.com"},
    {"id": 13, "type": "SIP", "asset_class": "stocks", "locale": "us", "name": "Consolidated Tape Association",
     "operating_mic": "XNYS", "participant_id": "S", "url": "https://www.nyse.com/data/cta"},
    {"id": 14, "type": "exchange", "asset_class": "stocks", "locale": "us", "name": "Long-Term Stock Exchange",
     "mic": "LTSE", "operating_mic": "LTSE", "participant_id": "L", "url": "https://www.ltse.com"},
    {"id": 15, "type": "exchange", "asset_class": "stocks", "locale": "us", "name": "Investors Exchange",
     "mic": "IEXG", "operating_mic": "IEXG", "participant_id": "V", "url": "https://www.iextrading.com"},
    {"id": 16, "type": "TRF", "asset_class": "stocks", "locale": "us", "name": "Cboe Stock Exchange",
     "mic": "CBSX", "operating_mic": "XCBO", "participant_id": "W", "url": "https://www.cboe.com"},
    {"id": 17, "type": "exchange", "asset_class": "stocks", "locale": "us", "name": "Nasdaq Philadelphia Exchange LLC",
     "mic": "XPHL", "operating_mic": "XNAS", "participant_id": "X", "url": "https://www.nasdaq.com/solutions/nasdaq-phlx"},
    {"id": 18, "type": "exchange", "asset_class": "stocks", "locale": "us", "name": "Cboe BYX",
     "mic": "BATY", "operating_mic": "XCBO", "participant_id": "Y", "url": "https://www.cboe.com/us/equities"},
    {"id": 19, "type": "exchange", "asset_class": "stocks", "locale": "us", "name": "Cboe BZX",
     "mic": "BATS", "operating_mic": "XCBO", "participant_id": "Z", "url": "https://www.cboe.com/us/equities"},
    {"id": 20, "type": "exchange", "asset_class": "stocks", "locale": "us", "name": "MIAX Pearl",
     "mic": "EPRL", "operating_mic": "MIHI", "participant_id": "H", "url": "https://www.miaxoptions.com/alerts/pearl-equities"},
    {"id": 21, "type": "exchange", "asset_class": "stocks", "locale": "us", "name": "Members Exchange",
     "mic": "MEMX", "operating_mic": "MEMX", "participant_id": "U", "url": "https://www.memx.com"},
    {"id": 22, "type": "exchange", "asset_class": "stocks", "locale": "us", "name": "24X National Exchange LLC",
     "acronym": "24X", "mic": "24EQ", "operating_mic": "24EQ", "participant_id": "G", "url": "https://24exchange.com/"},
    {"id": 23, "type": "exchange", "asset_class": "stocks", "locale": "us", "name": "Texas Stock Exchange LLC",
     "acronym": "TXSE", "mic": "TXSE", "operating_mic": "TXSE", "participant_id": "F", "url": "https://txse.com/"},
    {"id": 62, "type": "ORF", "asset_class": "stocks", "locale": "us", "name": "OTC Equity Security",
     "mic": "OOTC", "operating_mic": "FINR", "url": "https://www.finra.org/filing-reporting/over-the-counter-reporting-facility-orf"},
    {"id": 201, "type": "TRF", "asset_class": "stocks", "locale": "us", "name": "FINRA NYSE TRF",
     "mic": "FINY", "operating_mic": "FINR", "url": "https://www.finra.org"},
    {"id": 202, "type": "TRF", "asset_class": "stocks", "locale": "us", "name": "FINRA Nasdaq TRF Carteret",
     "mic": "FINN", "operating_mic": "FINR", "url": "https://www.finra.org"},
    {"id": 203, "type": "TRF", "asset_class": "stocks", "locale": "us", "name": "FINRA Nasdaq TRF Chicago",
     "mic": "FINC", "operating_mic": "FINR", "url": "https://www.finra.org"},
]

DEFAULT_EXCHANGES: List[Exchange] = parse_exchanges(_BUNDLED_REFERENCE_RAW)
DEFAULT_PARTICIPANT_INDEX: Dict[str, Exchange] = build_participant_index(DEFAULT_EXCHANGES)
DEFAULT_MIC_INDEX: Dict[str, Exchange] = build_mic_index(DEFAULT_EXCHANGES)
DEFAULT_ID_INDEX: Dict[int, Exchange] = build_id_index(DEFAULT_EXCHANGES)


def get_exchanges(api_key=None, prefer_live=True) -> List[Exchange]:
    """Live exchanges table when a key is available, else the bundled snapshot."""
    if prefer_live:
        live = fetch_all_exchanges(api_key=api_key)
        if live:
            return live
    return DEFAULT_EXCHANGES

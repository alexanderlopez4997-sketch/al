#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sale_conditions.py — stock trade sale-condition reference data.

Stores the SIP sale-condition table (Massive/Polygon-compatible
`/v3/reference/conditions` schema: per-tape code mappings plus
consolidated/market-center update rules) and classifies a trade's raw
condition codes into which OHLCV fields it is allowed to update.

A bundled snapshot ships offline so classification works without any
network access; fetch_all_conditions() can refresh it from the live API.
"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

MASSIVE_DEFAULT_KEY = os.environ.get("MASSIVE_KEY", "")
MASSIVE_API_BASE = os.environ.get("MASSIVE_API_BASE", "https://api.massive.com")

UPDATE_FIELDS = ("updates_high_low", "updates_open_close", "updates_volume")


@dataclass
class SaleCondition:
    """One SIP sale condition and its per-tape code + update rules."""
    id: int
    name: str
    asset_class: str
    sip_mapping: Dict[str, str] = field(default_factory=dict)
    update_rules: Dict[str, Dict[str, bool]] = field(default_factory=dict)
    data_types: List[str] = field(default_factory=list)
    legacy: bool = False

    def rules_for(self, scope: str) -> Dict[str, bool]:
        """Update rules for 'consolidated' or 'market_center', defaulting to all-True."""
        return self.update_rules.get(scope, {f: True for f in UPDATE_FIELDS})


def parse_conditions(raw_results: List[dict]) -> List[SaleCondition]:
    """Turn the API's `results` list into SaleCondition objects."""
    out = []
    for r in raw_results:
        out.append(SaleCondition(
            id=r["id"],
            name=r["name"],
            asset_class=r.get("asset_class", "stocks"),
            sip_mapping=r.get("sip_mapping", {}),
            update_rules=r.get("update_rules", {}),
            data_types=r.get("data_types", []),
            legacy=r.get("legacy", False),
        ))
    return out


def build_index(conditions: List[SaleCondition]) -> Dict[str, Dict[str, SaleCondition]]:
    """Index conditions by {tape: {sip_code: SaleCondition}} for O(1) lookup."""
    index: Dict[str, Dict[str, SaleCondition]] = {}
    for cond in conditions:
        for tape, code in cond.sip_mapping.items():
            index.setdefault(tape, {})[code] = cond
    return index


def fetch_all_conditions(
    api_key=None, asset_class="stocks", base_url=None, timeout=10, max_pages=50
):
    """Page through the live /v3/reference/conditions endpoint.

    Returns the full List[SaleCondition], or None if no key is configured
    or the request fails (mirrors quant_engine.massive_quote's fallback style).
    """
    api_key = api_key or MASSIVE_DEFAULT_KEY
    if not api_key:
        return None
    base_url = base_url or MASSIVE_API_BASE
    url = (f"{base_url}/v3/reference/conditions?asset_class={asset_class}"
           f"&limit=1000&order=asc&sort=asset_class&apiKey={api_key}")
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
        return parse_conditions(results)
    except (urllib.error.URLError, ValueError, KeyError, TimeoutError):
        return None


def classify_trade(
    condition_codes: List[str], tape: str, scope: str = "consolidated",
    index: Optional[Dict[str, Dict[str, SaleCondition]]] = None
) -> Dict[str, bool]:
    """Whether a trade's SIP condition codes let it update high/low, open/close, volume.

    A field is suppressed if ANY present, recognized condition marks it False
    for the given scope; unknown codes and trades with no codes update everything.
    `tape` is the SIP feed the codes belong to (CTA/UTP/FINRA_TDDS); `scope` is
    'consolidated' or 'market_center'.
    """
    if index is None:
        index = DEFAULT_INDEX
    result = {f: True for f in UPDATE_FIELDS}
    for code in condition_codes:
        cond = index.get(tape, {}).get(code)
        if cond is None:
            continue
        rules = cond.rules_for(scope)
        for f in UPDATE_FIELDS:
            if not rules.get(f, True):
                result[f] = False
    return result


def get_condition(
    code: str, tape: str, index: Optional[Dict[str, Dict[str, SaleCondition]]] = None
) -> Optional[SaleCondition]:
    """Look up the SaleCondition for a single SIP code on a given tape."""
    if index is None:
        index = DEFAULT_INDEX
    return index.get(tape, {}).get(code)


def build_id_index(conditions: List[SaleCondition]) -> Dict[int, SaleCondition]:
    """Index conditions by numeric id — the form real-time trade messages carry
    in their `conditions` array (Massive/Polygon normalize per-tape SIP codes
    into these ids, so live trades don't need a tape to classify)."""
    return {cond.id: cond for cond in conditions}


def classify_trade_by_id(
    condition_ids: List[int], scope: str = "consolidated",
    index: Optional[Dict[int, SaleCondition]] = None
) -> Dict[str, bool]:
    """Same as classify_trade(), keyed by the numeric condition ids a live
    trade message's `conditions` field carries instead of tape+SIP-code."""
    if index is None:
        index = DEFAULT_ID_INDEX
    result = {f: True for f in UPDATE_FIELDS}
    for cid in condition_ids:
        cond = index.get(cid)
        if cond is None:
            continue
        rules = cond.rules_for(scope)
        for f in UPDATE_FIELDS:
            if not rules.get(f, True):
                result[f] = False
    return result


def get_condition_by_id(condition_id: int, index: Optional[Dict[int, SaleCondition]] = None) -> Optional[SaleCondition]:
    """Look up the SaleCondition for a single numeric condition id."""
    if index is None:
        index = DEFAULT_ID_INDEX
    return index.get(condition_id)


# ------------------------------------------------------- bundled snapshot ---
# Offline default so classify_trade() works with zero network/API-key setup.
# Refresh via: parse_conditions(...) from fetch_all_conditions() output.
_BUNDLED_REFERENCE_RAW = [
    {"id": 1, "name": "Acquisition", "asset_class": "stocks",
     "sip_mapping": {"UTP": "A"},
     "update_rules": {
         "consolidated": {"updates_high_low": True, "updates_open_close": True, "updates_volume": True},
         "market_center": {"updates_high_low": True, "updates_open_close": True, "updates_volume": True}},
     "data_types": ["trade"]},
    {"id": 2, "name": "Average Price Trade", "asset_class": "stocks",
     "sip_mapping": {"CTA": "B", "UTP": "W", "FINRA_TDDS": "W"},
     "update_rules": {
         "consolidated": {"updates_high_low": False, "updates_open_close": False, "updates_volume": True},
         "market_center": {"updates_high_low": False, "updates_open_close": False, "updates_volume": True}},
     "data_types": ["trade"]},
    {"id": 3, "name": "Automatic Execution", "asset_class": "stocks",
     "sip_mapping": {"CTA": "E"},
     "update_rules": {
         "consolidated": {"updates_high_low": True, "updates_open_close": True, "updates_volume": True},
         "market_center": {"updates_high_low": True, "updates_open_close": True, "updates_volume": True}},
     "data_types": ["trade"]},
    {"id": 4, "name": "Bunched Trade", "asset_class": "stocks",
     "sip_mapping": {"UTP": "B"},
     "update_rules": {
         "consolidated": {"updates_high_low": True, "updates_open_close": True, "updates_volume": True},
         "market_center": {"updates_high_low": True, "updates_open_close": True, "updates_volume": True}},
     "data_types": ["trade"]},
    {"id": 5, "name": "Bunched Sold Trade", "asset_class": "stocks",
     "sip_mapping": {"UTP": "G"},
     "update_rules": {
         "consolidated": {"updates_high_low": True, "updates_open_close": False, "updates_volume": True},
         "market_center": {"updates_high_low": True, "updates_open_close": False, "updates_volume": True}},
     "data_types": ["trade"]},
    {"id": 6, "name": "CAP Election", "asset_class": "stocks",
     "sip_mapping": {"CTA": "I"},
     "update_rules": {
         "consolidated": {"updates_high_low": True, "updates_open_close": True, "updates_volume": True},
         "market_center": {"updates_high_low": True, "updates_open_close": True, "updates_volume": True}},
     "data_types": ["trade"], "legacy": True},
    {"id": 7, "name": "Cash Sale", "asset_class": "stocks",
     "sip_mapping": {"CTA": "C", "UTP": "C", "FINRA_TDDS": "C"},
     "update_rules": {
         "consolidated": {"updates_high_low": False, "updates_open_close": False, "updates_volume": True},
         "market_center": {"updates_high_low": False, "updates_open_close": False, "updates_volume": True}},
     "data_types": ["trade"]},
    {"id": 8, "name": "Closing Prints", "asset_class": "stocks",
     "sip_mapping": {"UTP": "6"},
     "update_rules": {
         "consolidated": {"updates_high_low": True, "updates_open_close": True, "updates_volume": True},
         "market_center": {"updates_high_low": True, "updates_open_close": True, "updates_volume": True}},
     "data_types": ["trade"]},
    {"id": 9, "name": "Cross Trade", "asset_class": "stocks",
     "sip_mapping": {"CTA": "X", "UTP": "X"},
     "update_rules": {
         "consolidated": {"updates_high_low": True, "updates_open_close": True, "updates_volume": True},
         "market_center": {"updates_high_low": True, "updates_open_close": True, "updates_volume": True}},
     "data_types": ["trade"]},
    {"id": 10, "name": "Derivatively Priced", "asset_class": "stocks",
     "sip_mapping": {"CTA": "4", "UTP": "4"},
     "update_rules": {
         "consolidated": {"updates_high_low": True, "updates_open_close": False, "updates_volume": True},
         "market_center": {"updates_high_low": True, "updates_open_close": False, "updates_volume": True}},
     "data_types": ["trade"]},
]

DEFAULT_CONDITIONS: List[SaleCondition] = parse_conditions(_BUNDLED_REFERENCE_RAW)
DEFAULT_INDEX: Dict[str, Dict[str, SaleCondition]] = build_index(DEFAULT_CONDITIONS)
DEFAULT_ID_INDEX: Dict[int, SaleCondition] = build_id_index(DEFAULT_CONDITIONS)


def get_conditions(api_key=None, prefer_live=True) -> List[SaleCondition]:
    """Live conditions table when a key is available, else the bundled snapshot."""
    if prefer_live:
        live = fetch_all_conditions(api_key=api_key)
        if live:
            return live
    return DEFAULT_CONDITIONS

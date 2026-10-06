#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Government Contract Monitor for Meridian — federal and DoD awards as an
early buying signal for small-cap defense, aerospace and tech contractors.

A $50M contract means nothing to Lockheed ($230B market cap). It moves a
$300M-cap company. This module fetches contract wins and scores them as:

    signal = contract_value / (market_cap or TTM revenue)

A contract is only valuable if matched to a ticker — we use the Quiver API
(the easiest path since your alt-data stack already authenticates there),
with USAspending.gov as a fallback for historical research.

Contract-to-revenue ratio thresholds:
  < 1%     micro signal, mostly noise
  1-5%     real signal, the size that moves a stock
  5-15%    major catalyst, expect >10% move
  > 15%    transformational (rare, usually on small-caps)

Age decay: a contract from 90 days ago is decaying and near-zero weight by
day 180. This prevents old wins from falsely signaling strength. Pair with
the daily watch-list to catch awards fresh.

Signal format matches congress/analyst/insider/whale/macro in quant_engine:
  {signal: -1..+1, confidence: 0..1, detail: str}

where signal is bounded by the contract-to-fundamentals ratio (never > 1.0,
can go negative if a contract is LOST or cancelled, though the API doesn't
currently expose that). Confidence scales with:
  - contract recency (fresh > old)
  - contract value (larger > smaller)
  - number of contracts in the window (multiple confirmations > single award)
"""

import json
import logging
import math
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np

logger = logging.getLogger("contracts")


def _finnhub_get(url, timeout=10):
    """GET Finnhub API, returning parsed JSON or None. Fail-open."""
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None

# ============================================================== QUIVER API ===
# Quiver is the fastest path: contracts already mapped to tickers. The /live/
# and /historical/ endpoints return real-time and historical data with minimal
# parsing. Requires QUIVER_API_TOKEN in environment.

QUIVER_API_BASE = "https://api.quiverquant.com/beta"
_QUIVER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")


def _quiver_get(path, token, timeout=15):
    """GET a Quiver endpoint, returning parsed JSON or None. Tries both auth
    schemes Quiver has used (Bearer/Token). Fail-open — never raises."""
    if not token:
        return None
    import json, urllib.request
    headers = {"Accept": "application/json", "User-Agent": _QUIVER_UA}
    for scheme in ("Bearer", "Token"):
        try:
            req = urllib.request.Request(f"{QUIVER_API_BASE}{path}",
                                         headers={**headers, "Authorization": f"{scheme} {token}"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception:
            continue
    return None


def quiver_contracts(ticker, token, lookback_days=180, timeout=10):
    """Recent gov contracts for ONE ticker via Quiver's historical endpoint.
    Returns a list of contract dicts, or empty list. Requires a Quiver token.

    Each contract carries: contractValue, agency, date (ISO 8601), description,
    and other metadata. The historical endpoint is ~4-week delayed; live endpoint
    is real-time but less complete."""
    if not token:
        return []
    try:
        data = _quiver_get(f"/historical/govcontracts/{ticker}", token, timeout)
        if isinstance(data, list):
            # Filter to lookback window (e.g. last 180d)
            cutoff = (pd.Timestamp.today() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            return [c for c in data if c.get("date", "")[:10] >= cutoff]
        return []
    except Exception:
        return []


def quiver_contracts_bulk(token, lookback_days=180, timeout=20):
    """Real-time gov contracts across ALL tickers via Quiver's live endpoint.
    Returns {ticker: [contracts]}, or {} on any failure. Much faster than
    calling quiver_contracts() per ticker. Requires a Quiver token."""
    if not token:
        return {}
    try:
        data = _quiver_get("/live/govcontracts", token, timeout)
        if not isinstance(data, list):
            return {}
        cutoff = (pd.Timestamp.today() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        groups = {}
        for c in data:
            ticker = c.get("Ticker") or c.get("ticker")
            if not ticker:
                continue
            if c.get("date", "")[:10] < cutoff:
                continue
            groups.setdefault(ticker, []).append(c)
        return groups
    except Exception:
        return {}


# ================================================================ MARKET CAP ===

def market_cap_from_finnhub(ticker, finnhub_key=None, timeout=10):
    """Fetch market cap in USD from Finnhub /stock/metric.
    Returns market cap (float) or None. Finnhub key required; uses
    FINNHUB_KEY env var if not provided."""
    if finnhub_key is None:
        finnhub_key = os.environ.get("FINNHUB_KEY", "")
    if not finnhub_key:
        return None
    try:
        url = (f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}"
               f"&metric=all&token={finnhub_key}")
        data = _finnhub_get(url, timeout)
        if not isinstance(data, dict):
            return None
        m = data.get("metric", {})
        # Finnhub has marketCapitalization in USD (millions)
        mcap_m = m.get("marketCapitalization")
        if mcap_m is not None:
            return float(mcap_m) * 1e6  # convert millions to dollars
        return None
    except Exception:
        return None


# ================================================================ SCORING ===

def _days_since(date_str):
    """Days from today to an ISO date string, clamped to [0, inf)."""
    try:
        return max(0.0, (pd.Timestamp.today().normalize() - pd.Timestamp(date_str[:10])).days)
    except Exception:
        return None


def _age_decay(days_since):
    """Exponential recency multiplier in (0,1] = exp(-age/60d). A contract today =
    1.00; 42 days ago ≈ 0.50; 60 days ≈ 0.37; 180 days ≈ 0.05. Contracts decay
    faster than insider Form 4s (30d tau) since a single award is less persistent
    signal than a repeated insider buyer. Uses 60d half-life."""
    if days_since is None or days_since < 0:
        return 1.0
    return float(math.exp(-days_since / 60.0))


def _contract_to_value_ratio(contracts, market_cap_or_revenue):
    """Total value across contracts ÷ market_cap_or_revenue, clamped to [-1, 1].

    Interpretation:
      signal=0.5 means contracts are worth 50% of market cap (huge signal)
      signal=0.2 means 20% (major catalyst)
      signal=0.05 means 5% (real but modest)
      signal=0.01 means 1% (micro, mostly noise)
      signal=-0.01 means LOST contract or negative news (rare in API)

    If fundamentals are missing or zero, return 0 (no signal, not error)."""
    if not contracts or market_cap_or_revenue is None or market_cap_or_revenue <= 0:
        return 0.0
    total = sum(float(c.get("contractValue") or c.get("value") or 0) for c in contracts)
    if total <= 0:
        return 0.0
    ratio = total / market_cap_or_revenue
    return float(np.clip(ratio, -1.0, 1.0))


def summarize_contracts(contracts, top=4, recent_days=180):
    """Condense a contract list into recent totals and top awards.

    Returns a dict with:
      total: count of all contracts in window
      recent_total: count in the last recent_days
      value_total: sum of all contract values
      value_recent: sum of recent values
      count_recent: count of recent values
      latest: [{agency, value, date, description}, ...]
      days_since: age of the most recent award (for decay calc)
    """
    contracts = sorted(contracts, key=lambda c: c.get("date", ""), reverse=True)
    cutoff = (pd.Timestamp.today() - pd.Timedelta(days=recent_days)).strftime("%Y-%m-%d")

    value_total = value_recent = 0.0
    count_recent = 0
    for c in contracts:
        v = float(c.get("contractValue") or c.get("value") or 0)
        value_total += v
        if c.get("date", "")[:10] >= cutoff:
            value_recent += v
            count_recent += 1

    latest = []
    for c in contracts[:top]:
        latest.append({
            "agency": c.get("agency") or c.get("Agency") or "?",
            "value": float(c.get("contractValue") or c.get("value") or 0),
            "date": c.get("date", "")[:10],
            "description": (c.get("description") or c.get("Description") or "")[:100],
        })

    days_since = _days_since(contracts[0].get("date")) if contracts else None
    return {
        "total": len(contracts),
        "recent_total": count_recent,
        "value_total": value_total,
        "value_recent": value_recent,
        "count_recent": count_recent,
        "recent_days": recent_days,
        "latest": latest,
        "days_since": days_since,
    }


def contract_signal(summary, market_cap=None, ttm_revenue=None):
    """Contract summary + fundamentals -> {signal, confidence, detail} or None.

    signal is the contract-to-fundamentals ratio, clipped to [-1, 1].
    confidence scales with:
      - contract value (larger > smaller)
      - recency (recent > old)
      - multiple awards (clustered > singleton)

    Use market_cap if available (preferred), else ttm_revenue, else return None.
    Decay confidence by age of the most recent award (old wins fade)."""
    if not summary or summary.get("total", 0) == 0:
        return None

    fundamentals = market_cap or ttm_revenue
    if fundamentals is None or fundamentals <= 0:
        return None

    ratio = _contract_to_value_ratio(summary.get("latest", []), fundamentals)
    if abs(ratio) < 0.001:  # sub-0.1% is noise
        return None

    # Confidence: scale with value, recency, and clustering
    val_conf = min(1.0, summary["value_recent"] / 1e7)  # ~$10M = full confidence
    recency_decay = _age_decay(summary.get("days_since"))
    cluster_conf = min(1.0, summary["count_recent"] / 3.0)  # 3+ recent = full confidence

    confidence = val_conf * recency_decay * cluster_conf

    # Detail string
    detail = f"${summary['value_recent']/1e6:.1f}M awarded ({summary['count_recent']} contracts)"
    if summary['value_total'] > summary['value_recent']:
        detail += f" · ${summary['value_total']/1e6:.1f}M all-time"
    if summary.get("days_since") is not None:
        decay = _age_decay(summary['days_since'])
        detail += f" · {summary['days_since']:.0f}d ago (decay ×{decay:.2f})"
    if summary["latest"]:
        top = summary["latest"][0]
        detail += f" · {top['agency']} {top['date']}"

    return {
        "signal": float(np.clip(ratio, -1, 1)),
        "confidence": float(confidence),
        "detail": detail,
    }


# ================================================================ PUBLIC API ===

def fetch_contracts_for_ticker(ticker, token=None, market_cap=None, ttm_revenue=None,
                                lookback_days=180, timeout=10):
    """Fetch gov contracts for ONE ticker and score them.

    Returns {signal, confidence, detail} or None.

    Args:
      ticker: stock symbol (e.g. 'LMT')
      token: Quiver API token (env var QUIVER_API_TOKEN if None)
      market_cap: market capitalization in USD (preferred for signal calc)
      ttm_revenue: trailing-12-month revenue in USD (fallback)
      lookback_days: how far back to search (default 180)
      timeout: HTTP timeout in seconds
    """
    if token is None:
        token = os.environ.get("QUIVER_API_TOKEN")

    contracts = quiver_contracts(ticker, token, lookback_days, timeout)
    if not contracts:
        return None

    summary = summarize_contracts(contracts, top=4, recent_days=lookback_days)
    return contract_signal(summary, market_cap, ttm_revenue)


def fetch_contracts_bulk(token=None, lookback_days=180, timeout=20, fundamentals_fn=None):
    """Fetch recent gov contracts for ALL tickers via Quiver's live endpoint.

    Returns {ticker: {signal, confidence, detail}} for tickers with contracts.

    Args:
      token: Quiver API token (env var if None)
      lookback_days: how far back to search
      timeout: HTTP timeout in seconds
      fundamentals_fn: optional callable(ticker) -> {market_cap, ttm_revenue}
                       If provided, scores each contract against fundamentals
    """
    if token is None:
        token = os.environ.get("QUIVER_API_TOKEN")

    groups = quiver_contracts_bulk(token, lookback_days, timeout)
    if not groups:
        return {}

    results = {}
    for ticker, contracts in groups.items():
        summary = summarize_contracts(contracts, top=4, recent_days=lookback_days)

        # Score: use provided fundamentals or None (signal will be skipped if no fundamentals)
        market_cap = ttm_revenue = None
        if fundamentals_fn:
            try:
                fund = fundamentals_fn(ticker)
                if fund:
                    market_cap = fund.get("market_cap")
                    ttm_revenue = fund.get("ttm_revenue")
            except Exception:
                pass  # fail-open: unscored contract is better than no result

        sig = contract_signal(summary, market_cap, ttm_revenue)
        if sig:
            results[ticker] = sig

    return results


if __name__ == "__main__":
    # Test: can be run with QUIVER_API_TOKEN in env
    token = os.environ.get("QUIVER_API_TOKEN")
    if not token:
        print("QUIVER_API_TOKEN not set; skipping live test")
    else:
        # Example: fetch contracts for LMT (Lockheed Martin)
        print("Fetching contracts for LMT...")
        cs = fetch_contracts_for_ticker("LMT", token, market_cap=230e9)
        if cs:
            print(json.dumps(cs, indent=2))
        else:
            print("No contracts found (or fundamentals missing)")

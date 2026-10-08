#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Historical event study for Meridian's government-contract signals.

Questions it answers, from real past awards instead of assumptions:

  1. Is there post-award drift at all, and does it grow with award size relative
     to market cap? (the < 1% / 1-5% / 5-15% / > 15% bands in contracts.py)
  2. How long does the drift last? (compared with contracts.py's 60-day decay)
  3. Do the cross-check flags in contract_crosscheck.py -- offering after the
     award, insider selling after, insider buying before -- actually separate
     good awards from bad ones, and is the SIZE of the confidence multipliers
     (x0.5 / x0.6 / x1.25) about right?

Method (market-adjusted event study on daily closes):
  * Event = one ticker's positive obligations on one action date, summed (a day's
    awards stack, like dod_scraper._combined_award), >= min_value. Later events for
    the same ticker inside `cooldown_days` are dropped so return windows don't overlap.
  * Information date = action date + lag_days (USAspending dates the signing, not the
    public disclosure). Entry = CLOSE of the first trading day on/after that date, so
    the announcement-day move is never counted as tradeable -- conservative, and the
    day-0 move is reported separately.
  * Abnormal return (AR) = stock return - benchmark (SPY) return over the same days.
  * Market cap = shares outstanding as FILED on or before the information date x the
    split-unadjusted close then. Point-in-time: later filings/splits cannot leak in.

Look-ahead discipline for the flags: a dilution/insider-selling flag only becomes
known days AFTER the award, so comparing "flagged vs clean" from the award date would
credit the flag with the offering's own price drop -- which nobody could trade. The
tradeable comparison therefore starts at a CHECKPOINT (the first trading day after the
cross-check window closes, when every flag is public). The from-entry comparison is
still printed, labelled descriptive. Insider BUYING is public by the award date, so it
is compared from entry.

The cross-check logic is NOT reimplemented here: events are flagged by calling
contract_crosscheck.evaluate(), so this tests the code that actually runs live.

HONEST LIMITS (also printed in every report):
  * Survivorship: tickers come from today's CONTRACTOR_TICKER_MAP and prices from a
    free feed; companies acquired or delisted since are mostly missing.
  * Daily bars: no intraday timing; USAspending's lag vs. real disclosure is a guess
    (--lag-days) -- run it at 0, 1 and 3 and see if the conclusion moves.
  * Events cluster (defense budget days, sector moves) so they are not independent;
    bootstrap CIs here treat them as if they were, i.e. are somewhat too narrow.
  * Many cells are tested: expect a few "significant" ones by chance. A cell with
    n < MIN_N is flagged underpowered, and nothing here auto-tunes a constant.

Network is needed only by the default loaders (USAspending, SEC, Yahoo). Everything
else takes injected callables, so the whole pipeline is testable offline.

CLI:  python3 contract_backtest.py --csv awards.csv --start 2018-01-01 --end 2025-12-31
      python3 contract_backtest.py --usaspending --start 2022-01-01 --end 2025-12-31
"""
import argparse
import csv
import json
import logging
import math
import os
import pickle
import re
import sys
import urllib.request
from datetime import date, timedelta

import numpy as np
import pandas as pd

import contract_crosscheck as cc

logger = logging.getLogger("contract_backtest")

HORIZONS = (1, 3, 5, 10, 20, 40, 60)       # trading days after entry
CHECKPOINT_HORIZONS = (5, 10, 20)
PROFILE_DAYS = 60
PRE_DAYS = 10                                # pre-entry window, leakage check
MIN_N = 15                                   # below this a cell is "underpowered"
MAG_TOL = 0.15                               # |implied - model| multiplier gap still "about as modelled"
DECAY_TAU_DAYS = 60.0                        # contracts._age_decay's time constant
BUCKETS = ((0.0, 0.01, "<1%"), (0.01, 0.05, "1-5%"), (0.05, 0.15, "5-15%"), (0.15, math.inf, ">15%"))
CACHE_DIR = os.path.expanduser("~/.meridian_cache/backtest")


class DataUnavailable(Exception):
    """A required data source could not be reached (usually the network policy)."""


# ======================================================================= awards ===

def _num(x):
    try:
        return float(str(x).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


def awards_from_csv(path):
    """Load awards from a CSV. Header names are matched case-insensitively:
    date|action_date, recipient|recipient_name|contractor, amount|value|value_usd|
    transaction_amount, and optional ticker, agency. -> (awards, n_skipped)."""
    alias = {"date": ("date", "action_date"), "recipient": ("recipient", "recipient_name", "contractor"),
             "amount": ("amount", "value", "value_usd", "transaction_amount"),
             "ticker": ("ticker",), "agency": ("agency",)}
    awards, bad = [], 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        cols = {(c or "").strip().lower(): c for c in (rd.fieldnames or [])}
        pick = {k: next((cols[a] for a in v if a in cols), None) for k, v in alias.items()}
        if not (pick["date"] and pick["amount"] and (pick["recipient"] or pick["ticker"])):
            raise ValueError("CSV needs a date, an amount and a recipient (or ticker) column; "
                             f"found {list(cols)}")
        for row in rd:
            d, amt = cc._to_date(row.get(pick["date"])), _num(row.get(pick["amount"]))
            if d is None or amt is None:
                bad += 1
                continue
            awards.append({"date": d, "amount": amt,
                           "recipient": (row.get(pick["recipient"]) or "").strip() if pick["recipient"] else "",
                           "ticker": ((row.get(pick["ticker"]) or "").strip().upper() or None) if pick["ticker"] else None,
                           "agency": (row.get(pick["agency"]) or "").strip() if pick["agency"] else ""})
    return awards, bad


USASPENDING_URL = "https://api.usaspending.gov/api/v2/search/spending_by_transaction/"
USASPENDING_FIELDS = ["Award ID", "Recipient Name", "Action Date", "Transaction Amount",
                      "Awarding Agency", "Transaction Description"]


def awards_from_usaspending_rows(rows):
    """USAspending spending_by_transaction `results` -> (awards, n_skipped).
    Obligations only: de-obligations (<= 0) are skipped."""
    awards, bad = [], 0
    for r in rows or []:
        try:
            d, amt = cc._to_date(r.get("Action Date")), _num(r.get("Transaction Amount"))
            if d is None or amt is None or amt <= 0:
                bad += 1
                continue
            awards.append({"date": d, "amount": amt, "recipient": (r.get("Recipient Name") or "").strip(),
                           "ticker": None, "agency": (r.get("Awarding Agency") or "").strip()})
        except Exception:
            bad += 1
    return awards, bad


def _http_post_json(url, payload, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Accept": "application/json",
                                          "User-Agent": "Meridian Research meridian-app contact@example.com"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        raise DataUnavailable(f"USAspending request failed ({e}). If this is a 403 on CONNECT, the "
                              "environment's network policy blocks api.usaspending.gov.") from e


def fetch_usaspending(start, end, min_amount=10e6, agency="Department of Defense",
                      max_pages=30, post=None):
    """Large contract obligations in [start, end], one month at a time, biggest first
    (so paging stops once amounts fall below `min_amount`). `post(payload)->dict` is
    injectable for tests. NOTE: the request/response shape follows USAspending's
    published API but has not been exercised live from this repo -- on the first real
    run check the first few parsed awards (the CLI prints them)."""
    post = post or (lambda p: _http_post_json(USASPENDING_URL, p))
    start, end = cc._to_date(start), cc._to_date(end)
    out, bad_total = [], 0
    cur = date(start.year, start.month, 1)
    while cur <= end:
        nxt = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        lo, hi = max(cur, start), min(nxt - timedelta(days=1), end)
        for page in range(1, max_pages + 1):
            payload = {"filters": {"award_type_codes": ["A", "B", "C", "D"],
                                   "time_period": [{"start_date": lo.isoformat(), "end_date": hi.isoformat()}],
                                   "agencies": [{"type": "awarding", "tier": "toptier", "name": agency}]},
                       "fields": USASPENDING_FIELDS, "sort": "Transaction Amount", "order": "desc",
                       "page": page, "limit": 100}
            resp = post(payload) or {}
            rows = resp.get("results") or []
            got, bad = awards_from_usaspending_rows(rows)
            bad_total += bad
            out += [a for a in got if a["amount"] >= min_amount]
            small = rows and (_num(rows[-1].get("Transaction Amount")) or 0) < min_amount
            if not rows or small or not (resp.get("page_metadata") or {}).get("hasNext"):
                break
        cur = nxt
    return out, bad_total


# ======================================================================= events ===

def build_events(awards, ticker_fn=None, min_value=10e6, cooldown_days=30):
    """Awards -> events [{ticker, date, value_usd, n, recipients}].

    Positive obligations are summed per (ticker, action date); days totalling less
    than `min_value` are dropped; later events for a ticker within `cooldown_days` of
    its previous KEPT event are skipped (overlapping windows would double-count).
    `ticker_fn(recipient)` defaults to dod_scraper's name->ticker map; an award that
    already carries a ticker uses it."""
    if ticker_fn is None:
        import dod_scraper
        ticker_fn = dod_scraper._contractor_to_ticker
    days = {}
    for a in awards or []:
        if not a.get("amount") or a["amount"] <= 0:
            continue
        tk = (a.get("ticker") or ticker_fn(a.get("recipient"))) or None
        if not tk:
            continue
        e = days.setdefault((tk.upper(), a["date"]), {"ticker": tk.upper(), "date": a["date"],
                                                      "value_usd": 0.0, "n": 0, "recipients": set()})
        e["value_usd"] += a["amount"]; e["n"] += 1; e["recipients"].add(a.get("recipient") or "")
    events, last = [], {}
    for e in sorted(days.values(), key=lambda e: (e["date"], e["ticker"])):
        if e["value_usd"] < min_value:
            continue
        prev = last.get(e["ticker"])
        if prev and (e["date"] - prev).days < cooldown_days:
            continue
        last[e["ticker"]] = e["date"]
        e["recipients"] = sorted(e["recipients"])
        events.append(e)
    return events


def bucket_for(ratio):
    for lo, hi, label in BUCKETS:
        if lo <= ratio < hi:
            return label
    return BUCKETS[-1][2]


# ======================================================================= prices ===

def unadjust(close, splits):
    """Split-ADJUSTED closes -> the prices actually quoted at the time (what shares
    outstanding as filed then must be multiplied by). `splits` is a Series of split
    ratios by date (4.0 = 4-for-1); a close is multiplied by every ratio dated after it."""
    if splits is None or len(splits) == 0:
        return close.copy()
    splits = splits[splits > 0]
    idx = pd.DatetimeIndex(close.index).tz_localize(None)
    sdates = pd.DatetimeIndex(splits.index).tz_localize(None)
    factor = np.ones(len(idx))
    for dt, ratio in zip(sdates, splits.values):
        factor[idx < dt] *= float(ratio)
    return pd.Series(close.values * factor, index=close.index)


def yf_price_loader(start, end, cache_dir=CACHE_DIR):
    """-> price_fn(ticker) -> DataFrame[adj, unadj] (daily) or None. Disk-cached.
    `adj` (dividend+split adjusted) is for returns; `unadj` for market cap."""
    os.makedirs(cache_dir, exist_ok=True)

    def load(ticker):
        path = os.path.join(cache_dir, f"{ticker}_{start}_{end}.pkl")
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
        try:
            import yfinance as yf
            df = yf.download(ticker, start=str(start), end=str(end), auto_adjust=False,
                             progress=False, actions=False)
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.DatetimeIndex(df.index).tz_localize(None)
            try:
                splits = yf.Ticker(ticker).splits
            except Exception:
                splits = None
            out = pd.DataFrame({"adj": df["Adj Close"], "unadj": unadjust(df["Close"], splits)}).dropna()
            with open(path, "wb") as f:
                pickle.dump(out, f)
            return out
        except Exception:
            return None
    return load


# ============================================================== event study core ===

def _ar(a, b, i, j):
    """Market-adjusted return from index i to j: stock return minus benchmark return."""
    return float((a[j] / a[i] - 1.0) - (b[j] / b[i] - 1.0))


def study_event(ev, px, bench, lag_days=1, horizons=HORIZONS):
    """Abnormal-return measurements for one event, or None if the price history can't
    support it. `px`: DataFrame[adj(, unadj)] ; `bench`: Series of benchmark closes.

    Entry index i0 = first trading day on/after (action date + lag_days); all forward
    returns start at that day's CLOSE, so nothing on or before i0 is ever counted as
    earned. Pure function of prices up to the horizon: edits to bars before i0 can only
    change `pre_ar`, never a forward return (tested)."""
    adj = px["adj"]
    common = adj.index.intersection(bench.index)
    if len(common) < PRE_DAYS + 2:
        return None
    a, b = adj.loc[common].values.astype(float), bench.loc[common].values.astype(float)
    info = pd.Timestamp(ev["date"]) + pd.Timedelta(days=lag_days)
    i0 = int(common.searchsorted(info))
    n = len(common)
    if i0 >= n or i0 < PRE_DAYS or (common[i0] - info).days > 5 or i0 + min(horizons) >= n:
        return None
    ar = {h: (_ar(a, b, i0, i0 + h) if i0 + h < n else float("nan")) for h in horizons}
    profile = [(_ar(a, b, i0, i0 + k) if i0 + k < n else float("nan")) for k in range(PROFILE_DAYS + 1)]
    # checkpoint: first trading day strictly AFTER the cross-check window has closed
    icp = max(int(common.searchsorted(pd.Timestamp(ev["date"]) + pd.Timedelta(days=cc.DILUTION_WINDOW_DAYS),
                                      side="right")), i0)
    ar_cp = {h: (_ar(a, b, icp, icp + h) if icp + h < n else float("nan")) for h in CHECKPOINT_HORIZONS}
    price0 = float(px["unadj"].loc[common[i0]]) if "unadj" in px else float(adj.loc[common[i0]])
    return {"entry_date": common[i0].date().isoformat(), "checkpoint_date": common[min(icp, n - 1)].date().isoformat(),
            "ar": ar, "profile": profile, "ar_cp": ar_cp, "pre_ar": _ar(a, b, i0 - PRE_DAYS, i0),
            "day0_ar": _ar(a, b, i0 - 1, i0), "price0": price0}


def classify(ev, ratio, filings_fn, insider_fn):
    """Run the PRODUCTION cross-check on a historical event. -> group label in
    {clean, dilution, insider_sell, insider_buy} (negative flags win, as in live), or
    "unknown" when no EDGAR source was supplied."""
    if filings_fn is None and insider_fn is None:
        return "unknown", None
    award = ev["date"]
    sig = {"signal": min(max(ratio, 1e-6), 1.0), "confidence": 1.0, "award_date": award.isoformat()}
    w = timedelta(days=cc.DILUTION_WINDOW_DAYS)
    dil = filings_fn(ev["ticker"], award, award + w) if filings_fn else []
    txs = (insider_fn(ev["ticker"], award - timedelta(days=cc.INSIDER_PRE_DAYS), award + w)
           if insider_fn else [])
    r = cc.evaluate(sig, dil or [], txs or [], today=award)
    if r is None:
        return "unknown", None
    if r["dilution"]:
        return "dilution", r
    if r["insider_sell_usd"] >= cc.MIN_INSIDER_USD:
        return "insider_sell", r
    if r["insider_buy_usd"] >= cc.MIN_INSIDER_USD:
        return "insider_buy", r
    return "clean", r


# ================================================================== statistics ===

def boot_summary(values, n_boot=2000, seed=0):
    """{n, mean, median, hit, ci: (lo, hi)} -- 95% percentile bootstrap CI of the mean.
    NaNs/None are dropped. n == 0 -> {"n": 0}."""
    v = np.array([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    n = len(v)
    if n == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, n, (n_boot, n))].mean(axis=1)
    return {"n": n, "mean": float(v.mean()), "median": float(np.median(v)), "hit": float((v > 0).mean()),
            "ci": (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))}


def boot_diff(a, b, n_boot=2000, seed=0):
    """Bootstrap CI of mean(a) - mean(b) (independent resamples). None if either is empty."""
    a = np.array([x for x in a if x is not None and np.isfinite(x)], dtype=float)
    b = np.array([x for x in b if x is not None and np.isfinite(x)], dtype=float)
    if len(a) == 0 or len(b) == 0:
        return None
    rng = np.random.default_rng(seed)
    d = (a[rng.integers(0, len(a), (n_boot, len(a)))].mean(axis=1)
         - b[rng.integers(0, len(b), (n_boot, len(b)))].mean(axis=1))
    return {"diff": float(a.mean() - b.mean()), "ci": (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))}


def _by_h(rows, key, horizons, **kw):
    return {h: boot_summary([r[key].get(h) for r in rows], **kw) for h in horizons}


MODEL_MULT = {"dilution": cc.DILUTION_MULT, "insider_sell": cc.INSIDER_SELL_MULT, "insider_buy": cc.INSIDER_BUY_MULT}


def multiplier_check(rows, seed=0, n_boot=2000):
    """Does each cross-check flag separate outcomes the way the model assumes?

    Negative flags (public only by the checkpoint) are compared from the CHECKPOINT;
    insider buying (public by the award date) from ENTRY -- each against clean events on
    the same basis. verdict is about DIRECTION and detectability only:
      supported     -- CI of (flagged - clean) excludes 0 on the side the model assumes
      contradicted  -- CI excludes 0 on the other side
      inconclusive  -- CI contains 0, or fewer than MIN_N flagged events
    `implied_ratio` = mean(flagged)/mean(clean) is given only when the clean mean's own CI
    excludes zero and is positive; it is the empirical analogue of the multiplier but is
    noisy -- read it next to the CI, never as a tuned constant. For a "supported" flag,
    `magnitude` says whether the point estimate is stronger/weaker than the multiplier
    (a ratio <= 0 means the flag erased or reversed the edge, which a shrink-only
    multiplier like x0.5 can never express)."""
    out = {}
    for g, mult in MODEL_MULT.items():
        key, hs = ("ar", (10, 20)) if g == "insider_buy" else ("ar_cp", (10, 20))
        for h in hs:
            gv = [r[key].get(h) for r in rows if r["group"] == g]
            cv = [r[key].get(h) for r in rows if r["group"] == "clean"]
            gs, cs = boot_summary(gv, n_boot, seed), boot_summary(cv, n_boot, seed)
            d = boot_diff(gv, cv, n_boot, seed)
            cell = {"basis": "entry" if g == "insider_buy" else "checkpoint", "model_mult": mult,
                    "flagged": gs, "clean": cs, "diff": d, "implied_ratio": None, "magnitude": None}
            if gs.get("n", 0) < MIN_N or cs.get("n", 0) < MIN_N or d is None:
                cell["verdict"] = "inconclusive (underpowered)"
            else:
                want_neg = mult < 1.0
                lo, hi = d["ci"]
                if (want_neg and hi < 0) or (not want_neg and lo > 0):
                    cell["verdict"] = "supported"
                elif (want_neg and lo > 0) or (not want_neg and hi < 0):
                    cell["verdict"] = "contradicted"
                else:
                    cell["verdict"] = "inconclusive (CI includes 0)"
                if cs["ci"][0] > 0:
                    cell["implied_ratio"] = gs["mean"] / cs["mean"]
                    if cell["verdict"] == "supported":
                        # point-estimate comparison only (no CI on the ratio): is the flag's
                        # effect bigger or smaller than the multiplier assumes?
                        gap = (mult - cell["implied_ratio"]) if want_neg else (cell["implied_ratio"] - mult)
                        cell["magnitude"] = ("stronger than modelled" if gap > MAG_TOL
                                             else "weaker than modelled" if gap < -MAG_TOL else "about as modelled")
            out[(g, h)] = cell
    return out


def decay_check(rows, seed=0, n_boot=2000):
    """Mean abnormal-return path after entry vs. the model's decay. `realized[k]` is the
    share of the day-60 mean AR already earned by day k; `model[k]` = 1 - exp(-k/tau).
    Only meaningful when the day-60 mean AR is itself distinguishable from zero."""
    final = boot_summary([r["profile"][PROFILE_DAYS] for r in rows], n_boot, seed)
    path = {k: boot_summary([r["profile"][k] for r in rows], n_boot, seed) for k in (1, 5, 10, 20, 40, 60)}
    out = {"path": path, "final": final, "usable": False, "realized": {}, "model": {}}
    if final.get("n", 0) >= MIN_N and (final["ci"][0] > 0 or final["ci"][1] < 0):
        out["usable"] = True
        for k in (5, 10, 20, 40):
            if path[k].get("n"):
                out["realized"][k] = path[k]["mean"] / final["mean"]
                out["model"][k] = 1.0 - math.exp(-k / DECAY_TAU_DAYS)
    return out


# =============================================================== orchestration ===

def run_backtest(events, price_fn, shares_fn, filings_fn=None, insider_fn=None,
                 lag_days=1, bench="SPY", n_boot=2000, seed=0):
    """Run the full study. All data access is injected:
      price_fn(ticker) -> DataFrame[adj(, unadj)] | None   (also called for `bench`)
      shares_fn(ticker, asof_date) -> shares outstanding as filed by then | None
      filings_fn(ticker, start, end) -> [{form, date, ...}] offering filings   (optional)
      insider_fn(ticker, start, end) -> [{code, usd, owner, accepted, ...}]     (optional)
    Events that can't be fully measured are dropped and counted in `dropped`."""
    dropped = {"no_prices": 0, "no_window": 0, "no_shares": 0}
    bpx = price_fn(bench)
    if bpx is None or len(bpx) == 0:
        raise DataUnavailable(f"no benchmark prices for {bench}")
    bser = bpx["adj"]
    rows = []
    for ev in events:
        px = price_fn(ev["ticker"])
        if px is None or len(px) == 0:
            dropped["no_prices"] += 1
            continue
        st = study_event(ev, px, bser, lag_days)
        if st is None:
            dropped["no_window"] += 1
            continue
        shares = shares_fn(ev["ticker"], pd.Timestamp(st["entry_date"]).date())
        if not shares or shares <= 0 or st["price0"] <= 0:
            dropped["no_shares"] += 1
            continue
        mcap = shares * st["price0"]
        ratio = ev["value_usd"] / mcap
        group, res = classify(ev, ratio, filings_fn, insider_fn)
        rows.append({**st, "ticker": ev["ticker"], "award_date": ev["date"].isoformat(),
                     "value_usd": ev["value_usd"], "mcap": mcap, "ratio": ratio,
                     "bucket": bucket_for(ratio), "group": group,
                     "flags": (res or {}).get("flags", [])})
    kw = {"n_boot": n_boot, "seed": seed}
    groups = sorted({r["group"] for r in rows} - {"unknown"})
    return {
        "params": {"lag_days": lag_days, "bench": bench, "seed": seed, "n_boot": n_boot},
        "n_events_in": len(events), "n_studied": len(rows), "dropped": dropped, "rows": rows,
        "overall": _by_h(rows, "ar", HORIZONS, **kw),
        "day0": boot_summary([r["day0_ar"] for r in rows], **kw),
        "pre_drift": boot_summary([r["pre_ar"] for r in rows], **kw),
        "by_bucket": {lab: _by_h([r for r in rows if r["bucket"] == lab], "ar", HORIZONS, **kw)
                      for _, _, lab in BUCKETS},
        "by_group_entry": {g: _by_h([r for r in rows if r["group"] == g], "ar", (5, 10, 20), **kw) for g in groups},
        "by_group_checkpoint": {g: _by_h([r for r in rows if r["group"] == g], "ar_cp", CHECKPOINT_HORIZONS, **kw)
                                for g in groups},
        "multiplier_check": multiplier_check(rows, seed, n_boot) if groups else {},
        "decay": decay_check(rows, seed, n_boot) if rows else None,
    }


# ===================================================================== report ===

def _pct(x):
    return "   n/a" if x is None or not np.isfinite(x) else f"{x*100:+6.2f}%"


def _cell(s):
    if not s or not s.get("n"):
        return "| n=0 | | | |"
    flag = " ⚠" if s["n"] < MIN_N else ""
    return (f"| {s['n']}{flag} | {_pct(s['mean'])} | [{_pct(s['ci'][0]).strip()}, {_pct(s['ci'][1]).strip()}] "
            f"| {s['hit']*100:.0f}% |")


def _inline(s):
    if not s or not s.get("n"):
        return "n=0"
    return (f"n={s['n']}{' ⚠' if s['n'] < MIN_N else ''}, mean {_pct(s['mean']).strip()}, "
            f"95% CI [{_pct(s['ci'][0]).strip()}, {_pct(s['ci'][1]).strip()}], hit {s['hit']*100:.0f}%")


def format_report(res):
    L = ["# Contract-award event study", "",
         f"Events in: {res['n_events_in']} · studied: {res['n_studied']} · dropped: {res['dropped']} · "
         f"lag {res['params']['lag_days']}d · benchmark {res['params']['bench']} · seed {res['params']['seed']}",
         "", "Abnormal return (AR) = stock − benchmark, from the close of the entry day. "
         "CI = 95% bootstrap of the mean. ⚠ = fewer than "
         f"{MIN_N} events (underpowered).", ""]
    if not res["n_studied"]:
        return "\n".join(L + ["**No events could be studied.** Check data access (SEC / USAspending / "
                              "Yahoo may be blocked by the network policy) and the dropped counts above."])

    def table(title, by_h, hs):
        if not any((by_h.get(h) or {}).get("n") for h in hs):
            L.extend([f"### {title}", "", "_no events_", ""])
            return
        L.extend([f"### {title}", "", "| horizon | n | mean AR | 95% CI | hit |", "|---|---|---|---|---|"])
        for h in hs:
            L.append(f"| {h}d {_cell(by_h.get(h))}")
        L.append("")

    L.append("## 1 · Is there drift?")
    table("All events (from entry)", res["overall"], HORIZONS)
    L.append(f"- Announcement-day move (not tradeable): {_inline(res['day0'])}\n"
             f"- Pre-entry drift, {PRE_DAYS}d (leakage/anticipation check): {_inline(res['pre_drift'])}\n")
    L.append("## 2 · Does size matter? (award ÷ market cap)")
    for _, _, lab in BUCKETS:
        table(f"Bucket {lab}", res["by_bucket"][lab], (5, 20, 60))
    d = res["decay"]
    L.append("## 3 · How long does it last?")
    L.append("")
    if d and d["usable"]:
        L.extend(["| day | share of day-60 AR realised | model 1−exp(−k/60) |", "|---|---|---|"])
        for k in sorted(d["realized"]):
            L.append(f"| {k} | {d['realized'][k]*100:.0f}% | {d['model'][k]*100:.0f}% |")
        L.append("")
    else:
        L.append("Day-60 mean AR is not distinguishable from zero (or too few events), so a decay "
                 "comparison would be meaningless. Mean AR path:")
        for k, st in (d or {}).get("path", {}).items():
            L.append(f"- day {k}: {_inline(st)}")
        L.append("")
    L.append("## 4 · Do the EDGAR cross-check flags work?")
    if res["by_group_checkpoint"]:
        L.extend(["_Tradeable comparison: returns start at the checkpoint, when every flag is public._", ""])
        for g, byh in res["by_group_checkpoint"].items():
            table(f"group `{g}` (from checkpoint)", byh, CHECKPOINT_HORIZONS)
        L.append("_From-entry numbers by group are descriptive only: flags that appear after the award "
                 "include the offering's own price reaction, which cannot be traded._\n")
        for g, byh in res["by_group_entry"].items():
            table(f"group `{g}` (from entry, descriptive)", byh, (5, 10, 20))
        L.extend(["### Multiplier check", "",
                  "| flag | basis | h | model × | flagged n / mean | clean n / mean | diff [95% CI] | implied × | verdict |",
                  "|---|---|---|---|---|---|---|---|---|"])
        for (g, h), c in res["multiplier_check"].items():
            fl, cl, df = c["flagged"], c["clean"], c["diff"]
            L.append(f"| {g} | {c['basis']} | {h}d | {c['model_mult']} | "
                     f"{fl.get('n', 0)} / {_pct(fl.get('mean')).strip()} | {cl.get('n', 0)} / {_pct(cl.get('mean')).strip()} | "
                     + (f"{_pct(df['diff']).strip()} [{_pct(df['ci'][0]).strip()}, {_pct(df['ci'][1]).strip()}]" if df else "n/a")
                     + f" | {('%.2f' % c['implied_ratio']) if c['implied_ratio'] is not None else '—'} | {c['verdict']}"
                     + (f" — {c['magnitude']}" if c.get("magnitude") else "") + " |")
        L.append("")
    else:
        L.append("No EDGAR flag data was supplied, so the cross-check was not tested.\n")
    L.extend(["## Limits (read before trusting any number)", "",
              "- **Survivorship**: tickers come from today's contractor map; acquired/delisted names are mostly absent.",
              "- **Disclosure timing**: USAspending dates signing, not public release. Re-run with `--lag-days 0/1/3`; "
              "a conclusion that flips with the lag is not a conclusion.",
              "- **Non-independence**: events cluster by date/sector, so the bootstrap CIs are somewhat too narrow.",
              "- **Multiple comparisons**: dozens of cells are shown; a few will look significant by chance.",
              "- **Verdicts speak to direction only.** `implied ×` is noisy; nothing here tunes a constant for you.",
              ""])
    return "\n".join(L)


def rows_to_csv(rows, path):
    keys = ["ticker", "award_date", "entry_date", "checkpoint_date", "value_usd", "mcap", "ratio", "bucket",
            "group", "day0_ar", "pre_ar"] + [f"ar_{h}d" for h in HORIZONS] + [f"ar_cp_{h}d" for h in CHECKPOINT_HORIZONS]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys + ["flags"])
        for r in rows:
            w.writerow([r.get(k) for k in keys[:11]] + [r["ar"].get(h) for h in HORIZONS]
                       + [r["ar_cp"].get(h) for h in CHECKPOINT_HORIZONS] + ["; ".join(r.get("flags", []))])


# ========================================================================= CLI ===

def main(argv=None):
    ap = argparse.ArgumentParser(description="Historical backtest of Meridian's gov-contract signals.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="awards CSV (date, recipient/ticker, amount)")
    src.add_argument("--usaspending", action="store_true", help="download DoD obligations from USAspending.gov")
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--min-value", type=float, default=10e6)
    ap.add_argument("--lag-days", type=int, default=1)
    ap.add_argument("--bench", default="SPY")
    ap.add_argument("--no-edgar-flags", action="store_true", help="skip the cross-check (faster; no section 4)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-md", default="contract_backtest_report.md")
    ap.add_argument("--out-csv", default="contract_backtest_events.csv")
    a = ap.parse_args(argv)
    try:
        if a.csv:
            awards, bad = awards_from_csv(a.csv)
        else:
            awards, bad = fetch_usaspending(a.start, a.end, a.min_value)
    except (DataUnavailable, ValueError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    awards = [x for x in awards if cc._to_date(a.start) <= x["date"] <= cc._to_date(a.end)]
    events = build_events(awards, min_value=a.min_value)
    print(f"{len(awards)} awards ({bad} unparseable) -> {len(events)} events; first awards: "
          f"{[(x['date'].isoformat(), x['recipient'][:30], x['amount']) for x in awards[:3]]}")
    if not events:
        print("No events (no awards mapped to a ticker above the minimum).", file=sys.stderr)
        return 1
    import edgar
    pad = timedelta(days=150)
    price_fn = yf_price_loader((cc._to_date(a.start) - pad).isoformat(), (cc._to_date(a.end) + pad).isoformat())
    filings_fn = insider_fn = None
    if not a.no_edgar_flags:
        filings_fn = lambda t, s, e: edgar.filings_between(t, edgar.DILUTIVE_FORMS, s, e)
        insider_fn = edgar.form4_transactions_between
    try:
        res = run_backtest(events, price_fn, edgar.shares_outstanding_asof, filings_fn, insider_fn,
                           a.lag_days, a.bench, seed=a.seed)
    except DataUnavailable as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    md = format_report(res)
    with open(a.out_md, "w") as f:
        f.write(md)
    rows_to_csv(res["rows"], a.out_csv)
    print(md)
    print(f"\nWrote {a.out_md} and {a.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

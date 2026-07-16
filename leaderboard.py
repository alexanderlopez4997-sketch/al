#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LEADERBOARD — cross-sectional factor ranking backend for Meridian.

Takes a universe of tickers (default: the ~120-name liquid US list), computes
the five-factor snapshot for every name, then scores them CROSS-SECTIONALLY:
each factor is percentile-ranked across the whole universe on the same day and
the ranks are blended with the standard factor weights. That answers "which
names look best RELATIVE TO THEIR PEERS right now", which is the right question
for a leaderboard — a +0.4 Trend factor means much more when the other 99
names are flat.

Honesty gate: a relative ranking always has a #1, even in a crashing market.
So "top BUY candidates" must ALSO pass the absolute BUY threshold (score ≥ 18
on the ticker's own history). If fewer than N names qualify, the leaderboard
says so rather than padding the list.

Pure backend — no Tk imports. Use as a library (the GUI Leaderboard tab calls
rank_universe with prefetched data) or from the shell:

    python3 leaderboard.py --demo                 # offline smoke test
    python3 leaderboard.py                        # liquid-US universe, live
    python3 leaderboard.py NVDA AMD MSFT ...      # explicit list
    python3 leaderboard.py --json                 # machine-readable output
"""
import argparse
import datetime as _dt
import json
import math
import sys

import numpy as np
import pandas as pd

import quant_engine as qe

TOP_N = 5
MIN_XSEC = 5        # below this many names, percentile ranks are meaningless


# ------------------------------------------------------------------ fetch ---
def fetch_batch(tickers, period="6mo", interval="1d", chunk=50):
    """Batched OHLCV download → {ticker: DataFrame}. Missing names skipped."""
    import yfinance as yf
    out = {}
    cols = ["Open", "High", "Low", "Close", "Volume"]
    for i in range(0, len(tickers), chunk):
        grp = tickers[i:i + chunk]
        try:
            df = yf.download(grp, period=period, interval=interval, progress=False,
                             auto_adjust=True, group_by="ticker", threads=True)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        if len(grp) == 1:
            sub = df.copy()
            if isinstance(sub.columns, pd.MultiIndex):
                sub.columns = sub.columns.get_level_values(-1)
            try:
                s = sub[cols].dropna()
                if len(s):
                    out[grp[0]] = s
            except Exception:
                pass
        else:
            lvl0 = df.columns.get_level_values(0)
            for t in grp:
                if t not in lvl0:
                    continue
                try:
                    s = df[t][cols].dropna()
                    if len(s):
                        out[t] = s
                except Exception:
                    continue
    return out


# ------------------------------------------------------------------- core ---
def factor_snapshot(ticker, df, min_bars=60):
    """Latest factor values + absolute score for one ticker, or raises."""
    if df is None or len(df) < min_bars:
        raise ValueError(f"{len(df) if df is not None else 0} bars — need {min_bars}")
    d = qe.enrich(df)
    F = qe.factor_matrix(d)
    comp = qe.composite(F)
    last = float(d["Close"].iloc[-1])
    atr_pct = float(d["atr"].iloc[-1] / last * 100)
    return {"ticker": ticker,
            "factors": {k: float(F[k].iloc[-1]) for k in qe.FACTORS},
            "abs_score": float(comp.iloc[-1]),
            "last": last,
            "chg": float((last / d["Close"].iloc[-2] - 1) * 100),
            "atr_pct": atr_pct,
            "dollar_vol": float((d["Close"] * d["Volume"]).tail(20).mean())}


def rank_universe(data, top_n=TOP_N, weights=None, min_bars=60):
    """The cross-sectional pipeline. `data` is {ticker: OHLCV DataFrame}.

    Returns {"ranked": [...], "top_buys": [...], "cross_sectional": bool,
             "universe": int, "errors": [(ticker, reason)], "generated": iso}.
    Each ranked row carries both the cross-sectional score (peer-relative,
    what the leaderboard sorts by) and the absolute score (own-history,
    what the BUY gate uses)."""
    rows, errors = [], []
    for t, df in data.items():
        try:
            rows.append(factor_snapshot(t, df, min_bars))
        except Exception as e:
            errors.append((t, str(e)))
    out = {"ranked": [], "top_buys": [], "cross_sectional": False,
           "universe": len(rows), "errors": errors,
           "generated": _dt.datetime.now().isoformat(timespec="seconds")}
    if not rows:
        return out
    w = weights or qe.BASE_WEIGHTS
    n = len(rows)
    if n >= MIN_XSEC:
        # percentile-rank each factor ACROSS the universe → [-1, +1]
        fm = pd.DataFrame([r["factors"] for r in rows], index=range(n))
        ranks = fm.rank(method="average")                 # 1..n, ties averaged
        xf = (2.0 * (ranks - 1) / (n - 1) - 1.0) if n > 1 else ranks * 0.0
        wv = np.array([w[k] for k in qe.FACTORS])
        xscores = 100.0 * (xf[qe.FACTORS].values @ wv)
        out["cross_sectional"] = True
        for i, r in enumerate(rows):
            r["xsec_factors"] = {k: float(xf[k].iloc[i]) for k in qe.FACTORS}
            r["xsec_score"] = float(xscores[i])
    else:
        # too few names for meaningful percentiles — fall back to absolute
        for r in rows:
            r["xsec_factors"] = dict(r["factors"])
            r["xsec_score"] = r["abs_score"]
    for r in rows:
        r["verdict"] = qe.verdict(r["abs_score"], r["atr_pct"])
    ranked = sorted(rows, key=lambda r: -r["xsec_score"])
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    out["ranked"] = ranked
    # BUY gate: peer-relative rank qualifies you; your OWN chart must agree.
    out["top_buys"] = [r for r in ranked if r["verdict"]["tone"] == "good"][:top_n]
    return out


def build_leaderboard(tickers=None, period="6mo", interval="1d",
                      top_n=TOP_N, demo=False):
    """Fetch + rank in one call (CLI / standalone use)."""
    tickers = list(dict.fromkeys(t.upper() for t in (tickers or qe.UNIVERSE_LIQUID)))
    if demo:
        data = {t: qe.demo_data(t) for t in tickers}
    else:
        data = fetch_batch(tickers, period, interval)
        for t in tickers:
            if t not in data:
                data[t] = None                     # recorded as an error row
    return rank_universe(data, top_n=top_n)


# ----------------------------------------------------------------- output ---
def _fmt_vol(x):
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if x >= div:
            return f"${x/div:.1f}{suf}"
    return f"${x:.0f}"


def print_leaderboard(lb, top_n=TOP_N):
    line = "─" * 74
    print(line)
    print(f"LEADERBOARD — {lb['universe']} names ranked cross-sectionally "
          f"({lb['generated']})")
    print(line)
    buys = lb["top_buys"]
    print(f"TOP {top_n} BUY CANDIDATES"
          + ("" if len(buys) >= top_n else
             f"  — only {len(buys)} name(s) pass the absolute BUY gate today"))
    if not buys:
        print("  none — nothing in this universe has a BUY-grade chart right now.")
    for r in buys:
        v = r["verdict"]
        risky = "  [RISKY]" if v["risky"] else ""
        print(f"  #{r['rank']:<3} {r['ticker']:<7} xsec {r['xsec_score']:>+6.1f} · "
              f"own-chart {r['abs_score']:>+5.0f} ({v['label']}){risky} · "
              f"last {r['last']:.2f} ({r['chg']:+.2f}%) · vol {_fmt_vol(r['dollar_vol'])}")
    print(line)
    print(f"{'#':<4}{'TICKER':<8}{'XSEC':>7}{'OWN':>6}  {'VERDICT':<20}"
          f"{'CHG%':>8}{'$VOL':>9}")
    for r in lb["ranked"][:30]:
        v = r["verdict"]
        print(f"{r['rank']:<4}{r['ticker']:<8}{r['xsec_score']:>+7.1f}"
              f"{r['abs_score']:>+6.0f}  {v['label']:<20}"
              f"{r['chg']:>+8.2f}{_fmt_vol(r['dollar_vol']):>9}")
    if lb["errors"]:
        skipped = " ".join(t for t, _ in lb["errors"][:12])
        more = f" +{len(lb['errors'])-12}" if len(lb["errors"]) > 12 else ""
        print(f"\nskipped ({len(lb['errors'])}): {skipped}{more}")
    print(line)
    print("Cross-sectional rank = strength vs peers TODAY; the BUY gate still "
          "requires the name's own chart to signal BUY. Not financial advice.")


def main():
    ap = argparse.ArgumentParser(description="Cross-sectional factor leaderboard.")
    ap.add_argument("tickers", nargs="*", help="universe (default: liquid US ~120)")
    ap.add_argument("--period", default="6mo")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--demo", action="store_true", help="synthetic data, offline")
    ap.add_argument("--json", action="store_true", help="print JSON for UI feeds")
    args = ap.parse_args()
    lb = build_leaderboard(args.tickers or None, args.period, args.interval,
                           args.top, args.demo)
    if args.json:
        print(json.dumps(lb, indent=2))
    else:
        print_leaderboard(lb, args.top)


if __name__ == "__main__":
    main()

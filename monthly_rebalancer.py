#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monthly Rebalancer — runs the quant_engine signal stack over the watchlist and
rebalances the portfolio: exits any open position whose verdict has turned
AVOID, then fills open slots with the highest-scoring qualifying BUY signals,
sized by ATR risk (quant_engine.position_size). A circuit breaker blocks new
entries — exits still run — once drawdown from the account's running peak
equity breaches CIRCUIT_BREAKER_DD_PCT.

USAGE
  python3 monthly_rebalancer.py                 # live run (yfinance)
  python3 monthly_rebalancer.py --demo           # offline synthetic data
  python3 monthly_rebalancer.py --dry-run        # show the plan, don't save
  python3 monthly_rebalancer.py --account 50000  # set starting account (first run only)

Intended to run on the 1st of each month @ 9 AM — wire to cron or a scheduled
Routine, e.g. cron expression "0 9 1 * *".

NOT FINANCIAL ADVICE. Rules-based technical signals on historical data.
"""
import argparse
import json
import os
import sys
from datetime import datetime

from quant_engine import (
    gr, rd, dim, bold,
    fetch, demo_data,
    analyze, position_size,
    load_positions, save_positions,
    DISCLAIMER,
)

WATCHLIST_FILE = "watchlist.txt"
ACCOUNT_FILE = "account_state.json"
REBALANCE_LOG_FILE = "rebalance_log.json"

DEFAULT_ACCOUNT = 100_000.0
RISK_PCT = 1.0                  # % of cash risked per new position (ATR stop)
MAX_POSITIONS = 8
MIN_CONVICTION = 60
CIRCUIT_BREAKER_DD_PCT = -10.0  # trip new entries once drawdown from peak equity breaches this


def load_watchlist(filepath=WATCHLIST_FILE):
    if not os.path.exists(filepath):
        return []
    with open(filepath) as f:
        return [ln.strip().upper() for ln in f if ln.strip() and not ln.startswith("#")]


def load_account(filepath=ACCOUNT_FILE):
    try:
        with open(filepath) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"start_date": datetime.now().isoformat(),
                "cash": DEFAULT_ACCOUNT, "equity_peak": DEFAULT_ACCOUNT}


def save_account(state, filepath=ACCOUNT_FILE):
    with open(filepath, "w") as f:
        json.dump(state, f, indent=2)


def append_log(entry, filepath=REBALANCE_LOG_FILE):
    try:
        with open(filepath) as f:
            log = json.load(f)
    except FileNotFoundError:
        log = []
    log.append(entry)
    with open(filepath, "w") as f:
        json.dump(log, f, indent=2)


def analyze_ticker(ticker, demo, period="6mo", interval="1d"):
    df = demo_data(ticker) if demo else fetch(ticker, period, interval)
    return analyze(ticker, df, interval)


def run(demo=False, dry_run=False, account_override=None):
    print(bold(f"\nMONTHLY REBALANCER — {datetime.now():%Y-%m-%d %H:%M}"))
    watchlist = load_watchlist()
    if not watchlist:
        print(rd("watchlist.txt is empty — nothing to evaluate"))
        return None

    pos_data = load_positions()
    acct = load_account()
    is_first_run = not pos_data["positions"] and not pos_data["closed"]
    if account_override is not None and is_first_run:
        acct["cash"] = account_override
        acct["equity_peak"] = account_override

    entered, exited, errors = [], [], []

    # 1. refresh + evaluate every open position — exit on an AVOID verdict
    for pos in list(pos_data["positions"]):
        t = pos["ticker"]
        try:
            res = analyze_ticker(t, demo)
        except Exception as e:
            errors.append(f"{t}: {e}")
            continue
        price = res["last"]
        pos["current_price"] = price
        pos["current_conviction"] = res["conviction"]
        pos["conviction_change"] = res["conviction"] - pos["entry_conviction"]
        pos["signal"] = res["verdict"]["label"]
        if res["verdict"]["tone"] == "bad":
            pnl = (price - pos["entry_price"]) * pos["shares"]
            print(f"  {rd('EXIT')}  {t:<6} @ ${price:.2f}  P&L {pnl:+.0f}  ({res['verdict']['label']})")
            exited.append((t, price, pnl))
            if not dry_run:
                pos["exit_price"] = price
                pos["exit_time"] = datetime.now().isoformat()
                pos["status"] = "closed"
                pos["pnl"] = pnl
                pos["pnl_pct"] = (price / pos["entry_price"] - 1) * 100
                pos_data["closed"].append(pos)
                pos_data["positions"].remove(pos)
                acct["cash"] += price * pos["shares"]

    held = {p["ticker"] for p in pos_data["positions"]}

    # 2. mark-to-market open positions -> account value & circuit breaker
    open_value = sum(p.get("current_price", p["entry_price"]) * p["shares"]
                      for p in pos_data["positions"])
    account_value = acct["cash"] + open_value
    acct["equity_peak"] = max(acct.get("equity_peak", account_value), account_value)
    drawdown_pct = (account_value / acct["equity_peak"] - 1) * 100 if acct["equity_peak"] else 0.0
    tripped = drawdown_pct <= CIRCUIT_BREAKER_DD_PCT

    print(f"\n  Account value ${account_value:,.0f}  |  peak ${acct['equity_peak']:,.0f}  "
          f"|  drawdown {drawdown_pct:+.1f}%")
    if tripped:
        print(rd(f"  CIRCUIT BREAKER TRIPPED ({drawdown_pct:+.1f}% <= {CIRCUIT_BREAKER_DD_PCT:.0f}%) "
                 f"— no new entries this cycle"))

    # 3. rank qualifying candidates not already held
    slots = MAX_POSITIONS - len(pos_data["positions"])
    candidates = []
    if not tripped and slots > 0:
        for t in watchlist:
            if t in held:
                continue
            try:
                res = analyze_ticker(t, demo)
            except Exception as e:
                errors.append(f"{t}: {e}")
                continue
            v = res["verdict"]
            if v["tone"] == "good" and res["conviction"] >= MIN_CONVICTION and not res["ineligible"]:
                candidates.append(res)
        candidates.sort(key=lambda r: r["score"], reverse=True)

    for res in candidates[:slots]:
        t = res["ticker"]
        sizing = position_size(res["last"], res["atr"], acct["cash"], RISK_PCT)
        if not sizing or sizing["notional"] > acct["cash"]:
            continue
        print(f"  {gr('ENTER')} {t:<6} @ ${res['last']:.2f}  {sizing['shares']} sh  "
              f"({res['verdict']['label']}, conviction {res['conviction']}%)")
        entered.append((t, res, sizing))
        if not dry_run:
            pos_data["positions"].append({
                "ticker": t, "entry_price": res["last"],
                "entry_time": datetime.now().isoformat(),
                "entry_conviction": res["conviction"],
                "shares": sizing["shares"], "status": "open",
                "current_price": res["last"], "current_conviction": res["conviction"],
                "pnl": 0.0, "pnl_pct": 0.0, "conviction_change": 0,
                "signal": res["verdict"]["label"],
            })
            acct["cash"] -= sizing["notional"]

    if not entered and not exited:
        print(dim("  no qualifying signals — no changes this cycle"))

    log_entry = {
        "date": datetime.now().isoformat(),
        "account_value_before": account_value,
        "watchlist_size": len(watchlist),
        "entered": [t for t, _, _ in entered],
        "exited": [t for t, _, _ in exited],
        "circuit_breaker_tripped": tripped,
        "drawdown_pct": drawdown_pct,
        "dry_run": dry_run,
        "errors": errors,
    }

    if not dry_run:
        save_positions(pos_data)
        save_account(acct)
        append_log(log_entry)

    for e in errors:
        print(rd("  error ") + e)
    print(dim(f"\n{DISCLAIMER}\n"))
    return log_entry


def main():
    ap = argparse.ArgumentParser(description="Monthly signal-based portfolio rebalancer.")
    ap.add_argument("--demo", action="store_true", help="synthetic data, no internet")
    ap.add_argument("--dry-run", action="store_true", help="show the plan without saving state")
    ap.add_argument("--account", type=float, default=None, help="starting account value (first run only)")
    args = ap.parse_args()
    run(demo=args.demo, dry_run=args.dry_run, account_override=args.account)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
After-Hours Watch — honest overnight early-warning for Meridian.

For each watchlist name it compares the LIVE extended-hours price (Alpaca, incl.
pre/post market) to the regular-session close, flags outsized overnight moves,
and notes when an intraday rally is fading after-hours (or a sell-off is bouncing).
It also attaches the legitimate structural datasets the app already pulls — FINRA
off-exchange (dark-pool) short ratio — clearly labelled as context.

HONEST SCOPE — read this before believing any "front-running" pitch:
  • What it CAN do: report observable price moves and public structural data, so
    you get an early warning ("RXRX is −8% after hours — go look") instead of
    waking up blindsided.
  • What it does NOT do: read "institutional intent", detect dark-pool block
    BUYERS vs SELLERS, or draw a "smart-money floor". That needs full
    consolidated-tape (paid SIP) data, and even then you cannot reliably infer
    who is trading or why. A print at $18.50 is a print at $18.50 — not proof a
    fund is "stepping in". Anything claiming otherwise is selling confidence,
    not signal. This module reports the move; it never fabricates the motive.
"""


def read_one(ticker, reg_close, prev_close, ah_price, dark=None, threshold=3.0):
    """Build one after-hours read, or None if inputs are unusable.
      ah_chg    — extended-hours price vs today's regular close (%)
      day_chg   — regular session vs prior close (%)
      diverges  — intraday direction and after-hours direction disagree
                  (a rally fading, or a sell-off bouncing — observed, not 'intent')
      flag      — |ah_chg| >= threshold, or a divergence
      dpi/dpi_avg — FINRA off-exchange short ratio + its recent average (context)"""
    if not (reg_close and ah_price and reg_close > 0):
        return None
    ah_chg = (ah_price / reg_close - 1.0) * 100.0
    day_chg = (reg_close / prev_close - 1.0) * 100.0 if prev_close else None
    diverges = bool(day_chg is not None and
                    ((day_chg > 2 and ah_chg < -1.5) or (day_chg < -2 and ah_chg > 1.5)))
    return {"ticker": ticker, "reg_close": float(reg_close), "ah_price": float(ah_price),
            "ah_chg": ah_chg, "day_chg": day_chg, "diverges": diverges,
            "flag": bool(abs(ah_chg) >= threshold or diverges),
            "dpi": (dark or {}).get("dpi"), "dpi_avg": (dark or {}).get("dpi_avg")}


def describe(r):
    """Plain, non-hyped one-liner for a flagged read."""
    arrow = "▼" if r["ah_chg"] < 0 else "▲"
    msg = f"{arrow} {r['ticker']} {r['ah_chg']:+.1f}% after-hours — {r['ah_price']:.2f} vs {r['reg_close']:.2f} close"
    if r["diverges"] and r["day_chg"] is not None:
        if r["day_chg"] > 0:
            msg += f" (rallied {r['day_chg']:+.1f}% intraday, now fading)"
        else:
            msg += f" (fell {r['day_chg']:+.1f}% intraday, now bouncing)"
    return msg

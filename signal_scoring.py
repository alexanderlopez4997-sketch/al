#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIGNAL SCORING — alt-data penalty layer + dash-ag-grid feed for Meridian.

Takes each ticker's technical_score and knocks it down with a rule-based
alt_data_penalty BEFORE the STRONG_BUY / ADJUSTED_NEUTRAL classification, so a
name can't ride a stale technical read past a live insider dump or a souring
macro tape. Two rules, both logged, no magic numbers left unexplained:

  1. Insider decay  — > $100M in insider sales over the trailing 30d halves
                       the score (INSIDER_DECAY_FACTOR).
  2. Macro penalty  — macro_sentiment < 0 subtracts MACRO_PENALTY_POINTS.

Every row carries an audit_trail (original score, which rules fired, the
adjusted predictive_score) so a demoted ticker can be explained instead of
disappearing into a black box — surface it as a dash-ag-grid tooltip.

Pure backend, no I/O. Not financial advice.
"""
import json

INSIDER_SELL_THRESHOLD = 100_000_000.0   # $100M in insider sales, trailing 30d
INSIDER_DECAY_FACTOR = 0.5               # applied to technical_score on breach
MACRO_PENALTY_POINTS = 10.0              # subtracted when macro_sentiment < 0
STRONG_BUY_THRESHOLD = 75.0              # predictive_score cutoff for STRONG_BUY


def calculate_final_score(ticker, technical_score, insider_activity_90d=None,
                           macro_sentiment=0.0, strong_buy_threshold=STRONG_BUY_THRESHOLD):
    """technical_score + alt-data penalty -> {predictive_score, status, audit_trail}.

    insider_activity_90d is a dict such as {"sales_last_30d": 125_000_000}; only
    the trailing-30d sales figure drives the decay rule (the 90d window is the
    lookback the figure was aggregated over upstream). macro_sentiment is a
    signed float, < 0 meaning risk-off. The STRONG_BUY / ADJUSTED_NEUTRAL split
    — the intersection filter — is evaluated on predictive_score, never on the
    raw technical_score, so the penalty always has teeth.
    """
    insider_activity_90d = insider_activity_90d or {}
    sales_last_30d = float(insider_activity_90d.get("sales_last_30d", 0.0))

    predictive_score = float(technical_score)
    penalties = []

    insider_triggered = sales_last_30d > INSIDER_SELL_THRESHOLD
    if insider_triggered:
        predictive_score *= INSIDER_DECAY_FACTOR
        penalties.append({
            "rule": "insider_sell_decay",
            "reason": f"insider sales ${sales_last_30d:,.0f} > "
                      f"${INSIDER_SELL_THRESHOLD:,.0f} in trailing 30d",
            "factor": INSIDER_DECAY_FACTOR,
        })

    macro_triggered = macro_sentiment < 0
    if macro_triggered:
        predictive_score -= MACRO_PENALTY_POINTS
        penalties.append({
            "rule": "macro_sentiment_penalty",
            "reason": f"macro_sentiment {macro_sentiment:+.2f} < 0",
            "points": -MACRO_PENALTY_POINTS,
        })

    predictive_score = round(predictive_score, 2)
    status = "STRONG_BUY" if predictive_score >= strong_buy_threshold else "ADJUSTED_NEUTRAL"

    return {
        "ticker": ticker,
        "technical_score": float(technical_score),
        "predictive_score": predictive_score,
        "alt_data_penalty": {
            "insider_decay_applied": insider_triggered,
            "macro_penalty_applied": macro_triggered,
        },
        "status": status,
        "audit_trail": {
            "original_score": float(technical_score),
            "penalties_applied": penalties,
            "final_score": predictive_score,
            "threshold": strong_buy_threshold,
            "explanation": (
                f"{float(technical_score):.2f} -> {predictive_score:.2f}"
                + ("; " + "; ".join(p["reason"] for p in penalties) if penalties else "; no penalties applied")
                + f"; classified {status} (threshold {strong_buy_threshold:.1f})"
            ),
        },
    }


def build_signal_grid(signals, strong_buy_threshold=STRONG_BUY_THRESHOLD):
    """Score a batch of tickers -> dash-ag-grid-ready {"columnDefs", "rowData"}.

    `signals` is an iterable of dicts accepted by calculate_final_score
    (ticker, technical_score, insider_activity_90d, macro_sentiment). Hand the
    result straight to dash_ag_grid.AgGrid(columnDefs=result["columnDefs"],
    rowData=result["rowData"]) — audit_trail.explanation is wired as a
    tooltipField so the reasoning shows up on hover.
    """
    row_data = [calculate_final_score(strong_buy_threshold=strong_buy_threshold, **s)
                for s in signals]
    column_defs = [
        {"field": "ticker", "headerName": "Ticker", "pinned": "left"},
        {"field": "technical_score", "headerName": "Technical", "type": "numericColumn"},
        {"field": "predictive_score", "headerName": "Predictive", "type": "numericColumn",
         "tooltipField": "audit_trail.explanation"},
        {"field": "status", "headerName": "Status"},
        {"field": "audit_trail.explanation", "headerName": "Reasoning",
         "tooltipField": "audit_trail.explanation"},
    ]
    return {"columnDefs": column_defs, "rowData": row_data}


def main():
    demo = [
        {"ticker": "AAPL", "technical_score": 88.0,
         "insider_activity_90d": {"sales_last_30d": 150_000_000}, "macro_sentiment": 0.3},
        {"ticker": "NVDA", "technical_score": 91.0,
         "insider_activity_90d": {"sales_last_30d": 0}, "macro_sentiment": -0.5},
        {"ticker": "MSFT", "technical_score": 80.0,
         "insider_activity_90d": {"sales_last_30d": 0}, "macro_sentiment": 0.1},
    ]
    print(json.dumps(build_signal_grid(demo), indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Morning Brief — turn overnight activity into a ranked "what to watch at the open".

A move alone is noise (thin after-hours gaps fade). What carries into the morning
is a move backed by a SOURCED CATALYST, so this blends the overnight signals the
app already collects into one Overnight Catalyst Score (−100..+100), weighting the
best-documented edges highest:

  insider open-market BUY (Form 4)  — highest; a director spending own cash
  earnings / material 8-K           — post-announcement drift (real anomaly)
  S-3 / 424B offering               — dilution; a durable AVOID
  overnight news sentiment          — medium
  after-hours move + whale footprint + technical alignment — confirmation

The insider read (`insider_sig`) is expected in the shape produced by either
quant_engine.insider_signal() (Finnhub) or edgar.form4_insider_bias() (parsed
Form 4 XML, role/decay-weighted, 10b5-1-discounted) — {signal, confidence,
detail}. `filings` is edgar.recent_filings()'s output; 8-K entries there carry
{items, impact, volatility_multiplier, session, gap_multiplier}, used below to
scale the overnight-move contribution — a high-impact 8-K filed pre-market
(thinnest order book of the day) implies a bigger realized gap than the same
after-hours move carrying no fresh disclosure.

HONEST SCOPE: this is a morning-PREP radar, not a prediction. It says where
something happened overnight and WHY (with the source), ranked by how actionable
the catalyst historically is. It does not promise the gap holds, and it reads the
DISCLOSED catalyst — never inferred institutional order flow.
"""


def catalyst_score(tech_score, ah_chg, insider_sig, filings, sentiment, whale):
    """Combine overnight signals → {score -100..+100, verdict, reasons}.
    reasons is a list of (text, direction) with direction in {+1, 0, -1}."""
    score = 0.0
    reasons = []

    # 0. Gap-risk multiplier from any 8-K on the tape — highest of (item-impact
    # x session) across filings, so one high-impact pre-market 8-K dominates.
    # Missing fields (older callers, or non-8-K forms) default to neutral 1.0.
    gap_risk = 1.0
    for f in (filings or []):
        if f.get("form") == "8-K":
            gap_risk = max(gap_risk, f.get("volatility_multiplier", 1.0) * f.get("gap_multiplier", 1.0))

    # 1. After-hours move — the trigger (bounded so a wild thin print can't
    # dominate; the bound itself scales with gap_risk so a confirmed
    # high-impact pre-market 8-K is allowed a bigger say than an unexplained move).
    if ah_chg:
        cap = 25.0 * min(gap_risk, 2.0)
        score += max(-cap, min(cap, ah_chg * 2.5 * gap_risk))
        if abs(ah_chg) >= 1.5:
            tag = f" (gap risk ×{gap_risk:.1f})" if gap_risk > 1.0 else ""
            reasons.append((f"{ah_chg:+.1f}% after-hours{tag}", 1 if ah_chg > 0 else -1))

    # 2. Insider open-market flow — highest weight (buys already weighted 2x in the signal)
    if insider_sig:
        c = insider_sig["signal"] * insider_sig["confidence"]
        score += c * 30.0
        if abs(c) >= 0.15:
            reasons.append((insider_sig.get("detail", "insider flow"), 1 if c > 0 else -1))

    # 3. Material SEC filings — offering = dilution (bearish), bullish stake/8-K
    for f in (filings or [])[:3]:
        if f["bias"] < 0:
            score -= 20.0
            reasons.append((f"{f['form']} — {f['note']}", -1))
        elif f["bias"] > 0:
            score += 12.0
            reasons.append((f"{f['form']} — {f['note']}", 1))
        elif f["form"] == "8-K":
            items = f.get("items") or []
            item_txt = ("Item " + ", ".join(items)) if items else "8-K"
            impact = f.get("impact", "low")
            session_note = {"pre_market": " · filed pre-market (thin liquidity)",
                            "after_hours": " · filed after-hours"}.get(f.get("session"), "")
            reasons.append((f"{item_txt} — {impact}-impact material event{session_note}", 0))

    # 4. Overnight news sentiment
    if sentiment:
        s = sentiment["signal"] * sentiment["confidence"]
        score += s * 15.0
        if abs(s) >= 0.25:
            reasons.append((f"news tone {sentiment['signal']:+.2f}", 1 if s > 0 else -1))

    # 5. Whale footprint confirmation
    if whale and whale.get("whale"):
        score += whale["signal"] * 10.0
        reasons.append((f"whale {whale['direction']} {whale['rvol']:.1f}× vol",
                        1 if whale["signal"] > 0 else -1))

    # 6. Technical alignment — does the chart already agree with the overnight direction?
    if ah_chg and tech_score:
        score += 10.0 if (ah_chg > 0) == (tech_score > 0) else -5.0

    score = max(-100.0, min(100.0, score))
    verdict = ("BUY candidate" if score >= 30 else
               "RISK / avoid" if score <= -25 else "watch")
    return {"score": int(round(score)), "verdict": verdict, "reasons": reasons}

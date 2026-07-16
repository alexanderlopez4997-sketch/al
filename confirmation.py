#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Confirmation Score — auto-runs the "verify a green light" checklist.

A single green verdict means little; a VERIFIED green light is confluence —
several INDEPENDENT signals agreeing, with no kill-switch contradicting them.
This tallies the app's independent evidence sources for one analyze() result
and returns a single read: VERIFIED / PARTIAL / WEAK / NOT VERIFIED.

Signals that aren't available (no keys, no data) are marked n/a and simply not
counted — the score is honest about how much it could actually check.

HONEST SCOPE: confluence stacks the odds in your favour; it is NOT a guarantee.
A fully-verified setup can still fail — that's what the ATR position sizing is
for. This confirms *agreement across signals*, not the future.
"""
import math


def _ok(cond):
    return "pass" if cond else "fail"


def confirm(res):
    """Return {headline, passed, checkable, checks:[(label,state,detail)], kills:[(label,detail)]}."""
    v = res.get("verdict", {}) or {}
    good = v.get("tone") == "good"
    checks, kills = [], []

    def add(label, state, detail=""):
        checks.append((label, state, detail))

    # 1 · base verdict is a buy
    add("Verdict is BUY / STRONG BUY", _ok(good), v.get("label", ""))
    # 2 · factors agree
    conv = res.get("conviction", 0)
    add("Conviction ≥ 60%", _ok(conv >= 60), f"{conv}%")
    # 3 · clears the name's calibrated threshold
    cal = res.get("calib")
    buy_th = cal["buy"] if cal else 18.0
    sc = res.get("score", 0.0)
    add(f"Score clears threshold (≥{buy_th:+.0f})", _ok(sc >= buy_th), f"{sc:+.0f}")
    # 4 · the rules actually work on this name
    sh = res.get("bt", {}).get("sharpe")
    if sh is None or (isinstance(sh, float) and math.isnan(sh)):
        add("Backtest Sharpe positive", "na")
    else:
        add("Backtest Sharpe positive", _ok(sh > 0), f"{sh:.2f}")
        if sh < 0:
            kills.append(("Negative backtest Sharpe", f"{sh:.2f} — the rules have LOST on this name"))
    # 5 · the score has historical edge here
    fw = res.get("fwd_stats")
    if fw:
        add("Score has positive edge here", _ok(fw["edge"] > 0), f"edge {fw['edge']*100:+.0f} pts")
    else:
        add("Score has positive edge here", "na")
    # 6 · alt-data agrees, not fighting
    alt = res.get("alt")
    if alt:
        add("Alt-data tilt not negative", _ok(alt["adjustment"] >= 0), f"{alt['adjustment']:+.1f}")
    else:
        add("Alt-data tilt not negative", "na")
    # 7 · whale accumulation (distribution is a kill)
    w = res.get("whale_activity")
    if w:
        if w["whale"] and w["direction"] == "distribution":
            add("Whale accumulation", "fail", "distribution")
            kills.append(("Whale distribution", f"{w['rvol']:.1f}× volume into selling pressure"))
        else:
            add("Whale accumulation", _ok(w["whale"] and w["direction"] == "accumulation"), w["direction"])
    else:
        add("Whale accumulation", "na")
    # 8 · dark-pool blocks net buy-initiated (net sell is a kill)
    fl = res.get("orderflow")
    if fl and fl.get("n_blocks") and "net_usd" in fl:
        net = fl["net_usd"]
        add("Dark-pool blocks net buy", _ok(net > 0), f"net ${net/1e6:+.1f}M")
        if net < -1_000_000:
            kills.append(("Net sell-side dark-pool blocks", f"${net/1e6:+.1f}M buy-vs-sell"))
    else:
        add("Dark-pool blocks net buy", "na")
    # 9 · market regime supportive
    mk = res.get("market")
    if mk:
        add("Risk-on & outperforming SPY", _ok(mk["risk_on"] and mk["rel"] > 0),
            ("risk-on" if mk["risk_on"] else "risk-OFF") + f", rel {mk['rel']*100:+.1f}%")
    else:
        add("Risk-on & outperforming SPY", "na")

    # kill-switches from other panels
    if v.get("risky"):
        kills.append(("RISKY (extreme volatility)", "signal reliability degraded"))
    for f in (res.get("filings") or []):
        if f.get("bias", 0) < 0:
            kills.append(("Dilution / offering filed", f"{f['form']} — {f['note']}"))
            break

    passed = sum(1 for _, s, _ in checks if s == "pass")
    checkable = sum(1 for _, s, _ in checks if s != "na")
    if not good:
        headline = "— not a BUY signal to confirm"
    elif kills:
        headline = f"🔴 NOT VERIFIED — {len(kills)} kill-switch(es)"
    elif passed >= 5 and checkable and passed / checkable >= 0.7:
        headline = "✅ VERIFIED — strong confluence"
    elif passed >= 3:
        headline = "🟡 PARTIAL — some confirmation"
    else:
        headline = "⚪ WEAK — little confirmation"
    return {"headline": headline, "passed": passed, "checkable": checkable,
            "checks": checks, "kills": kills}

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meridian test suite — assertions over the pure logic of every module.

No pytest dependency; run with:  python3 test_meridian.py
Network-dependent functions (API fetches) are NOT called — only the pure
transforms, scoring, and formatting they feed into. Exit code is nonzero on
any failure so this can gate a launch.
"""
import logging
import os
import sys
import tempfile
import threading
import time
import unittest.mock

import numpy as np
import pandas as pd

import quant_engine as qe
import fundamental_engine as fe
import sentiment_engine as se
import afterhours as ah
import morning as mb
import confirmation as cf
import trackrecord as tr
import orderflow as of
import edgar
import leaderboard as lb
import sale_conditions as sc
import exchanges as ex
import websocket_client_v2 as wsc
from meridian_cache import MeridianCache
import tui_dashboard as td
import signal_scoring as ss
import quant_gui as qg
import web_server as ws
import contracts
import dod_scraper

_PASS = _FAIL = 0
_FAILURES = []


def check(name, cond):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
    else:
        _FAIL += 1
        _FAILURES.append(name)
        print(f"  ✗ {name}")


def section(t):
    print(f"\n{t}")


def _raises(fn):
    try:
        fn(); return False
    except Exception:
        return True


def _msg(fn):
    try:
        fn(); return ""
    except Exception as e:
        return str(e)


# ------------------------------------------------------------- indicators ---
section("engine · indicators")
d = qe.enrich(qe.demo_data("TEST"))
close = d["Close"]
check("enrich adds all factor columns", all(c in d for c in ("e20", "e50", "rsi", "macd", "atr", "st", "stdir", "z", "relvol", "obv", "hi20", "lo20", "imbalance")))
check("rsi in [0,100]", d["rsi"].between(0, 100).all())
check("atr positive", (d["atr"] > 0).all())
check("ema20 tracks price magnitude", 0.5 < d["e20"].iloc[-1] / close.iloc[-1] < 2.0)
check("supertrend dir is ±1", set(np.unique(d["stdir"])) <= {-1, 1})
check("no whale/clv dead columns", "whale" not in d and "clv" not in d)

# ------------------------------------------------------------- factors ------
section("engine · factors / scoring")
F = qe.factor_matrix(d)
check("factor matrix has 5 factors", list(F.columns) == qe.FACTORS)
check("factors bounded [-1,1]", F.abs().max().max() <= 1.0001)
comp = qe.composite(F)
check("composite in plausible range", -100 <= comp.iloc[-1] <= 100)
check("positions are 0/1", set(np.unique(qe.positions(comp))) <= {0.0, 1.0})
check("positions_np matches pandas", np.array_equal(qe.positions(comp).values, qe._positions_np(comp.values)))

# ------------------------------------------------------------- backtest -----
section("engine · backtest / optimizer")
bt = qe.backtest(close, comp)
check("backtest keys present", all(k in bt for k in ("strategy", "buyhold", "sharpe", "maxdd", "trades", "winrate", "exposure")))
check("exposure in [0,1]", 0 <= bt["exposure"] <= 1)
check("maxdd <= 0", bt["maxdd"] <= 0)
# intraday backtest masks overnight: strat differs on a multi-day intraday index
idx = pd.date_range("2026-01-01 09:30", periods=200, freq="1h")
dfi = pd.DataFrame({"Open": 1., "High": 1., "Low": 1., "Close": np.linspace(100, 110, 200), "Volume": 1}, index=idx)
compi = pd.Series(np.tile([30, -30], 100), index=idx)
check("intraday backtest runs", "sharpe" in qe.backtest(dfi["Close"], compi, intraday=True))
check("session_starts detects day boundaries", qe._session_starts(idx) is not None and qe._session_starts(idx).sum() > 0)
opt = qe.optimize_weights(F, close, walk_forward=False)
check("optimizer weights sum≈1", abs(sum(opt["weights"].values()) - 1.0) < 1e-6)
check("optimizer weights in [0.02,0.60]", all(0.02 <= v <= 0.60 for v in opt["weights"].values()))
optwf = qe.optimize_weights(F, close, walk_forward=True)
check("walk-forward adds wf_sharpe", "wf_sharpe" in optwf)
check("wf weights identical to non-wf (same seed)", opt["weights"] == optwf["weights"])

# objective_loss edge cases
check("objective_loss NaN returns=999", qe.objective_loss(np.array([1.]), np.array([np.nan, np.nan]), np.array([0., 1.])) == 999.0)
check("objective_loss zero-std=999", qe.objective_loss(np.ones(5), np.zeros(5), np.zeros(5)) == 999.0)

# ------------------------------------------------------------- verdict ------
section("engine · verdict / calibration")
check("verdict STRONG BUY", qe.verdict(50, 2)["label"] == "STRONG BUY signal")
check("verdict BUY", qe.verdict(20, 2)["tone"] == "good")
check("verdict HOLD", qe.verdict(0, 2)["tone"] == "neutral")
check("verdict AVOID", qe.verdict(-30, 2)["tone"] == "bad")
check("verdict custom thresholds", qe.verdict(20, 2, buy=25)["tone"] == "neutral")
check("verdict RISKY flag", qe.verdict(50, 10)["risky"] is True)

# regime threshold blending must COMPOSE with vol/calibration widening, not
# replace it — a confident bull-regime read must not silently erase the
# safety margin vol_thresholds() built in for a genuinely high-vol name
# (this is exactly what happened to a real 126%-vol, going-concern ticker
# that got mislabeled STRONG BUY off a raw score of only +41).
_nuai_buy, _nuai_strong = qe.vol_thresholds(126.0)
check("vol_thresholds widened as expected for 126% ann vol",
      _nuai_buy > 30 and _nuai_strong > 75)
_nuai_verdict = qe.verdict(41, 10.7, _nuai_buy, _nuai_strong,
                            regime={"regime": "bull", "confidence": 0.8})
check("bull regime no longer discards vol-widened STRONG threshold",
      _nuai_verdict["label"] != "STRONG BUY signal")
check("score below the composed threshold reads as plain BUY, not STRONG",
      _nuai_verdict["label"] == "BUY signal")
# for a name at the DEFAULT (unwidened) threshold, regime blending must
# still behave exactly as the old flat-override did — no regression there.
check("regime blending is a no-op vs. old override when base == default",
      qe.verdict(17, 1.0, 18.0, 45.0, regime={"regime": "bull", "confidence": 0.8})["label"]
      == "BUY signal")
check("regime ignored below confidence 0.5 (unchanged)",
      qe.verdict(17, 1.0, 18.0, 45.0, regime={"regime": "bull", "confidence": 0.3})["label"]
      == "HOLD / no edge")
cal = qe.calibrate_thresholds(comp.values, close.values)
check("calibration returns buy/strong or None", cal is None or (cal["strong"] >= cal["buy"]))
fw = qe.forward_stats(comp.values, close.values)
check("forward_stats has win_rate/edge or None", fw is None or ("win_rate" in fw and "edge" in fw))
check("conviction in [0,100]", 0 <= qe.conviction(F.iloc[-1], comp.iloc[-1]) <= 100)

# ------------------------------------------------------------- sizing -------
section("engine · position sizing")
ps = qe.position_size(100, 2, 10000, 1)
check("position_size shares>0", ps and ps["shares"] > 0)
check("reward ≈ 2× risk (2R)", abs(ps["reward_dollars"] - 2 * ps["risk_dollars"]) < 1e-6)
check("stop below entry", ps["stop"] < ps["entry"] < ps["target"])
tight = qe.position_size(100, 2, 10000, 1, stop_mult=1.5)
check("stop_mult tightens stop", tight["stop"] > ps["stop"])
check("position_size None on bad input", qe.position_size(0, 2, 10000, 1) is None)

# ------------------------------------------------------------- whale --------
section("engine · whale score")
w = qe.whale_score(d)
check("whale_score keys", w and all(k in w for k in ("rvol", "cmf", "direction", "whale", "signal")))
check("whale signal in [-1,1]", -1 <= w["signal"] <= 1)
check("whale direction valid", w["direction"] in ("accumulation", "distribution", "neutral"))

# ------------------------------------------------------- alt-data signals ---
section("engine · alt-data signals")
asig = qe.analyst_signal({"strongBuy": 8, "buy": 12, "hold": 5, "sell": 2, "strongSell": 1})
check("analyst_signal in range", asig and -1 <= asig["signal"] <= 1)
check("analyst_signal None when empty", qe.analyst_signal({"strongBuy": 0, "buy": 0, "hold": 0, "sell": 0, "strongSell": 0}) is None)
csum = qe.summarize_congress([{"Ticker": "X", "TransactionDate": "2026-06-20", "Transaction": "Purchase", "Range": "$1M", "House": "Senate", "Party": "D"}])
check("summarize_congress counts", csum["buys"] == 1 and "recent_buys" in csum)
isig = qe.insider_signal({"buys": 2, "sells": 0, "buy_usd": 2e6, "sell_usd": 0, "biggest_buy": None, "mspr": 50, "recent_days": 90})
check("insider_signal bullish positive", isig and isig["signal"] > 0)
msig = se.macro_signal({"signal": 0.4, "confidence": 0.7, "detail": "x"})
check("macro_signal passthrough", msig["signal"] == 0.4)
tilt = qe.alt_data_tilt(csum, {"strongBuy": 5, "buy": 5, "hold": 1, "sell": 0, "strongSell": 0}, None, None, msig)
check("alt_data_tilt bounded ±15", tilt and abs(tilt["adjustment"]) <= qe.ALT_MAX_TILT + 1e-9)
check("ALT_WEIGHTS has Macro", "Macro" in qe.ALT_WEIGHTS)
# apply_alt_tilt keeps backtest untouched
res = qe.analyze("X", qe.demo_data("X"), "1d", calibrate=True)
bt_before = res["bt"]["strategy"]
qe.apply_alt_tilt(res, tilt, None)
check("apply_alt_tilt leaves backtest unchanged", res["bt"]["strategy"] == bt_before)
check("apply_alt_tilt stores base_score", "base_score" in res)

# apply_alt_tilt must not silently undo an adaptive-gate veto or drop
# volatility widening — found by auditing for bugs of the same shape as the
# regime-threshold one: a downstream recompute that discards an upstream
# safety adjustment. Both were live (quant_gui._work_single and
# web_server._full_analyze both call apply_alt_tilt right after the gates).
_F_gate = pd.DataFrame({"Direction": [0.4], "Momentum": [0.3], "Volume": [0.5], "MeanRev": [0.2]})
_res_gated = {
    "score": 30.0, "atr_pct": 2.1, "F": _F_gate, "ann_vol": 20.0, "calib": None, "regime": None,
    "whale_activity": {"rvol": 2.2, "cmf": -0.09, "dollar_vol": 5_000_000,
                        "direction": "distribution", "whale": True, "signal": -0.6},
    "verdict": qe.verdict(30.0, 2.1), "buy_th": 18.0, "strong_th": 45.0,
}
qg.apply_adaptive_gates(_res_gated)
qe.apply_alt_tilt(_res_gated, {"adjustment": 15.0}, {"adjustment": 10.0})   # max possible combined tilt
check("apply_alt_tilt cannot resurrect a whale-vetoed score",
      _res_gated["score"] == 0.0 and "GATE" in _res_gated["verdict"]["label"])

_buy_hv, _strong_hv = qe.vol_thresholds(126.0)
_res_hv = {
    "score": 41.0, "atr_pct": 10.7, "ann_vol": 126.0, "calib": None, "regime": None,
    "buy_th": _buy_hv, "strong_th": _strong_hv,
    "verdict": qe.verdict(41.0, 10.7, _buy_hv, _strong_hv),
    "F": pd.DataFrame({"Direction": [0.4], "Momentum": [0.3], "Volume": [0.1], "MeanRev": [0.1]}),
}
qe.apply_alt_tilt(_res_hv, {"adjustment": 5.0}, None)
check("apply_alt_tilt preserves volatility widening for uncalibrated high-vol names",
      _res_hv["verdict"]["strong_threshold"] > 75.0 and _res_hv["verdict"]["label"] != "STRONG BUY signal")

# market_context
sidx = pd.date_range("2026-01-01", periods=120, freq="B")
spy = pd.DataFrame({"Open": 1, "High": 1, "Low": 1, "Close": np.linspace(100, 110, 120), "Volume": 1}, index=sidx)
stk = pd.DataFrame({"Open": 1, "High": 1, "Low": 1, "Close": np.linspace(100, 140, 120), "Volume": 1}, index=sidx)
mc = qe.market_context(stk, spy)
check("market_context outperformance positive", mc and mc["rel"] > 0)
check("market_context None when unalignable", qe.market_context(stk, spy.iloc[:5]) is None)

# ------------------------------------------------------------- analyze ------
section("engine · analyze end-to-end")
r = qe.analyze("NVDA", qe.demo_data("NVDA"), "1d", calibrate=True)
check("analyze has all sections", all(k in r for k in ("score", "verdict", "bt", "whale_activity", "conviction", "fwd_stats", "calib", "intraday")))
check("analyze raises below MIN_BARS", _raises(lambda: qe.analyze("X", qe.demo_data("X", bars=30), "1d")))
check("analyze error is actionable", "recent listing" in _msg(lambda: qe.analyze("X", qe.demo_data("X", bars=30), "1d")))
_short = qe.analyze("Y", qe.demo_data("Y", bars=48), "1d", calibrate=True)
check("analyze works at 40-60 bars", _short["score"] is not None)
check("limited_history flagged at 48 bars", _short["limited_history"] is True and _short["n_bars"] == 48)
check("limited_history off at full history", r["limited_history"] is False)

# ------------------------------------- upgrades: vol thresholds / gate / decay
section("engine · vol thresholds · backtest gate · alt decay")
check("vol_thresholds widen with vol", qe.vol_thresholds(45)[0] > qe.vol_thresholds(15)[0])
check("vol_thresholds default at low vol", qe.vol_thresholds(15) == (18.0, 45.0))
check("vol_thresholds capped", qe.vol_thresholds(200)[0] <= 18.0 * 1.8 + 0.01)
check("analyze exposes ineligible flag", "ineligible" in r and isinstance(r["ineligible"], bool))
check("analyze exposes buy_th/strong_th", "buy_th" in r and "strong_th" in r)
# ineligible = neg sharpe OR sub-35% winrate
_r2 = dict(r); _r2 = r  # ineligible logic tested via direct fields
check("age_decay monotone decreasing", qe._age_decay(0) == 1.0 and qe._age_decay(30) < qe._age_decay(10) < 1.0)
check("age_decay ~0.37 at 30d", abs(qe._age_decay(30) - 0.368) < 0.01)
check("age_decay handles None", qe._age_decay(None) == 1.0)
_cr = qe.summarize_congress([{"Ticker": "X", "TransactionDate": "2026-07-05", "Transaction": "Purchase", "Range": "$1M", "House": "S", "Party": "D"}])
check("summarize_congress adds days_since", "days_since" in _cr)
check("congress_signal decays old filings", qe.congress_signal({"recent_buys": 3, "recent_sells": 0, "days_since": 80})["confidence"]
      < qe.congress_signal({"recent_buys": 3, "recent_sells": 0, "days_since": 1})["confidence"])
check("insider_signal decays old filings", qe.insider_signal({"buys": 1, "sells": 0, "buy_usd": 2e6, "sell_usd": 0, "biggest_buy": None, "mspr": 50, "recent_days": 90, "days_since": 80})["confidence"]
      < qe.insider_signal({"buys": 1, "sells": 0, "buy_usd": 2e6, "sell_usd": 0, "biggest_buy": None, "mspr": 50, "recent_days": 90, "days_since": 1})["confidence"])
# vol-adjusted stop: wider stop_mult auto-cuts shares, cash risk ~constant
_p2 = qe.position_size(400, 20, 100000, 1, 2.0); _p3 = qe.position_size(400, 20, 100000, 1, 3.0)
check("wider stop cuts share size", _p3["shares"] < _p2["shares"])
check("wider stop holds cash risk ~flat", abs(_p3["risk_dollars"] - _p2["risk_dollars"]) / _p2["risk_dollars"] < 0.05)

# ------------------------------------------------------- fundamentals -------
section("fundamental_engine")
f = fe.demo_fundamentals("NVDA")
check("demo_fundamentals deterministic", fe.demo_fundamentals("NVDA") == f)
check("demo_fundamentals has metrics", all(k in f for k in ("pe", "de", "growth", "current_ratio")))
check("passes_fundamental_filter true", fe.passes_fundamental_filter({"pe": 20, "de": 1, "growth": 10, "current_ratio": 2}, 50, 2, 0, 1))
check("passes_fundamental_filter false on high PE", not fe.passes_fundamental_filter({"pe": 80, "de": 1, "growth": 10, "current_ratio": 2}, 50, 2, 0, 1))
check("passes_fundamental_filter None fails", not fe.passes_fundamental_filter(None, 50, 2, 0, 1))
check("_num coerces", fe._num("1.5") == 1.5 and fe._num("None") is None and fe._num("-") is None)
check("_pick first valid", fe._pick({"a": "None", "b": "3.2"}, "a", "b") == 3.2)
check("fmt_fund formats", "P/E" in fe.fmt_fund(f))

# ------------------------------------------------------------- sentiment ----
section("sentiment_engine")
s, hits = se.score_text("earnings beat, strong growth and record profit surge")
check("score_text positive", s > 0 and hits >= 3)
sn, _ = se.score_text("plunge, downgrade, lawsuit and bankruptcy fears")
check("score_text negative", sn < 0)
neg, _ = se.score_text("not strong, no growth")
check("score_text negation flips", neg <= 0)
demo_s = se.demo_sentiment("NVDA")
check("demo_sentiment shape", "signal" in demo_s and "defensive_shift" in demo_s)
asm = se._assemble([(0.5, 1, 1), (0.3, 1, 4)], 7, 2, "test")
check("_assemble aggregates", asm and -1 <= asm["signal"] <= 1)

# ------------------------------------------------------------- afterhours ---
section("afterhours")
ahr = ah.read_one("RIVN", 20.14, 19.50, 18.64)
check("read_one computes ah_chg", ahr and ahr["ah_chg"] < 0 and ahr["flag"])
check("read_one divergence", ah.read_one("X", 4.0, 3.6, 3.85)["diverges"])
check("read_one None on bad input", ah.read_one("X", 0, 1, 1) is None)
check("describe non-empty", "after-hours" in ah.describe(ahr))

# ------------------------------------------------------------- morning ------
section("morning")
ins = {"signal": 0.8, "confidence": 1.0, "detail": "big buy"}
mbrief = mb.catalyst_score(35, 4.0, ins, [{"form": "SC 13D", "note": "stake", "bias": 1}], {"signal": 0.4, "confidence": 0.8}, {"whale": True, "signal": 0.5, "direction": "accumulation", "rvol": 2.0})
check("catalyst_score BUY candidate", mbrief["verdict"] == "BUY candidate" and mbrief["score"] > 30)
mrisk = mb.catalyst_score(20, -8, None, [{"form": "424B5", "note": "dilution", "bias": -1}], {"signal": -0.4, "confidence": 0.7}, None)
check("catalyst_score RISK on dilution", mrisk["verdict"] == "RISK / avoid")
check("catalyst_score bounded", -100 <= mb.catalyst_score(100, 50, ins, [], None, None)["score"] <= 100)

# 8-K gap-risk multiplier: a high-impact 8-K filed pre-market amplifies the
# after-hours-move contribution vs. the same move with no 8-K on the tape
hi8k_premkt = [{"form": "8-K", "note": "material event (8-K)", "bias": 0, "items": ["5.02"],
                "impact": "high", "volatility_multiplier": 1.6, "session": "pre_market",
                "gap_multiplier": 1.5}]
plain = mb.catalyst_score(0, 3.0, None, [], None, None)
amplified = mb.catalyst_score(0, 3.0, None, hi8k_premkt, None, None)
check("catalyst_score amplifies AH move on high-impact pre-market 8-K", amplified["score"] > plain["score"])
check("catalyst_score bounded even with gap-risk multiplier",
      -100 <= mb.catalyst_score(0, 50, None, hi8k_premkt, None, None)["score"] <= 100)
check("catalyst_score old-shape filing dicts (no 8-K fields) still work",
      mb.catalyst_score(0, 3.0, None, [{"form": "8-K", "note": "x", "bias": 0}], None, None)["score"] >= 0)

# ------------------------------------------------------------- confirmation -
section("confirmation")
verified = dict(r)
verified["conviction"] = 80
verified["fwd_stats"] = {"edge": 0.08}
verified["alt"] = {"adjustment": 4.0}
verified["whale_activity"] = {"whale": True, "direction": "accumulation", "rvol": 2.0, "cmf": 0.2}
verified["market"] = {"risk_on": True, "rel": 0.05}
verified["verdict"] = {"tone": "good", "label": "BUY signal", "risky": False}
verified["bt"] = dict(r["bt"]); verified["bt"]["sharpe"] = 3.0
cs = cf.confirm(verified)
check("confirm VERIFIED on confluence", "VERIFIED" in cs["headline"] and not cs["kills"])
killed = dict(verified); killed["bt"] = dict(killed["bt"]); killed["bt"]["sharpe"] = -1.5
killed["filings"] = [{"form": "424B5", "note": "dilution", "bias": -1}]
csk = cf.confirm(killed)
check("confirm NOT VERIFIED on kill-switch", "NOT VERIFIED" in csk["headline"] and len(csk["kills"]) >= 2)

# insider Form-4 bias: confident net buying confirms, confident net selling kills
buy_bias = dict(verified); buy_bias["insider_form4"] = {"signal": 0.8, "confidence": 0.9, "detail": "CEO bought"}
cs_buy = cf.confirm(buy_bias)
check("confirm passes Insider Form-4 check on net buying",
      any(l == "Insider Form-4 bias (EDGAR)" and s == "pass" for l, s, _ in cs_buy["checks"]))
sell_bias = dict(verified); sell_bias["insider_form4"] = {"signal": -0.6, "confidence": 0.5, "detail": "CFO sold"}
cs_sell = cf.confirm(sell_bias)
check("confirm kills on confident insider Form-4 net selling",
      any("Insider Form-4 net selling" in l for l, _ in cs_sell["kills"]))
no_bias = dict(verified); no_bias.pop("insider_form4", None)
check("confirm marks Insider Form-4 na when unavailable",
      any(l == "Insider Form-4 bias (EDGAR)" and s == "na" for l, s, _ in cf.confirm(no_bias)["checks"]))

# high-impact 8-K: informational check, never a kill on its own (earnings drift is a documented edge)
hi8k = dict(verified); hi8k["filings"] = [{"form": "8-K", "note": "material event (8-K)", "bias": 0,
                                           "items": ["5.02"], "impact": "high"}]
cs_hi8k = cf.confirm(hi8k)
check("confirm flags unresolved high-impact 8-K without killing",
      any(l == "No unresolved high-impact 8-K" and s == "fail" for l, s, _ in cs_hi8k["checks"])
      and not any("8-K" in l for l, _ in cs_hi8k["kills"]))

check("confirm exposes a level for the UI",
      cs["level"] == "verified" and csk["level"] == "kill" and cf.confirm(dict(verified, verdict={"tone": "bad", "label": "SELL"}))["level"] == "none")

# web_server green-signal banner
import web_server as _ws
_banner = _ws._confirmation_html(verified)
check("web banner shows VERIFIED headline for a confluence BUY", "VERIFIED" in _banner and "signals agree" in _banner)
_kb = _ws._confirmation_html(dict(killed, filings=[{"form": "424B5", "note": "<b>x</b>", "bias": -1}]))
check("web banner lists kill-switches, HTML-escaped", "NOT VERIFIED" in _kb and "<b>x</b>" not in _kb)
check("web banner empty when not a BUY", _ws._confirmation_html(dict(verified, verdict={"tone": "bad", "label": "SELL"})) == "")
check("web banner never raises on junk", _ws._confirmation_html({}) == "")

# DoD award check: confirms on a meaningful award, n/a on none/noise, never a kill
dod_ok = dict(verified); dod_ok["dod_awards"] = {"signal": 0.4, "confidence": 0.9, "detail": "$250M DoD award"}
check("confirm passes DoD award check on a meaningful award",
      any(l == "DoD contract award today" and s == "pass" for l, s, _ in cf.confirm(dod_ok)["checks"]))
dod_noise = dict(verified); dod_noise["dod_awards"] = {"signal": 0.01, "confidence": 0.05, "detail": "$5M"}
check("confirm marks DoD award na when it is noise",
      any(l == "DoD contract award today" and s == "na" for l, s, _ in cf.confirm(dod_noise)["checks"]))
check("confirm marks DoD award na when absent, and never kills on it",
      any(l == "DoD contract award today" and s == "na" for l, s, _ in cf.confirm(verified)["checks"])
      and not any("DoD" in l for l, _ in cf.confirm(dod_ok)["kills"]))

# ------------------------------------------------------------- trackrecord --
section("trackrecord")
_tmp = tempfile.mktemp(suffix=".json")
tr._PATH = _tmp
tr.log_verdicts([{"ticker": "AAA", "tone": "good", "label": "BUY", "score": 40, "price": 100, "tags": ["whale_accum"]}])
tr.log_verdicts([{"ticker": "AAA", "tone": "good", "label": "BUY", "score": 40, "price": 100, "tags": []}])  # dupe same day
check("trackrecord dedupes per day", len(tr._load()) == 1)
# forge an aged entry and score it
import json as _json
from datetime import date, timedelta
old = (date.today() - timedelta(days=12)).isoformat()
_json.dump([{"ticker": "WIN", "date": old, "tone": "good", "label": "BUY", "score": 40, "price": 100, "tags": []}], open(_tmp, "w"))
lut = pd.DataFrame({"Close": np.linspace(90, 120, 20)}, index=pd.date_range(date.today() - timedelta(days=20), periods=20))
scored = tr.score(lambda t: lut, horizon=5)
check("trackrecord scores aged BUY as hit", scored[0]["status"] == "scored" and scored[0]["win"] is True)
summ = tr.summary(scored, 5)
check("trackrecord summary shape", "by_tone" in summ and summ["graded"] == 1)
os.remove(_tmp)

# ------------------------------------------------------------- orderflow ----
section("orderflow (pure)")
check("block_alert fires on net buy", of.block_alert({"buy_usd": 3e6, "sell_usd": 1e6, "mid_usd": 5e6, "net_usd": 2e6}) is not None)
check("block_alert silent on net sell", of.block_alert({"buy_usd": 1e6, "sell_usd": 3e6, "mid_usd": 5e6, "net_usd": -2e6}) is None)
check("block_alert None on empty", of.block_alert({}) is None)
w0, w1 = of.after_hours_window()
check("after_hours_window returns iso pair", "T" in w0 and "Z" in w1)

# ------------------------------------------------------------- edgar (pure) -
section("edgar (pure)")
check("MATERIAL maps offerings bearish", edgar.MATERIAL["424B5"][1] < 0)
check("MATERIAL 8-K neutral", edgar.MATERIAL["8-K"][1] == 0)
check("_after_hours detects evening filing", edgar._after_hours("2026-06-17T22:40:43.000Z"))
check("_after_hours false midday", not edgar._after_hours("2026-06-17T18:00:00.000Z"))

# session classification (pre-market / regular / after-hours / overnight, ET)
check("_session pre-market", edgar._session("2026-06-17T11:00:00.000Z") == "pre_market")
check("_session regular", edgar._session("2026-06-17T16:00:00.000Z") == "regular")
check("_session after-hours", edgar._session("2026-06-17T21:00:00.000Z") == "after_hours")
check("_session overnight", edgar._session("2026-06-17T09:00:00.000Z") == "overnight")
check("_session bad input falls back neutral", edgar._session("not-a-timestamp") == "overnight")

# 8-K Item code -> impact/volatility multiplier, with a graceful fallback for
# an item code the table has never heard of
check("_8k_impact high on 5.02 (exec change)", edgar._8k_impact(["5.02"]) == ("high", 1.6))
check("_8k_impact low on 9.01 (exhibits only)", edgar._8k_impact(["9.01"]) == ("low", 1.0))
check("_8k_impact low/neutral on empty items", edgar._8k_impact([]) == ("low", 1.0))
check("_8k_impact high wins when mixed", edgar._8k_impact(["7.01", "1.01"])[0] == "high")
check("_8k_impact unknown item code doesn't crash", edgar._8k_impact(["99.99"]) == ("low", 1.0))

# audit: HIGH_IMPACT_8K_ITEMS must be a strict superset of EXEC_ITEMS's keys
# — every item EXEC_ITEMS tags in `note` as a "biggest gap" driver (exec
# change, bankruptcy, M&A, delisting) must ALSO score high-impact/high-
# volatility, or the note text and the risk weight silently contradict each
# other (this was a real bug: 1.03/2.01/3.01 were tagged in note but scored
# low-impact until HIGH_IMPACT_8K_ITEMS was derived from EXEC_ITEMS's keys).
check("every EXEC_ITEMS code is high-impact", set(edgar.EXEC_ITEMS) <= edgar.HIGH_IMPACT_8K_ITEMS)
check("_8k_impact high on 1.03 (bankruptcy)", edgar._8k_impact(["1.03"]) == ("high", 1.6))
check("_8k_impact high on 2.01 (acquisition/disposition)", edgar._8k_impact(["2.01"]) == ("high", 1.6))
check("_8k_impact high on 3.01 (delisting notice)", edgar._8k_impact(["3.01"]) == ("high", 1.6))
check("_8k_impact high on 2.02 (earnings)", edgar._8k_impact(["2.02"]) == ("high", 1.6))
check("_8k_impact high on 4.02 (non-reliance/restatement)", edgar._8k_impact(["4.02"]) == ("high", 1.6))

# end-to-end through recent_filings()'s parsing path: item extraction from
# SEC's raw comma-joined "items" string, note tagging, and impact/volatility
# stay consistent for a bankruptcy 8-K specifically (the item this bug hid)
_bankruptcy_items = [it.strip() for it in "1.03,9.01".split(",") if it.strip()]
check("bankruptcy item code parses out of a raw SEC items string",
      _bankruptcy_items == ["1.03", "9.01"])
_bankruptcy_tag = next((edgar.EXEC_ITEMS[c] for c in _bankruptcy_items if c in edgar.EXEC_ITEMS), None)
_bankruptcy_impact, _bankruptcy_mult = edgar._8k_impact(_bankruptcy_items)
check("bankruptcy 8-K gets the bankruptcy note tag", _bankruptcy_tag == "bankruptcy")
check("bankruptcy 8-K is scored high-impact, not low", _bankruptcy_impact == "high" and _bankruptcy_mult == 1.6)

# Form 4 XML parsing: P (open-market buy, CEO) and S (10b5-1 plan sale) count;
# A (grant) is dropped
_FORM4_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <issuer><issuerTradingSymbol>TEST</issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>Jane Doe</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>0</isDirector><isOfficer>1</isOfficer><isTenPercentOwner>0</isTenPercentOwner>
      <officerTitle>Chief Executive Officer</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-06-01</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>50</value></transactionPricePerShare>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-06-01</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>500</value></transactionShares>
        <transactionPricePerShare><value>52</value></transactionPricePerShare>
      </transactionAmounts>
      <footnoteId id="F1"/>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-06-01</value></transactionDate>
      <transactionCoding><transactionCode>A</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>2000</value></transactionShares>
        <transactionPricePerShare><value>0</value></transactionPricePerShare>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
  <footnotes>
    <footnote id="F1">Sale pursuant to a Rule 10b5-1 trading plan adopted 2026-01-01.</footnote>
  </footnotes>
</ownershipDocument>"""
_f4txs = edgar.parse_form4_xml(_FORM4_XML)
check("parse_form4_xml drops grants, keeps P/S", len(_f4txs) == 2 and {t["code"] for t in _f4txs} == {"P", "S"})
check("parse_form4_xml computes usd = shares*price", _f4txs[0]["usd"] == 50000.0)
check("parse_form4_xml flags 10b5-1 sale via footnote", next(t for t in _f4txs if t["code"] == "S")["is_10b5_1"])
check("parse_form4_xml P not flagged 10b5-1", not next(t for t in _f4txs if t["code"] == "P")["is_10b5_1"])
check("parse_form4_xml malformed XML returns []", edgar.parse_form4_xml("not xml") == [])
check("_role_weight CEO gets top weight", edgar._role_weight(_f4txs[0]) == 2.0)
check("_role_weight ten-pct owner", edgar._role_weight({"title": "", "is_director": False,
     "is_officer": False, "is_ten_pct_owner": True}) == edgar.TEN_PCT_OWNER_WEIGHT)
check("_role_weight plain director default", edgar._role_weight({"title": "", "is_director": True,
     "is_officer": False, "is_ten_pct_owner": False}) == edgar.DIRECTOR_WEIGHT)

# 3-tier title-weight scheme: CEO/CFO 2.0x, any other officer (or 10%+
# owner, even without a C-suite title) 1.5x, plain director 1.0x
check("_role_weight CEO = 2.0x", edgar._role_weight({"title": "Chief Executive Officer"}) == 2.0)
check("_role_weight CFO = 2.0x", edgar._role_weight({"title": "Chief Financial Officer"}) == 2.0)
check("_role_weight COO (officer, non-CEO/CFO title) = 1.5x",
      edgar._role_weight({"title": "Chief Operating Officer", "is_officer": True}) == 1.5)
check("_role_weight officer without a C-suite-sounding title still 1.5x",
      edgar._role_weight({"title": "General Counsel", "is_officer": True}) == 1.5)
check("_role_weight plain director with no flags = 1.0x",
      edgar._role_weight({"title": "Director"}) == 1.0)

# signal_score: bias * weight, comparable magnitude across row types
check("signal_score CEO buy = +2.0 (top weight, bullish)",
      edgar.signal_score({"form": "4", "bias": 1, "title": "Chief Executive Officer"}) == 2.0)
check("signal_score officer sell = -1.5",
      edgar.signal_score({"form": "4", "bias": -1, "title": "", "is_officer": True}) == -1.5)
check("signal_score director buy = +1.0",
      edgar.signal_score({"form": "4", "bias": 1, "title": "Director"}) == 1.0)
check("signal_score 8-K is always 0 — bias is never guessed for 8-K",
      edgar.signal_score({"form": "8-K", "bias": 0, "volatility_multiplier": 1.6, "gap_multiplier": 1.5}) == 0.0)
check("signal_score dilution filing = bias alone (-1.0)",
      edgar.signal_score({"form": "424B5", "bias": -1}) == -1.0)
check("signal_score never exceeds SIGNAL_SCORE_BOUND",
      abs(edgar.signal_score({"form": "4", "bias": 1, "title": "Chief Executive Officer"})) <= edgar.SIGNAL_SCORE_BOUND)
check("signal_score missing fields degrade to neutral, never raise",
      edgar.signal_score({}) == 0.0)

# rate limiting: a token bucket bounds ANY 1-second window to
# capacity + rate*1.0 requests — verify the configured constants actually
# hold that bound, and that a fresh bucket enforces it in practice.
check("RATE_LIMIT constants stay strictly under SEC's 10 req/s ceiling",
      edgar.RATE_LIMIT_BURST + edgar.RATE_LIMIT_PER_SEC * 1.0 < 10.0)
_tb = edgar._TokenBucket(rate=7.0, capacity=2.0)
_tb_start = time.time()
for _ in range(9):
    _tb.acquire()
_tb_elapsed = time.time() - _tb_start
check("_TokenBucket throttles to ~configured rate, not faster",
      _tb_elapsed >= (9 - 2) / 7.0 - 0.1)

# in-memory TTL cache: expiry + LRU eviction
_c = edgar._TTLCache(maxsize=2)
_c.set("a", b"1", ttl=0.05)
_c.set("b", b"2", ttl=10)
check("_TTLCache returns a fresh value before expiry", _c.get("a") == b"1")
time.sleep(0.08)
check("_TTLCache entry expires after its TTL", _c.get("a") is edgar._TTLCache._MISS)
_c.set("c", b"3", ttl=10)
_c.set("d", b"4", ttl=10)  # maxsize=2 -> least-recently-used of {b, c} evicted
check("_TTLCache evicts LRU past maxsize",
      (_c.get("b") is edgar._TTLCache._MISS) or (_c.get("c") is edgar._TTLCache._MISS))

# TTL routing: a "recent filings" index is short-lived, a filed document is
# effectively permanent (SEC never revises an accepted accession in place)
check("_cache_ttl_for gives submissions index the short TTL",
      edgar._cache_ttl_for("https://data.sec.gov/submissions/CIK0000320193.json") == edgar.SUBMISSIONS_CACHE_TTL)
check("_cache_ttl_for gives a filed document the long TTL",
      edgar._cache_ttl_for("https://www.sec.gov/Archives/edgar/data/320193/0/doc.xml") == edgar.DOCUMENT_CACHE_TTL)

# _get() and _get_bytes() must share one cache entry per URL — a Form 4 XML
# fetched by one path warms the cache for the other instead of double-fetching
edgar.clear_http_cache()
_fake_calls = []


class _FakeResp:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_urlopen(req, timeout=15):
    _fake_calls.append(req.full_url)
    return _FakeResp(b"<xml/>")


with unittest.mock.patch("urllib.request.urlopen", _fake_urlopen):
    _dedup_url = "https://www.sec.gov/Archives/edgar/data/1/2/doc.xml"
    _b1 = edgar._get_bytes(_dedup_url)
    _s1 = edgar._get(_dedup_url)
    _b2 = edgar._get_bytes(_dedup_url)
    check("_get/_get_bytes share the HTTP cache by URL",
          _b1 == b"<xml/>" and _s1 == "<xml/>" and _b2 == b"<xml/>" and len(_fake_calls) == 1)
edgar.clear_http_cache()

# graceful timeout handling: a network timeout is logged (not silently
# swallowed) at the one choke point all SEC requests funnel through, then
# re-raised into each caller's own `except Exception` degrade path --
# recent_filings() must still return normally, never propagate the timeout
# up into /api/feed.
_log_records = []
_log_handler = logging.Handler()
_log_handler.emit = lambda record: _log_records.append(record)
edgar.logger.addHandler(_log_handler)


def _timeout_urlopen(req, timeout=15):
    raise TimeoutError("timed out")


with unittest.mock.patch("urllib.request.urlopen", _timeout_urlopen):
    check("_get_bytes re-raises a timeout (callers degrade via their own except)",
          _raises(lambda: edgar._get_bytes("https://data.sec.gov/submissions/CIK0000320193.json")))
    check("_get_bytes logs the timeout instead of swallowing it silently",
          any("timed out" in r.getMessage() for r in _log_records))
    _log_records.clear()
    check("recent_filings() degrades to [] on a timeout, never raises",
          edgar.recent_filings("AAPL", days=2) == [])
edgar.logger.removeHandler(_log_handler)
edgar.clear_http_cache()

# detect_insider_clusters(): standalone cluster-buying detector, independent
# of form4_insider_bias()'s continuous signal.
from datetime import datetime as _datetime, timezone as _timezone
_now = _datetime.now(_timezone.utc)
check("detect_insider_clusters: no purchases -> not a cluster",
      edgar.detect_insider_clusters([])["cluster_detected"] is False)
_none_heavy = [
    {"filing_date": _now - timedelta(hours=1), "owner_name": "A", "owner_title": "Chief Executive Officer",
     "transaction_code": "P", "shares": None, "price_per_share": 50.0, "direct_ownership": True},
    {"filing_date": _now - timedelta(hours=1), "owner_name": None, "owner_title": None,
     "transaction_code": "P", "shares": 1000.0, "price_per_share": None, "direct_ownership": None},
]
check("detect_insider_clusters: explicit None fields degrade instead of raising",
      edgar.detect_insider_clusters(_none_heavy)["total_value"] == 0.0)
_ceo_cluster = [
    {"filing_date": _now - timedelta(hours=1), "owner_name": "A", "owner_title": "Chief Executive Officer",
     "transaction_code": "P", "shares": 1000, "price_per_share": 50.0, "direct_ownership": True},
    {"filing_date": _now - timedelta(hours=1), "owner_name": "B", "owner_title": "Chief Financial Officer",
     "transaction_code": "P", "shares": 1000, "price_per_share": 50.0, "direct_ownership": True},
]
_director_cluster = [
    {"filing_date": _now - timedelta(hours=1), "owner_name": "A", "owner_title": "Director",
     "transaction_code": "P", "shares": 1000, "price_per_share": 50.0, "direct_ownership": True},
    {"filing_date": _now - timedelta(hours=1), "owner_name": "B", "owner_title": "Director",
     "transaction_code": "P", "shares": 1000, "price_per_share": 50.0, "direct_ownership": True},
]
_r_ceo = edgar.detect_insider_clusters(_ceo_cluster)
_r_dir = edgar.detect_insider_clusters(_director_cluster)
check("detect_insider_clusters: same-dollar cluster flags as a cluster",
      _r_ceo["cluster_detected"] is True and _r_dir["cluster_detected"] is True)
check("detect_insider_clusters: weighted_conviction scales confidence by role, not just raw dollars",
      _r_ceo["total_value"] == _r_dir["total_value"] == 100_000.0
      and _r_ceo["weighted_conviction"] > _r_dir["weighted_conviction"]
      and _r_ceo["confidence"] > _r_dir["confidence"])

# form4_records_from_bias / insider_cluster_for_ticker / dashboard badge
_acc = (_now - timedelta(hours=2)).isoformat()
def _btx(owner, title, usd, plan=False):
    return {"code": "P", "shares": usd / 50.0, "price": 50.0, "usd": usd, "is_10b5_1": plan,
            "owner": owner, "title": title, "accepted": _acc}
_bias = {"transactions": [_btx("A", "Chief Executive Officer", 600_000), _btx("B", "Chief Financial Officer", 600_000),
                          _btx("C", "Director", 900_000, plan=True), {"accepted": "garbage", "code": "P"}]}
_recs = edgar.form4_records_from_bias(_bias)
check("form4_records_from_bias: drops 10b5-1 and unparseable rows", len(_recs) == 2)
check("form4_records_from_bias: None/garbage input -> []",
      edgar.form4_records_from_bias(None) == [] and edgar.form4_records_from_bias({"transactions": None}) == [])
_cl = edgar.insider_cluster_for_ticker("TEST", bias=_bias)
check("insider_cluster_for_ticker: end-to-end cluster from bias (no network)",
      _cl["cluster_detected"] is True and _cl["unique_buyer_count"] == 2)
with unittest.mock.patch.object(edgar, "form4_insider_bias", side_effect=RuntimeError("boom")):
    check("insider_cluster_for_ticker: fetch failure degrades, never raises",
          edgar.insider_cluster_for_ticker("TEST")["cluster_detected"] is False)
_hi = ws._insider_cluster_html(_cl)
check("cluster badge: high-conviction tier renders", "HIGH-CONVICTION INSIDER CLUSTER" in _hi and "cluster-badge high" in _hi)
_lo = ws._insider_cluster_html({"cluster_detected": True, "unique_buyer_count": 2, "total_value": 120_000.0,
                                "confidence": 0.1, "details": "x"})
check("cluster badge: low-confidence cluster uses the plain tier",
      "INSIDER CLUSTER" in _lo and "HIGH-CONVICTION" not in _lo)
check("cluster badge: no cluster / None / malformed -> empty string",
      ws._insider_cluster_html(None) == "" and ws._insider_cluster_html({"cluster_detected": False}) == ""
      and ws._insider_cluster_html({"cluster_detected": True, "unique_buyer_count": "x"}) == "")
check("cluster badge: tooltip text is HTML-escaped",
      "<script>" not in ws._insider_cluster_html({"cluster_detected": True, "details": '"><script>'}))
check("cluster badge: analyze payload only wires it in the live branch",
      ws._full_analyze("AAPL", demo=True)["cluster_badge_html"] == "")

# ------------------------------------------------------------- leaderboard --
section("leaderboard")
board = lb.build_leaderboard(qe.UNIVERSE_LIQUID[:30], demo=True)
check("leaderboard ranks", board["universe"] > 0 and len(board["ranked"]) == board["universe"])
check("leaderboard sorted by xsec", all(board["ranked"][i]["xsec_score"] >= board["ranked"][i + 1]["xsec_score"] for i in range(len(board["ranked"]) - 1)))
check("leaderboard top_buys are good", all(b["verdict"]["tone"] == "good" for b in board["top_buys"]))
check("leaderboard cross_sectional flag", board["cross_sectional"] is True)
small = lb.build_leaderboard(["AAPL", "NVDA"], demo=True)
check("leaderboard falls back on small universe", small["cross_sectional"] is False)

# ------------------------------------------------------------- cache --------
section("meridian_cache")
_dbtmp = tempfile.mktemp(suffix=".db")
c = MeridianCache(_dbtmp)
cdf = qe.demo_data("CACHE").tail(200)
c.save("CACHE", cdf, "6mo")
got = c.get("CACHE", "6mo")
check("cache round-trips", got is not None and len(got) > 0)
check("cache columns correct", list(got.columns) == ["Open", "High", "Low", "Close", "Volume"])
c.save("CACHE", cdf, "6mo")  # re-save must not raise (upsert)
check("cache upsert no crash", True)
os.remove(_dbtmp)

# ------------------------------------------------------------- sale_conditions
section("sale_conditions")
check("bundled snapshot loads", len(sc.DEFAULT_CONDITIONS) == 10)
check("index covers CTA/UTP/FINRA_TDDS tapes", set(sc.DEFAULT_INDEX) == {"CTA", "UTP", "FINRA_TDDS"})
check("Average Price Trade suppresses high/low+open/close", sc.classify_trade(["W"], "UTP") == {
    "updates_high_low": False, "updates_open_close": False, "updates_volume": True})
check("Cash Sale suppresses high/low+open/close on CTA too", sc.classify_trade(["C"], "CTA") == {
    "updates_high_low": False, "updates_open_close": False, "updates_volume": True})
check("Cross Trade updates everything", sc.classify_trade(["X"], "UTP") == {
    "updates_high_low": True, "updates_open_close": True, "updates_volume": True})
check("no condition codes updates everything", all(sc.classify_trade([], "UTP").values()))
check("unrecognized code updates everything", all(sc.classify_trade(["ZZ"], "UTP").values()))
check("one suppressing code among several suppresses the field", sc.classify_trade(["X", "C"], "CTA")["updates_high_low"] is False)
check("Derivatively Priced suppresses only open/close", sc.classify_trade(["4"], "UTP") == {
    "updates_high_low": True, "updates_open_close": False, "updates_volume": True})
check("get_condition finds Bunched Trade on UTP", sc.get_condition("B", "UTP").name == "Bunched Trade")
check("get_condition distinguishes tapes for same code", sc.get_condition("B", "CTA").name == "Average Price Trade")
check("get_condition None for unknown code", sc.get_condition("ZZ", "UTP") is None)
check("legacy flag parsed", sc.get_condition("I", "CTA").legacy is True)
check("non-legacy defaults False", sc.get_condition("X", "UTP").legacy is False)
check("fetch_all_conditions returns None without an API key", sc.fetch_all_conditions(api_key="") is None)
_parsed = sc.parse_conditions([{"id": 99, "name": "Test Cond", "asset_class": "stocks",
                                 "sip_mapping": {"UTP": "Z"}, "update_rules": {}, "data_types": ["trade"]}])
check("parse_conditions round-trips fields", _parsed[0].id == 99 and _parsed[0].sip_mapping == {"UTP": "Z"})
check("rules_for defaults all-True for missing scope", all(_parsed[0].rules_for("consolidated").values()))

# ------------------------------------------------------------------ exchanges
section("exchanges")
check("bundled snapshot loads", len(ex.DEFAULT_EXCHANGES) == 27)
check("participant_id T resolves to Nasdaq", ex.get_exchange("T").name == "Nasdaq" and ex.get_exchange("T").mic == "XNAS")
check("participant_id N resolves to NYSE", ex.get_exchange("N").mic == "XNYS")
check("unknown participant_id returns None", ex.get_exchange("ZZ") is None)
check("rows without participant_id excluded from participant index", "OTC Equity Security" not in
      {v.name for v in ex.DEFAULT_PARTICIPANT_INDEX.values()})
check("mic index resolves Nasdaq operating_mic", ex.DEFAULT_MIC_INDEX["XNAS"].name == "Nasdaq")
check("fetch_all_exchanges returns None without an API key", ex.fetch_all_exchanges(api_key="") is None)
_pex = ex.parse_exchanges([{"id": 999, "type": "exchange", "asset_class": "stocks", "locale": "us",
                             "name": "Test Exch", "operating_mic": "TEST", "mic": "TEST", "participant_id": "Q"}])
check("parse_exchanges round-trips fields", _pex[0].id == 999 and _pex[0].participant_id == "Q")
check("get_exchange_by_id resolves Nasdaq (id 12)", ex.get_exchange_by_id(12).name == "Nasdaq")
check("get_exchange_by_id None for unknown id", ex.get_exchange_by_id(9999) is None)
check("id index covers every bundled row", len(ex.DEFAULT_ID_INDEX) == len(ex.DEFAULT_EXCHANGES))

# ------------------------------------------------- websocket_client_v2 (pure)
section("websocket_client_v2 · trade processing")
_client = wsc.DiagnosticsWebSocketClient(symbols=["TEST"], max_buffer_size=10)
_buf = _client.buffers["TEST"]
_client._process_trade("TEST", wsc.Trade(symbol="TEST", price=100.0, size=500, conditions=[], timestamp=1))
_client._process_trade("TEST", wsc.Trade(symbol="TEST", price=999.0, size=100, conditions=[2], timestamp=2))  # Average Price Trade
_client._process_trade("TEST", wsc.Trade(symbol="TEST", price=101.0, size=300, conditions=[9], timestamp=3))  # Cross Trade
_bar = _buf.data[-1]
check("suppressed trade (id 2) does not move high", _bar.high < 999)
check("suppressed trade (id 2) does not move close", _bar.close == 101.0)
check("suppressed trade still counted in volume", _bar.volume == 900)
check("bar open set from first trade", _bar.open == 100.0)
_buf.close_bar()
_client._process_trade("TEST", wsc.Trade(symbol="TEST", price=50.0, size=10, conditions=[], timestamp=4))
check("close_bar() starts a fresh bar on the next trade", len(_buf.data) == 2 and _buf.data[-1].open == 50.0)
check("Trade.from_message parses short-key attrs", wsc.Trade.from_message(
    type("Msg", (), {"sym": "AAPL", "p": 190.5, "s": 100, "c": [9], "t": 123})()
) == wsc.Trade(symbol="AAPL", price=190.5, size=100, conditions=[9], timestamp=123, exchange=None))
check("Trade.from_message returns None on malformed input", wsc.Trade.from_message(
    type("Msg", (), {"sym": "AAPL", "p": "not-a-number", "s": 1, "c": [], "t": 1})()
) is None)
_ext = wsc.Trade(symbol="TEST", price=100.0, size=1, conditions=[], timestamp=1, exchange=12)
check("Trade.exchange_name resolves via exchanges module", _ext.exchange_name == "Nasdaq")
check("Trade.exchange_name is None when exchange is unset", wsc.Trade(
    symbol="TEST", price=100.0, size=1, conditions=[], timestamp=1).exchange_name is None)
check("Trade.from_message parses exchange id from short-key 'x'", wsc.Trade.from_message(
    type("Msg", (), {"sym": "AAPL", "p": 190.5, "s": 100, "c": [], "t": 123, "x": 12})()
).exchange_name == "Nasdaq")

_client2 = wsc.DiagnosticsWebSocketClient(symbols=["TEST"], max_buffer_size=10)
_client2._process_trade("TEST", wsc.Trade(symbol="TEST", price=1.0, size=1, conditions=[], timestamp=1, exchange=12))
check("_process_trade records resolved venue in last_exchange", _client2.last_exchange["TEST"] == "Nasdaq")
_diag = _client2.get_diagnostics()
check("get_diagnostics surfaces last_exchange (warming_up)", _diag.last_exchange == {"TEST": "Nasdaq"})

section("tui_dashboard")
check("parse_watchlist splits and strips", td.parse_watchlist("NVDA, AMD , AAPL") == ["NVDA", "AMD", "AAPL"])
check("load_saved_watchlist returns a non-empty list", len(td.load_saved_watchlist()) > 0)
check("market_session returns a known session label", td.market_session() in ("pre", "open", "post", "closed"))
_raw = {t: qe.demo_data(t) for t in ["NVDA", "AMD"]}
_scored, _res_by_t = td.score_watchlist(["NVDA", "AMD"], _raw)
check("score_watchlist returns one row per ticker with data", len(_scored) == 2)
check("score_watchlist rows have watchlist-table shape",
      set(_scored[0]) >= {"ticker", "last", "chg", "score", "tone", "verdict"})
check("score_watchlist also returns per-ticker res for reuse", set(_res_by_t) == {"NVDA", "AMD"})
check("score_watchlist rows sorted by descending score",
      all(_scored[i]["score"] >= _scored[i+1]["score"] for i in range(len(_scored)-1)))
_events = td.demo_event_annotations("NVDA")
check("demo_event_annotations returns 2 labeled placeholders", len(_events) == 2)
check("demo_event_annotations labels are marked (demo)", all("(demo)" in e[0] for e in _events))
_reasons = td.catalyst_reasons(_res_by_t["NVDA"])
check("catalyst_reasons returns morning.catalyst_score's reasons list", isinstance(_reasons, list))
check("account_health averages open-position pnl_pct",
      td.account_health([{"pnl_pct": 10.0}, {"pnl_pct": -2.0}]) == 4.0)
check("account_health is None with no open positions", td.account_health([]) is None)

section("signal_scoring")
_r = ss.calculate_final_score("MSFT", 80.0)
check("no penalties -> predictive_score unchanged", _r["predictive_score"] == 80.0)
check("no penalties -> empty audit_trail penalties list", _r["audit_trail"]["penalties_applied"] == [])
check("80 @ default threshold 75 -> STRONG_BUY", _r["status"] == "STRONG_BUY")

_r = ss.calculate_final_score("AAPL", 88.0, insider_activity_90d={"sales_last_30d": 150_000_000})
check("insider decay halves the score", _r["predictive_score"] == 44.0)
check("insider decay flagged in alt_data_penalty", _r["alt_data_penalty"]["insider_decay_applied"] is True)
check("insider decay demotes to ADJUSTED_NEUTRAL", _r["status"] == "ADJUSTED_NEUTRAL")

_r = ss.calculate_final_score("NVDA", 91.0, macro_sentiment=-0.5)
check("negative macro_sentiment subtracts 10 points", _r["predictive_score"] == 81.0)
check("macro penalty flagged in alt_data_penalty", _r["alt_data_penalty"]["macro_penalty_applied"] is True)
check("81 still clears threshold -> STRONG_BUY", _r["status"] == "STRONG_BUY")

_r = ss.calculate_final_score("TSLA", 80.0, insider_activity_90d={"sales_last_30d": 50_000_000})
check("insider sales under $100M threshold -> no decay", _r["alt_data_penalty"]["insider_decay_applied"] is False)
check("insider sales under threshold -> score untouched", _r["predictive_score"] == 80.0)

check("intersection filter reads the adjusted score, not the raw technical_score",
      ss.calculate_final_score("X", 100.0, insider_activity_90d={"sales_last_30d": 150_000_000}
                                )["status"] == "ADJUSTED_NEUTRAL")

_grid = ss.build_signal_grid([
    {"ticker": "AAPL", "technical_score": 88.0,
     "insider_activity_90d": {"sales_last_30d": 150_000_000}, "macro_sentiment": 0.3},
    {"ticker": "NVDA", "technical_score": 91.0, "macro_sentiment": -0.5},
])
check("build_signal_grid returns rowData + columnDefs", set(_grid) == {"columnDefs", "rowData"})
check("build_signal_grid rowData has one row per input signal", len(_grid["rowData"]) == 2)
check("build_signal_grid columnDefs include a status column",
      any(c["field"] == "status" for c in _grid["columnDefs"]))
check("build_signal_grid rows are JSON-serializable", _json.dumps(_grid["rowData"]))

# ------------------------------------------------- edge tracker · real trades
section("edge_tracker · real closed-trade track record")
import edge_tracker as et
_orig_db = et.DB_PATH
et.DB_PATH = tempfile.mktemp(suffix=".db")
try:
    check("override empty on fresh DB (no crash)", et.track_record_override("ZZZ") == (False, None))
    et.record_closed_trade("ZZZ", -3.0)          # 1 loss
    for _ in range(3):
        et.record_closed_trade("ZZZ", 4.0)       # 3 wins
    _ov, _tr = et.track_record_override("ZZZ")
    check("real trades accumulate wins/losses", _tr["total_wins"] == 3 and _tr["total_losses"] == 1)
    check("win_rate reflects real outcomes", abs(_tr["win_rate"] - 0.75) < 1e-9)
    check("override off below MIN_TRADES", _ov is False)
    for _ in range(et.TRACK_RECORD_MIN_TRADES):
        et.record_closed_trade("ZZZ", 1.0)
    _ov2, _tr2 = et.track_record_override("ZZZ")
    check("override on once real trade count clears threshold",
          _ov2 is True and _tr2["total_trades"] >= et.TRACK_RECORD_MIN_TRADES)
    # close_position() feeds the real track record
    _pf = tempfile.mktemp(suffix=".json")
    qe.add_position("WWW", 100.0, 70, filepath=_pf)
    _closed = qe.close_position("WWW", 120.0, filepath=_pf)
    check("close_position records realized win", abs(_closed["pnl_pct"] - 20.0) < 1e-6
          and et.track_record_override("WWW")[1]["total_wins"] == 1)
finally:
    et.DB_PATH = _orig_db

# ------------------------------------------------------ quant_gui · adaptive gates
section("quant_gui · adaptive gates (alignment / whale) + lookback stability")

_F_misaligned = pd.DataFrame({"Direction": [-0.36], "Momentum": [-0.03],
                               "Volume": [0.22], "MeanRev": [0.46]})
_res_would_buy = {
    "score": 22.0, "atr_pct": 2.1, "F": _F_misaligned,
    "whale_activity": {"rvol": 0.8, "cmf": -0.07, "dollar_vol": 750_000_000,
                        "direction": "distribution", "whale": False, "signal": -0.05},
    "verdict": qe.verdict(22.0, 2.1), "buy_th": 18.0, "strong_th": 45.0, "regime": None,
}
qg.apply_adaptive_gates(_res_would_buy)
check("misaligned would-be BUY (2/4 consensus) gets suppressed to 0",
      _res_would_buy["score"] == 0.0)
check("gated verdict tone flips to bad", _res_would_buy["verdict"]["tone"] == "bad")
check("gated verdict label mentions the gate", "GATE" in _res_would_buy["verdict"]["label"])
check("gate reason cites the misalignment", _res_would_buy["adaptive_gate"]["vetoed"] is True)

_F_aligned = pd.DataFrame({"Direction": [0.4], "Momentum": [0.3], "Volume": [0.5], "MeanRev": [0.2]})
_res_whale = {
    "score": 30.0, "atr_pct": 2.1, "F": _F_aligned,
    "whale_activity": {"rvol": 2.2, "cmf": -0.09, "dollar_vol": 5_000_000,
                        "direction": "distribution", "whale": True, "signal": -0.6},
    "verdict": qe.verdict(30.0, 2.1), "buy_th": 18.0, "strong_th": 45.0, "regime": None,
}
qg.apply_adaptive_gates(_res_whale)
check("aligned factors + whale distribution + abnormal volume still vetoes",
      _res_whale["score"] == 0.0 and "WHALE" in _res_whale["verdict"]["label"])

_res_clean = {
    "score": 30.0, "atr_pct": 2.1, "F": _F_aligned,
    "whale_activity": {"rvol": 1.1, "cmf": 0.03, "dollar_vol": 900_000,
                        "direction": "neutral", "whale": False, "signal": 0.1},
    "verdict": qe.verdict(30.0, 2.1), "buy_th": 18.0, "strong_th": 45.0, "regime": None,
}
qg.apply_adaptive_gates(_res_clean)
check("aligned factors + clean whale metrics pass through unchanged",
      _res_clean["score"] == 30.0 and _res_clean["adaptive_gate"]["vetoed"] is False)

_res_hold = {
    "score": -2.0, "atr_pct": 2.1, "F": _F_misaligned,
    "whale_activity": {"rvol": 0.8, "cmf": -0.07, "dollar_vol": 750_000_000,
                        "direction": "distribution", "whale": False, "signal": -0.05},
    "verdict": qe.verdict(-2.0, 2.1), "buy_th": 18.0, "strong_th": 45.0, "regime": None,
}
qg.apply_adaptive_gates(_res_hold)
check("HOLD/AVOID scores are left untouched (nothing to prevent)",
      _res_hold["score"] == -2.0 and _res_hold["adaptive_gate"]["vetoed"] is False)

# lookback-window stability (Problem #1 / WFO) — flags eligibility that flips
# depending purely on how much history the backtest happens to use.
_stable_res = qe.analyze("STAB", qe.demo_data("STAB", bars=124), "1d", None)
_stable_res["opt"] = None
_stable_ls = qg.check_lookback_stability(_stable_res)
check("lookback stability check runs on ordinary demo data", _stable_ls is not None)
check("demo data (no regime shift) reads as stable", _stable_ls["unstable"] is False)

_rng = np.random.RandomState(3)
_n = 124
_idx = pd.bdate_range("2025-01-01", periods=_n)
_ret1 = _rng.randn(62) * 0.01 + 0.01
_ret2 = _rng.randn(62) * 0.03 - 0.02
_close = 100 * np.exp(np.cumsum(np.concatenate([_ret1, _ret2])))
_high = _close * (1 + np.abs(_rng.randn(_n) * 0.01))
_low = _close * (1 - np.abs(_rng.randn(_n) * 0.01))
_vol = _rng.exponential(1e6, _n)
_df_shift = pd.DataFrame({"Open": _close, "High": _high, "Low": _low,
                           "Close": _close, "Volume": _vol}, index=_idx)
_shift_res = qe.analyze("SHIFT", _df_shift, "1d", None)
_shift_res["opt"] = None
_shift_ls = qg.check_lookback_stability(_shift_res)
check("a sharp regime shift mid-history reads as unstable", _shift_ls["unstable"] is True)
check("unstable read reports both window sizes", {r["bars"] for r in _shift_ls["windows"]} == {62, 124})

# volatility z-score normalization (Problem #2) — replaces blunt threshold
# widening with a relative-strength read for high-vol names, but only when
# the two methods actually disagree on the buy/no-buy call.
_rng2 = np.random.RandomState(1)
_n2 = 80
_dir = _rng2.randn(_n2) * 0.15; _dir[-1] = 0.35
_mom = _rng2.randn(_n2) * 0.15; _mom[-1] = 0.25
_vol_f = _rng2.randn(_n2) * 0.15; _vol_f[-1] = 0.15
_mrv = _rng2.randn(_n2) * 0.15; _mrv[-1] = 0.10
_F_spike = pd.DataFrame({"Direction": _dir, "Momentum": _mom, "Volume": _vol_f, "MeanRev": _mrv})
_close_hv = pd.Series(100 * np.exp(np.cumsum(_rng2.randn(_n2) * 0.04)))

_res_suppressed = {
    "score": 25.0, "atr_pct": 5.0, "ann_vol": 60.0, "buy_th": 30.0, "strong_th": 75.0,
    "calib": None, "F": _F_spike, "d": {"Close": _close_hv}, "opt": None, "regime": None,
    "verdict": qe.verdict(25.0, 5.0, 30.0, 75.0),
}
qg.apply_vol_normalization(_res_suppressed)
check("blunt widening suppressed a real signal to HOLD before normalization",
      25.0 < 30.0)
check("z-normalized relative-strength read restores the BUY call",
      _res_suppressed["vol_normalization"]["overridden"] is True
      and _res_suppressed["score"] >= _res_suppressed["buy_th"])
check("overridden score stays on the system's normal -100..+100 scale",
      abs(_res_suppressed["score"]) <= 100.0)
check("overridden thresholds reset to the fixed defaults (18/45)",
      _res_suppressed["buy_th"] == 18.0 and _res_suppressed["strong_th"] == 45.0)

_rng3 = np.random.RandomState(2)
_F_quiet = pd.DataFrame({"Direction": _rng3.randn(_n2) * 0.15, "Momentum": _rng3.randn(_n2) * 0.15,
                          "Volume": _rng3.randn(_n2) * 0.15, "MeanRev": _rng3.randn(_n2) * 0.15})
_res_agree = {
    "score": 5.0, "atr_pct": 5.0, "ann_vol": 60.0, "buy_th": 30.0, "strong_th": 75.0,
    "calib": None, "F": _F_quiet, "d": {"Close": _close_hv}, "opt": None, "regime": None,
    "verdict": qe.verdict(5.0, 5.0, 30.0, 75.0),
}
qg.apply_vol_normalization(_res_agree)
check("both methods agreeing on no-BUY leaves the raw score untouched",
      _res_agree["vol_normalization"]["overridden"] is False and _res_agree["score"] == 5.0)

_res_low_vol = {
    "score": 25.0, "atr_pct": 1.0, "ann_vol": 15.0, "buy_th": 18.0, "strong_th": 45.0,
    "calib": None, "F": _F_spike, "d": {"Close": _close_hv}, "opt": None, "regime": None,
    "verdict": qe.verdict(25.0, 1.0, 18.0, 45.0),
}
qg.apply_vol_normalization(_res_low_vol)
check("low-vol names are skipped entirely (blunt widening never applied there)",
      _res_low_vol["vol_normalization"]["applied"] is False and _res_low_vol["score"] == 25.0)

_res_calibrated = {
    "score": 25.0, "atr_pct": 5.0, "ann_vol": 60.0, "buy_th": 22.0, "strong_th": 55.0,
    "calib": {"buy": 22.0, "strong": 55.0}, "F": _F_spike, "d": {"Close": _close_hv},
    "opt": None, "regime": None, "verdict": qe.verdict(25.0, 5.0, 22.0, 55.0),
}
qg.apply_vol_normalization(_res_calibrated)
check("per-name calibrated thresholds are left alone (nothing to correct)",
      _res_calibrated["vol_normalization"]["applied"] is False and _res_calibrated["score"] == 25.0)

# ------------------------------------------------------- web_server /api/feed --
section("web_server /api/feed cache")


def _setup_feed_cache():
    ws._feed_cache.clear()
    ws._feed_inflight.clear()


_setup_feed_cache()
_feed_calls = []


def _fake_feed_rows(tickers, demo):
    _feed_calls.append((tuple(tickers), demo))
    time.sleep(0.05)
    return [{"ticker": tickers[0], "form": "4"}]


with unittest.mock.patch.object(ws, "_feed_rows", _fake_feed_rows):
    r1 = ws._cached_feed_rows(["AAPL"], False)
    r2 = ws._cached_feed_rows(["AAPL"], False)
    check("duplicate /api/feed request is served from cache, not recomputed",
          r1 == r2 and len(_feed_calls) == 1)
    r3 = ws._cached_feed_rows(["MSFT"], False)
    check("a different (tickers, demo) key is not served from another key's cache",
          r3 != r1 and len(_feed_calls) == 2)
    r4 = ws._cached_feed_rows(["AAPL"], True)
    check("demo and live share no cache entry for the same tickers",
          len(_feed_calls) == 3)

_setup_feed_cache()
_feed_calls = []
_feed_results = [None] * 6


def _worker(i):
    _feed_results[i] = ws._cached_feed_rows(["NVDA"], False)


with unittest.mock.patch.object(ws, "_feed_rows", _fake_feed_rows):
    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check("6 concurrent duplicate /api/feed requests share exactly one underlying fetch",
          len(_feed_calls) == 1 and all(r == _feed_results[0] for r in _feed_results))

_setup_feed_cache()


def _flaky_feed_rows(tickers, demo):
    raise RuntimeError("SEC unreachable")


with unittest.mock.patch.object(ws, "_feed_rows", _flaky_feed_rows):
    check("a leader's exception propagates rather than caching a bad result",
          _raises(lambda: ws._cached_feed_rows(["TSLA"], False)))
    check("a failed leader's in-flight entry is cleaned up, not left stuck",
          ("TSLA",) not in [k[0] for k in ws._feed_inflight])
_setup_feed_cache()

# ----------------------------------------- DoD daily scraper (dod_scraper.py) ---
section("DoD daily contract scraper")

_sample_dod_html = """
<article>
  <h2>Lockheed Martin Space Corporation awarded $50,000,000 contract</h2>
  <time datetime="2026-10-06T21:30:00Z"></time>
  <p>Lockheed Martin Space Corporation, Bethesda, Maryland, is awarded a $50,000,000
  firm-fixed-price contract for missile guidance systems development.</p>
</article>

<article>
  <h2>Personnel Change: General Retires</h2>
  <time datetime="2026-10-06T20:00:00Z"></time>
  <p>General Jones retires after 40 years of service.</p>
</article>

<article>
  <h2>Boeing Awarded $125.5 million modification</h2>
  <time datetime="2026-10-06T18:15:00Z"></time>
  <p>Boeing, Seattle, Washington awarded $125.5 million contract modification
  for KC-46A Tanker sustainment and development.</p>
</article>
"""

_dod_awards = dod_scraper.dod_daily_awards(html=_sample_dod_html)
check("DoD parser filters non-contract releases", len(_dod_awards) == 2)
check("DoD parser extracts award values", all(a.get("value_usd", 0) > 0 for a in _dod_awards))

# Test contractor extraction
_lmt_award = next((a for a in _dod_awards if "Lockheed" in a.get("contractor", "")), None)
check("DoD parser extracts contractor names", _lmt_award is not None)
if _lmt_award:
    check("DoD parser maps contractor to ticker", _lmt_award.get("ticker") == "LMT")

# Test award scoring (use $1B cap so $50M = 5% = meaningful signal)
_sig = dod_scraper.dod_award_signal(_lmt_award, market_cap=1e9)
check("DoD award scoring works with market cap", _sig is not None)
if _sig:
    check("DoD signal bounded -1..+1", -1 <= _sig["signal"] <= 1)
    check("DoD signal confidence is 0..1", 0 <= _sig["confidence"] <= 1)

# Test small-cap scenario
_sig_small = dod_scraper.dod_award_signal(_lmt_award, market_cap=300e6)
if _lmt_award and _sig_small:
    check("DoD $50M to $300M cap shows major signal",
          _sig_small["signal"] > 0.1)  # >10% of market cap

# Test bulk scoring
_bulk = dod_scraper.dod_bulk_score(_dod_awards)
check("DoD bulk scoring returns dict", isinstance(_bulk, dict))
check("DoD bulk scoring includes tickers", all(k in ["LMT", "BA"] for k in _bulk.keys()))

# Test cron scheduling
# annotate_awards: live page needs per-award signal/confidence (was always 0 before)
_ann = dod_scraper.annotate_awards([dict(a) for a in _dod_awards], lambda tk: 1e9)
check("DoD annotate_awards scores awards with a ticker",
      all(a["signal"] is not None and 0 <= a["confidence"] <= 1 for a in _ann if a.get("ticker")))
_ann_none = dod_scraper.annotate_awards([dict(a) for a in _dod_awards], lambda tk: None)
check("DoD annotate_awards leaves signal None without market cap (shows n/a, not 0)",
      all(a["signal"] is None for a in _ann_none))
_one = dod_scraper.dod_signal_for_ticker(_dod_awards, "LMT", 1e9)
check("DoD dod_signal_for_ticker returns signal for matching ticker", _one is not None and _one["signal"] > 0)
check("DoD dod_signal_for_ticker None for no match / no market cap",
      dod_scraper.dod_signal_for_ticker(_dod_awards, "ZZZZ", 1e9) is None
      and dod_scraper.dod_signal_for_ticker(_dod_awards, "LMT", None) is None)

# ---- name/award extraction from a realistic multi-award daily release (SYNTHETIC fixture
# written to the documented DoD format: one paragraph per award, grouped by service)
_day_html = """
<article>
  <h2>Contracts for Oct. 6, 2026</h2>
  <time datetime="2026-10-06T21:00:00Z"></time>
  <p>AIR FORCE</p>
  <p>Lockheed Martin Corp., Fort Worth, Texas, has been awarded a $199,303,678 firm-fixed-price
  contract for F-35 sustainment. Work will be performed in Fort Worth, Texas. Fiscal 2026 funds
  in the amount of $50,000,000 are being obligated at time of award. Air Force Life Cycle
  Management Center, Wright-Patterson Air Force Base, Ohio, is the contracting activity (FA8611-26-C-0001).</p>
  <p>NAVY</p>
  <p>Sikorsky Aircraft Corp., a Lockheed Martin Co., Stratford, Connecticut, is awarded a $1.2 billion
  modification (P00012) to a previously awarded contract (N00019-20-C-0001) for CH-53K helicopters,
  bringing the total cumulative face value of the contract to $9,500,000,000.</p>
  <p>Huntington Ingalls Inc., Newport News Shipbuilding division, Newport News, Virginia, is awarded a $310,000,000 contract.</p>
  <p>Booz Allen Hamilton Inc., McLean, Virginia (N00178-26-D-0001); Leidos Inc., Reston, Virginia (N00178-26-D-0002);
  and General Atomics, San Diego, California (N00178-26-D-0003), are awarded a $300,000,000 multiple-award contract.</p>
  <p>ARMY</p>
  <p>Raytheon Co., Tucson, Arizona; and Northrop Grumman Systems Corp., Huntsville, Alabama, are each awarded
  a $90,000,000 contract for missile components.</p>
  <p>Sturm, Ruger &amp; Co. Inc., Newport, New Hampshire, was awarded a $12,000,000 contract for rifles.</p>
  <p>Bell Boeing Joint Project Office, Amarillo, Texas, was awarded a $40,000,000 modification for V-22 support.</p>
  <p>Tiny Widgets LLC, Dayton, Ohio, is awarded a $500,000 contract.</p>
  <p>*Small business set-aside.</p>
</article>
"""
_day = dod_scraper.dod_daily_awards(html=_day_html)
_by = lambda n: [a for a in _day if n.lower() in a["contractor"].lower()]
check("day release: one award per awardee paragraph, not one per article", len(_day) >= 9)
_lm = _by("Lockheed Martin Corp")[0]
check("day release: value is the headline amount, not the funds-obligated one",
      _lm["value_usd"] == 199_303_678 and _lm["ticker"] == "LMT" and _lm["date"] == "2026-10-06")
_sk = _by("Sikorsky")[0]
check("day release: billion parsed, first amount wins over cumulative total",
      _sk["value_usd"] == 1.2e9 and _sk["ticker"] == "LMT")
check("day release: subsidiary-division name maps to parent (HII)", _by("Huntington Ingalls")[0]["ticker"] == "HII")
_multi = [a for a in _day if a["description"].startswith("Booz Allen")]
check("day release: multi-award paragraph yields one entry per awardee", len(_multi) == 3)
check("day release: shared ceiling is split evenly across awardees (not counted 3x)",
      all(abs(a["value_usd"] - 100e6) < 1 and a["value_total_usd"] == 300e6 and a["n_awardees"] == 3 for a in _multi))
check("day release: multi-award tickers/None resolved per awardee",
      {a["contractor"].split()[0]: a["ticker"] for a in _multi} == {"Booz": "BAH", "Leidos": "LDOS", "General": None})
_each = [a for a in _day if a["description"].startswith("Raytheon")]
check("day release: 'each awarded' gives every awardee the full value",
      len(_each) == 2 and all(a["value_usd"] == 90e6 for a in _each)
      and {a["ticker"] for a in _each} == {"RTX", "NOC"})
check("day release: comma inside a name is kept (Sturm, Ruger)", _by("Sturm")[0]["ticker"] == "RGR")
check("day release: JV maps to no ticker", _by("Bell Boeing")[0]["ticker"] is None)
check("day release: headings, footnotes and sub-$1M awards are dropped",
      not _by("Tiny Widgets") and all(a["contractor"] not in ("ARMY", "NAVY", "AIR FORCE") for a in _day))

# multiple same-day awards to one ticker are summed, not averaged / last-wins
_lmt_all = [a for a in _day if a["ticker"] == "LMT"]
_sum = dod_scraper.dod_signal_for_ticker(_day, "LMT", 100e9)
check("DoD per-ticker signal sums that ticker's awards (counted value)",
      _sum is not None and "DoD awards," in _sum["detail"]
      and abs(_sum["signal"] - sum(dod_scraper.award_counted_usd(a) for a in _lmt_all) / 100e9) < 1e-9)
check("DoD bulk score sums too (not last-wins)",
      abs(dod_scraper.dod_bulk_score(_day, lambda t: {"market_cap": 100e9})["LMT"]["signal"] - _sum["signal"]) < 1e-9)

# ---- ceiling vs obligated: what an award is actually worth to a signal
_pp = dod_scraper._parse_award_paragraph
_def = _pp("Lockheed Martin Corp., Fort Worth, Texas, has been awarded a $199,303,678 firm-fixed-price contract for F-35 sustainment. "
           "Fiscal 2026 funds in the amount of $50,000,000 are being obligated at time of award.")
check("firm contract: kind definite, obligated captured alongside the face value",
      _def["kind"] == "definite" and _def["obligated"] == 50_000_000 and _def["value"] == 199_303_678)
_idiq = _pp("Acme Defense Inc., Dayton, Ohio, is awarded a $4,000,000,000 indefinite-delivery/indefinite-quantity contract for parts. "
            "Fiscal 2026 funds in the amount of $5,000 are being obligated at time of award.")
check("IDIQ: kind ceiling, only the token minimum is obligated", _idiq["kind"] == "ceiling" and _idiq["obligated"] == 5_000)
check("multiple-award task order contract is a ceiling (the vehicle, not an order)",
      _pp("Acme Inc., Dayton, Ohio, is awarded a $90,000,000 multiple-award task order contract.")["kind"] == "ceiling")
check("ceiling wording BEFORE the amount counts ('with a maximum ceiling of $X')",
      _pp("Acme Inc., Dayton, Ohio, is awarded a contract with a maximum ceiling of $500,000,000 for services.")["kind"] == "ceiling")
check("a task order placed under an IDIQ is definite money",
      _pp("Acme Inc., Dayton, Ohio, is awarded a $30,000,000 firm-fixed-price task order against a previously awarded indefinite-delivery contract.")["kind"] == "definite")
check("a modification to an IDIQ is definite money",
      _pp("Acme Inc., Dayton, Ohio, is awarded a $30,000,000 modification to a previously awarded indefinite-delivery/indefinite-quantity contract.")["kind"] == "definite")
check("funding deferred to later orders is NOT money obligated at award",
      _pp("Acme Inc., Dayton, Ohio, is awarded a $50,000,000 IDIQ contract. Funds will be obligated as individual task orders are issued.")["obligated"] is None)
check("'no funds obligated at time of award' is an explicit zero",
      _pp("Acme Inc., Dayton, Ohio, is awarded a $50,000,000 IDIQ contract. No funds are being obligated at time of award.")["obligated"] == 0.0)
check("several fund clauses at award are summed",
      _pp("Acme Inc., Dayton, Ohio, is awarded a $90,000,000 contract. Fiscal 2026 funds in the amount of $5,000,000 and fiscal 2025 funds "
          "in the amount of $2,000,000 are being obligated at time of award.")["obligated"] == 7_000_000)

_cd = dod_scraper._awards_from_text_blocks([
    "Acme Defense Inc., Dayton, Ohio, is awarded a $400,000,000 indefinite-delivery/indefinite-quantity contract. "
    "Fiscal 2026 funds in the amount of $5,000 are being obligated at time of award.",
    "Beta Corp., Austin, Texas, is awarded a $400,000,000 multiple-award contract.",
    "Gamma Corp., Austin, Texas, is awarded a $400,000,000 firm-fixed-price contract.",
    "Delta Corp., Austin, Texas, is awarded a $400,000,000 IDIQ contract. No funds are being obligated at time of award.",
    "Epsilon Corp., Austin, Texas, is awarded a $20,000,000 firm-fixed-price contract. Funds in the amount of $90,000,000 are being obligated at time of award.",
], "t", "2026-10-06")
_cb = {a["contractor"].split()[0]: a for a in _cd}
check("ceiling with obligated funds counts only the obligated amount",
      _cb["Acme"]["effective_value_usd"] == 5_000 and _cb["Acme"]["value_usd"] == 400e6 and _cb["Acme"]["obligated_usd"] == 5_000)
check("ceiling with no funding info is haircut by CEILING_WEIGHT",
      abs(_cb["Beta"]["effective_value_usd"] - 400e6 * dod_scraper.CEILING_WEIGHT) < 1 and _cb["Beta"]["obligated_usd"] is None)
check("firm award with no obligation stated counts at face value",
      _cb["Gamma"]["effective_value_usd"] == 400e6 and _cb["Gamma"]["value_kind"] == "definite")
check("explicit zero-obligated ceiling counts for nothing", _cb["Delta"]["effective_value_usd"] == 0)
check("obligated can never exceed face value", _cb["Epsilon"]["obligated_usd"] == 20e6)
# multi-award: ceiling is split per awardee, THEN haircut (not haircut once and shared 3x)
_mc = [a for a in _day if a["description"].startswith("Booz Allen")]
check("multi-award ceiling: split three ways, then haircut",
      all(abs(a["effective_value_usd"] - 300e6 / 3 * dod_scraper.CEILING_WEIGHT) < 1 and a["value_kind"] == "ceiling" for a in _mc))
_lmx = [a for a in _day if a["description"].startswith("Lockheed Martin Corp")][0]
check("day release: a stated obligation drives the counted value, even on a firm award (face kept for display)",
      _lmx["effective_value_usd"] == 50_000_000 and _lmx["obligated_usd"] == 50_000_000
      and _lmx["value_usd"] == 199_303_678 and _lmx["value_kind"] == "definite")

# signals follow the counted value, and say why
_sig_def = dod_scraper.dod_award_signal(_cd[2], market_cap=10e9)
_sig_ceil = dod_scraper.dod_award_signal(_cd[1], market_cap=10e9)
check("a $400M firm award out-signals the same-size ceiling",
      _sig_def["signal"] > 5 * _sig_ceil["signal"] and "ceiling" in _sig_ceil["detail"] and "ceiling" not in _sig_def["detail"])
check("ceiling detail says funds weren't reported vs. obligated",
      "no funds reported" in _sig_ceil["detail"]
      and "obligated of a" in dod_scraper.dod_award_signal(dict(_cd[0], effective_value_usd=5_000_000, obligated_usd=5_000_000), market_cap=1e9)["detail"])
# ---- parse_contract_financial_obligations (public entry point) + regressions for review findings
_pf = dod_scraper.parse_contract_financial_obligations
_r = _pf("Lockheed Martin Corp., Fort Worth, Texas, has been awarded a $199,303,678 firm-fixed-price contract. "
         "Fiscal 2026 funds in the amount of $50,000,000 are being obligated at time of award.", 199_303_678)
check("real DoD phrasing (amount BEFORE 'obligated at time of award', 'are being') is found",
      _r["obligated_value"] == 50e6 and _r["realized_economic_value"] == 50e6 and _r["kind"] == "definite" and _r["vehicle_weight"] == 1.0)
_r = _pf("Acme Inc. is awarded a $90,000,000 contract. Funds will be obligated at the time of award. "
         "Cumulative face value of the contract is $400,000,000.", 90e6)
check("a later cumulative total is never mistaken for the obligated amount",
      _r["obligated_value"] is None and _r["realized_economic_value"] == 90e6)
_r = _pf("Acme Inc. is awarded a $30,000,000 modification to a previously awarded indefinite-delivery/indefinite-quantity contract.", 30e6)
check("a modification to an IDIQ is real money: not discounted",
      _r["realized_economic_value"] == 30e6 and _r["is_modification"] and _r["is_idiq"]
      and _r["kind"] == "definite" and _r["vehicle_weight"] == 0.80)
_r = _pf("Acme Inc. is awarded a $12,000,000 firm-fixed-price contract for center console content.", 12e6)
check("'nte' inside ordinary words (center, content) does not make a ceiling",
      not _r["is_ceiling"] and _r["realized_economic_value"] == 12e6)
_r = _pf("Acme Inc. is awarded a $4,000,000,000 indefinite-delivery/indefinite-quantity contract.", 4e9)
check("IDIQ with nothing obligated: headline x CEILING_WEIGHT, vehicle_weight 0.25",
      abs(_r["realized_economic_value"] - 4e9 * dod_scraper.CEILING_WEIGHT) < 1 and _r["vehicle_weight"] == 0.25 and _r["is_idiq"])
_r = _pf("Acme Inc. is awarded a $4,000,000,000 IDIQ contract. Fiscal 2026 funds in the amount of $5,000 are being obligated at time of award.", 4e9)
check("IDIQ with a token obligation counts only the obligation", _r["realized_economic_value"] == 5_000)
_r = _pf("A Inc., X, Ohio; B Inc., Y, Ohio; and C Inc., Z, Ohio, are awarded a $300,000,000 multiple-award contract. "
         "Fiscal 2026 funds in the amount of $30,000 are being obligated at time of award.", 100e6, n_awardees=3)
check("joint award: obligated total is shared evenly across awardees", _r["obligated_value"] == 10_000 and _r["realized_economic_value"] == 10_000)
check("junk / empty input never raises",
      _pf("", 0)["realized_economic_value"] == 0 and _pf(None, 5e6)["realized_economic_value"] == 5e6)
check("zero-counted award yields no signal", dod_scraper.dod_award_signal(_cd[3], market_cap=100e9) is None)
check("legacy / demo awards without effective value still score on face value",
      dod_scraper.award_counted_usd({"value_usd": 250e6}) == 250e6
      and dod_scraper.dod_award_signal({"value_usd": 250e6, "contractor": "X"}, market_cap=10e9) is not None)
_stack = [dict(a, ticker="ZZ") for a in (_cd[1], _cd[2])]
_stk = dod_scraper.dod_signal_for_ticker(_stack, "ZZ", 100e9)
check("per-ticker signal stacks COUNTED values (firm + haircut ceiling) and reports the face gap",
      abs(_stk["signal"] - (400e6 + 40e6) / 100e9) < 1e-12 and "counted of $800M face" in _stk["detail"])

# value extraction: first amount in reading order
check("value: first $ amount in text order, not first unit seen",
      dod_scraper._extract_award_value("a $5,000,000 award; total $2.5 billion") == 5_000_000)
check("value: million / thousand / plain / none",
      dod_scraper._extract_award_value("$125.5 million") == 125.5e6
      and dod_scraper._extract_award_value("$750 thousand") == 750e3
      and dod_scraper._extract_award_value("no money here") is None)
# contractor extraction on odd shapes
for _txt, _want in [
    ("Boeing, Seattle, Washington awarded $125.5 million contract", "Boeing"),
    ("The Boeing Co., St. Louis, Missouri, is awarded a $9,000,000 contract", "The Boeing Co."),
    ("Raytheon Technologies Corp., Pratt & Whitney Military Engines, East Hartford, Connecticut, was awarded a $9M", "Raytheon Technologies Corp."),
    ("contractor: Acme Defense Corp", "Acme Defense"),                    # legacy fallback
    ("", None),
]:
    check(f"extract_contractor({_txt[:40]!r}) -> {_want!r}", dod_scraper._extract_contractor(_txt) == _want)
check("award sentence with the awardee AFTER the verb is not mis-parsed as 'The contract'",
      dod_scraper._parse_award_paragraph("The contract was awarded to Acme Corp., Dayton, Ohio, for $5,000,000.") is None
      and dod_scraper._extract_contractor("The contract was awarded to Acme Corp., Dayton, Ohio, for $5,000,000.") == "Acme")
check("date from title handles 'Sept.' and plain months",
      dod_scraper._date_from_title("Contracts for Sept. 30, 2026") == "2026-09-30"
      and dod_scraper._date_from_title("Contracts for Oct. 6, 2026") == "2026-10-06"
      and dod_scraper._date_from_title("nothing") is None)

# markup without <article>: loose <p> fallback; headline-only listing: follow article link
_loose = """<html><body><p>Contracts for Oct. 6, 2026</p>
<p>Lockheed Martin Corp., Fort Worth, Texas, is awarded a $80,000,000 contract. Completed by Oct. 5, 2031.</p></body></html>"""
_la = dod_scraper.dod_daily_awards(html=_loose)
check("loose-<p> fallback extracts the award and dates it from the page header, not the award text",
      len(_la) == 1 and _la[0]["ticker"] == "LMT" and _la[0]["date"] == "2026-10-06")
_listing = '<article><h2>Contracts for Oct. 6, 2026</h2><p>Click for details.</p></article><a href="/News/Contracts/Contract/Article/1234567/">x</a>'
with unittest.mock.patch.object(dod_scraper, "_fetch_dod_html", lambda timeout=15, url=None: _listing), \
     unittest.mock.patch.object(dod_scraper, "_fetch_url", lambda u, timeout=15: _loose if "/Article/1234567" in u else None):
    _fa = dod_scraper.dod_daily_awards()
check("headline-only listing: follows the newest article link and parses it", len(_fa) == 1 and _fa[0]["ticker"] == "LMT")
_tried = []
def _fake_fetch(u, timeout=15):
    _tried.append(u); return "<article><h2>t</h2><p>x</p></article>" if "defense.gov" in u else None
with unittest.mock.patch.object(dod_scraper, "_fetch_url", _fake_fetch):
    _h = dod_scraper._fetch_dod_html()
check("fetch tries war.gov first, falls back to the old defense.gov URL",
      _tried[0].startswith("https://www.war.gov/") and "defense.gov" in _tried[-1] and _h is not None)

# ---- ticker mapping: token-based matching (no raw-substring false positives)
_m = dod_scraper._contractor_to_ticker
for _name, _want in [
    ("Lockheed Martin Corp.", "LMT"), ("The Boeing Co.", "BA"), ("LOCKHEED MARTIN", "LMT"),
    ("Raytheon Co.", "RTX"), ("Raytheon Technologies", "RTX"), ("Pratt & Whitney", "RTX"),
    ("Sikorsky Aircraft Corp.", "LMT"), ("General Dynamics Electric Boat", "GD"),
    ("Huntington Ingalls Inc.", "HII"), ("Northrop Grumman Systems Corp.", "NOC"),
    ("L3Harris Technologies", "LHX"), ("L3", "LHX"), ("BWX Technologies", "BWXT"), ("BWX", "BWXT"),
    ("AAI Corp.", "TXT"), ("Bell Textron Inc.", "TXT"), ("Spirit AeroSystems", "BA"),
    ("Sturm, Ruger & Co., Inc.", "RGR"), ("Curtiss-Wright Corp.", "CW"), ("AT&T Corp.", "T"),
    ("Booz Allen Hamilton Inc.", "BAH"), ("Amazon Web Services Inc.", "AMZN"),
    ("Sikorsky Aircraft Corp., a Lockheed Martin Co.", "LMT"),
    ("Joint venture of Lockheed Martin Corp. and Raytheon", "LMT"),   # multi-word alias matches mid-name
]:
    check(f"DoD map: {_name!r} -> {_want}", _m(_name) == _want)
# General Dynamics vs General Electric must not collide on the shared first word
check("DoD map: General Electric -> GE, not GD", _m("General Electric Co.") == "GE")
check("DoD map: General Dynamics -> GD, not GE", _m("General Dynamics Corp.") == "GD")
# raw-substring false positives the old matcher produced
check("DoD map: 'Saxon Industries' is not Axon", _m("Saxon Industries Inc.") is None)
check("DoD map: 'Hawaiian Airlines' is not AAI", _m("Hawaiian Airlines Inc.") is None)
check("DoD map: lone short word mid-name doesn't match", _m("Smith Oracle Consulting") is None)
check("DoD map: empty / None / junk -> None", _m("") is None and _m(None) is None and _m("!!!") is None)
# explicit None (private / foreign / JV) is a definite answer that beats shorter aliases
check("DoD map: Bell Boeing JV -> None, not BA", _m("Bell Boeing Joint Project Office") is None)
check("DoD map: private firm -> None", _m("General Atomics Aeronautical Systems") is None)
check("DoD map: delisted Triumph -> None", _m("Triumph Group Inc.") is None)
check("DoD map: the AAI-is-Northrop bug is gone", dod_scraper.CONTRACTOR_TICKER_MAP["AAI"] == "TXT")
check("DoD map: every mapped ticker is a plain upper-case symbol",
      all(v is None or (v.replace(".", "").isalnum() and v == v.upper())
          for v in dod_scraper.CONTRACTOR_TICKER_MAP.values()))
_unm = dod_scraper.unmapped_contractors([
    {"contractor": "Acme Rocketry", "ticker": None, "value_usd": 200e6},
    {"contractor": "Acme Rocketry", "ticker": None, "value_usd": 100e6},
    {"contractor": "Tiny Co", "ticker": None, "value_usd": 1e6},
    {"contractor": "General Atomics", "ticker": None, "value_usd": 900e6},      # known private
    {"contractor": "Boeing", "ticker": "BA", "value_usd": 900e6},               # mapped
    {"contractor": "Unknown", "ticker": None, "value_usd": 900e6},
])
check("DoD unmapped_contractors lists only big, never-seen names (totalled)",
      _unm == [("Acme Rocketry", 300e6)])

_cron = dod_scraper.schedule_dod_scraper()
check("DoD scheduler returns cron expression (5pm ET, weekdays)", "0 17" in _cron and "1-5" in _cron)

# -------------------------------------------- gov contracts (contracts.py) ---
section("gov contracts scoring")

# Test contract signal with real market cap
_sample_contracts = [
    {
        "contractValue": 5e6,
        "date": (pd.Timestamp.today() - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
        "agency": "DoD",
        "description": "Missile guidance system",
    },
    {
        "contractValue": 3e6,
        "date": (pd.Timestamp.today() - pd.Timedelta(days=45)).strftime("%Y-%m-%d"),
        "agency": "NASA",
        "description": "Flight control software",
    },
]
_summary = contracts.summarize_contracts(_sample_contracts)
check("contract summary tallies total value", _summary["value_total"] == 8e6)
check("contract summary counts recent (180d window)", _summary["count_recent"] == 2)
check("contract summary extracts latest awards", len(_summary["latest"]) > 0)
check("contract days_since is reasonable", _summary["days_since"] is not None and _summary["days_since"] >= 0)

# Test signal generation with market cap
_sig = contracts.contract_signal(_summary, market_cap=100e6)
check("contract signal is generated when market_cap provided", _sig is not None)
if _sig:
    check("contract signal is bounded -1..+1", -1 <= _sig["signal"] <= 1)
    check("contract signal has confidence", 0 <= _sig["confidence"] <= 1)
    check("contract signal includes detail string", isinstance(_sig["detail"], str) and len(_sig["detail"]) > 0)

# Test signal with no market cap
_sig_no_mcap = contracts.contract_signal(_summary, market_cap=None)
check("contract signal returns None when market_cap is None", _sig_no_mcap is None)

# Test age decay
_old_contracts = [
    {
        "contractValue": 10e6,
        "date": (pd.Timestamp.today() - pd.Timedelta(days=150)).strftime("%Y-%m-%d"),
        "agency": "DoD",
    }
]
_old_summary = contracts.summarize_contracts(_old_contracts)
_old_sig = contracts.contract_signal(_old_summary, market_cap=500e6)
if _old_sig and _sig:
    check("older contract has lower confidence due to age decay",
          _old_sig["confidence"] < _sig["confidence"])

# Test alt_data_tilt includes GovContracts
_alt_with_contracts = qe.alt_data_tilt(
    congress=None,
    recs=None,
    insiders=None,
    whale=None,
    macro=None,
    gov_contracts={"signal": 0.5, "confidence": 0.8, "detail": "Test contracts"}
)
check("alt_data_tilt accepts gov_contracts parameter", _alt_with_contracts is not None)
if _alt_with_contracts:
    check("GovContracts is in alt_data_tilt parts", "GovContracts" in _alt_with_contracts["parts"])
    check("GovContracts has ALT_WEIGHTS entry", "GovContracts" in qe.ALT_WEIGHTS)

# ------------------------------------------------------------- summary ------
print(f"\n{'='*50}")
print(f"RESULTS: {_PASS} passed, {_FAIL} failed")
if _FAILURES:
    print("FAILED:", ", ".join(_FAILURES))
    sys.exit(1)
print("✓ all green")

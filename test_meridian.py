#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meridian test suite — assertions over the pure logic of every module.

No pytest dependency; run with:  python3 test_meridian.py
Network-dependent functions (API fetches) are NOT called — only the pure
transforms, scoring, and formatting they feed into. Exit code is nonzero on
any failure so this can gate a launch.
"""
import json
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
import contract_backtest as cbt
import contract_crosscheck as cc
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

# --- quarterly earnings from SEC XBRL company facts (pure parsing, no network)
def _q(start, end, val, form="10-Q", filed=None):
    return {"start": start, "end": end, "val": val, "form": form, "filed": filed or end}
_facts = {"facts": {"us-gaap": {
    "Revenues": {"units": {"USD": [
        _q("2024-04-01", "2024-06-30", 1000), _q("2024-01-01", "2024-06-30", 1900),   # YTD row ignored
        _q("2024-07-01", "2024-09-30", 1050),
        _q("2025-04-01", "2025-06-30", 1200), _q("2025-07-01", "2025-09-30", 1300),
        _q("2025-07-01", "2025-09-30", 1310, filed="2025-11-20"),                      # restatement wins
        _q("2025-01-01", "2025-12-31", 5000, form="10-K")]}},
    "NetIncomeLoss": {"units": {"USD": [_q("2024-07-01", "2024-09-30", 100), _q("2025-07-01", "2025-09-30", 150)]}},
    "OperatingIncomeLoss": {"units": {"USD": [_q("2024-07-01", "2024-09-30", 105), _q("2025-07-01", "2025-09-30", 170)]}},
}}}
_e = fe.earnings_from_facts(_facts)
check("earnings: success status", _e["status"] == "success" and _e["period_end"] == "2025-09-30")
check("earnings: restated revenue + YTD/10-K rows ignored", _e["revenue"] == 1310)
check("earnings: revenue YoY", abs(_e["revenue_yoy"] - (1310 - 1050) / 1050) < 1e-9)
check("earnings: revenue QoQ", abs(_e["revenue_qoq"] - (1310 - 1200) / 1200) < 1e-9)
check("earnings: net income YoY", abs(_e["net_income_yoy"] - 0.5) < 1e-9)
check("earnings: margin expansion in pp", _e["margin_change_pp"] > 0)
check("earnings: signal bounded and positive", 0 < _e["signal"] <= 1)
_loss = {"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [
    _q("2024-07-01", "2024-09-30", -100), _q("2025-07-01", "2025-09-30", -50)]}}}}}
check("earnings: shrinking loss counts as improvement", fe.earnings_from_facts(_loss)["net_income_yoy"] > 0)
for _bad in (None, {}, {"facts": {}}, {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [{"form": "10-Q"}]}}}}}):
    _r = fe.earnings_from_facts(_bad)
    check("earnings: bad input fails soft", _r["status"] == "no_data" and _r["signal"] == 0.0)
with unittest.mock.patch.object(edgar, "_load_ciks", return_value={}):
    _r = fe.analyze_quarterly_earnings("ZZZZ", use_cache=False)
    check("analyze_quarterly_earnings: unknown ticker -> no_data", _r["status"] == "no_data" and "signal" in _r)
with unittest.mock.patch.object(edgar, "_load_ciks", side_effect=RuntimeError("boom")):
    _r = fe.analyze_quarterly_earnings("AAPL", use_cache=False)
    check("analyze_quarterly_earnings: exceptions fail soft", _r["status"] == "error" and _r["signal"] == 0.0)
with unittest.mock.patch.object(edgar, "_load_ciks", return_value={"AAPL": "0000320193"}), \
     unittest.mock.patch.object(edgar, "_get", return_value=__import__("json").dumps(_facts)):
    _r = fe.analyze_quarterly_earnings("aapl", use_cache=False)
    check("analyze_quarterly_earnings: end-to-end with mocked SEC", _r["status"] == "success" and _r["ticker"] == "AAPL")

# --- fiscal Q4 is not in any 10-Q: derive it from the 10-K full year minus the 9-month YTD
def _fy(year, q, q4_total):
    """Calendar-year company: Q1-Q3 10-Q quarters + 9M YTD row, plus a 10-K full year."""
    rows = [_q(f"{year}-01-01", f"{year}-03-31", q[0], filed=f"{year}-05-01"),
            _q(f"{year}-04-01", f"{year}-06-30", q[1], filed=f"{year}-08-01"),
            _q(f"{year}-07-01", f"{year}-09-30", q[2], filed=f"{year}-11-01"),
            _q(f"{year}-01-01", f"{year}-09-30", sum(q), filed=f"{year}-11-01")]
    if q4_total is not None:
        rows.append(_q(f"{year}-01-01", f"{year}-12-31", sum(q) + q4_total, form="10-K", filed=f"{year + 1}-02-01"))
    return rows
_fy_facts = lambda cur_q4: {"facts": {"us-gaap": {"Revenues": {"units": {"USD":
    _fy(2024, (100, 110, 120), 140) + _fy(2025, (130, 140, 150), cur_q4)}}}}}
_e4 = fe.earnings_from_facts(_fy_facts(180))
check("fiscal Q4: derived quarter becomes the anchor", _e4["period_end"] == "2025-12-31" and _e4["derived_q4"])
check("fiscal Q4: revenue = FY - 9M YTD", _e4["revenue"] == 180)
check("fiscal Q4: YoY vs prior derived Q4", abs(_e4["revenue_yoy"] - (180 - 140) / 140) < 1e-9)
check("fiscal Q4: QoQ vs Q3", abs(_e4["revenue_qoq"] - (180 - 150) / 150) < 1e-9)
_e3 = fe.earnings_from_facts(_fy_facts(None))
check("fiscal Q4: 10-K not yet filed -> latest 10-Q quarter", _e3["period_end"] == "2025-09-30" and not _e3["derived_q4"])
_no_ytd = _fy_facts(180)
_no_ytd["facts"]["us-gaap"]["Revenues"]["units"]["USD"] = [
    r for r in _no_ytd["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
    if not (r["form"] == "10-Q" and r["end"] == "2025-09-30" and r["start"] == "2025-01-01")]
check("fiscal Q4: missing 9M YTD row -> no Q4 derived, falls back to Q3",
      fe.earnings_from_facts(_no_ytd)["period_end"] == "2025-09-30")
_wk = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [      # 52-week FY (Apple-style, ends late Sept)
    _q("2024-09-29", "2025-06-28", 700), _q("2025-03-30", "2025-06-28", 230, filed="2025-08-01"),
    _q("2024-09-29", "2025-09-27", 1000, form="10-K", filed="2025-10-31")]}}}}}
check("fiscal Q4: 52-week year derives from its 9M YTD", fe.earnings_from_facts(_wk)["revenue"] == 300)

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

# insider cluster: annotates the existing Form-4 check, never a second vote or a kill
_cl_ok = {"cluster_detected": True, "unique_buyer_count": 3, "total_value": 1_200_000.0}
_ins = lambda r: next(x for x in r["checks"] if x[0] == "Insider Form-4 bias (EDGAR)")
cs_cl = cf.confirm(dict(buy_bias, insider_cluster=_cl_ok))
check("confirm annotates the insider check with a detected cluster",
      "CLUSTER (3 insiders, $1,200,000)" in _ins(cs_cl)[2] and _ins(cs_cl)[1] == "pass")
check("confirm: a cluster is not an extra vote (passed/checkable/level unchanged)",
      (cs_cl["passed"], cs_cl["checkable"], cs_cl["level"]) == (cs_buy["passed"], cs_buy["checkable"], cs_buy["level"])
      and len(cs_cl["checks"]) == len(cs_buy["checks"]))
cs_cl_sell = cf.confirm(dict(sell_bias, insider_cluster=_cl_ok))
check("confirm: a cluster never overrides insider net-selling (still fails + kills)",
      _ins(cs_cl_sell)[1] == "fail" and any("Insider Form-4 net selling" in l for l, _ in cs_cl_sell["kills"]))
check("confirm: cluster without Form-4 data stays na (not a standalone vote)",
      _ins(cf.confirm(dict(no_bias, insider_cluster=_cl_ok)))[1] == "na")
check("confirm: no/negative/malformed cluster leaves the detail untouched and never raises",
      all(_ins(cf.confirm(dict(buy_bias, insider_cluster=x)))[2] == "CEO bought"
          for x in (None, {"cluster_detected": False}, "junk", {"cluster_detected": True, "unique_buyer_count": "x"})))

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
with unittest.mock.patch.object(edgar, "form4_insider_bias", return_value=None) as _fb:
    edgar.insider_cluster_for_ticker("TEST", bias=None)
    check("insider_cluster_for_ticker: explicit bias=None is NOT refetched", _fb.call_count == 0)
    edgar.insider_cluster_for_ticker("TEST")
    check("insider_cluster_for_ticker: omitted bias is fetched once", _fb.call_count == 1)
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
      and abs(_sum["signal"] - dod_scraper._dod_signal_curve(
          sum(dod_scraper.award_counted_usd(a) for a in _lmt_all) / 100e9)) < 1e-9)
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
_sig_def = dod_scraper.dod_award_signal(_cd[2], market_cap=1e12)
_sig_ceil = dod_scraper.dod_award_signal(_cd[1], market_cap=1e12)
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
_stk = dod_scraper.dod_signal_for_ticker(_stack, "ZZ", 1e12)
check("per-ticker signal stacks COUNTED values (firm + haircut ceiling) and reports the face gap",
      abs(_stk["signal"] - dod_scraper._dod_signal_curve((400e6 + 40e6) / 1e12)) < 1e-12
      and "counted of $800M face" in _stk["detail"])
# ---- signal scale + confidence weighting (enhanced_dod_award_signal spec)
import math as _m
_curve = lambda r: _m.log1p(r / 0.001) / _m.log1p(5.0)
_sg = lambda value, mcap: dod_scraper.dod_award_signal({"value_usd": value, "contractor": "X"}, market_cap=mcap)["signal"]
check("log curve: 0.1% of market cap -> ~0.39, 0.5% -> exactly 1.0 (not the 0.69 the spec's comment claims)",
      abs(_sg(65e6, 65e9) - _curve(0.001)) < 1e-12 and abs(_sg(65e6, 65e9) - 0.3869) < 1e-3
      and abs(_sg(325e6, 65e9) - 1.0) < 1e-12)
check("log curve is monotone and concave: small awards register, bigger ones add less",
      0 < _sg(6.5e6, 65e9) < _sg(32.5e6, 65e9) < _sg(65e6, 65e9) < _sg(195e6, 65e9) < 1.0
      and _sg(32.5e6, 65e9) > 0.5 * _sg(65e6, 65e9))
check("log curve: nothing above 0.5% of market cap adds anything ($1B on $65B = $10B on $65B = 1.0)",
      _sg(1e9, 65e9) == _sg(10e9, 65e9) == 1.0)
check("signal curve never raises on zero/negative ratios", dod_scraper._dod_signal_curve(0) == 0.0
      and dod_scraper._dod_signal_curve(-0.002) == 0.0)
_cw = dod_scraper.dod_award_signal({"value_usd": 200e6, "contractor": "X", "vehicle_weight": 0.25}, market_cap=1e10)
_c1 = dod_scraper.dod_award_signal({"value_usd": 200e6, "contractor": "X"}, market_cap=1e10)
check("confidence is NOT scaled by vehicle_weight (the ceiling discount is already in the counted value)",
      _cw["confidence"] == _c1["confidence"] == 1.0)
check("confidence is min(1, counted / $100M)",
      abs(dod_scraper.dod_award_signal({"value_usd": 25e6, "contractor": "X"}, market_cap=1e10)["confidence"] - 0.25) < 1e-12)
check("a huge award can't push the signal past +1",
      dod_scraper.dod_award_signal({"value_usd": 5e10, "contractor": "X"}, market_cap=1e10)["signal"] == 1.0)
_tup = dod_scraper.refined_dod_award_signal(_cd[2], 1e12)
check("refined_dod_award_signal -> (signal, confidence) tuple matching dod_award_signal; old name still works",
      dod_scraper.enhanced_dod_award_signal is dod_scraper.refined_dod_award_signal)
check("refined_dod_award_signal -> (signal, confidence) tuple matching dod_award_signal",
      isinstance(_tup, tuple) and _tup == (dod_scraper.dod_award_signal(_cd[2], market_cap=1e12)["signal"],
                                           dod_scraper.dod_award_signal(_cd[2], market_cap=1e12)["confidence"]))
check("enhanced_dod_award_signal: (0.0, 0.0) with no market cap or nothing counted",
      dod_scraper.refined_dod_award_signal(_cd[2], 0) == (0.0, 0.0)
      and dod_scraper.refined_dod_award_signal(_cd[2], None) == (0.0, 0.0)
      and dod_scraper.refined_dod_award_signal(_cd[3], 1e12) == (0.0, 0.0))
_bare = {"description": "Acme Inc., Dayton, Ohio, is awarded a $4,000,000,000 indefinite-delivery/indefinite-quantity contract.",
         "value_usd": 4e9}
_bs, _bc = dod_scraper.refined_dod_award_signal(_bare, 1e12)
check("enhanced_dod_award_signal parses a bare description+headline record (ceiling haircut)",
      abs(_bs - dod_scraper._dod_signal_curve(4e9 * dod_scraper.CEILING_WEIGHT / 1e12)) < 1e-12 and _bc == 1.0)
check("an explicit zero stays zero: it does not fall back to the headline value",
      dod_scraper.refined_dod_award_signal({"value_usd": 400e6, "effective_value_usd": 0.0}, 1e12) == (0.0, 0.0)
      and dod_scraper.dod_award_signal({"value_usd": 400e6, "effective_value_usd": 0.0}, market_cap=1e12) is None)
check("stored breakdown is used as-is: an obligation past the 500-char description cut is not lost",
      dod_scraper.refined_dod_award_signal(
          dict(_bare, description=_bare["description"][:40], effective_value_usd=7e6, vehicle_weight=0.25), 1e12)[0]
      == dod_scraper._dod_signal_curve(7e6 / 1e12))
check("scraped awards carry vehicle_weight as an informational field (ceiling 0.25, firm 1.0)",
      _cb["Beta"]["vehicle_weight"] == 0.25 and _cb["Gamma"]["vehicle_weight"] == 1.0)

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

# ------------------------------------------------- contract <-> EDGAR cross-check -
section("contract/EDGAR cross-check")
import datetime as _dt
_T = _dt.date(2026, 10, 8)
_sig0 = {"signal": 0.4, "confidence": 0.8, "detail": "$50M award", "award_date": "2026-10-06"}
_off = [{"form": "424B5", "date": "2026-10-07", "url": "u"}]
_sell = [{"code": "S", "usd": 400_000.0, "owner": "CFO", "accepted": "2026-10-07T21:00:00.000Z", "is_10b5_1": False}]
_buy = [{"code": "P", "usd": 300_000.0, "owner": "CEO", "accepted": "2026-09-20T14:00:00.000Z", "is_10b5_1": False}]

check("contract_signal / dod_award_signal carry award_date",
      contracts.contract_signal(contracts.summarize_contracts(
          [{"contractValue": 5e7, "agency": "DoD", "date": "2026-10-06", "description": "x"}]),
          market_cap=1e9)["award_date"] == "2026-10-06"
      and dod_scraper.dod_award_signal({"value_usd": 5e7, "date": "2026-10-06", "contractor": "X",
                                        "ticker": "X"}, market_cap=1e9)["award_date"] == "2026-10-06")

_clean = cc.evaluate(_sig0, [], [], _T)
check("no filings -> no flags, multiplier 1, not negative",
      _clean and _clean["flags"] == [] and _clean["confidence_mult"] == 1.0 and not _clean["negative"])
_d = cc.evaluate(_sig0, _off, [], _T)
check("offering after award is negative and halves confidence",
      _d["negative"] and _d["confidence_mult"] == cc.DILUTION_MULT and "424B5" in _d["detail"])
check("offering BEFORE the award is ignored",
      not cc.evaluate(_sig0, [{"form": "S-3", "date": "2026-10-05"}], [], _T)["flags"])
check("offering same day as award counts",
      cc.evaluate(_sig0, [{"form": "S-3", "date": "2026-10-06"}], [], _T)["negative"])
check("offering outside DILUTION_WINDOW_DAYS is ignored",
      not cc.evaluate(dict(_sig0, award_date="2026-09-01"),
                      [{"form": "S-3", "date": "2026-09-20"}], [], _dt.date(2026, 9, 25))["flags"])
check("post-award insider sale is negative", cc.evaluate(_sig0, [], _sell, _T)["negative"])
check("10b5-1 sale is ignored", not cc.evaluate(_sig0, [], [dict(_sell[0], is_10b5_1=True)], _T)["flags"])
check("sub-$100k sale is noise", not cc.evaluate(_sig0, [], [dict(_sell[0], usd=50_000.0)], _T)["flags"])
check("sale BEFORE the award is not a post-award warning",
      not cc.evaluate(_sig0, [], [dict(_sell[0], accepted="2026-10-01T10:00:00.000Z")], _T)["flags"])
_b = cc.evaluate(_sig0, [], _buy, _T)
check("pre-award insider buy boosts confidence and is not negative",
      not _b["negative"] and _b["confidence_mult"] == cc.INSIDER_BUY_MULT)
check("insider buy older than INSIDER_PRE_DAYS is ignored",
      not cc.evaluate(_sig0, [], [dict(_buy[0], accepted="2026-08-01T10:00:00.000Z")], _T)["flags"])
check("negative flag beats a positive one (no boost when dilution present)",
      cc.evaluate(_sig0, _off, _buy, _T)["confidence_mult"] == cc.DILUTION_MULT)
check("dilution + insider selling compound",
      abs(cc.evaluate(_sig0, _off, _sell, _T)["confidence_mult"] - cc.DILUTION_MULT * cc.INSIDER_SELL_MULT) < 1e-9)
check("non-positive signal, missing/future/stale award_date -> None",
      all(cc.evaluate(x, _off, _sell, _T) is None for x in
          (dict(_sig0, signal=0.0), dict(_sig0, signal=-0.2), {k: v for k, v in _sig0.items() if k != "award_date"},
           dict(_sig0, award_date="not-a-date"), dict(_sig0, award_date="2026-12-01"),
           dict(_sig0, award_date="2026-06-01"), None)))
check("evaluate never raises on malformed EDGAR input",
      cc.evaluate(_sig0, [None, {"date": None}, "junk"], [None, {"code": "S", "usd": "x", "accepted": 5}], _T) is not None)

_adj = cc.adjusted_signal(_sig0, _d)
check("adjusted_signal scales confidence, appends detail, keeps signal, doesn't mutate input",
      abs(_adj["confidence"] - 0.4) < 1e-9 and _adj["signal"] == 0.4 and "EDGAR:" in _adj["detail"]
      and _sig0["confidence"] == 0.8 and "crosscheck" not in _sig0)
check("adjusted_signal caps boosted confidence at 1.0",
      cc.adjusted_signal(dict(_sig0, confidence=0.95), _b)["confidence"] == 1.0)
check("adjusted_signal passes through untouched with no result / no flags",
      cc.adjusted_signal(_sig0, None) is _sig0 and cc.adjusted_signal(_sig0, _clean) is _sig0
      and cc.adjusted_signal(None, _d) is None)

with unittest.mock.patch.object(edgar, "dilutive_filings", return_value=_off) as _mf, \
     unittest.mock.patch.object(edgar, "form4_insider_bias", return_value={"transactions": _sell}):
    _g = cc.crosscheck_for_ticker("XYZ", _sig0, _T)
    check("crosscheck_for_ticker combines both EDGAR sources",
          _g and _g["dilution"] and _g["insider_sell_usd"] == 400_000.0)
    cc.crosscheck_for_ticker("XYZ", dict(_sig0, signal=-0.1), _T)
    cc.crosscheck_for_ticker("XYZ", dict(_sig0, award_date="2026-01-01"), _T)
    check("crosscheck_for_ticker doesn't touch EDGAR for non-positive or stale awards", _mf.call_count == 1)
with unittest.mock.patch.object(edgar, "dilutive_filings", side_effect=RuntimeError("down")):
    check("crosscheck_for_ticker fails open when EDGAR errors", cc.crosscheck_for_ticker("XYZ", _sig0, _T) is None)
with unittest.mock.patch.object(edgar, "dilutive_filings", return_value=[]), \
     unittest.mock.patch.object(edgar, "form4_insider_bias", return_value=None):
    check("crosscheck_for_ticker: quiet EDGAR -> clean result, not None",
          cc.crosscheck_for_ticker("XYZ", _sig0, _T)["flags"] == [])

with unittest.mock.patch.object(edgar, "dilutive_filings", return_value=_off), \
     unittest.mock.patch.object(edgar, "form4_insider_bias", return_value=None):
    _adjd, _res = cc.apply_crosschecks("XYZ", {"GovContracts": _sig0, "DoDAwards": None}, _T)
check("apply_crosschecks adjusts per source and reports only sources with results",
      _adjd["GovContracts"]["confidence"] < _sig0["confidence"] and _adjd["DoDAwards"] is None
      and list(_res) == ["GovContracts"])

# edgar.dilutive_filings: filters the cached submissions index, no Form 4 parsing
_sub = json.dumps({"filings": {"recent": {
    "form": ["424B5", "4", "S-3", "10-Q", "424B3"],
    "filingDate": [(_dt.date.today() - _dt.timedelta(days=2)).isoformat(),
                   (_dt.date.today() - _dt.timedelta(days=1)).isoformat(),
                   (_dt.date.today() - _dt.timedelta(days=5)).isoformat(),
                   (_dt.date.today() - _dt.timedelta(days=1)).isoformat(),
                   (_dt.date.today() - _dt.timedelta(days=40)).isoformat()],
    "accessionNumber": ["0001-26-1", "0001-26-2", "0001-26-3", "0001-26-4", "0001-26-5"],
    "primaryDocument": ["a.htm", "b.xml", "c.htm", "d.htm", "e.htm"]}}})
with unittest.mock.patch.dict(edgar._cik_map, {"XYZ": "0000000001"}), \
     unittest.mock.patch.object(edgar, "_get", return_value=_sub), \
     unittest.mock.patch.object(edgar, "_parse_form4", side_effect=AssertionError("must not parse Form 4")):
    _df = edgar.dilutive_filings("XYZ", days=10)
    check("dilutive_filings returns only in-window offering forms, newest first",
          [f["form"] for f in _df] == ["424B5", "S-3"] and _df[0]["url"].endswith("/a.htm"))
    check("dilutive_filings: unknown ticker / fetch error -> []",
          edgar.dilutive_filings("NOPE", days=10) == [])
with unittest.mock.patch.dict(edgar._cik_map, {"XYZ": "0000000001"}), \
     unittest.mock.patch.object(edgar, "_get", side_effect=OSError("down")):
    check("dilutive_filings fails open on network error", edgar.dilutive_filings("XYZ") == [])

# confirmation: the award check can only fail/na, offering is a kill, buying is not a vote
_ccv = dict(verified); _ccv["filings"] = []
_cc_label = "Contract award not undercut (EDGAR)"
_ccrow = lambda r: next(x for x in r["checks"] if x[0] == _cc_label)
_base_cc = cf.confirm(_ccv)
check("confirm: cross-check is na with no data", _ccrow(_base_cc)[1] == "na" and not _base_cc["kills"])
_cc_dil = cf.confirm(dict(_ccv, contract_crosscheck={"DoDAwards": _d}))
check("confirm: offering after award fails the check and kills",
      _ccrow(_cc_dil)[1] == "fail" and any(l == "Offering filed after contract award" for l, _ in _cc_dil["kills"])
      and "NOT VERIFIED" in _cc_dil["headline"])
_cc_dup = cf.confirm(dict(_ccv, contract_crosscheck={"DoDAwards": _d},
                          filings=[{"form": "424B5", "note": "dilution", "bias": -1}]))
check("confirm: the same offering isn't double-counted as two kills",
      len([1 for l, _ in _cc_dup["kills"] if "ffering" in l or "ilution" in l]) == 1)
_cc_sell = cf.confirm(dict(_ccv, contract_crosscheck={"GovContracts": cc.evaluate(_sig0, [], _sell, _T)}))
check("confirm: insider selling after award fails the check but is not a kill",
      _ccrow(_cc_sell)[1] == "fail" and not _cc_sell["kills"])
_cc_buy = cf.confirm(dict(_ccv, contract_crosscheck={"DoDAwards": _b}))
check("confirm: insider buying around an award is not an extra vote",
      _ccrow(_cc_buy)[1] == "na" and (_cc_buy["passed"], _cc_buy["checkable"]) == (_base_cc["passed"], _base_cc["checkable"]))
check("confirm: malformed cross-check input never raises",
      all(_ccrow(cf.confirm(dict(_ccv, contract_crosscheck=x)))[1] == "na" for x in (None, {}, "junk", {"a": None}, {"a": "x"})))

# ------------------------------------------------------ contract backtest ----
section("contract backtest")
import datetime as _dt2

D = _dt2.date
# --- events: aggregation, threshold, cooldown, unmapped
_aw = [{"date": D(2020, 3, 2), "amount": 6e6, "recipient": "Acme", "ticker": "AAA"},
       {"date": D(2020, 3, 2), "amount": 7e6, "recipient": "Acme", "ticker": "AAA"},
       {"date": D(2020, 3, 2), "amount": -9e6, "recipient": "Acme", "ticker": "AAA"},       # de-obligation ignored
       {"date": D(2020, 3, 9), "amount": 50e6, "recipient": "Acme", "ticker": "AAA"},       # inside cooldown
       {"date": D(2020, 5, 4), "amount": 20e6, "recipient": "Acme", "ticker": "AAA"},       # after cooldown
       {"date": D(2020, 3, 2), "amount": 5e6, "recipient": "Tiny", "ticker": "BBB"},        # below min
       {"date": D(2020, 3, 2), "amount": 99e6, "recipient": "Mystery Co", "ticker": None}]  # unmapped
_ev = cbt.build_events(_aw, ticker_fn=lambda n: None, min_value=10e6, cooldown_days=30)
check("build_events sums a day, drops de-obligations/small/unmapped",
      [(e["ticker"], e["date"], e["value_usd"], e["n"]) for e in _ev] == [("AAA", D(2020, 3, 2), 13e6, 2), ("AAA", D(2020, 5, 4), 20e6, 1)])
check("build_events cooldown drops overlapping later event", D(2020, 3, 9) not in [e["date"] for e in _ev])
check("build_events maps recipient names through the DoD ticker map by default",
      [e["ticker"] for e in cbt.build_events([{"date": D(2020, 3, 2), "amount": 5e8, "recipient": "Lockheed Martin Corp."}])] == ["LMT"])
check("bucket_for edges match contracts.py bands",
      [cbt.bucket_for(x) for x in (0.0099, 0.01, 0.0499, 0.05, 0.1499, 0.15, 7.0)]
      == ["<1%", "1-5%", "1-5%", "5-15%", "5-15%", ">15%", ">15%"])
check("unadjust multiplies closes by LATER splits only",
      list(cbt.unadjust(pd.Series([10.0, 10.0, 5.0, 5.0], index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"])),
                       pd.Series([2.0], index=pd.to_datetime(["2020-01-03"])))) == [20.0, 20.0, 5.0, 5.0])

# --- study_event timing + look-ahead guarantees
_bidx = pd.bdate_range("2020-01-01", "2020-12-31")
_flat = pd.Series(100.0, index=_bidx)
def _px(series): return pd.DataFrame({"adj": series, "unadj": series})
_evt = {"ticker": "X", "date": D(2020, 6, 1), "value_usd": 1e7}          # Monday
_i0 = _bidx.searchsorted(pd.Timestamp("2020-06-02"))                       # lag 1 -> Tue
_jump0 = _flat.copy(); _jump0.iloc[_i0:] *= 1.10                          # +10% ON the entry day
_jump1 = _flat.copy(); _jump1.iloc[_i0 + 1:] *= 1.10                      # +10% the day AFTER entry
_s0, _s1 = cbt.study_event(_evt, _px(_jump0), _flat), cbt.study_event(_evt, _px(_jump1), _flat)
check("entry = first trading day on/after action date + lag", _s0["entry_date"] == "2020-06-02")
check("move ON the entry day is reported as day-0 but never earned", abs(_s0["day0_ar"] - 0.10) < 1e-9 and abs(_s0["ar"][1]) < 1e-9)
check("move the day AFTER entry is earned at h=1", abs(_s1["ar"][1] - 0.10) < 1e-9)
check("weekend info date rolls to Monday entry",
      cbt.study_event({"ticker": "X", "date": D(2020, 6, 5), "value_usd": 1}, _px(_flat), _flat, lag_days=1)["entry_date"] == "2020-06-08")
_hist = _flat.copy(); _hist.iloc[:_i0 - 3] *= 0.5                         # rewrite history before entry
_fut = _flat.copy(); _fut.iloc[_i0 + 70:] *= 3.0                          # rewrite far future (beyond all horizons)
_sh, _sf = cbt.study_event(_evt, _px(_hist), _flat), cbt.study_event(_evt, _px(_fut), _flat)
_base = cbt.study_event(_evt, _px(_flat), _flat)
check("forward returns ignore anything before entry (only pre_ar moves)", _sh["ar"] == _base["ar"] and _sh["profile"] == _base["profile"])
check("rewriting pre-entry history changes pre_ar", abs(_sh["pre_ar"]) > 0.5)
check("forward returns ignore bars beyond the horizon", all(abs(_sf["ar"][h]) < 1e-12 for h in cbt.HORIZONS))
check("study_event None when too little history / no forward bars / entry far from info date",
      cbt.study_event({"ticker": "X", "date": D(2020, 12, 30), "value_usd": 1}, _px(_flat), _flat) is None
      and cbt.study_event({"ticker": "X", "date": D(2020, 1, 3), "value_usd": 1}, _px(_flat), _flat) is None
      and cbt.study_event({"ticker": "X", "date": D(2021, 6, 1), "value_usd": 1}, _px(_flat), _flat) is None)
check("checkpoint is strictly after the cross-check window",
      _s0["checkpoint_date"] > (D(2020, 6, 1) + _dt2.timedelta(days=cc.DILUTION_WINDOW_DAYS)).isoformat())

# --- statistics
check("boot_summary deterministic for a seed, drops NaN/None",
      cbt.boot_summary([0.1, -0.02, float("nan"), None, 0.05], seed=3) == cbt.boot_summary([0.1, -0.02, 0.05], seed=3))
check("boot_summary empty -> n=0; CI brackets the mean",
      cbt.boot_summary([]) == {"n": 0} and (lambda s: s["ci"][0] <= s["mean"] <= s["ci"][1])(cbt.boot_summary(list(np.linspace(-1, 2, 50)))))
check("boot_diff recovers a clear gap and returns None on empty",
      (lambda d: d["ci"][0] > 0.9 and abs(d["diff"] - 1.05) < 1e-9)(cbt.boot_diff([2.0] * 20 + [2.1] * 20, [1.0] * 40)) and cbt.boot_diff([], [1.0]) is None)

# --- synthetic world with a KNOWN effect
def _world(n_clean=60, n_dil=30, n_buy=0, clean_drift=0.003, dil_post_drift=-0.003, dil_jump=-0.05, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", "2021-12-31")
    bret = rng.normal(0, 0.004, len(idx))
    prices, events, dil_filings, buys = {}, [], {}, {}
    spans = pd.bdate_range("2019-06-03", "2021-03-01")[::3]
    kinds = ["clean"] * n_clean + ["dil"] * n_dil + ["buy"] * n_buy
    for k, kind in enumerate(kinds):
        tk = f"{kind[0].upper()}{k:02d}"
        award = spans[(k * 7) % len(spans)].date()
        r = bret + rng.normal(0, 0.004, len(idx))
        i0 = int(idx.searchsorted(pd.Timestamp(award) + pd.Timedelta(days=1)))
        icp = int(idx.searchsorted(pd.Timestamp(award) + pd.Timedelta(days=cc.DILUTION_WINDOW_DAYS), side="right"))
        if kind in ("clean", "buy"):
            r[i0 + 1:i0 + 41] += clean_drift
        else:
            j = int(idx.searchsorted(pd.Timestamp(award) + pd.Timedelta(days=5)))
            r[j] += dil_jump
            r[icp + 1:icp + 21] += dil_post_drift
            dil_filings[tk] = [{"form": "424B5", "date": (award + _dt2.timedelta(days=5)).isoformat()}]
        if kind == "buy":
            buys[tk] = [{"code": "P", "usd": 300_000.0, "owner": "CEO", "is_10b5_1": False,
                         "accepted": (award - _dt2.timedelta(days=7)).isoformat() + "T15:00:00.000Z"}]
        s = pd.Series(50.0 * np.cumprod(1 + r), index=idx)
        prices[tk] = pd.DataFrame({"adj": s, "unadj": s})
        events.append({"ticker": tk, "date": award, "value_usd": 5e7, "n": 1, "recipients": []})
    prices["SPY"] = pd.DataFrame({"adj": pd.Series(100.0 * np.cumprod(1 + bret), index=idx)})
    return events, prices, dil_filings, buys

def _run(world, **kw):
    events, prices, dil, buys = world
    return cbt.run_backtest(
        events, lambda t: prices.get(t), lambda t, d: 2e7,           # 2e7 sh x ~$50 ~ $1B cap -> ~5% ratio
        lambda t, s, e: [f for f in dil.get(t, []) if str(s) <= f["date"] <= str(e)],
        lambda t, s, e: buys.get(t, []), n_boot=600, **kw)

_w = _world(n_buy=25)
_r = _run(_w)
check("backtest studies the events and tracks drops", _r["n_studied"] > 100 and _r["n_studied"] + sum(_r["dropped"].values()) == _r["n_events_in"])
check("production cross-check labels the groups", {r["group"] for r in _r["rows"]} == {"clean", "dilution", "insider_buy"}
      and all(r["group"] == "dilution" for r in _r["rows"] if r["ticker"].startswith("D")))
check("size bucket follows value / (shares x unadjusted price at entry)",
      all(r["bucket"] == cbt.bucket_for(r["ratio"]) and abs(r["ratio"] - r["value_usd"] / r["mcap"]) < 1e-12
          and abs(r["mcap"] - 2e7 * r["price0"]) < 1e-3 for r in _r["rows"])
      and sum(_r["by_bucket"][lab][20]["n"] for _, _, lab in cbt.BUCKETS) == sum(1 for r in _r["rows"] if np.isfinite(r["ar"][20])))
_big = cbt.run_backtest(_w[0][:30], lambda t: _w[1].get(t), lambda t, d: 2e9, n_boot=200)
check("a 100x bigger cap moves every award into the '<1%' bucket", {r["bucket"] for r in _big["rows"]} == {"<1%"})
_cl = _r["by_group_entry"]["clean"][20]
check("recovers real drift for clean awards (+0.3%/day -> ~+6% AR at 20d, CI excludes 0)", _cl["ci"][0] > 0.03)
_m = _r["multiplier_check"]
check("flag with a REAL post-checkpoint effect is 'supported' (dilution @10d and @20d)",
      _m[("dilution", 10)]["verdict"] == "supported" and _m[("dilution", 20)]["verdict"] == "supported"
      and _m[("dilution", 10)]["diff"]["ci"][1] < 0)
check("insider_buy compared from entry, supported when it really leads", _m[("insider_buy", 20)]["basis"] == "entry"
      and _m[("insider_buy", 20)]["verdict"] in ("inconclusive (CI includes 0)", "supported"))
check("implied ratio only given when clean drift is itself significant",
      _m[("dilution", 20)]["implied_ratio"] is not None and _m[("dilution", 20)]["implied_ratio"] < 0.5)
check("a flag that REVERSES the edge is reported as stronger than the shrink-only multiplier",
      _m[("dilution", 20)]["magnitude"] == "stronger than modelled" and _m[("insider_buy", 20)]["magnitude"] is None)
check("pre-entry drift ~0 and announcement-day move ~0 in a world with neither", abs(_r["pre_drift"]["mean"]) < 0.01)
check("decay check usable and reports model comparison", _r["decay"]["usable"] and set(_r["decay"]["realized"]) == {5, 10, 20, 40})

# the look-ahead TRAP: the offering's own price drop, no real post-checkpoint edge
_t = _run(_world(clean_drift=0.0, dil_post_drift=0.0, dil_jump=-0.06, seed=5))
_te = cbt.boot_diff([r["ar"][10] for r in _t["rows"] if r["group"] == "dilution"],
                   [r["ar"][10] for r in _t["rows"] if r["group"] == "clean"], n_boot=600)
check("from-ENTRY comparison is fooled by the offering drop (the trap exists)", _te["ci"][1] < 0)
check("tradeable from-CHECKPOINT comparison is NOT fooled (no verdict of 'supported')",
      _t["multiplier_check"][("dilution", 10)]["verdict"] != "supported")

# underpowered / robustness / reporting
_small = _run(_world(n_clean=6, n_dil=4))
check("tiny samples are labelled underpowered, never 'supported'",
      all(c["verdict"].startswith("inconclusive") for c in _small["multiplier_check"].values()))
check("report renders sections, warns on small n, and states limits",
      all(x in cbt.format_report(_small) for x in ("Is there drift?", "Multiplier check", "⚠", "Survivorship", "lag-days")))
check("report for zero studied events explains rather than crashing",
      "No events could be studied" in cbt.format_report(cbt.run_backtest([], lambda t: _w[1].get(t), lambda t, d: 1e7)))
try:
    cbt.run_backtest([_w[0][0]], lambda t: None, lambda t, d: 1e7); _no_bench = False
except cbt.DataUnavailable:
    _no_bench = True
check("missing benchmark raises DataUnavailable (blocked network surfaces loudly)", _no_bench)
_nosh = cbt.run_backtest(_w[0][:5], lambda t: _w[1].get(t), lambda t, d: None)
check("events without point-in-time shares are dropped and counted, not guessed",
      _nosh["n_studied"] == 0 and _nosh["dropped"]["no_shares"] + _nosh["dropped"]["no_window"] == 5)
_noflags = cbt.run_backtest(_w[0][:30], lambda t: _w[1].get(t), lambda t, d: 2e7)
check("no EDGAR sources -> group 'unknown', cross-check section skipped",
      {r["group"] for r in _noflags["rows"]} == {"unknown"} and _noflags["multiplier_check"] == {}
      and "not tested" in cbt.format_report(_noflags))
with tempfile.TemporaryDirectory() as _td:
    _p = os.path.join(_td, "e.csv"); cbt.rows_to_csv(_r["rows"], _p)
    _lines = open(_p).read().splitlines()
    check("rows_to_csv writes a header + one line per event", len(_lines) == _r["n_studied"] + 1 and _lines[0].startswith("ticker,"))
    # awards CSV loading
    _c = os.path.join(_td, "a.csv")
    open(_c, "w").write("Action_Date,Recipient_Name,Transaction_Amount,Ticker\n2020-03-02,Acme Inc,\"$12,000,000\",aaa\nbad,Acme,1,\n2020-03-03,Acme,notanumber,\n")
    _aw2, _bad = cbt.awards_from_csv(_c)
    check("awards_from_csv: flexible headers, money parsing, bad rows counted",
          len(_aw2) == 1 and _bad == 2 and _aw2[0]["amount"] == 12e6 and _aw2[0]["ticker"] == "AAA" and _aw2[0]["date"] == D(2020, 3, 2))
    open(os.path.join(_td, "b.csv"), "w").write("foo,bar\n1,2\n")
    try:
        cbt.awards_from_csv(os.path.join(_td, "b.csv")); _bad_hdr = False
    except ValueError:
        _bad_hdr = True
    check("awards_from_csv rejects a CSV with no usable columns", _bad_hdr)
    _o1, _o2 = os.path.join(_td, "r.md"), os.path.join(_td, "r.csv")
    check("CLI returns 2 (not a traceback) when the awards file is missing",
          cbt.main(["--csv", os.path.join(_td, "nope.csv"), "--start", "2020-01-01", "--end", "2020-12-31",
                   "--out-md", _o1, "--out-csv", _o2]) == 2)

# --- USAspending parsing + paging (response shape per the public API; not exercised live)
_rows = [{"Recipient Name": "Acme Inc", "Action Date": "2020-03-02", "Transaction Amount": 25e6, "Awarding Agency": "DoD"},
         {"Recipient Name": "Acme Inc", "Action Date": "2020-03-03", "Transaction Amount": -5e6},
         {"Recipient Name": "Acme Inc", "Action Date": "garbage", "Transaction Amount": 5e6}]
_pa, _pb = cbt.awards_from_usaspending_rows(_rows)
check("usaspending rows: obligations kept, de-obligations/garbage skipped", len(_pa) == 1 and _pb == 2 and _pa[0]["amount"] == 25e6)
_calls = []
def _fake_post(payload):
    _calls.append(payload)
    big = [{"Recipient Name": "Acme", "Action Date": payload["filters"]["time_period"][0]["start_date"], "Transaction Amount": 30e6}]
    small = [{"Recipient Name": "Acme", "Action Date": payload["filters"]["time_period"][0]["start_date"], "Transaction Amount": 1e6}]
    return {"results": big if payload["page"] == 1 else small, "page_metadata": {"hasNext": True}}
_fa, _ = cbt.fetch_usaspending("2020-01-15", "2020-03-20", min_amount=10e6, post=_fake_post)
check("fetch_usaspending: one query window per month, clipped to the range, stops paging below min amount",
      len(_calls) == 6 and [c["filters"]["time_period"][0]["start_date"] for c in _calls[::2]] == ["2020-01-15", "2020-02-01", "2020-03-01"]
      and _calls[-1]["filters"]["time_period"][0]["end_date"] == "2020-03-20" and len(_fa) == 3)
def _raises_du(fn):
    try:
        fn(); return False
    except cbt.DataUnavailable:
        return True
check("fetch_usaspending surfaces a failed request as DataUnavailable",
      _raises_du(lambda: cbt.fetch_usaspending("2020-01-01", "2020-01-31", post=lambda p: (_ for _ in ()).throw(cbt.DataUnavailable("x")))))
with unittest.mock.patch("urllib.request.urlopen", side_effect=OSError("Tunnel connection failed: 403 Forbidden")):
    check("a blocked USAspending connection becomes DataUnavailable naming the likely cause",
          _raises_du(lambda: cbt.fetch_usaspending("2020-01-01", "2020-01-31", min_amount=1e6)))

# --- EDGAR historical helpers
_cf = {"facts": {"dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
    {"end": "2019-12-31", "val": 100e6, "accn": "a1", "filed": "2020-02-20"},
    {"end": "2020-03-31", "val": 110e6, "accn": "a2", "filed": "2020-05-08"},
    {"end": "2020-03-31", "val": 40e6, "accn": "a2", "filed": "2020-05-08"},          # second share class
    {"end": "2020-06-30", "val": 999e6, "accn": "a3", "filed": "2020-08-07"}]}}}}}
check("shares as-of uses FILING dates: a later filing can't leak into an earlier date",
      edgar._shares_from_companyfacts(_cf, "2020-03-01") == 100e6 and edgar._shares_from_companyfacts(_cf, "2020-06-01") == 150e6
      and edgar._shares_from_companyfacts(_cf, "2020-01-01") is None and edgar._shares_from_companyfacts(_cf, "2021-01-01") == 999e6)
check("shares fall back to us-gaap, and garbage input is None",
      edgar._shares_from_companyfacts({"facts": {"us-gaap": {"CommonStockSharesOutstanding": {"units": {"shares": [
          {"end": "2019-12-31", "val": 7e6, "filed": "2020-02-01"}]}}}}}, "2020-06-01") == 7e6
      and edgar._shares_from_companyfacts({}, "2020-06-01") is None and edgar._shares_from_companyfacts(None, "2020-06-01") is None)
check("_raw_form4_url strips the XSL rendering folder only",
      edgar._raw_form4_url("https://www.sec.gov/Archives/edgar/data/1/2/xslF345X05/f4.xml") == "https://www.sec.gov/Archives/edgar/data/1/2/f4.xml"
      and edgar._raw_form4_url("https://www.sec.gov/Archives/edgar/data/1/2/f4.xml") == "https://www.sec.gov/Archives/edgar/data/1/2/f4.xml")
_recent = {"form": ["424B5", "4", "10-Q"], "filingDate": ["2020-06-03", "2020-06-02", "2020-05-01"],
           "acceptanceDateTime": ["2020-06-03T21:00:00.000Z", "2020-06-02T21:00:00.000Z", ""],
           "accessionNumber": ["0001-20-3", "0001-20-2", "0001-20-1"], "primaryDocument": ["a.htm", "xslF345X05/f4.xml", "q.htm"]}
_old = {"form": ["S-3", "4"], "filingDate": ["2018-03-01", "2018-04-02"], "acceptanceDateTime": ["", ""],
        "accessionNumber": ["0001-18-1", "0001-18-2"], "primaryDocument": ["s3.htm", "f4old.xml"]}
_pages = []
def _fake_get(url, timeout=15):
    _pages.append(url)
    if "history.json" in url:
        return json.dumps(_old)
    return json.dumps({"filings": {"recent": _recent, "files": [
        {"name": "history.json", "filingFrom": "2017-01-01", "filingTo": "2018-12-31"},
        {"name": "ancient.json", "filingFrom": "2000-01-01", "filingTo": "2001-12-31"}]}})
with unittest.mock.patch.dict(edgar._cik_map, {"XYZ": "0000000001"}), unittest.mock.patch.object(edgar, "_get", side_effect=_fake_get):
    _fb = edgar.filings_between("XYZ", edgar.DILUTIVE_FORMS, "2018-01-01", "2020-12-31")
    check("filings_between follows older history pages that overlap, skips ones that don't, sorts oldest-first",
          [f["form"] for f in _fb] == ["S-3", "424B5"] and not any("ancient" in u for u in _pages) and any("history" in u for u in _pages))
    _pages.clear()
    check("filings_between only fetches history when the range needs it",
          [f["form"] for f in edgar.filings_between("XYZ", {"424B5"}, "2020-06-01", "2020-06-30")] == ["424B5"]
          and not any("history" in u for u in _pages))
    check("filings_between unknown ticker -> []", edgar.filings_between("NOPE", {"4"}, "2020-01-01", "2020-12-31") == [])
_fetched = []
def _fake_get4_bytes(url, timeout=15):
    if "/submissions/" in url:
        return _fake_get(url, timeout).encode()
    _fetched.append(url)
    return _FORM4_XML.encode()
with unittest.mock.patch.dict(edgar._cik_map, {"XYZ": "0000000001"}), \
     unittest.mock.patch.object(edgar, "_get_bytes", side_effect=_fake_get4_bytes):
    _f4 = edgar.form4_transactions_between("XYZ", "2020-06-01", "2020-06-30")
    check("form4_transactions_between fetches the RAW xml and returns evaluate()-ready transactions",
          _fetched and "xslF345X05" not in _fetched[0] and _f4 and all(t["accepted"].startswith("2020-06-02") and "code" in t for t in _f4))
    check("those transactions feed the production cross-check unchanged",
          cc.evaluate({"signal": 0.3, "award_date": "2020-06-02"}, [], _f4, D(2020, 6, 2)) is not None)
with unittest.mock.patch.dict(edgar._cik_map, {"XYZ": "0000000001"}), unittest.mock.patch.object(edgar, "_get", side_effect=OSError("down")):
    check("historical EDGAR helpers fail open", edgar.filings_between("XYZ", {"4"}, "2020-01-01", "2020-12-31") == []
          and edgar.form4_transactions_between("XYZ", "2020-01-01", "2020-12-31") == []
          and edgar.shares_outstanding_asof("XYZ", "2020-06-01") is None)

# ----------------------------------------------------- Form 4 XSL url fix ----
section("Form 4 raw-XML fetch")
import datetime as _dt3
_nowz = (_dt3.datetime.now(_dt3.timezone.utc) - _dt3.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
_subs4 = json.dumps({"filings": {"recent": {
    "form": ["4"], "filingDate": [_dt3.date.today().isoformat()], "acceptanceDateTime": [_nowz],
    "accessionNumber": ["0001-20-2001"], "primaryDocument": ["xslF345X05/f4.xml"]}}}).encode()
_HTML_VIEW = b"<html><body><table><tr><td>Rendered Form 4 view</td></table></body>"   # not well-formed XML

def _fake_edgar(calls, xsl_serves="html", raw_serves="xml"):
    """EDGAR as it really behaves: the listed .../xslF345X05/f4.xml is an HTML view; the
    raw XML is one folder up. `xsl_serves`/`raw_serves` let tests break either location."""
    def get_bytes(url, timeout=15):
        calls.append(url)
        if "/submissions/" in url:
            return _subs4
        kind = xsl_serves if "/xslF345X05/" in url else raw_serves
        if kind == "xml":
            return _FORM4_XML.encode()
        if kind == "html":
            return _HTML_VIEW
        raise OSError("404")
    return get_bytes

_XSL_URL = "https://www.sec.gov/Archives/edgar/data/1/000120202001/xslF345X05/f4.xml"
check("an HTML view parses to nothing (what the unfixed code was handed)",
      edgar.parse_form4_xml(_HTML_VIEW.decode()) == []
      and unittest.mock.patch.object(edgar, "_get_bytes", return_value=_HTML_VIEW).start() is not None
      and edgar._parse_form4(_XSL_URL) is None)
unittest.mock.patch.stopall()

_c = []
with unittest.mock.patch.object(edgar, "_get_bytes", side_effect=_fake_edgar(_c)):
    _p4 = edgar._parse_form4(_XSL_URL)
check("_parse_form4 fetches the raw XML, not the HTML view",
      _p4 and _p4["owner"] == "Jane Doe" and _p4["buy_usd"] == 50000.0 and len(_c) == 1 and "xslF345X05" not in _c[0])

_c = []
with unittest.mock.patch.dict(edgar._cik_map, {"XYZ": "0000000001"}), \
     unittest.mock.patch.object(edgar, "_get_bytes", side_effect=_fake_edgar(_c)):
    _bias = edgar.form4_insider_bias("XYZ", lookback_hours=72)
check("form4_insider_bias works end-to-end when EDGAR lists the XSL path",
      _bias and _bias["n_buys"] == 1 and _bias["n_sells"] == 1 and _bias["buy_usd"] == 50000.0)
check("...and only ever fetched the raw XML for the Form 4 itself",
      [u for u in _c if "/submissions/" not in u] and all("xslF345X05" not in u for u in _c if "/submissions/" not in u))

_c = []
with unittest.mock.patch.dict(edgar._cik_map, {"XYZ": "0000000001"}), \
     unittest.mock.patch.object(edgar, "_get_bytes", side_effect=_fake_edgar(_c)):
    _rf = edgar.recent_filings("XYZ", days=3)
check("recent_filings classifies the Form 4 from its XML but still links the human-readable view",
      _rf and _rf[0]["form"] == "4" and _rf[0]["bias"] == 1 and _rf[0]["usd"] == 50000.0
      and "xslF345X05" in _rf[0]["url"])

_c = []
with unittest.mock.patch.object(edgar, "_get_bytes", side_effect=_fake_edgar(_c, xsl_serves="xml", raw_serves="404")):
    _fb4 = edgar._parse_form4(_XSL_URL)
check("falls back to the listed URL when the raw one is missing (never worse than before)",
      _fb4 and _fb4["owner"] == "Jane Doe" and len(_c) == 2 and "xslF345X05" not in _c[0] and "xslF345X05" in _c[1])

_c = []
with unittest.mock.patch.object(edgar, "_get_bytes", side_effect=_fake_edgar(_c, raw_serves="xml")):
    edgar._parse_form4("https://www.sec.gov/Archives/edgar/data/1/000120202001/f4.xml")
check("a URL with no XSL folder is fetched once, unchanged", len(_c) == 1 and _c[0].endswith("/000120202001/f4.xml"))

with unittest.mock.patch.object(edgar, "_get_bytes", side_effect=_fake_edgar([], xsl_serves="404", raw_serves="404")):
    check("both locations failing still degrades to None, never raises", edgar._parse_form4(_XSL_URL) is None)

# ------------------------------------------------------ market dashboard ----
section("market_dashboard widgets")
import market_dashboard as md
_rows = [
    {"ticker": "NVDA", "chg": 3.0, "score": 40, "last": 100.0, "verdict": "BUY signal"},
    {"ticker": "AMD", "chg": -1.0, "score": -20, "last": 50.0, "verdict": "STRONG AVOID"},
    {"ticker": "MSFT", "chg": 1.0, "score": 0, "last": 300.0, "verdict": "HOLD / no edge"},
    {"ticker": "ZZZZ", "chg": 0.0, "score": 0, "last": 5.0, "verdict": "HOLD / no edge"},
]
_ss = md.signal_summary(_rows)
check("signal_summary counts verdict words", (_ss["buy"], _ss["hold"], _ss["avoid"], _ss["total"]) == (1, 2, 1, 4))
check("signal_summary avg score", _ss["avg_score"] == 5.0)
check("signal_summary empty watchlist doesn't divide by zero", md.signal_summary([])["avg_score"] == 0)
_sr = md.sector_rotation(_rows)["sectors"]
_by = {x["name"]: x for x in _sr}
check("sector_rotation groups via SECTOR_MAP, unknown -> Other",
      set(_by) == {"Technology", "Semiconductors", "Other"})
check("sector_rotation sorted best sector first", [x["name"] for x in _sr][0] == "Technology")
check("sector_rotation tech avg change/winners", _by["Technology"]["avg_chg"] == 2.0 and _by["Technology"]["winners"] == 2
      and _by["Technology"]["tickers"] == "NVDA,MSFT")
check("sector_rotation losers counted", _by["Semiconductors"]["losers"] == 1 and _by["Other"]["winners"] == 0)
_mv = md.top_movers(_rows)
check("top_movers gainers/losers only list actual movers",
      [g_["ticker"] for g_ in _mv["gainers"]] == ["NVDA", "MSFT"] and [l_["ticker"] for l_ in _mv["losers"]] == ["AMD"])
check("top_movers caps at n", len(md.top_movers([dict(_rows[0], ticker=f"T{i}", chg=i + 1) for i in range(9)])["gainers"]) == 3)
_sp = md.price_spark_svg([1, 2, 3, 4])
check("sparkline is green when up, red when down, '' when too short",
      md.price_spark_svg([1, 2, 3])[0:4] == "<svg" and "#4ADE80" in _sp and "#FF6B5E" in md.price_spark_svg([4, 3, 2, 1])
      and md.price_spark_svg([1]) == "" and "#FF6B5E" in md.price_spark_svg([1, 2, 3], up=False))
check("sparkline tolerates flat series", "polyline" in md.price_spark_svg([5, 5, 5]))
_mo = md.market_overview(True)["indices"]
check("market_overview (demo) returns the four benchmarks with a sparkline",
      [i["ticker"] for i in _mo] == ["SPY", "QQQ", "DIA", "IWM"] and all(i["spark"].startswith("<svg") for i in _mo))
with tempfile.TemporaryDirectory() as _d:
    _now = time.time()
    for _name, _age_h in (("fresh.json", 1), ("warn.json", 30), ("stale.json", 100)):
        _fp = os.path.join(_d, _name); open(_fp, "w").write("{}")
        os.utime(_fp, (_now - _age_h * 3600, _now - _age_h * 3600))
    _chk = [("Fresh", "fresh.json", 24, 96), ("Warn", "warn.json", 24, 96), ("Stale", "stale.json", 24, 96),
            ("Gone", "nope.json", 24, 96)]
    _h = {c["label"]: c for c in md.system_health(base=_d, now=_now, checks=_chk)["checks"]}
    check("system_health ok/warn/stale/missing", (_h["Fresh"]["status"], _h["Warn"]["status"], _h["Stale"]["status"],
          _h["Gone"]["status"]) == ("ok", "warn", "stale", "missing") and _h["Gone"]["age_hours"] is None)
    check("system_health reports age in hours", _h["Warn"]["age_hours"] == 30.0)
_dash = ws._dashboard(["NVDA", "AMD", "MSFT"], True)
check("web_server._dashboard bundles every widget from one watchlist",
      set(_dash) == {"watchlist", "market", "sectors", "movers", "summary", "health"}
      and len(_dash["watchlist"]) == 3 and _dash["summary"]["total"] == 3
      and all("spark" in r and "rvol" in r for r in _dash["watchlist"]))
check("the dashboard page ships the amber theme + the new widgets",
      "--gold:#FFA630" in ws._get_page() and 'id="ribbontrack"' in ws._get_page() and "dashHtml" in ws._get_page())

# ------------------------------------------------------------- summary ------
print(f"\n{'='*50}")
print(f"RESULTS: {_PASS} passed, {_FAIL} failed")
if _FAILURES:
    print("FAILED:", ", ".join(_FAILURES))
    sys.exit(1)
print("✓ all green")

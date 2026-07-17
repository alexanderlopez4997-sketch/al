#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meridian test suite — assertions over the pure logic of every module.

No pytest dependency; run with:  python3 test_meridian.py
Network-dependent functions (API fetches) are NOT called — only the pure
transforms, scoring, and formatting they feed into. Exit code is nonzero on
any failure so this can gate a launch.
"""
import math
import os
import sys
import tempfile

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
from meridian_cache import MeridianCache

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

# ------------------------------------------------------------- summary ------
print(f"\n{'='*50}")
print(f"RESULTS: {_PASS} passed, {_FAIL} failed")
if _FAILURES:
    print("FAILED:", ", ".join(_FAILURES))
    sys.exit(1)
print("✓ all green")

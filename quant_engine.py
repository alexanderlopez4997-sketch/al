#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUANT ENGINE  —  single-file stock analyzer.

Fetches price history, runs a full technical-indicator stack, scores the
stock across five factors, and issues a transparent BUY / HOLD / AVOID
verdict with position sizing, a rules backtest, and an optional
annealing optimizer that tunes the factor weights (with a train/test
honesty check).

USAGE
  pip3 install yfinance pandas numpy
  python3 quant_engine.py --demo                 # offline, no internet
  python3 quant_engine.py NVDA                    # one full report
  python3 quant_engine.py SDOT TC LHAI            # ranked scan of several
  python3 quant_engine.py NVDA --interval 1h --period 60d
  python3 quant_engine.py NVDA --account 10000 --risk 1
  python3 quant_engine.py NVDA --optimize         # anneal weights, 70/30 test

NOT FINANCIAL ADVICE. Backtests here ignore fees/slippage and are in-sample;
optimized weights can overfit (compare train vs unseen-test Sharpe). Past
patterns do not predict future returns.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# --------------------------------------------------------------- colors ---
COLOR = sys.stdout.isatty()
def C(t, c): return f"\033[{c}m{t}\033[0m" if COLOR else str(t)
def gr(t): return C(t, "32")
def rd(t): return C(t, "31")
def yl(t): return C(t, "33")
def dim(t): return C(t, "90")
def bold(t): return C(t, "1")
def signed(x, fmt="{:+.2f}"):
    s = fmt.format(x)
    return gr(s) if x >= 0 else rd(s)

FINNHUB_DEFAULT_KEY = "d930h31r01qpou38ope0d930h31r01qpou38opeg"
FACTORS = ["Direction", "Momentum", "Volume", "MeanRev"]
BASE_WEIGHTS = {"Direction": 0.38, "Momentum": 0.27, "Volume": 0.20, "MeanRev": 0.15}

REGIME_WEIGHTS = {
    "bull": {"Direction": 0.45, "Momentum": 0.30, "Volume": 0.15, "MeanRev": 0.10},
    "bear": {"Direction": 0.30, "Momentum": 0.20, "Volume": 0.15, "MeanRev": 0.35},
    "range": {"Direction": 0.20, "Momentum": 0.15, "Volume": 0.20, "MeanRev": 0.45},
}

REGIME_THRESHOLDS = {
    "bull": {"enter": 16.0, "strong": 40.0},
    "bear": {"enter": 20.0, "strong": 50.0},
    "range": {"enter": 15.0, "strong": 38.0},
}

SLIPPAGE_PCT = 0.05
PPY = {"1d": 252, "1wk": 52, "1mo": 12, "1h": 252 * 7, "60m": 252 * 7,
       "90m": 252 * 4, "30m": 252 * 13, "15m": 252 * 26, "5m": 252 * 78,
       "3m": 252 * 130, "2m": 252 * 195, "1m": 252 * 390}   # 390 min/session
INTRADAY_INTERVALS = {"1m", "2m", "3m", "5m", "15m", "30m", "1h", "60m", "90m"}

def is_crypto(ticker):
    """True for crypto symbols (BTC/USD, BTC-USD, ETH/USD…). Crypto trades 24/7,
    so it has no market sessions — the overnight-gap handling must NOT apply."""
    t = (ticker or "").upper()
    return ("/" in t) or t.endswith(("-USD", "-USDT", "-USDC"))
ENTER, EXIT = 18.0, 0.0
RISKY_ATR_PCT = 8.0
MIN_BARS = 40       # hard floor — below this no read is meaningful
STABLE_BARS = 60    # below this (but ≥ MIN_BARS) signals fire with a LIMITED-HISTORY flag

# A curated universe of liquid US names (screener discovery fallback and the
# default leaderboard universe). Engine-level so headless scripts can use it.
UNIVERSE_LIQUID = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "AVGO", "NFLX",
    "ADBE", "CRM", "ORCL", "INTC", "QCOM", "CSCO", "TXN", "MU", "AMAT", "PLTR",
    "SMCI", "ARM", "PANW", "SNOW", "NOW", "SHOP", "UBER", "ABNB", "COIN", "SQ",
    "PYPL", "SOFI", "HOOD", "DKNG", "RBLX", "U", "DDOG", "NET", "CRWD", "ZS",
    "MDB", "TTD", "MRVL", "ON", "MPWR", "LRCX", "KLAC", "ASML", "TSM", "SMH",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "V", "MA", "AXP",
    "BRK-B", "BLK", "SPGI", "COF", "USB", "PNC", "JNJ", "UNH", "LLY", "PFE",
    "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY", "AMGN", "GILD", "CVS", "MRNA",
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "PSX", "WMT", "COST",
    "HD", "LOW", "TGT", "NKE", "SBUX", "MCD", "CMG", "DIS", "KO", "PEP",
    "PG", "CAT", "DE", "BA", "GE", "HON", "LMT", "RTX", "UPS", "FDX",
    "F", "GM", "RIVN", "LCID", "T", "VZ", "TMUS", "CMCSA", "SPY", "QQQ",
]
# Alternative-data tilt: max points alt-data can move the live score
# (never the backtest — that stays technical-only to avoid lookahead).
# Insiders get the biggest weight: dollar-weighted open-market exec buying is
# the best-documented of these signals. Whale flow is the noisiest.
ALT_MAX_TILT = 15.0
ALT_WEIGHTS = {"Insider": 0.28, "Analyst": 0.22, "Congress": 0.20,
               "Macro": 0.15, "WhaleFlow": 0.15}

# ================================================================ PIPELINE ===
# Clean modular architecture:
#   Market Data → Regime Detector → Factor Analyzer → Risk Engine → Metrics
# Each stage transforms input and passes to next. All state is explicit.

class PipelineData:
    """Explicit data container flowing through pipeline stages."""
    def __init__(self, ticker, df):
        self.ticker = ticker
        self.df = df
        self.d = None           # enriched data (indicators)
        self.regime = None      # market regime classification
        self.F = None           # factor matrix
        self.comp = None        # composite score
        self.account = None     # account metrics (size, risk %)

class MarketDataStage:
    """Stage 1: Load and enrich market data with technical indicators."""
    @staticmethod
    def execute(data):
        data.d = enrich(data.df)
        return data

class RegimeDetectorStage:
    """Stage 2: Classify market regime (BULL/BEAR/RANGE)."""
    @staticmethod
    def execute(data):
        data.regime = detect_regime(data.d)
        return data

class FactorAnalyzerStage:
    """Stage 3: Compute factor scores and correlations."""
    @staticmethod
    def execute(data):
        data.F = factor_matrix(data.d)
        data.comp = composite(data.F, regime=data.regime)
        data.ir = information_ratio(data.F, data.d["Close"])
        data.corr = factor_correlation(data.F)
        return data

class RiskEngineStage:
    """Stage 4: Position sizing, risk metrics, backtests."""
    @staticmethod
    def execute(data, account=None, risk_pct=1.0):
        data.account = account
        data.risk_pct = risk_pct
        score = float(data.comp.iloc[-1])
        last_price = float(data.d["Close"].iloc[-1])
        atr = float(data.d["atr"].iloc[-1])

        if account:
            data.position = position_size(last_price, atr, account, risk_pct)
        data.backtest = backtest(data.d["Close"], data.comp, intraday=False)
        data.backtest_by_regime = backtest_by_regime(data.d["Close"], data.comp, data.d)
        return data

def detect_red_flags(res):
    """Detect major red flags: insider selling, offerings, negative catalysts.
    Returns {has_flags: bool, flags: [list], risk_level: 'LOW'|'MEDIUM'|'HIGH'}."""
    flags = []
    score = 0

    insiders = res.get("insiders", {})
    if insiders:
        buy_usd = insiders.get("buy_usd", 0)
        sell_usd = insiders.get("sell_usd", 0)
        if sell_usd > 0 and buy_usd == 0:
            ratio = sell_usd / max(buy_usd, 1)
            if sell_usd > 1e6:
                flags.append(f"Insider selling: ${sell_usd/1e6:.1f}M sold, ${buy_usd/1e6:.1f}M bought")
                score += 2
        if buy_usd > sell_usd and buy_usd > 0.5e6:
            flags.append(f"Insider buying: ${buy_usd/1e6:.1f}M accumulated")
            score -= 1

    news_tone = res.get("macro_signal", {}).get("signal", 0) if res.get("macro_signal") else 0
    if news_tone < -0.4:
        flags.append(f"Negative news tone: {news_tone:.2f}")
        score += 1
    elif news_tone > 0.4:
        flags.append(f"Positive news tone: {news_tone:.2f}")
        score -= 1

    backtest = res.get("bt", {})
    if backtest.get("sharpe", 0) < -0.5:
        flags.append(f"Poor backtest: Sharpe {backtest.get('sharpe', 0):.2f}")
        score += 1

    verdict = res.get("verdict", {})
    if verdict.get("tone") == "bad":
        flags.append(f"Sell signal: {verdict.get('label')}")
        score += 1

    risk_level = "HIGH" if score >= 3 else "MEDIUM" if score >= 1 else "LOW"
    return {
        "has_flags": len(flags) > 0,
        "flags": flags,
        "risk_level": risk_level,
        "score": score
    }

def categorize_portfolio(results):
    """Categorize stocks into BUY/RISK/NEUTRAL based on signals and catalysts.
    Returns {buy: [list], risk: [list], neutral: [list]}."""
    buy_candidates = []
    risk_avoid = []
    neutral = []

    for res in results:
        score = res.get("score", 0)
        verdict = res.get("verdict", {})
        confirmation = res.get("confirmation", {})
        red_flags = detect_red_flags(res)

        is_buy = (
            score >= 40 and
            verdict.get("tone") == "good" and
            confirmation.get("level") in ("FULL", "PARTIAL") and
            not red_flags["has_flags"]
        )

        is_risk = (
            score <= -20 or
            verdict.get("tone") == "bad" or
            red_flags["risk_level"] == "HIGH"
        )

        res["red_flags"] = red_flags

        if is_buy:
            buy_candidates.append(res)
        elif is_risk:
            risk_avoid.append(res)
        else:
            neutral.append(res)

    return {
        "buy": sorted(buy_candidates, key=lambda x: -x["score"]),
        "risk": sorted(risk_avoid, key=lambda x: -x["score"]),
        "neutral": sorted(neutral, key=lambda x: -x["score"])
    }

def confirmation_checklist(score, verdict, conviction, backtest, fwd_stats, buy_th, strong_th):
    """Multi-criteria validation checklist for signal reliability.
    Returns {passed: int, total: int, checks: [list], level: 'FULL'|'PARTIAL'|'WEAK'}."""
    checks = []

    checks.append({
        "name": "Verdict is BUY / STRONG BUY",
        "pass": verdict.get("tone") in ("good",),
        "detail": verdict.get("label", "unknown")
    })

    checks.append({
        "name": "Conviction >= 60%",
        "pass": conviction >= 60,
        "detail": f"{conviction}%"
    })

    checks.append({
        "name": f"Score clears threshold (>={buy_th:.0f})",
        "pass": score >= buy_th,
        "detail": f"{score:+.0f}"
    })

    checks.append({
        "name": "Backtest Sharpe positive",
        "pass": backtest.get("sharpe", float("nan")) > 0,
        "detail": f"{backtest.get('sharpe', 0):.2f}"
    })

    checks.append({
        "name": "Score has positive edge here",
        "pass": fwd_stats.get("edge", 0) > 0 if fwd_stats else None,
        "detail": f"edge {fwd_stats.get('edge', 0):+.2f}" if fwd_stats else "no history"
    })

    checks.append({
        "name": "Winning streak (win rate > 50%)",
        "pass": backtest.get("winrate", 0) > 0.50 if not math.isnan(backtest.get("winrate", float("nan"))) else None,
        "detail": f"{backtest.get('winrate', 0)*100:.0f}%" if not math.isnan(backtest.get("winrate", float("nan"))) else "n/a"
    })

    checks.append({
        "name": "Exposure >= 25% (not sitting idle)",
        "pass": backtest.get("exposure", 0) >= 0.25,
        "detail": f"{backtest.get('exposure', 0)*100:.0f}%"
    })

    checks.append({
        "name": "Max drawdown < 40%",
        "pass": backtest.get("maxdd", 0) > -0.40,
        "detail": f"{backtest.get('maxdd', 0)*100:.0f}%"
    })

    passed = sum(1 for c in checks if c["pass"] is True)
    total = sum(1 for c in checks if c["pass"] is not None)
    pct = (passed / total * 100) if total > 0 else 0

    if pct >= 75:
        level = "FULL"
    elif pct >= 50:
        level = "PARTIAL"
    else:
        level = "WEAK"

    return {
        "passed": passed,
        "total": total,
        "pct": pct,
        "level": level,
        "checks": [c for c in checks if c["pass"] is not None]
    }

class MetricsStage:
    """Stage 5: Compute final verdict and performance metrics."""
    @staticmethod
    def execute(data):
        score = float(data.comp.iloc[-1])
        last = float(data.d["Close"].iloc[-1])
        atr_pct = float(data.d["atr"].iloc[-1] / last * 100)

        buy_th = 18.0
        strong_th = 45.0
        if data.regime and data.regime.get("confidence", 0) > 0.5:
            buy_th = REGIME_THRESHOLDS.get(data.regime["regime"], {}).get("enter", buy_th)
            strong_th = REGIME_THRESHOLDS.get(data.regime["regime"], {}).get("strong", strong_th)

        data.verdict = verdict(score, atr_pct, buy_th, strong_th, data.regime)
        data.score = score
        data.last = last
        data.atr_pct = atr_pct
        data.chg = float((last/data.d["Close"].iloc[-2]-1)*100)
        data.conviction = conviction(data.F.iloc[-1], score)
        return data

class Pipeline:
    """Orchestrate stages: Market Data → Regime → Factors → Risk → Metrics."""
    @staticmethod
    def execute(ticker, df, account=None, risk_pct=1.0):
        data = PipelineData(ticker, df)
        data = MarketDataStage.execute(data)
        data = RegimeDetectorStage.execute(data)
        data = FactorAnalyzerStage.execute(data)
        data = RiskEngineStage.execute(data, account, risk_pct)
        data = MetricsStage.execute(data)
        return data

# ----------------------------------------------------------- indicators ---
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def _ema_multi(close, spans):
    """Batch EMA computation for multiple spans — avoid redundant iterations."""
    return {n: ema(close, n) for n in spans}

def rsi(close, n=14):
    d = close.diff()
    ag = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    al = (-d).clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50.0)

def atr(df, n=14):
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"]-df["Low"], (df["High"]-pc).abs(),
                    (df["Low"]-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def supertrend(df, n=10, mult=3.0):
    a = atr(df, n).values
    hl2 = ((df["High"]+df["Low"])/2).values
    close = df["Close"].values
    ub, lb = hl2+mult*a, hl2-mult*a
    fu, fl = ub.copy(), lb.copy()
    st = np.zeros(len(df)); dr = np.ones(len(df), dtype=int)
    st[0], dr[0] = ub[0], -1
    for i in range(1, len(df)):
        fu[i] = ub[i] if (ub[i] < fu[i-1] or close[i-1] > fu[i-1]) else fu[i-1]
        fl[i] = lb[i] if (lb[i] > fl[i-1] or close[i-1] < fl[i-1]) else fl[i-1]
        if st[i-1] == fu[i-1] and close[i] <= fu[i]:
            st[i], dr[i] = fu[i], -1
        elif st[i-1] == fu[i-1] and close[i] > fu[i]:
            st[i], dr[i] = fl[i], 1
        elif st[i-1] == fl[i-1] and close[i] >= fl[i]:
            st[i], dr[i] = fl[i], 1
        else:
            st[i], dr[i] = fu[i], -1
    return pd.Series(st, index=df.index), pd.Series(dr, index=df.index)

def obv(df):
    return (np.sign(df["Close"].diff().fillna(0)) * df["Volume"]).cumsum()

def enrich(df):
    d = df.copy()
    c = d["Close"]

    emas = _ema_multi(c, [12, 20, 26, 50])
    d["e20"] = emas[20]; d["e50"] = emas[50]

    d["rsi"] = rsi(c)
    m = emas[12] - emas[26]
    d["macd"] = m
    d["macds"] = m.ewm(span=9, adjust=False).mean()
    d["mach"] = d["macd"] - d["macds"]
    d["atr"] = atr(d)
    d["st"], d["stdir"] = supertrend(d)

    roll20 = c.rolling(20)
    mid = roll20.mean()
    sd = roll20.std()
    d["bb_bw"] = ((mid+2*sd)-(mid-2*sd))/mid*100
    d["z"] = ((c-mid)/sd.replace(0, np.nan)).fillna(0.0)

    d["relvol"] = d["Volume"]/d["Volume"].rolling(20).mean()
    d["obv"] = obv(d)

    d["hi20"] = d["High"].rolling(20).max()
    d["lo20"] = d["Low"].rolling(20).min()

    up = d["Volume"].where(d["Close"] >= d["Open"], 0.0)
    dn = d["Volume"].where(d["Close"] < d["Open"], 0.0)
    ru, rdn = up.rolling(20).sum(), dn.rolling(20).sum()
    d["imbalance"] = ((ru-rdn)/(ru+rdn).replace(0, np.nan)).fillna(0.0)
    return d

# --------------------------------------------------------------- scoring ---
def factor_matrix(d):
    """Compute 4 core factors (reduced from 5 by merging Trend+Structure).
    Direction is a composite of EMA-based trend + price structure."""
    ret = d["Close"].pct_change().fillna(0.0)
    # Direction: composite of trend + structure (were 0.80-0.97 correlated separately)
    trend = (np.sign(d["Close"]-d["e20"]) + np.sign(d["e20"]-d["e50"]) + d["stdir"])/3.0
    span = (d["hi20"]-d["lo20"]).replace(0, np.nan)
    struct = ((((d["Close"]-d["lo20"])/span).clip(0, 1)).fillna(0.5)-0.5)*2.0
    direction = (trend + struct) / 2.0  # Merged to reduce redundancy
    # Momentum: from MACD + RSI
    mom = (np.sign(d["mach"]) + np.sign(d["mach"].diff().fillna(0))
           + ((d["rsi"]-50)/50).clip(-1, 1))/3.0
    # Volume: OBV + relative volume + imbalance
    volf = (np.sign(d["obv"].diff(10).fillna(0))
            + np.where(d["relvol"] > 1.3, np.sign(ret), 0.0) + d["imbalance"])/3.0
    # Mean reversion: Z-score based
    mrev = (-(d["z"])/2.5).clip(-1, 1)
    return pd.DataFrame({"Direction": direction, "Momentum": mom, "Volume": volf,
                         "MeanRev": mrev}, index=d.index).fillna(0.0)

def detect_regime(d, window=50):
    """Classify market regime into bull/bear/range.
    Bull: price above 50-EMA AND EMA rising with low drawdown.
    Bear: price below 50-EMA AND EMA falling.
    Range: oscillating around midpoint with low directional bias.
    Returns {regime, confidence 0..1, details}."""
    if len(d) < window + 10:
        return {"regime": "unknown", "confidence": 0.0, "details": "insufficient history"}

    price = d["Close"].values
    ema50 = ema(d["Close"], window).values
    last_price = price[-1]
    last_ema = ema50[-1]

    recent_ema = ema50[-window:]
    ema_trend = (recent_ema[-1] - recent_ema[0]) / recent_ema[0]

    price_vs_ema = (last_price - last_ema) / last_ema

    recent_prices = price[-window:]
    recent_range = (recent_prices.max() - recent_prices.min()) / recent_prices.mean()

    ema_up = ema_trend > 0.01
    ema_down = ema_trend < -0.01
    price_above = price_vs_ema > 0.01
    price_below = price_vs_ema < -0.01
    low_volatility = recent_range < 0.06

    if price_above and ema_up and not low_volatility:
        regime, conf = "bull", min(1.0, abs(ema_trend) * 20 + abs(price_vs_ema) * 5)
    elif price_below and ema_down and not low_volatility:
        regime, conf = "bear", min(1.0, abs(ema_trend) * 20 + abs(price_vs_ema) * 5)
    else:
        regime, conf = "range", min(1.0, low_volatility * 2.0)

    return {
        "regime": regime,
        "confidence": float(np.clip(conf, 0.0, 1.0)),
        "ema_trend": float(ema_trend * 100),
        "price_vs_ema": float(price_vs_ema * 100),
        "volatility": float(recent_range * 100),
    }

def factor_correlation(F, window=60):
    """Analyze rolling correlation between factors to detect redundancy.
    Returns a matrix of pairwise correlations and identifies which factors move together."""
    if len(F) < window:
        return None

    corr = F[FACTORS].iloc[-window:].corr()

    redundant_pairs = []
    for i, f1 in enumerate(FACTORS):
        for f2 in FACTORS[i+1:]:
            c = float(corr.loc[f1, f2])
            if abs(c) > 0.65:
                redundant_pairs.append((f1, f2, c))

    return {
        "correlation_matrix": corr,
        "redundant_pairs": redundant_pairs,
        "n_highly_correlated": len([p for p in redundant_pairs if abs(p[2]) > 0.70]),
    }

def information_ratio(F, close, horizon=5, min_samples=50):
    """Score each factor's predictive power (correlation with forward returns).
    High IR = factor has edge; low IR = dead weight."""
    if len(F) < horizon + min_samples:
        return None

    fwd = close.pct_change(horizon).shift(-horizon).fillna(0.0)
    irs = {}
    for factor in FACTORS:
        valid = ~(F[factor].isna() | fwd.isna())
        if valid.sum() < min_samples:
            irs[factor] = 0.0
            continue
        corr = float(np.corrcoef(F[factor][valid], fwd[valid])[0, 1])
        irs[factor] = float(np.clip(abs(corr), 0.0, 1.0))

    return irs

def composite(F, weights=None, regime=None):
    """Composite score with optional regime-based weight adjustment.
    If regime provided and confidence is high, use regime-optimized weights."""
    w = weights or BASE_WEIGHTS

    if regime and regime.get("confidence", 0) > 0.6:
        regime_w = REGIME_WEIGHTS.get(regime["regime"])
        if regime_w:
            alpha = regime["confidence"]
            w = {k: (1 - alpha) * w[k] + alpha * regime_w[k] for k in FACTORS}

    return pd.Series(100*(F[FACTORS].values @ np.array([w[k] for k in FACTORS])),
                     index=F.index)

def positions(comp, enter=ENTER, exit_=EXIT):
    raw = np.where(comp > enter, 1.0, np.where(comp < exit_, 0.0, np.nan))
    return pd.Series(raw, index=comp.index).ffill().fillna(0.0)

def _positions_np(comp, enter=ENTER, exit_=EXIT):
    """Pure-numpy equivalent of positions(): forward-fill via index accumulation,
    leading NaNs -> 0. Used in the annealing hot loop to avoid pandas overhead."""
    raw = np.where(comp > enter, 1.0, np.where(comp < exit_, 0.0, np.nan))
    idx = np.where(~np.isnan(raw), np.arange(len(raw)), 0)
    np.maximum.accumulate(idx, out=idx)
    return np.nan_to_num(raw[idx], nan=0.0)

def _sharpe(strat, ppy):
    sd = strat.std()
    return float("nan") if (not np.isfinite(sd) or sd == 0) else float(strat.mean()/sd*math.sqrt(ppy))

def _session_starts(index):
    """Boolean mask, True at the first bar of each new trading day — used to drop
    overnight-gap returns for intraday series (a day trader is flat overnight, so
    those gaps aren't their P&L and shouldn't inflate volatility)."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return None
    days = index.normalize().values
    mask = np.empty(len(days), dtype=bool)
    mask[0] = False
    mask[1:] = days[1:] != days[:-1]
    return mask if mask.any() else None

def backtest(close, comp, ppy=252, intraday=False, slippage_pct=SLIPPAGE_PCT):
    """Backtest with optional slippage modeling.
    slippage_pct: cost per entry + exit (e.g. 0.05 = 0.05% entry + 0.05% exit = 0.10% round trip)."""
    ret_full = close.pct_change().fillna(0.0)
    ret = ret_full
    pos = positions(comp)
    if intraday:
        ns = _session_starts(close.index)
        if ns is not None:
            ret = ret_full.mask(pd.Series(ns, index=close.index), 0.0)
    strat = pos.shift(1).fillna(0.0)*ret

    if slippage_pct > 0:
        slippage_cost = np.abs(pos.diff().fillna(0)) * (slippage_pct / 100.0)
        strat = strat - slippage_cost

    eq = (1+strat).cumprod(); bh = (1+ret_full).cumprod()
    trades, inpos, start = [], False, None
    e, p = eq.values, pos.values
    for i in range(1, len(p)):
        if p[i] == 1 and not inpos: inpos, start = True, i
        elif p[i] == 0 and inpos: trades.append(e[i]/e[start]-1); inpos = False
    if inpos: trades.append(e[-1]/e[start]-1)
    wins = sum(1 for t in trades if t > 0)
    total_slippage = sum(np.abs(pos.diff().fillna(0)) * slippage_pct / 100.0) if slippage_pct > 0 else 0.0
    return {"strategy": float(eq.iloc[-1]-1), "buyhold": float(bh.iloc[-1]-1),
            "maxdd": float((eq/eq.cummax()-1).min()),
            "maxdd_bh": float((bh/bh.cummax()-1).min()), "trades": len(trades),
            "winrate": (wins/len(trades)) if trades else float("nan"),
            "exposure": float(pos.mean()), "sharpe": _sharpe(strat, ppy),
            "slippage_cost": float(total_slippage)}

def backtest_by_regime(close, comp, d, ppy=252, intraday=False):
    """Run separate backtests for BULL/BEAR/RANGE regimes to understand
    which market conditions the strategy thrives in. Returns {regime: backtest_result}."""
    regime_data = {}
    for i in range(len(d)):
        r = detect_regime(d.iloc[:i+1])
        regime = r["regime"]
        if regime not in regime_data:
            regime_data[regime] = []
        regime_data[regime].append(i)

    results = {}
    for regime_name, indices in regime_data.items():
        if not indices:
            continue
        if len(indices) < MIN_BARS:
            continue
        lo, hi = min(indices), max(indices) + 1
        res = backtest(close.iloc[lo:hi], comp.iloc[lo:hi], ppy, intraday)
        results[regime_name] = res
    return results

def objective_loss(equity_curve, strat_returns, signals, ppy=252,
                   lambda_1=3.0, lambda_2=2.0):
    """Annealing loss: -Sharpe + λ1·maxDD + λ2·turnover. Lower is better —
    rewards risk-adjusted return, punishes deep drawdowns and signal
    flip-flopping (which is what racks up fees in live trading)."""
    sd = np.std(strat_returns)
    if not np.isfinite(sd) or sd == 0:
        return 999.0
    sharpe = math.sqrt(ppy) * float(np.mean(strat_returns)) / float(sd)
    peak = np.maximum.accumulate(equity_curve)
    peak = np.where(peak <= 0, 1.0, peak)
    max_dd = float(np.max((peak - equity_curve) / peak))
    turnover = float(np.mean(np.abs(np.diff(signals)))) if len(signals) > 1 else 0.0
    loss = -sharpe + lambda_1 * max_dd + lambda_2 * turnover
    return 999.0 if not np.isfinite(loss) else float(loss)

def _anneal_core(FA, ret, ppy, n_iter=500, seed=11, slippage_pct=SLIPPAGE_PCT):
    """Simulated-annealing over a single train segment given as numpy arrays.
    Returns (best_weight_vector, best_loss). Kept pure/numpy so it can be reused
    per fold in walk-forward validation without pandas overhead."""
    rng = np.random.default_rng(seed)
    n_factors = FA.shape[1]
    n_ret = len(ret)

    pos_diff_cache = np.empty(n_ret)
    strat_cache = np.empty(n_ret)
    eq_cache = np.empty(n_ret)

    def loss_of(wv):
        nonlocal pos_diff_cache, strat_cache, eq_cache
        comp = 100.0*(FA @ wv)
        pos = _positions_np(comp)
        strat_cache[0] = 0.0
        strat_cache[1:] = pos[:-1] * ret[1:]
        if slippage_pct > 0:
            np.abs(np.diff(pos, prepend=0), out=pos_diff_cache)
            strat_cache -= pos_diff_cache * (slippage_pct / 100.0)
        np.cumprod(1.0 + strat_cache, out=eq_cache)
        return objective_loss(eq_cache, strat_cache, pos, ppy)

    base = np.array([BASE_WEIGHTS[k] for k in FACTORS])
    w = base.copy(); loss = loss_of(w); best_w, best_l = w.copy(), loss; T = 0.6
    no_improve = 0

    for _ in range(n_iter):
        cand = np.clip(w + rng.normal(0, 0.08, n_factors)*max(T, 0.05), 0.02, 0.60)
        cand /= cand.sum(); l = loss_of(cand)
        if l < loss or rng.random() < math.exp(min(0.0, (loss-l))/max(T, 1e-3)):
            w, loss = cand, l
            if l < best_l: best_w, best_l = cand.copy(), l; no_improve = 0
            else: no_improve += 1
        else:
            no_improve += 1
        T *= 0.985
        if no_improve > 80 and T < 0.01: break
    return best_w, best_l

def _seg_sharpe_w(F, close, wv, lo, hi, ppy):
    """Sharpe of a weight set on bars [lo:hi], returns computed within the slice."""
    FA_slice = F[FACTORS].values[lo:hi]
    comp_vals = 100.0 * (FA_slice @ wv)
    ret_slice = close.iloc[lo:hi].pct_change().fillna(0.0).values
    pos = _positions_np(comp_vals)
    strat = np.empty_like(pos); strat[0] = 0.0; strat[1:] = pos[:-1] * ret_slice[1:]
    s = _sharpe(strat, ppy)
    return s

def walk_forward_validate(F, close, ppy=252, folds=4, n_iter=300, seed=11):
    """True walk-forward: expand the train window fold by fold, re-optimize on
    each, and measure Sharpe on the NEXT (unseen) segment. Averaging several
    out-of-sample periods is far more honest than a single 30% hold-out —
    weights that only work on one test slice get exposed here. Returns None if
    there isn't enough history to split meaningfully."""
    n = len(F)
    step = n // (folds + 1)
    if n < 150 or step < 40:
        return None
    FA_full = F[FACTORS].values
    close_vals = close.values
    ret_full = np.diff(close_vals, prepend=close_vals[0]) / close_vals.clip(min=1e-9) - 1.0
    ret_full[0] = 0.0
    oos = []
    for k in range(1, folds + 1):
        tr_hi = step * k
        te_lo, te_hi = tr_hi, (n if k == folds else step * (k + 1))
        if te_hi - te_lo < 30:
            continue
        FA_tr = FA_full[0:tr_hi]
        ret_tr = ret_full[0:tr_hi]
        w, _ = _anneal_core(FA_tr, ret_tr, ppy, n_iter, seed)
        oos.append(_seg_sharpe_w(F, close, w, te_lo, te_hi, ppy))
    valid = [s for s in oos if np.isfinite(s)]
    if not valid:
        return None
    return {"wf_sharpe": float(np.mean(valid)),
            "wf_folds": [round(float(s), 2) if np.isfinite(s) else None for s in oos],
            "wf_pos_frac": float(np.mean([1.0 if s > 0 else 0.0 for s in valid]))}

def optimize_weights(F, close, ppy=252, n_iter=500, seed=11, walk_forward=True):
    n = len(F); cut = max(60, int(n*0.7))
    FA_full = F[FACTORS].values
    close_vals = close.values
    ret_full = np.diff(close_vals, prepend=close_vals[0]) / close_vals.clip(min=1e-9) - 1.0
    ret_full[0] = 0.0

    FA_train = FA_full[0:cut]
    ret_train = ret_full[0:cut]
    best_w, best_l = _anneal_core(FA_train, ret_train, ppy, n_iter, seed)

    def seg_metrics(wv, lo, hi):
        FA_slice = FA_full[lo:hi]
        comp = 100.0 * (FA_slice @ wv)
        pos = _positions_np(comp)
        ret_slice = ret_full[lo:hi]
        strat = np.empty(len(pos))
        strat[0] = 0.0
        strat[1:] = pos[:-1] * ret_slice[1:]
        eq = np.cumprod(1.0 + strat)
        return eq, strat, pos

    def seg_loss(wv, lo, hi):
        eq, strat, pos = seg_metrics(wv, lo, hi)
        return objective_loss(eq, strat, pos, ppy)
    def seg_sharpe(wv, lo, hi):
        eq, strat, pos = seg_metrics(wv, lo, hi)
        s = _sharpe(strat, ppy)
        return -9.0 if not np.isfinite(s) else s

    base = np.array([BASE_WEIGHTS[k] for k in FACTORS])
    wr = [round(float(v), 3) for v in best_w]
    imax = int(np.argmax(best_w))
    wr[imax] = round(wr[imax] + (1.0 - sum(wr)), 3)
    out = {"weights": {k: wr[i] for i, k in enumerate(FACTORS)},
           "train_sharpe": seg_sharpe(best_w, 0, cut),
           "test_sharpe": seg_sharpe(best_w, cut, n),
           "base_test_sharpe": seg_sharpe(base, cut, n),
           "train_loss": best_l, "test_loss": seg_loss(best_w, cut, n)}
    if walk_forward:
        wf = walk_forward_validate(F, close, ppy, n_iter=max(200, n_iter // 2), seed=seed)
        if wf:
            out.update(wf)
    return out

# --------------------------------------------------------------- verdict ---
HIGH_VOL = 35.0   # annualized-vol % above which a name is treated as high-beta

def vol_thresholds(ann_vol):
    """Widen the default BUY/STRONG cutoffs for high-volatility names. A hyper-
    volatile name's composite score swings far more than a slow mega-cap's, so a
    flat +18 fires on noise. Scales from 1.0× at ≤25% vol toward 1.8× at ~65%+."""
    factor = min(1.8, 1.0 + max(0.0, (ann_vol - 25.0) / 40.0))
    return round(ENTER * factor, 1), round(45.0 * factor, 1)

def verdict(score, atr_pct, buy=ENTER, strong=45.0, regime=None):
    if regime and regime.get("confidence", 0) > 0.5:
        buy = REGIME_THRESHOLDS.get(regime["regime"], {}).get("enter", buy)
        strong = REGIME_THRESHOLDS.get(regime["regime"], {}).get("strong", strong)

    if score >= strong: lab, tone = "STRONG BUY signal", "good"
    elif score >= buy: lab, tone = "BUY signal", "good"
    elif score > -buy: lab, tone = "HOLD / no edge", "neutral"
    elif score > -strong: lab, tone = "AVOID / sell signal", "bad"
    else: lab, tone = "STRONG AVOID", "bad"
    return {"label": lab, "tone": tone, "risky": bool(atr_pct >= RISKY_ATR_PCT)}

def calibrate_thresholds(comp, close, horizon=5, min_bars=150):
    """Derive per-name BUY/STRONG cutoffs from history instead of the fixed
    18/45: for each bar, look at the forward `horizon`-bar return, then find the
    score above which forward returns actually turned positive (BUY) and where
    the top tercile of scores sits (STRONG). Returns None if too little data or
    if the calibration is unstable (no monotone edge), so callers fall back to
    the defaults. IN-SAMPLE by construction — like the weight optimizer, treat it
    as a description of this history, not a promise about the future."""
    c = np.asarray(comp, dtype=float)
    px = np.asarray(close, dtype=float)
    n = len(c)
    if n < min_bars:
        return None
    fwd = np.full(n, np.nan)
    fwd[:-horizon] = px[horizon:] / px[:-horizon] - 1.0
    m = np.isfinite(fwd) & np.isfinite(c)
    c, fwd = c[m], fwd[m]
    if len(c) < min_bars:
        return None
    # Scan candidate BUY thresholds; pick the lowest score whose forward-return
    # mean is positive AND beats the mean below it (a real edge, not noise).
    order = np.argsort(c)
    cs, fs = c[order], fwd[order]
    best_buy = None
    for thr in np.arange(5.0, 45.0, 2.5):
        above = fs[cs >= thr]; below = fs[cs < thr]
        if len(above) < 20 or len(below) < 20:
            continue
        if above.mean() > 0 and above.mean() > below.mean():
            best_buy = float(thr)
            break
    if best_buy is None:
        return None
    strong = float(np.nanpercentile(c[c >= best_buy], 66)) if np.any(c >= best_buy) else best_buy + 20
    strong = max(strong, best_buy + 10)
    above = fwd[c >= best_buy]
    return {"buy": round(best_buy, 1), "strong": round(strong, 1),
            "horizon": horizon, "fwd_mean": float(above.mean()) if len(above) else float("nan"),
            "n": int(len(c))}

def forward_stats(comp, close, horizon=5, band=12.0, min_bars=120):
    """The score's empirical track record ON THIS NAME: of all past bars whose
    score was similar to today's, how did the next `horizon` bars actually go?
    Returns win rate, mean forward return, sample size, and the same figures for
    ALL bars (the baseline) so you can see whether a high score has historically
    meant anything here. IN-SAMPLE and descriptive — it reports what this history
    did, not what the future will do. None if too little data."""
    c = np.asarray(comp, dtype=float)
    px = np.asarray(close, dtype=float)
    n = len(c)
    if n < min_bars:
        return None
    fwd = np.full(n, np.nan)
    fwd[:-horizon] = px[horizon:] / px[:-horizon] - 1.0
    valid = np.isfinite(fwd)
    cur = float(c[-1])
    # cohort = past bars within `band` of today's score; widen to a directional
    # cohort (same side, at least as extreme) if the tight band is too thin.
    cohort = valid & (np.abs(c - cur) <= band)
    if cohort.sum() < 15:
        cohort = valid & ((c >= cur) if cur >= 0 else (c <= cur))
    if cohort.sum() < 15:
        return None
    fc, base = fwd[cohort], fwd[valid]
    win = float((fc > 0).mean()); base_win = float((base > 0).mean())
    return {"horizon": horizon, "score": cur, "n": int(cohort.sum()),
            "win_rate": win, "mean_fwd": float(fc.mean()),
            "median_fwd": float(np.median(fc)),
            "base_win_rate": base_win, "base_mean_fwd": float(base.mean()),
            "edge": win - base_win}

def conviction(F_last, score):
    sgn = np.sign(score)
    agree = sum(1 for k in FACTORS if np.sign(F_last[k]) == sgn and sgn != 0)
    return int(round(100*agree/len(FACTORS)))

def whale_score(d, rvol_flag=1.5, cmf_flag=0.05):
    """Detect large-money FOOTPRINTS from observable price+volume — nothing paid,
    nothing inferred about identity. Two legitimate signals:
      RVOL — today's volume / 20-day average. >2x means abnormally large activity.
      CMF  — Chaikin Money Flow (20): volume weighted by where price closes in its
             range. >0 = buying pressure (accumulation), <0 = selling (distribution).
    A 'whale' bar = high RVOL AND directional CMF. HONEST SCOPE: this flags that
    big money moved and which way pressure leaned — it CANNOT identify who traded
    or why (a block could be a fund, a market maker hedging, or an index rebalance).
    Returns {rvol, cmf, dollar_vol, direction, whale, signal -1..+1} or None."""
    if d is None or len(d) < 20:
        return None
    vol, close, high, low = d["Volume"], d["Close"], d["High"], d["Low"]
    avg20 = float(vol.tail(20).mean())
    rvol = float(vol.iloc[-1] / avg20) if avg20 > 0 else 1.0
    hl = (high - low).replace(0, np.nan)
    mfv = (((close - low) - (high - close)) / hl * vol).fillna(0.0)   # money-flow volume
    denom = float(vol.rolling(20).sum().iloc[-1])
    cmf = float(mfv.rolling(20).sum().iloc[-1] / denom) if denom > 0 else 0.0
    mag = float(np.clip((rvol - 1.0) / 2.0, 0.0, 1.0))               # 1x→0, 3x→1
    return {"rvol": rvol, "cmf": cmf,
            "dollar_vol": float(close.iloc[-1] * vol.iloc[-1]),
            "direction": "accumulation" if cmf > 0.02 else "distribution" if cmf < -0.02 else "neutral",
            "whale": bool(rvol >= rvol_flag and abs(cmf) >= cmf_flag),
            "signal": float(np.clip(np.sign(cmf) * mag, -1.0, 1.0))}

def position_size(price, atr_val, account, risk_pct, stop_mult=2.0):
    # stop_mult defaults to 2×ATR; a defensive news-tone shift tightens it (e.g. 1.5).
    stop = stop_mult*atr_val
    if not (stop > 0) or not (price > 0): return None
    shares = min(int((account*risk_pct/100.0)//stop), int(account//price))
    if shares <= 0: return None
    return {"shares": shares, "entry": price, "stop": price-stop,
            "target": price+2*stop, "risk_dollars": shares*stop, "notional": shares*price,
            "reward_dollars": shares*2*stop, "stop_mult": stop_mult}

# ------------------------------------------------------------------ data ---
def fetch(ticker, period, interval):
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance not installed. Run:  pip3 install yfinance pandas numpy")
    df = yf.download(ticker, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"no data for '{ticker}' (check symbol / period / interval)")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

def demo_data(label="DEMO", bars=380):
    seed = int(hashlib.md5(label.upper().encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    mom, price, rows = 0.0, 40+rng.random()*160, []
    for _ in range(bars):
        mom = mom*0.9 + rng.normal(0, 0.012)
        r = mom*0.35 + rng.normal(0, 0.014)
        o = price; c = max(0.5, price*(1+r))
        h = max(o, c)*(1+abs(rng.normal(0, 0.006)))
        l = min(o, c)*(1-abs(rng.normal(0, 0.006)))
        v = int(2e5*math.exp(abs(r)*40 + rng.normal(0, 0.5)))
        rows.append((o, h, l, c, v)); price = c
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=bars, freq="B")
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close", "Volume"])

# ------------------------------------------------------------- analysts ---
def key_candidates(k):
    k = (k or "").strip()
    if not k: return []
    if len(k) >= 40:
        a, b = k[:20], k[-20:]
        if sum(x == y for x, y in zip(a, b)) >= 17: return [b, a, k]
    return [k]

def finnhub_recs(ticker, key):
    import json, urllib.request
    for cand in key_candidates(key):
        try:
            url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={ticker}&token={cand}"
            with urllib.request.urlopen(url, timeout=6) as r:
                data = json.loads(r.read().decode())
            if isinstance(data, list) and data: return data[0]
        except Exception:
            continue
    return None

# ------------------------------------------------------- real-time quote ---
def finnhub_quote(ticker, key, timeout=5):
    """Real-time last price from Finnhub /quote (free, 60/min, reuses the app's
    key). Returns {price, prev_close, change_pct, ts} or None. This is the
    cheapest way to un-delay the app: the bars are ~15-min lagged, but the last
    price here is live, so patching it onto the latest bar makes the current
    price/score/stops reflect NOW."""
    if not key:
        return None
    import json, urllib.request
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={key}"
        with urllib.request.urlopen(url, timeout=timeout) as r:
            d = json.loads(r.read().decode())
        c = d.get("c")
        if not c or c <= 0:
            return None
        return {"price": float(c), "prev_close": float(d.get("pc") or 0.0),
                "change_pct": float(d.get("dp") or 0.0), "ts": d.get("t"),
                "source": "Finnhub", "session": "regular"}
    except Exception:
        return None

def _session_from_et(et):
    """Classify a US-market datetime (in ET) into its trading session."""
    if et.weekday() >= 5:
        return "closed"
    mins = et.hour * 60 + et.minute
    if 570 <= mins < 960:   return "regular"       # 09:30–16:00
    if 960 <= mins < 1200:  return "after-hours"   # 16:00–20:00
    if 240 <= mins < 570:   return "pre-market"    # 04:00–09:30
    return "closed"

def alpaca_latest_price(ticker, key, secret, timeout=5):
    """Latest trade from Alpaca (IEX feed) — INCLUDES pre-market and after-hours
    trades, unlike Finnhub /quote (which only sees the regular session). Returns
    {price, ts, source, session} where session ∈ regular/pre-market/after-hours/
    closed, or None on failure."""
    if not (key and secret):
        return None
    import json, re, urllib.request
    from datetime import datetime, timezone
    try:
        url = f"https://data.alpaca.markets/v2/stocks/{ticker}/trades/latest?feed=iex"
        req = urllib.request.Request(url, headers={"APCA-API-KEY-ID": key,
                                                   "APCA-API-SECRET-KEY": secret})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            tr = json.loads(r.read().decode()).get("trade", {})
        px = tr.get("p"); ts = tr.get("t")
        if not px or px <= 0:
            return None
        session = "regular"
        try:
            from zoneinfo import ZoneInfo
            m = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", ts or "")
            if m:
                utc = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                session = _session_from_et(utc.astimezone(ZoneInfo("America/New_York")))
        except Exception:
            pass
        return {"price": float(px), "ts": ts, "source": "Alpaca", "session": session}
    except Exception:
        return None

def realtime_quote(ticker, finnhub_key=None, alpaca_key=None, alpaca_secret=None):
    """Best available live price: Alpaca first (sees extended hours), Finnhub
    fallback (regular session only). None if neither works."""
    if alpaca_key and alpaca_secret:
        q = alpaca_latest_price(ticker, alpaca_key, alpaca_secret)
        if q:
            return q
    return finnhub_quote(ticker, finnhub_key)

def alpaca_latest_trade(ticker, key, secret, timeout=5):
    """Latest trade price from Alpaca (IEX). Unlike a regular-session quote this
    INCLUDES extended hours — during pre/post-market it returns the aftermarket
    price. Returns a float or None."""
    if not (key and secret):
        return None
    import json, urllib.request
    try:
        url = f"https://data.alpaca.markets/v2/stocks/{ticker}/trades/latest?feed=iex"
        req = urllib.request.Request(url, headers={"APCA-API-KEY-ID": key,
                                                   "APCA-API-SECRET-KEY": secret})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            p = json.loads(r.read().decode()).get("trade", {}).get("p")
        return float(p) if p and p > 0 else None
    except Exception:
        return None

def realtime_quote(ticker, finnhub_key=None, alpaca_key=None, alpaca_secret=None):
    """Best available live price. Prefers Alpaca's latest trade (real-time AND
    includes pre/post-market), falls back to Finnhub /quote (regular session).
    Returns {price, source} or None."""
    p = alpaca_latest_trade(ticker, alpaca_key, alpaca_secret)
    if p:
        return {"price": p, "source": "Alpaca (incl. extended hrs)"}
    q = finnhub_quote(ticker, finnhub_key)
    if q:
        return {"price": q["price"], "source": "Finnhub"}
    return None

def patch_realtime(df, quote):
    """Overwrite the most recent bar's Close with the live quote price (and widen
    that bar's High/Low to contain it), so the current read is live rather than
    ~15-min delayed. Returns a copy; returns df unchanged if no usable quote."""
    if not quote or df is None or df.empty:
        return df
    px = quote.get("price")
    if not px or px <= 0:
        return df
    df = df.copy()
    i = df.index[-1]
    df.loc[i, "High"] = max(float(df.loc[i, "High"]), px)
    df.loc[i, "Low"] = min(float(df.loc[i, "Low"]), px)
    df.loc[i, "Close"] = px
    return df

# ---------------------------------------------------- congress (Quiver) ---
QUIVER_API_BASE = "https://api.quiverquant.com/beta"
# A browser-like User-Agent is REQUIRED: Quiver sits behind Cloudflare, which
# blocks urllib's default UA with a 403 (error 1010) before auth is checked.
_QUIVER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

def _quiver_get(path, token, timeout):
    """GET a Quiver endpoint, returning parsed JSON or None. Tries both auth
    schemes Quiver has used (Bearer/Token). Fail-open — never raises."""
    if not token:
        return None
    import json, urllib.request
    headers = {"Accept": "application/json", "User-Agent": _QUIVER_UA}
    for scheme in ("Bearer", "Token"):
        try:
            req = urllib.request.Request(f"{QUIVER_API_BASE}{path}",
                                         headers={**headers, "Authorization": f"{scheme} {token}"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception:
            continue
    return None

def _congress_date(t):
    return t.get("TransactionDate") or t.get("Date") or t.get("ReportDate") or ""

ALT_DECAY_TAU = 30.0   # days; exp(-age/τ) recency multiplier for alt-data

def _age_decay(days_since):
    """Exponential recency multiplier in (0,1] = exp(-age/30d). A filing today =
    1.00; 21 days ago ≈ 0.50; 30 days ≈ 0.37; 80 days ≈ 0.07 — so stale alt-data
    shrinks toward zero instead of carrying a fixed bias forever (a Congress buy
    from 80 days ago is old news)."""
    if days_since is None or days_since < 0:
        return 1.0
    return float(math.exp(-days_since / ALT_DECAY_TAU))

def _days_since(date_str):
    try:
        return max(0.0, (pd.Timestamp.today().normalize() - pd.Timestamp(date_str[:10])).days)
    except Exception:
        return None

def summarize_congress(trades, top=4, recent_days=90):
    """Condense a Quiver congress-trading list into buy/sell counts (all-time and
    a recent window) and the most recent disclosures. Tolerant of missing fields."""
    trades = sorted(trades, key=_congress_date, reverse=True)
    cutoff = (pd.Timestamp.today() - pd.Timedelta(days=recent_days)).strftime("%Y-%m-%d")
    buys = sells = rbuys = rsells = 0
    for t in trades:
        tx = str(t.get("Transaction", "")).lower()
        is_buy = "purchase" in tx or "buy" in tx
        is_sell = "sale" in tx or "sell" in tx
        if is_buy: buys += 1
        elif is_sell: sells += 1
        if _congress_date(t)[:10] >= cutoff:      # ISO dates compare lexically
            if is_buy: rbuys += 1
            elif is_sell: rsells += 1
    latest = [{
        "rep": t.get("Representative") or t.get("Name") or "?",
        "party": (t.get("Party") or "")[:1],
        "house": t.get("House") or "",
        "tx": t.get("Transaction") or "",
        "range": t.get("Range") or t.get("Amount") or "",
        "date": _congress_date(t)[:10],
    } for t in trades[:top]]
    days_since = _days_since(_congress_date(trades[0])) if trades else None
    return {"total": len(trades), "buys": buys, "sells": sells,
            "recent_buys": rbuys, "recent_sells": rsells,
            "recent_days": recent_days, "latest": latest, "days_since": days_since}

def quiver_congress(ticker, token, timeout=6):
    """Recent congressional trades for ONE ticker (historical endpoint).
    Returns a summary dict (possibly total=0), or None. Needs a Quiver token."""
    data = _quiver_get(f"/historical/congresstrading/{ticker}", token, timeout)
    return summarize_congress(data) if isinstance(data, list) else None

def quiver_congress_bulk(token, timeout=20):
    """ONE call for the most recent congressional trades across ALL tickers
    (Quiver's live feed), grouped into a {ticker: summary} map. Lets the Screener
    surface congressional activity for its whole universe without a call per
    name. Returns {} on any failure (fail-open)."""
    data = _quiver_get("/live/congresstrading", token, timeout)
    if not isinstance(data, list) or not data:
        return {}
    groups = {}
    for t in data:
        tk = t.get("Ticker")
        if tk:
            groups.setdefault(tk, []).append(t)
    return {tk: summarize_congress(rows) for tk, rows in groups.items()}

# --------------------------------------------------- insiders (Finnhub) ---
def _finnhub_json(path_qs, timeout=8):
    import json, urllib.request
    with urllib.request.urlopen(f"https://finnhub.io/api/v1/{path_qs}", timeout=timeout) as r:
        return json.loads(r.read().decode())

def finnhub_insiders(ticker, key, recent_days=90):
    """Corporate-insider summary from SEC Form 4 filings (via Finnhub).
    Only NON-DERIVATIVE open-market trades count: code P (purchase) and
    S (sale). Gifts (G), grants (A), option exercises (M) and tax
    withholding (F) are excluded — they say nothing about conviction.
    Dollar-weights each trade (|shares| x price). Also pulls Finnhub's
    aggregated MSPR (monthly net-buying score, -100..+100). None on failure."""
    if not key:
        return None
    try:
        tx = _finnhub_json(f"stock/insider-transactions?symbol={ticker}&token={key}")
        rows = tx.get("data") or []
        cutoff = (pd.Timestamp.today() - pd.Timedelta(days=recent_days)).strftime("%Y-%m-%d")
        buy_usd = sell_usd = 0.0; buys = sells = 0; biggest = None; newest = ""
        for r in rows:
            if r.get("isDerivative"):
                continue
            code = r.get("transactionCode"); px = r.get("transactionPrice") or 0
            chg = r.get("change") or 0
            tdate = r.get("transactionDate") or ""
            if tdate < cutoff or px <= 0:
                continue
            if tdate > newest:
                newest = tdate
            usd = abs(chg) * px
            if code == "P" and chg > 0:
                buys += 1; buy_usd += usd
                if biggest is None or usd > biggest[1]:
                    biggest = (r.get("name", "?").title(), usd, tdate)
            elif code == "S" and chg < 0:
                sells += 1; sell_usd += usd
        mspr = None
        try:
            to = pd.Timestamp.today(); frm = to - pd.Timedelta(days=180)
            sent = _finnhub_json(f"stock/insider-sentiment?symbol={ticker}"
                                 f"&from={frm:%Y-%m-%d}&to={to:%Y-%m-%d}&token={key}")
            vals = [m.get("mspr") for m in (sent.get("data") or []) if m.get("mspr") is not None]
            if vals:
                mspr = float(np.mean(vals[-3:]))          # last ~3 months
        except Exception:
            pass
        if buys == sells == 0 and mspr is None:
            return None
        return {"buys": buys, "sells": sells, "buy_usd": buy_usd, "sell_usd": sell_usd,
                "biggest_buy": biggest, "mspr": mspr, "recent_days": recent_days,
                "days_since": _days_since(newest) if newest else None}
    except Exception:
        return None

def insider_signal(ins):
    """Insider summary -> {signal -1..+1, confidence 0..1} or None.
    Blends dollar-weighted net open-market flow with MSPR. Buys are weighted
    2x sells: purchases have one motive, sales have many (tax, diversification)."""
    if not ins:
        return None
    parts, conf = [], 0.0
    tot = ins["buy_usd"] + ins["sell_usd"]
    if tot > 0:
        parts.append(np.clip((2*ins["buy_usd"] - ins["sell_usd"]) / (2*ins["buy_usd"] + ins["sell_usd"]), -1, 1))
        conf = max(conf, min(1.0, tot / 2e6))             # ~$2M flow = full confidence
    if ins.get("mspr") is not None:
        parts.append(ins["mspr"] / 100.0)
        conf = max(conf, 0.5)
    if not parts:
        return None
    decay = _age_decay(ins.get("days_since"))                # stale Form 4s fade out
    detail = f"${ins['buy_usd']/1e6:.1f}M bought / ${ins['sell_usd']/1e6:.1f}M sold ({ins['recent_days']}d)"
    if ins.get("biggest_buy"):
        n, usd, dt_ = ins["biggest_buy"]
        detail += f" · top buy {n} ${usd/1e6:.1f}M {dt_}"
    if ins.get("days_since") is not None:
        detail += f" · {ins['days_since']:.0f}d ago (decay ×{decay:.2f})"
    return {"signal": float(np.clip(np.mean(parts), -1, 1)), "confidence": float(conf) * decay,
            "detail": detail}

# --------------------------------------- whale flow (options + dark pool) ---
def options_whale_flow(ticker, min_dte=7, max_dte=60, max_expiries=4):
    """Unusual-positioning read from Yahoo option chains (delayed, free).
    Uses expiries 7-60 days out — near-dated chains are day-trader noise, and
    that window is where positioning ahead of a move shows up. Two components:
      pcr    — put/call volume skew (calls dominating = bullish)
      fresh  — volume vs open interest (today's volume >> existing OI on the
               call side = NEW positioning, not old holders trading around)
    Returns raw components + totals, or None if chains are unavailable."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        today = pd.Timestamp.today().normalize()
        exps = [e for e in t.options
                if min_dte <= (pd.Timestamp(e) - today).days <= max_dte][:max_expiries]
        if not exps:
            return None
        cv = pv = coi = poi = 0.0
        for e in exps:
            ch = t.option_chain(e)
            cv += float(ch.calls["volume"].fillna(0).sum())
            pv += float(ch.puts["volume"].fillna(0).sum())
            coi += float(ch.calls["openInterest"].fillna(0).sum())
            poi += float(ch.puts["openInterest"].fillna(0).sum())
        if cv + pv < 100:                                  # too thin to mean anything
            return None
        pcr = pv / max(cv, 1.0)
        cturn = cv / max(coi, 1.0); pturn = pv / max(poi, 1.0)
        return {"call_vol": cv, "put_vol": pv, "pcr": pcr,
                "call_turnover": cturn, "put_turnover": pturn,
                "expiries": len(exps)}
    except Exception:
        return None

def quiver_darkpool(ticker, token, lookback=20):
    """Dark-pool proxy: FINRA off-exchange short ratio (DPI) via Quiver.
    Falling DPI vs its own recent average = short pressure easing (mildly
    bullish). A daily aggregate, NOT real-time prints. None on failure."""
    data = _quiver_get(f"/historical/offexchange/{ticker}", token, timeout=8)
    if not isinstance(data, list) or len(data) < lookback + 1:
        return None
    rows = sorted(data, key=lambda r: r.get("Date", ""))[-(lookback + 1):]
    dpis = [r.get("DPI") for r in rows if r.get("DPI") is not None]
    if len(dpis) < lookback:
        return None
    today = float(dpis[-1]); avg = float(np.mean(dpis[:-1]))
    return {"dpi": today, "dpi_avg": avg, "delta": avg - today}

def whale_signal(flow, dark=None):
    """Options flow (+ optional dark pool) -> {signal, confidence, detail} or None.
    'Whale Flow Score': 70% options positioning, 30% dark-pool short-pressure
    trend when available. The noisiest alt signal — weighted lowest in the tilt."""
    if not flow:
        return None
    sig_pcr = np.clip((1.0 - flow["pcr"]) / 0.5, -1, 1)    # PCR 0.5→+1 · 1.0→0 · 1.5→-1
    ct, pt = flow["call_turnover"], flow["put_turnover"]
    sig_fresh = np.clip((ct - pt) / max(ct + pt, 1e-9), -1, 1)
    sig = 0.5 * sig_pcr + 0.5 * sig_fresh
    detail = (f"P/C {flow['pcr']:.2f} · call turn {ct:.2f}x vs put {pt:.2f}x "
              f"({flow['expiries']} expiries 7-60d)")
    if dark:
        sig = 0.7 * sig + 0.3 * np.clip(dark["delta"] / 0.10, -1, 1)
        detail += f" · DPI {dark['dpi']:.2f} vs {dark['dpi_avg']:.2f} avg"
    conf = min(1.0, (flow["call_vol"] + flow["put_vol"]) / 50000.0)
    return {"signal": float(np.clip(sig, -1, 1)), "confidence": float(conf), "detail": detail}

# ------------------------------------------------------- alt-data tilt ---
def analyst_signal(recs):
    """Finnhub recommendation counts -> {signal -1..+1, confidence 0..1} or None.
    Confidence scales with how many analysts cover the name."""
    if not recs:
        return None
    sb = recs.get("strongBuy", 0); b = recs.get("buy", 0); h = recs.get("hold", 0)
    s = recs.get("sell", 0); ss = recs.get("strongSell", 0)
    tot = sb + b + h + s + ss
    if tot <= 0:
        return None
    sig = (2*sb + b - s - 2*ss) / (2*tot)                     # [-1, +1]
    return {"signal": float(np.clip(sig, -1, 1)), "confidence": min(1.0, tot/10.0),
            "detail": f"{sb+b} buy / {h} hold / {s+ss} sell"}

def congress_signal(congress):
    """Quiver congress summary -> {signal -1..+1, confidence 0..1} or None.
    Uses the RECENT net (last ~90d), then decays confidence by how long ago the
    most recent trade was — an 80-day-old buy barely moves the tilt."""
    if not congress:
        return None
    rb = congress.get("recent_buys", 0); rs = congress.get("recent_sells", 0)
    n = rb + rs
    if n <= 0:
        return None
    decay = _age_decay(congress.get("days_since"))
    ds = congress.get("days_since")
    return {"signal": float((rb - rs) / n), "confidence": min(1.0, n/5.0) * decay,
            "detail": (f"{rb} buys / {rs} sells last {congress.get('recent_days', 90)}d"
                       + (f" · {ds:.0f}d ago (decay ×{decay:.2f})" if ds is not None else ""))}

def alt_data_tilt(congress, recs, insiders=None, whale=None, macro=None):
    """Blend congress/analyst/insider/whale-flow/macro-text signals into a bounded
    point adjustment for the live score. Each source is confidence-scaled, so thin
    coverage tilts little. Returns {parts, blended, adjustment} or None."""
    parts = {}
    for name, sig in (("Congress", congress_signal(congress)),
                      ("Analyst", analyst_signal(recs)),
                      ("Insider", insider_signal(insiders)),
                      ("WhaleFlow", whale if (whale and "signal" in whale) else None),
                      ("Macro", macro if (macro and "signal" in macro) else None)):
        if sig:
            parts[name] = sig
    if not parts:
        return None
    num = wsum = 0.0
    for k, p in parts.items():
        w = ALT_WEIGHTS.get(k, 0.0)
        num += w * p["confidence"] * p["signal"]
        wsum += w
    blended = num / wsum if wsum else 0.0        # low confidence shrinks toward 0
    return {"parts": parts, "blended": blended,
            "adjustment": ALT_MAX_TILT * float(np.clip(blended, -1, 1))}

MARKET_MAX_TILT = 10.0

def market_context(stock_df, spy_df, window=60):
    """Rate the stock relative to the market (SPY). Returns relative strength
    over `window` bars, the market regime (SPY above/below its 50-EMA), and a
    bounded score adjustment: outperformance helps, underperformance hurts, and a
    lone gainer in a falling tape is discounted (headwind). None if unalignable."""
    if spy_df is None or len(stock_df) < 30 or len(spy_df) < 30:
        return None
    s, m = stock_df["Close"], spy_df["Close"]
    common = s.index.intersection(m.index)
    if len(common) < 30:
        return None
    s, m = s.loc[common], m.loc[common]
    w = min(window, len(common) - 1)
    s_ret = float(s.iloc[-1] / s.iloc[-1-w] - 1.0)
    m_ret = float(m.iloc[-1] / m.iloc[-1-w] - 1.0)
    rel = s_ret - m_ret
    rs_signal = float(np.clip(rel / 0.15, -1, 1))          # ±15% rel = full signal
    risk_on = bool(m.iloc[-1] >= ema(m, 50).iloc[-1])
    adj = MARKET_MAX_TILT * rs_signal
    if not risk_on and adj > 0:
        adj *= 0.5                                          # discount strength in a weak tape
    return {"rel": rel, "rs_signal": rs_signal, "risk_on": risk_on,
            "adjustment": float(adj), "window": w,
            "stock_ret": s_ret, "spy_ret": m_ret}

def apply_alt_tilt(res, tilt, market=None):
    """Apply alt-data and/or market-relative adjustments to an analyze() result
    in place: shift the live score, stash the pre-adjustment score, and recompute
    verdict + conviction (respecting calibrated thresholds if present). The
    backtest in res is untouched — it stays a pure technical measure."""
    if not (tilt or market):
        return res
    adj = 0.0
    if tilt:
        res["alt"] = tilt; adj += tilt["adjustment"]
    if market:
        res["market"] = market; adj += market["adjustment"]
    res["base_score"] = res["score"]
    res["score"] = float(np.clip(res["score"] + adj, -100, 100))
    cal = res.get("calib")
    buy_th = cal["buy"] if cal else ENTER
    strong_th = cal["strong"] if cal else 45.0
    res["verdict"] = verdict(res["score"], res["atr_pct"], buy_th, strong_th)
    res["conviction"] = conviction(res["F"].iloc[-1], res["score"])
    return res

# ------------------------------------------------------------- analyze ---
def analyze_pipeline(ticker, df, interval, account=None, risk_pct=1.0):
    """New architecture: explicit pipeline with clear data flow.
    Market Data → Regime Detector → Factor Analyzer → Risk Engine → Metrics."""
    data = Pipeline.execute(ticker, df, account, risk_pct)

    ppy = PPY.get(interval, 252)
    intraday = (interval in INTRADAY_INTERVALS) and not is_crypto(ticker)
    rets = data.d["Close"].pct_change()
    if intraday:
        ns = _session_starts(data.d.index)
        if ns is not None:
            rets = rets.mask(pd.Series(ns, index=data.d.index))
    ann_vol = float(rets.std() * math.sqrt(ppy) * 100)

    return {
        "ticker": ticker,
        "stage_1_data": {"d": data.d, "n_bars": len(data.d)},
        "stage_2_regime": data.regime,
        "stage_3_factors": {"F": data.F, "ir": data.ir, "corr": data.corr, "comp": data.comp},
        "stage_4_risk": {"account": data.account, "risk_pct": data.risk_pct, "position": data.position if hasattr(data, 'position') else None},
        "stage_5_metrics": {
            "score": data.score, "last": data.last, "chg": data.chg,
            "atr": float(data.d["atr"].iloc[-1]), "atr_pct": data.atr_pct,
            "ann_vol": ann_vol, "maxdd": float((data.d["Close"]/data.d["Close"].cummax()-1).min()*100),
            "verdict": data.verdict, "conviction": data.conviction,
            "backtest": data.backtest, "backtest_by_regime": data.backtest_by_regime,
        }
    }

def analyze(ticker, df, interval, weights=None, d=None, F=None, calibrate=False):
    # d/F may be passed in pre-computed (e.g. the optimize path already enriched
    # once) to avoid recomputing enrich()/factor_matrix() — the heaviest steps.
    if d is None:
        d = enrich(df)
    nb = len(d)
    if nb < MIN_BARS:
        raise ValueError(
            f"only {nb} bars — likely a recent listing. Need ≥{MIN_BARS} for any read. "
            f"Try a finer interval (1h / 30m): a young stock has far more intraday bars.")
    limited_history = nb < STABLE_BARS      # 40–60 bars: EMA50/backtest not fully formed
    ppy = PPY.get(interval, 252)
    # crypto is 24/7 — no sessions, so no overnight-gap masking even intraday
    intraday = (interval in INTRADAY_INTERVALS) and not is_crypto(ticker)
    if F is None:
        F = factor_matrix(d)

    regime = detect_regime(d)
    corr_analysis = factor_correlation(F)
    ir = information_ratio(F, d["Close"])

    comp = composite(F, weights, regime)
    score = float(comp.iloc[-1]); last = float(d["Close"].iloc[-1])
    atr_pct = float(d["atr"].iloc[-1]/last*100)
    # Annualized vol: for intraday, drop the overnight gap returns (they aren't a
    # day trader's risk and would otherwise dominate the estimate).
    rets = d["Close"].pct_change()
    if intraday:
        ns = _session_starts(d.index)
        if ns is not None:
            rets = rets.mask(pd.Series(ns, index=d.index))
    ann_vol = float(rets.std()*math.sqrt(ppy)*100)
    # Optional per-name thresholds calibrated to forward returns (Analyze tab).
    # The Screener leaves this off so its BUYs stay comparable across names.
    calib = calibrate_thresholds(comp.values, d["Close"].values) if calibrate else None
    fwd = forward_stats(comp.values, d["Close"].values) if calibrate else None
    # Thresholds: use per-name calibration if available, else volatility-scaled
    # defaults (a high-beta name needs a higher bar than a slow mega-cap).
    if calib:
        buy_th, strong_th = calib["buy"], calib["strong"]
    else:
        buy_th, strong_th = vol_thresholds(ann_vol)
    bt = backtest(d["Close"], comp, ppy, intraday, slippage_pct=SLIPPAGE_PCT)
    bt_by_regime = backtest_by_regime(d["Close"], comp, d, ppy, intraday)
    # Backtest GATE: if the rules have historically lost on this name (neg Sharpe
    # or sub-35% win rate), the asset is INELIGIBLE — no green light can rescue a
    # losing strategy, and callers skip the alt-data fetches for it.
    wr = bt["winrate"]
    ineligible = bool(bt["sharpe"] < 0 or (not math.isnan(wr) and bt["trades"] >= 3 and wr < 0.35))

    vrd = verdict(score, atr_pct, buy_th, strong_th, regime)
    conf = confirmation_checklist(score, vrd, conviction(F.iloc[-1], score), bt, fwd, buy_th, strong_th)

    return {"ticker": ticker, "d": d, "F": F, "score": score, "last": last,
            "chg": float((last/d["Close"].iloc[-2]-1)*100),
            "atr": float(d["atr"].iloc[-1]), "atr_pct": atr_pct,
            "ann_vol": ann_vol,
            "maxdd": float((d["Close"]/d["Close"].cummax()-1).min()*100),
            "verdict": vrd,
            "buy_th": buy_th, "strong_th": strong_th,
            "conviction": conviction(F.iloc[-1], score),
            "confirmation": conf,
            "bt": bt, "bt_by_regime": bt_by_regime, "ppy": ppy, "ineligible": ineligible,
            "intraday": intraday, "calib": calib, "fwd_stats": fwd,
            "limited_history": limited_history, "n_bars": nb,
            "whale_activity": whale_score(d),
            "regime": regime, "factor_correlation": corr_analysis, "information_ratio": ir}

# -------------------------------------------------------------- reports ---
def reasons(row):
    out = []
    out.append((row["Close"] > row["e20"], f"price {'above' if row['Close']>row['e20'] else 'below'} EMA20 ({row['e20']:.2f})"))
    out.append((row["e20"] > row["e50"], f"EMA20 {'>' if row['e20']>row['e50'] else '<'} EMA50 ({row['e50']:.2f})"))
    out.append((row["stdir"] > 0, f"Supertrend {'UP' if row['stdir']>0 else 'DOWN'} (line {row['st']:.2f})"))
    out.append((row["mach"] > 0, f"MACD histogram {row['mach']:+.3f}"))
    if row["rsi"] >= 70: out.append((False, f"RSI {row['rsi']:.0f} overbought"))
    elif row["rsi"] <= 30: out.append((True, f"RSI {row['rsi']:.0f} oversold"))
    else: out.append((row["rsi"] >= 50, f"RSI {row['rsi']:.0f}"))
    out.append((row["relvol"] >= 1.0, f"relative volume {row['relvol']:.2f}x avg"))
    if abs(row["z"]) >= 2:
        out.append((row["z"] < 0, f"z-score {row['z']:+.2f} — stretched {'below' if row['z']<0 else 'above'} mean"))
    return out

def report_pipeline(res, args):
    """Display pipeline output with clear stage-by-stage flow."""
    print(dim("\n" + "="*80))
    print(bold("PIPELINE VIEW: Market Data → Regime → Factors → Risk → Metrics"))
    print(dim("="*80))

    ticker = res["ticker"]
    stage1 = res["stage_1_data"]
    stage2 = res["stage_2_regime"]
    stage3 = res["stage_3_factors"]
    stage4 = res["stage_4_risk"]
    stage5 = res["stage_5_metrics"]

    print(f"\n{bold('STAGE 1: Market Data')} [{stage1['n_bars']} bars loaded]")
    print(f"  Ticker: {ticker}")
    print(f"  Period: {args.period} @ {args.interval}")

    print(f"\n{bold('STAGE 2: Regime Detector')} [{stage2['regime'].upper()}]")
    if stage2.get("confidence", 0) > 0:
        print(f"  Confidence: {stage2['confidence']*100:.0f}%")
        print(f"  EMA trend: {signed(stage2.get('ema_trend', 0), '{:+.1f}%')}")
        print(f"  Price vs EMA: {signed(stage2.get('price_vs_ema', 0), '{:+.1f}%')}")

    print(f"\n{bold('STAGE 3: Factor Analyzer')} [{len(stage3['F'].columns)} factors]")
    ir = stage3['ir']
    for k in FACTORS:
        if k in ir:
            ir_val = ir[k]
            bar = "█" * int(ir_val * 20) + "░" * (20 - int(ir_val * 20))
            print(f"  {k:<12} {signed(float(stage3['F'][k].iloc[-1])):<8} [IR {ir_val:.2f}] {bar}")

    print(f"\n{bold('STAGE 4: Risk Engine')} [Position Sizing]")
    if stage4.get("position"):
        pos = stage4["position"]
        print(f"  Account: ${stage4['account']:,.0f}")
        print(f"  Risk: {stage4['risk_pct']:.1f}% per trade")
        print(f"  Shares: {pos['shares']} @ {pos['entry']:.2f}")
        print(f"  Stop: {rd(format(pos['stop'], '.2f'))} | Target: {gr(format(pos['target'], '.2f'))}")

    print(f"\n{bold('STAGE 5: Metrics & Verdict')} [Final Output]")
    v = stage5["verdict"]
    score = stage5["score"]
    print(f"  Score: {signed(score, '{:+.0f}')}")
    print(f"  Verdict: {v['label']}")
    print(f"  Conviction: {stage5['conviction']}%")
    bt = stage5["backtest"]
    print(f"  Backtest Sharpe: {bt['sharpe']:.2f}")
    print(f"  Slippage cost: {bt.get('slippage_cost', 0)*100:.2f}%")

    print(dim("="*80 + "\n"))

def report(res, args, opt=None):
    d, F, row = res["d"], res["F"], res["d"].iloc[-1]
    score = res["score"]; v = res["verdict"]
    tone = {"good": gr, "neutral": yl, "bad": rd}[v["tone"]]
    line = dim("\u2500"*66)
    print(line)
    ticker = res["ticker"]
    last_px = f'{res["last"]:.2f}'
    chg_pct = res["chg"]
    bars_info = f'{len(d)} bars · {args.interval} · {args.period}'
    print(f"{bold(ticker):<16} last {bold(last_px)}  "
          f"{signed(chg_pct, '{:+.2f}%')}   "
          f"{dim(bars_info)}")

    conf = res.get("confirmation", {})
    if conf:
        level_color = {"FULL": gr, "PARTIAL": yl, "WEAK": rd}.get(conf["level"], dim)
        print(f"CONFIRMATION  {level_color(conf['level'])} - {conf['passed']}/{conf['total']} checks pass")
        for check in conf["checks"]:
            status = gr("✓") if check["pass"] else rd("✗")
            print(f"  {status} {check['name']:<45} {check['detail']}")
        print(line)

    print(bold("FACTORS") + dim("  (each -1..+1, weighted into the score)"))
    ir = res.get("information_ratio", {})
    for k in FACTORS:
        ir_str = ""
        if k in ir:
            ir_val = ir[k]
            ir_str = dim(f"  [IR {ir_val:.2f}]")
        print(f"  {k:<10} {signed(float(F[k].iloc[-1]))}{ir_str}")

    corr = res.get("factor_correlation", {})
    if corr and corr.get("n_highly_correlated", 0) > 0:
        print(dim(f"\n  [!] {corr['n_highly_correlated']} factor pairs highly correlated "
                  f"(redundancy)"))
        for f1, f2, c in corr.get("redundant_pairs", []):
            if abs(c) > 0.70:
                print(dim(f"      {f1} <-> {f2}: {c:+.2f}"))

    regime = res.get("regime", {})
    if regime.get("regime") != "unknown":
        regime_label = regime["regime"].upper()
        conf_pct = int(regime.get("confidence", 0) * 100)
        print(f"\n{bold('REGIME')} {regime_label} ({conf_pct}% confidence)")
        print(f"  EMA trend {signed(regime.get('ema_trend', 0), '{:+.1f}%')} · "
              f"price vs EMA {signed(regime.get('price_vs_ema', 0), '{:+.1f}%')} · "
              f"volatility {regime.get('volatility', 0):.1f}%")

    print()
    print(bold("EVIDENCE"))
    for good, txt in reasons(row):
        print(f"  {gr('▲') if good else rd('▼')} {txt}")
    print()
    print(bold("RISK"))
    print(f"  ATR {res['atr']:.2f} ({res['atr_pct']:.1f}% of price) · "
          f"ann vol {res['ann_vol']:.0f}% · max drawdown {res['maxdd']:.0f}% · "
          f"BB width {row['bb_bw']:.1f}%")
    if v["risky"]:
        print(rd("  ! RISKY: volatility this extreme sharply degrades signal reliability"))
    print()
    bt = res["bt"]
    wr = "n/a" if math.isnan(bt["winrate"]) else f"{bt['winrate']*100:.0f}%"
    sh = "n/a" if math.isnan(bt["sharpe"]) else f"{bt['sharpe']:.2f}"
    print(bold("BACKTEST") + dim(" (long score>18, flat <0 · includes slippage · in-sample)"))
    print(f"  strategy {signed(bt['strategy']*100, '{:+.1f}%')} vs buy&hold "
          f"{signed(bt['buyhold']*100, '{:+.1f}%')} · trades {bt['trades']} · "
          f"win {wr} · Sharpe {sh} · maxDD {bt['maxdd']*100:.0f}% · exposure {bt['exposure']*100:.0f}%")
    if bt.get("slippage_cost", 0) > 0:
        print(f"  slippage cost: {bold(format(bt['slippage_cost']*100, '.2f'))}% total · "
              f"{bt['slippage_cost']*100/max(1, bt['trades']):.3f}% per trade")

    bt_by_regime = res.get("bt_by_regime", {})
    if bt_by_regime and len(bt_by_regime) > 1:
        print(f"\n  {dim('performance by regime:')}")
        for regime_name in ["bull", "bear", "range"]:
            if regime_name in bt_by_regime:
                r = bt_by_regime[regime_name]
                sh_r = "n/a" if math.isnan(r["sharpe"]) else f"{r['sharpe']:.2f}"
                print(f"    {regime_name:<6} {signed(r['strategy']*100, '{:+.1f}%'):>7} · "
                      f"trades {r['trades']:<3} · Sharpe {sh_r:<6} · maxDD {r['maxdd']*100:>6.0f}%")
    print()
    if args.account:
        ps = position_size(res["last"], res["atr"], args.account, args.risk)
        print(bold("POSITION SIZING") + dim(f" (risk {args.risk}% of {args.account:,.0f}, stop 2xATR)"))
        if ps:
            print(f"  {ps['shares']} shares (~{ps['notional']:,.0f}) · entry {ps['entry']:.2f} · "
                  f"stop {rd(format(ps['stop'], '.2f'))} · target {gr(format(ps['target'], '.2f'))} · "
                  f"risk if stopped ~{rd(format(ps['risk_dollars'], ',.0f'))} · "
                  f"est. profit if target ~{gr(format(ps['reward_dollars'], ',.0f'))}")
        else:
            print(dim("  position too small to size at this risk level"))
        print()
    if opt:
        f = lambda x: "n/a" if (x is None or x <= -9 or math.isnan(x)) else f"{x:.2f}"
        fl = lambda x: "n/a" if (x is None or x >= 999 or math.isnan(x)) else f"{x:.2f}"
        print(bold("ANNEALED WEIGHTS ") + dim(str(opt["weights"])))
        print(f"  Sharpe — train {f(opt['train_sharpe'])} · unseen test {f(opt['test_sharpe'])} · "
              f"default weights on same test {f(opt['base_test_sharpe'])}")
        print(dim(f"  objective (−Sharpe + 3·MaxDD + 2·Turnover) — "
                  f"train {fl(opt['train_loss'])} · test {fl(opt['test_loss'])}"))
        if "wf_sharpe" in opt:
            folds = " ".join(f(s) for s in opt["wf_folds"])
            print(f"  walk-forward OOS Sharpe {f(opt['wf_sharpe'])} "
                  f"({opt['wf_pos_frac']*100:.0f}% of {len(opt['wf_folds'])} folds positive) "
                  + dim(f"folds [{folds}]"))
            if opt["wf_pos_frac"] < 0.5:
                print(yl("  caution: most walk-forward folds negative — weights don't generalize"))
        elif isinstance(opt['test_sharpe'], float) and isinstance(opt['train_sharpe'], float) \
                and opt['test_sharpe'] < opt['train_sharpe']*0.4:
            print(yl("  caution: big train→test drop = the tuning likely overfit this history"))
        print()
    tag = tone(bold(v["label"])) + (rd("  [RISKY]") if v["risky"] else "")
    print(bold("VERDICT ") + f"score {signed(score, '{:+.0f}')}  " + tag
          + dim(f"  · conviction {res['conviction']}%"))
    print(line)

def scan(results):
    tone = {"good": gr, "neutral": yl, "bad": rd}
    print(dim("\u2500"*75))
    print(bold(f"{'TICKER':<8}{'LAST':>10}{'CHG%':>9}{'SCORE':>8}  {'REGIME':<7}  VERDICT"))
    for r in sorted(results, key=lambda x: -x["score"]):
        v = r["verdict"]
        regime = r.get("regime", {}).get("regime", "?").upper()[:5]
        lab = tone[v["tone"]](v["label"]) + (rd(" [RISKY]") if v["risky"] else "")
        print(f"{r['ticker']:<8}{r['last']:>10.2f}{signed(r['chg'], '{:+.2f}'):>9}"
              f"{signed(r['score'], '{:+.0f}'):>8}  {regime:<7}  {lab}")
    print(dim("\u2500"*75))

def dashboard(results):
    """Portfolio dashboard: compact watchlist view for multiple stocks."""
    tone_map = {"good": gr, "neutral": yl, "bad": rd}
    conf_symbol = {"FULL": gr("\u25cf"), "PARTIAL": yl("\u25cf"), "WEAK": rd("\u25cf")}

    print()
    print(dim("="*95))
    print(bold("WATCHLIST DASHBOARD"))
    print(dim("="*95))
    print(f"{bold('SCORE'):<8} {bold('TICKER'):<8} {bold('PRICE'):>12} {bold('CHG%'):>8} {bold('VERDICT'):>20} {bold('CONF'):>6}")
    print(dim("-"*95))

    for r in sorted(results, key=lambda x: -x["score"]):
        score = r["score"]
        ticker = r["ticker"]
        price = r["last"]
        chg = r["chg"]
        v = r["verdict"]
        conf = r.get("confirmation", {})

        score_color = gr if score >= 40 else yl if score >= 0 else rd
        verdict_str = v["label"][:18]
        verdict_colored = tone_map[v["tone"]](verdict_str)
        conf_level = conf.get("level", "?")
        conf_display = conf_symbol.get(conf_level, "?")

        print(f"{score_color(f'{score:+.0f}'):<8} {ticker:<8} {price:>12.2f} "
              f"{signed(chg, '{:+.1f}%'):>8} {verdict_colored:>20} {conf_display:>6}")

    print(dim("="*95))
    print(dim(f"Updated {pd.Timestamp.now().strftime('%I:%M:%S %p')}"))

def load_positions(filepath="positions.json"):
    """Load positions from JSON file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"positions": [], "closed": []}

def save_positions(data, filepath="positions.json"):
    """Save positions to JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def add_position(ticker, entry_price, conviction, filepath="positions.json"):
    """Add a new open position."""
    data = load_positions(filepath)
    position = {
        "ticker": ticker,
        "entry_price": entry_price,
        "entry_time": datetime.now().isoformat(),
        "entry_conviction": conviction,
        "shares": 100,
        "status": "open"
    }
    data["positions"].append(position)
    save_positions(data, filepath)
    return position

def close_position(ticker, exit_price, filepath="positions.json"):
    """Close an open position."""
    data = load_positions(filepath)
    for pos in data["positions"]:
        if pos["ticker"] == ticker and pos["status"] == "open":
            pos["exit_price"] = exit_price
            pos["exit_time"] = datetime.now().isoformat()
            pos["status"] = "closed"
            pnl = (exit_price - pos["entry_price"]) * pos["shares"]
            pnl_pct = (exit_price / pos["entry_price"] - 1) * 100
            pos["pnl"] = pnl
            pos["pnl_pct"] = pnl_pct
            data["closed"].append(pos)
            data["positions"].remove(pos)
            save_positions(data, filepath)
            return pos
    return None

def update_positions_live(results, filepath="positions.json"):
    """Update open positions with live prices from results."""
    data = load_positions(filepath)
    result_map = {r["ticker"]: r for r in results}

    for pos in data["positions"]:
        if pos["ticker"] in result_map:
            res = result_map[pos["ticker"]]
            current_price = res["last"]
            current_conviction = res.get("conviction", 0)

            pnl = (current_price - pos["entry_price"]) * pos["shares"]
            pnl_pct = (current_price / pos["entry_price"] - 1) * 100

            pos["current_price"] = current_price
            pos["current_conviction"] = current_conviction
            pos["pnl"] = pnl
            pos["pnl_pct"] = pnl_pct
            pos["conviction_change"] = current_conviction - pos["entry_conviction"]
            pos["signal"] = res.get("verdict", {}).get("label", "?")

    save_positions(data, filepath)
    return data

def show_positions(filepath="positions.json"):
    """Display open and closed positions with performance."""
    data = load_positions(filepath)
    tone_map = {"good": gr, "neutral": yl, "bad": rd}

    print()
    print(dim("="*120))
    print(bold("POSITION TRACKING - LIVE P&L"))
    print(dim("="*120))

    if data["positions"]:
        print(f"\n{bold('OPEN POSITIONS')} ({len(data['positions'])})")
        print(dim("-"*120))
        print(f"{bold('TICKER'):<10} {bold('ENTRY'):>12} {bold('CURRENT'):>12} {bold('P&L'):>15} {bold('P&L%'):>8} {bold('CONV'):>6} {bold('CHG'):>6} {bold('SIGNAL'):<20}")
        print(dim("-"*120))

        total_pnl = 0
        for pos in sorted(data["positions"], key=lambda x: -x.get("pnl_pct", 0)):
            ticker = pos["ticker"]
            entry = pos["entry_price"]
            current = pos.get("current_price", entry)
            pnl = pos.get("pnl", 0)
            pnl_pct = pos.get("pnl_pct", 0)
            entry_conv = pos.get("entry_conviction", 0)
            curr_conv = pos.get("current_conviction", 0)
            conv_chg = pos.get("conviction_change", 0)
            signal = pos.get("signal", "?")[:19]

            total_pnl += pnl
            pnl_color = gr if pnl >= 0 else rd
            conv_color = gr if conv_chg >= 0 else rd

            print(f"{ticker:<10} ${entry:>11.2f} ${current:>11.2f} "
                  f"{pnl_color(f'${pnl:>13.0f}'):>15} {pnl_color(f'{pnl_pct:>7.1f}%'):>8} "
                  f"{entry_conv:>5.0f}% {conv_color(f'{conv_chg:>+5.0f}%'):>6} {signal:<20}")

        print(dim("-"*120))
        total_color = gr if total_pnl >= 0 else rd
        print(f"{bold('TOTAL'):<10} {total_color(f'${total_pnl:>13.0f}'):>43} ")
    else:
        print(f"\n{dim('No open positions')}")

    if data["closed"]:
        print(f"\n{bold('CLOSED POSITIONS')} ({len(data['closed'])})")
        print(dim("-"*120))
        print(f"{bold('TICKER'):<10} {bold('ENTRY'):>12} {bold('EXIT'):>12} {bold('P&L'):>15} {bold('P&L%'):>8} {bold('HOLD TIME'):<15}")
        print(dim("-"*120))

        total_closed_pnl = 0
        wins = 0
        for pos in sorted(data["closed"], key=lambda x: -x.get("pnl_pct", 0))[:10]:
            ticker = pos["ticker"]
            entry = pos["entry_price"]
            exit_p = pos.get("exit_price", 0)
            pnl = pos.get("pnl", 0)
            pnl_pct = pos.get("pnl_pct", 0)

            total_closed_pnl += pnl
            if pnl > 0:
                wins += 1

            entry_time = datetime.fromisoformat(pos["entry_time"])
            exit_time = datetime.fromisoformat(pos.get("exit_time", pos["entry_time"]))
            hold_time = (exit_time - entry_time).days
            hold_str = f"{hold_time}d" if hold_time > 0 else "<1d"

            pnl_color = gr if pnl >= 0 else rd

            print(f"{ticker:<10} ${entry:>11.2f} ${exit_p:>11.2f} "
                  f"{pnl_color(f'${pnl:>13.0f}'):>15} {pnl_color(f'{pnl_pct:>7.1f}%'):>8} {hold_str:<15}")

        print(dim("-"*120))
        win_rate = (wins / len(data["closed"]) * 100) if data["closed"] else 0
        total_color = gr if total_closed_pnl >= 0 else rd
        print(f"{bold('CLOSED TOTAL'):<10} {total_color(f'${total_closed_pnl:>13.0f}'):>43} Win rate: {win_rate:.0f}%")

    print(dim("="*120))

def portfolio_alerts(results):
    """Portfolio alerts: categorize stocks by opportunity/risk with catalysts."""
    categorized = categorize_portfolio(results)
    tone_map = {"good": gr, "neutral": yl, "bad": rd}

    print()
    print(dim("="*100))
    print(bold("PORTFOLIO ALERTS - BUY CANDIDATES & RISKS"))
    print(dim("="*100))

    buy = categorized["buy"]
    if buy:
        print(f"\n{gr('GREEN - BUY CANDIDATES')} ({len(buy)} stocks)")
        print(dim("-"*100))
        for r in buy:
            score = r["score"]
            ticker = r["ticker"]
            verdict = r["verdict"]["label"]
            insiders = r.get("insiders", {})

            print(f"  {gr(ticker):<8} {gr(f'{score:+.0f}'):>4} {verdict}")
            if insiders and insiders.get("buy_usd", 0) > 0:
                print(f"    Insider buy: ${insiders['buy_usd']/1e6:.1f}M")
            if r.get("macro_signal", {}).get("signal", 0) > 0.4:
                print(f"    News tone: {r['macro_signal']['signal']:+.2f}")
            print(f"    Risk: {r.get('red_flags', {}).get('risk_level', 'MEDIUM')}")
    else:
        print(f"\n{gr('GREEN - BUY CANDIDATES')} (none)")

    risk = categorized["risk"]
    if risk:
        print(f"\n{rd('RED - RISK / AVOID')} ({len(risk)} stocks)")
        print(dim("-"*100))
        for r in risk:
            score = r["score"]
            ticker = r["ticker"]
            verdict = r["verdict"]["label"]
            red_flags = r.get("red_flags", {})

            print(f"  {rd(ticker):<8} {rd(f'{score:+.0f}'):>4} {verdict}")
            for flag in red_flags.get("flags", [])[:3]:
                print(f"    {rd('DOWN')} {flag}")
            print(f"    Risk: {red_flags.get('risk_level', 'UNKNOWN')}")
    else:
        print(f"\n{rd('RED - RISK / AVOID')} (none)")

    neutral = categorized["neutral"]
    neutral_tickers = ", ".join([r["ticker"] for r in neutral[:10]])
    remaining = f" +{len(neutral)-10} more" if len(neutral) > 10 else ""
    print(f"\n{yl('YELLOW - NEUTRAL')} ({len(neutral)} stocks)")
    print(f"  {neutral_tickers}{remaining}")

    print(dim("="*100))

DISCLAIMER = ("Rules-based technical signals on historical data — not financial advice. "
              "Backtest ignores fees/slippage and is in-sample; optimized weights can "
              "overfit. Past patterns do not predict future returns.")

# ------------------------------------------------------------------ main ---
def main():
    ap = argparse.ArgumentParser(description="Quant engine — scores stocks, issues buy/hold/avoid verdicts.")
    ap.add_argument("tickers", nargs="*", default=[])
    ap.add_argument("--period", default="6mo")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--demo", action="store_true", help="synthetic data, no internet")
    ap.add_argument("--optimize", action="store_true", help="anneal factor weights (70/30 train-test)")
    ap.add_argument("--account", type=float, default=None, help="account size for position sizing")
    ap.add_argument("--risk", type=float, default=1.0, help="percent risked per trade (default 1)")
    ap.add_argument("--pipeline", action="store_true", help="show modular pipeline view (stages 1-5)")
    ap.add_argument("--dashboard", action="store_true", help="show portfolio dashboard view (multi-stock)")
    ap.add_argument("--alerts", action="store_true", help="show portfolio alerts (BUY/RISK categorization)")
    ap.add_argument("--positions", action="store_true", help="show open/closed positions with live P&L")
    ap.add_argument("--add-position", nargs=2, metavar=("TICKER", "ENTRY_PRICE"), help="add a new position (ticker, entry_price)")
    ap.add_argument("--close-position", nargs=2, metavar=("TICKER", "EXIT_PRICE"), help="close a position (ticker, exit_price)")
    ap.add_argument("--finnhub-key", default=os.environ.get("FINNHUB_KEY", FINNHUB_DEFAULT_KEY))
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    global COLOR
    if args.no_color: COLOR = False

    if args.positions:
        show_positions()
        sys.exit(0)

    if args.add_position:
        ticker, entry_price = args.add_position
        pos = add_position(ticker, float(entry_price), 75)
        print(f"\nAdded position: {ticker} @ ${pos['entry_price']:.2f}")
        sys.exit(0)

    if args.close_position:
        ticker, exit_price = args.close_position
        pos = close_position(ticker, float(exit_price))
        if pos:
            print(f"\nClosed position: {ticker}")
            print(f"  Entry: ${pos['entry_price']:.2f}, Exit: ${pos['exit_price']:.2f}")
            print(f"  P&L: ${pos['pnl']:.2f} ({pos['pnl_pct']:+.1f}%)")
            sys.exit(0)
        else:
            print(f"Position {ticker} not found")
            sys.exit(1)

    tickers = [t.upper() for t in args.tickers] or (["DEMO"] if args.demo else [])
    if not tickers:
        ap.error("give at least one ticker, or use --demo")

    results, errors = [], []
    for t in tickers:
        try:
            df = demo_data(t) if args.demo else fetch(t, args.period, args.interval)
            weights, opt = None, None
            if args.optimize:
                d0 = enrich(df); F0 = factor_matrix(d0)
                opt = optimize_weights(F0, d0["Close"], PPY.get(args.interval, 252))
                weights = opt["weights"]
            if args.pipeline:
                res = analyze_pipeline(t, df, args.interval, args.account, args.risk)
            else:
                res = analyze(t, df, args.interval, weights)
            results.append((res, opt))
        except Exception as e:
            errors.append(f"{t}: {e}")

    if len(results) == 1:
        res, opt = results[0]
        if args.pipeline:
            report_pipeline(res, args)
        else:
            report(res, args, opt)
        if not args.demo and args.finnhub_key:
            rec = finnhub_recs(res["ticker"], args.finnhub_key)
            if rec:
                print(bold("ANALYSTS (Finnhub)") +
                      f"  strong buy {rec.get('strongBuy',0)} · buy {rec.get('buy',0)} · "
                      f"hold {rec.get('hold',0)} · sell {rec.get('sell',0)} · "
                      f"strong sell {rec.get('strongSell',0)}  {dim(rec.get('period',''))}")
                print()
    elif results:
        results_list = [r for r, _ in results]
        update_positions_live(results_list)
        if args.alerts:
            portfolio_alerts(results_list)
        elif args.dashboard:
            dashboard(results_list)
        else:
            scan(results_list)

    for e in errors:
        print(rd("error ") + e)
    if results:
        print(dim(DISCLAIMER))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)

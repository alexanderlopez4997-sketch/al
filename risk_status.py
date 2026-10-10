#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio risk snapshot for the web terminal's dashboard bar: kill-switch
state, drawdowns, open-risk "heat", 1-day VaR and position count.

Read-only port of the Quantitative Terminal's risk_manager.py
(PortfolioRiskManager.dashboard_data and the helpers it calls) with the same
rules and the same on-disk layout, so Meridian's ``paper_trading/`` folder can
be copied over as-is to see real numbers:

    paper_trading/positions.json     {SYMBOL: {entry_price, stop_loss, shares, notional, risk_dollars, sector?}}
    paper_trading/account.json       {"initial_capital": ..., "current_value": ...}
    paper_trading/equity_curve.json  [{"date": ISO8601, "value": ...}, ...]
    logs/trades.csv                  date,symbol,...,pnl_pct,... (recent closed trades)

Missing or corrupt files degrade to the same defaults Meridian uses ($100,000,
no positions) rather than raising. Trade evaluation / sizing / the Massive and
filing checks that live in risk_manager.py are NOT part of this module, and
nothing here writes to those files.
"""
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:                                  # VaR just reports n/a without it
    yf = None

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "paper_trading"
TRADES_LOG = BASE_DIR / "logs" / "trades.csv"

DEFAULT_EQUITY = 100000

# Synthetic, mildly drawn-down book for demo mode (same numbers as Meridian's --demo).
_DEMO_CURVE = [100000, 101200, 100500, 99800, 98600, 99100, 97900]
_DEMO_TRADE_PNLS = [-1, -1, -1, -1]


@dataclass
class RiskConfig:
    max_portfolio_heat_pct: float = 6.0     # cap on total open risk across all positions
    max_positions: int = 15
    daily_loss_limit_pct: float = 3.0       # halt new entries if equity drops this much day-over-day
    max_intraday_drawdown_pct: float = 2.0  # halt (latched) if equity falls this far from TODAY's peak
    max_drawdown_pct: float = 10.0          # halt new entries if equity is this far below its running peak
    consecutive_loss_halt: int = 4          # halt after this many losing trades in a row


# ------------------------------------------------------------------- data I/O ---
def _load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _validate_position(symbol, pos):
    required = {"entry_price", "stop_loss", "shares", "notional", "risk_dollars"}
    missing = required - set(pos.keys())
    if missing:
        raise ValueError(f"Position {symbol} missing keys: {missing}")
    if not isinstance(pos["entry_price"], (int, float)) or pos["entry_price"] <= 0:
        raise ValueError(f"Position {symbol} invalid entry_price: {pos['entry_price']}")
    if not isinstance(pos["shares"], int) or pos["shares"] <= 0:
        raise ValueError(f"Position {symbol} invalid shares: {pos['shares']}")
    for k in ("stop_loss", "notional", "risk_dollars"):
        if not isinstance(pos[k], (int, float)) or pos[k] < 0:
            raise ValueError(f"Position {symbol} invalid {k}: {pos[k]}")


def load_positions(data_dir=DATA_DIR):
    """Validated positions, or {} when the file is missing or corrupt."""
    positions = _load_json(Path(data_dir) / "positions.json", {})
    try:
        if not isinstance(positions, dict):
            raise ValueError("positions.json is not an object")
        for symbol, pos in positions.items():
            _validate_position(symbol, pos)
    except (ValueError, AttributeError, TypeError):
        return {}
    return positions


def load_account(data_dir=DATA_DIR):
    default = {"initial_capital": DEFAULT_EQUITY, "current_value": DEFAULT_EQUITY}
    account = _load_json(Path(data_dir) / "account.json", default)
    try:
        initial = account.get("initial_capital", DEFAULT_EQUITY)
        current = account.get("current_value", DEFAULT_EQUITY)
        if not isinstance(initial, (int, float)) or initial <= 0:
            raise ValueError("invalid initial_capital")
        if not isinstance(current, (int, float)) or current < 0:
            raise ValueError("invalid current_value")
    except (ValueError, AttributeError):
        return default
    return account


def load_equity_curve(data_dir=DATA_DIR):
    """[{"date", "value"}, ...] oldest first; entries without both fields are dropped."""
    curve = _load_json(Path(data_dir) / "equity_curve.json", [])
    if not isinstance(curve, list):
        return []
    return [pt for pt in curve if isinstance(pt, dict) and "date" in pt
            and isinstance(pt.get("value"), (int, float))]


def load_recent_trade_pnls(n, trades_log=TRADES_LOG):
    try:
        with open(trades_log, newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return []
    pnls = []
    for r in rows[-n:]:
        try:
            pnls.append(float(r["pnl_pct"]))
        except (KeyError, ValueError, TypeError):
            continue
    return pnls


# ----------------------------------------------------------------- risk math ---
def drawdown_circuit_breaker(portfolio_history, max_allowed_drawdown=0.10):
    """Account-level kill switch: HALT_TRADING once equity is more than
    max_allowed_drawdown below its running peak (same rule as
    portfolio_drawdown_circuit_breaker in the Meridian repo's quant_engine)."""
    curve = pd.Series([float(v) for v in portfolio_history])
    if len(curve) == 0:
        return {"status": "ACTIVE", "current_drawdown": "0.00%"}
    dd = (curve - curve.cummax()) / curve.cummax().replace(0, np.nan)
    current = float(dd.iloc[-1]) if dd.iloc[-1] == dd.iloc[-1] else 0.0
    status = "HALT_TRADING" if current <= -max_allowed_drawdown else "ACTIVE"
    return {"status": status, "current_drawdown": f"{current * 100:.2f}%"}


def historical_var(returns, confidence_level=0.95):
    """Historical 1-period VaR as a positive percentage (e.g. 3.2 means daily
    losses worse than -3.2% happened ~1 day in 20)."""
    clean = pd.Series(returns).dropna()
    if len(clean) == 0:
        return 0.0
    return max(0.0, float(-np.percentile(clean, (1.0 - confidence_level) * 100)))


def _daily_returns(symbols, period="3mo"):
    if yf is None or not symbols:
        return pd.DataFrame()
    try:
        data = yf.download(list(symbols), period=period, progress=False, auto_adjust=True)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame(symbols[0])
        returns = data.pct_change().dropna(how="all")
        return returns[[c for c in returns.columns if returns[c].notna().any()]]
    except Exception:
        return pd.DataFrame()


class RiskSnapshot:
    """Loads the state files once and answers the dashboard-bar questions."""

    def __init__(self, config=None, demo=False, data_dir=DATA_DIR, trades_log=TRADES_LOG):
        self.config = config or RiskConfig()
        self.demo = demo
        self.data_dir = Path(data_dir)
        self.trades_log = trades_log
        self.positions = load_positions(self.data_dir)
        self.account = load_account(self.data_dir)

    def equity(self):
        return float(self.account.get("current_value", self.account.get("initial_capital", DEFAULT_EQUITY)))

    def _curve(self):
        return list(_DEMO_CURVE) if self.demo else load_equity_curve(self.data_dir)

    def intraday_status(self, now=None):
        """Drawdown from TODAY's peak, latched: once it crosses the limit at any
        point today it stays HALT for the rest of the day even if equity recovers."""
        if self.demo:
            points = list(_DEMO_CURVE)
        else:
            today = (now or datetime.now()).date().isoformat()
            points = [pt["value"] for pt in load_equity_curve(self.data_dir) if str(pt["date"])[:10] == today]
        if len(points) < 2:
            return {"status": "ACTIVE", "reason": None, "current_drawdown_pct": 0.0}
        peak, tripped_at = points[0], None
        limit = self.config.max_intraday_drawdown_pct / 100.0
        for v in points[1:]:
            peak = max(peak, v)
            dd = (peak - v) / peak if peak else 0.0
            if tripped_at is None and dd >= limit:
                tripped_at = dd
        final_dd = (peak - points[-1]) / peak if peak else 0.0
        if tripped_at is not None:
            return {"status": "HALT", "current_drawdown_pct": round(final_dd * 100, 2),
                    "reason": f"Intraday drawdown from today's peak hit {tripped_at * 100:.2f}% "
                              f"(limit {self.config.max_intraday_drawdown_pct:.1f}%) — "
                              f"halted for the rest of today regardless of recovery"}
        return {"status": "ACTIVE", "reason": None, "current_drawdown_pct": round(final_dd * 100, 2)}

    def kill_switch(self, now=None):
        cfg, reasons = self.config, []
        curve = self._curve()
        values = curve if self.demo else [pt["value"] for pt in curve]
        dd = drawdown_circuit_breaker(values, cfg.max_drawdown_pct / 100.0)
        if dd["status"] == "HALT_TRADING":
            reasons.append(f"Max drawdown breached ({dd['current_drawdown']}, limit -{cfg.max_drawdown_pct:.0f}%)")
        if len(values) >= 2 and values[-2] > 0:
            move = (values[-1] - values[-2]) / values[-2] * 100.0
            if move <= -cfg.daily_loss_limit_pct:
                reasons.append(f"Daily loss limit breached ({move:+.2f}%, limit -{cfg.daily_loss_limit_pct:.0f}%)")
        intraday = self.intraday_status(now)
        if intraday["status"] == "HALT":
            reasons.append(intraday["reason"])
        pnls = _DEMO_TRADE_PNLS if self.demo else load_recent_trade_pnls(cfg.consecutive_loss_halt, self.trades_log)
        streak = pnls[-cfg.consecutive_loss_halt:]
        if len(streak) == cfg.consecutive_loss_halt and all(p < 0 for p in streak):
            reasons.append(f"{cfg.consecutive_loss_halt} consecutive losing trades")
        return {"status": "HALT" if reasons else "ACTIVE", "reasons": reasons,
                "current_drawdown": dd["current_drawdown"], "intraday": intraday}

    def heat_pct(self):
        equity = self.equity()
        total = sum(p["risk_dollars"] for p in self.positions.values())
        return round(total / equity * 100.0, 3) if equity else 0.0

    def var_1d_pct(self):
        """1-day 95% VaR of the book in %, 0 with no positions, None when no price data."""
        if not self.positions:
            return 0.0
        if self.demo:
            return 1.9
        symbols = list(self.positions)
        returns = _daily_returns(symbols)
        if returns.empty:
            return None
        notional = sum(p["notional"] for p in self.positions.values()) or 1.0
        weights = {s: self.positions[s]["notional"] / notional for s in symbols if s in returns.columns}
        if not weights:
            return None
        book = sum(returns[s].fillna(0) * w for s, w in weights.items())
        return round(historical_var(book) * 100, 2)

    def status(self, now=None):
        """The dict /api/dashboard sends to the bar."""
        ks = self.kill_switch(now)
        var = self.var_1d_pct()
        return {
            "available": True,
            "equity": round(self.equity(), 2),
            "status": ks["status"],
            "reasons": ks["reasons"],
            "current_drawdown": ks["current_drawdown"],
            "intraday_drawdown_pct": ks["intraday"]["current_drawdown_pct"],
            "portfolio_heat_pct": round(self.heat_pct(), 2),
            "portfolio_heat_cap_pct": self.config.max_portfolio_heat_pct,
            "portfolio_var_1d_pct": var,
            "open_positions": len(self.positions),
            "max_positions": self.config.max_positions,
        }


def risk_status(demo=False, **kw):
    """Never raises: a broken state file must not take the dashboard down."""
    try:
        return RiskSnapshot(demo=demo, **kw).status()
    except Exception as e:
        return {"available": False, "error": str(e)}

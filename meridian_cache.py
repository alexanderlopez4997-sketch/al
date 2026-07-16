#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persistent daily-bar price cache for Meridian (SQLite).

Scope: DAILY (1d) bars only — immutable history, which is where the repeated
re-download cost lives (a 120-name Discover scan pulls 6 months of daily data
every time). Intraday intervals are NOT cached here; they change intra-session
and the free feed is delayed, so the caller passes them straight to yfinance.

Freshness rule: a ticker's cache is valid if it was written today (US/Eastern)
AND the stored history is at least as long as the requested period. Worst case
on a market holiday is one harmless extra download — it never serves stale data.

Fails open: every public method is wrapped by callers in try/except so a cache
problem degrades to a normal live fetch instead of breaking the app.
"""
import os
import sqlite3
from contextlib import closing
from datetime import datetime

import pandas as pd

try:
    from zoneinfo import ZoneInfo          # stdlib 3.9+ — no pytz dependency
    _ET = ZoneInfo("America/New_York")
except Exception:                          # pragma: no cover
    _ET = None

# Rough calendar-day span of each period string, used to check whether cached
# history is long enough and to trim it back to the requested window.
_PERIOD_DAYS = {"1mo": 31, "3mo": 93, "6mo": 186, "1y": 372, "2y": 744}


class MeridianCache:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "meridian_cache.db")
        self.db_path = db_path
        self._init_db()

    def _conn(self):
        c = sqlite3.connect(self.db_path, timeout=10)
        c.execute("PRAGMA journal_mode=WAL")      # concurrent reads during a write
        c.execute("PRAGMA busy_timeout=10000")    # wait out locks from worker threads
        return c

    def _init_db(self):
        with closing(self._conn()) as c, c:
            c.execute("""CREATE TABLE IF NOT EXISTS price_history (
                ticker TEXT, date TEXT,
                open REAL, high REAL, low REAL, close REAL, volume INTEGER,
                PRIMARY KEY (ticker, date))""")
            c.execute("""CREATE TABLE IF NOT EXISTS cache_meta (
                ticker TEXT PRIMARY KEY, last_updated TEXT, period TEXT)""")

    def _now_et(self):
        return datetime.now(_ET) if _ET else datetime.now()

    def is_valid(self, ticker, period):
        """True if the cache for `ticker` was written today (ET) and covers `period`."""
        with closing(self._conn()) as c:
            row = c.execute("SELECT last_updated, period FROM cache_meta WHERE ticker=?",
                            (ticker,)).fetchone()
        if not row:
            return False
        last_updated, cached_period = row
        try:
            lu = datetime.fromisoformat(last_updated)
        except (ValueError, TypeError):
            return False
        if _ET and lu.tzinfo is not None:
            lu = lu.astimezone(_ET)
        if lu.date() != self._now_et().date():
            return False
        return _PERIOD_DAYS.get(cached_period, 0) >= _PERIOD_DAYS.get(period, 10**9)

    def save(self, ticker, df, period):
        """Upsert daily OHLCV bars (INSERT OR REPLACE — safe to re-save overlaps)."""
        if df is None or df.empty:
            return
        rows = []
        for ts, r in df.iterrows():
            d = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
            rows.append((ticker, d, float(r["Open"]), float(r["High"]), float(r["Low"]),
                         float(r["Close"]), int(r["Volume"])))
        with closing(self._conn()) as c, c:
            c.executemany("""INSERT OR REPLACE INTO price_history
                (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)""", rows)
            c.execute("""INSERT OR REPLACE INTO cache_meta (ticker, last_updated, period)
                VALUES (?, ?, ?)""", (ticker, self._now_et().isoformat(), period))

    def get(self, ticker, period):
        """Return cached bars trimmed to `period` as an OHLCV DataFrame
        (DatetimeIndex, columns matching quant_engine's expectations)."""
        with closing(self._conn()) as c:
            df = pd.read_sql_query(
                "SELECT date, open, high, low, close, volume FROM price_history "
                "WHERE ticker=? ORDER BY date ASC",
                c, params=(ticker,), parse_dates=["date"])
        if df.empty:
            return pd.DataFrame()
        df.set_index("date", inplace=True)
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        days = _PERIOD_DAYS.get(period)
        if days:
            cutoff = pd.Timestamp(self._now_et().date()) - pd.Timedelta(days=days)
            df = df[df.index >= cutoff]
        return df

    def clear(self):
        with closing(self._conn()) as c, c:
            c.execute("DELETE FROM price_history")
            c.execute("DELETE FROM cache_meta")

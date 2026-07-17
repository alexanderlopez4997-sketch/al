#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NO EDGE detection and Track Record override logic.

Implements:
1. check_edge(factor_id, ir, win_rate) → edge_status, is_active
2. track_record_override(factor_id) → override_active, track_record_data
3. Database schema for storing factor performance history
"""
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

DB_PATH = os.path.expanduser("~/.meridian_cache/meridian_cache.db")

EDGE_THRESHOLD_IR = 0.05
EDGE_THRESHOLD_WINRATE = 0.50

TRACK_RECORD_MIN_TRADES = 100
TRACK_RECORD_MIN_SHARPE = 1.2
TRACK_RECORD_LOOKBACK_DAYS = 365


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS factor_track_record (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            factor_id TEXT NOT NULL,
            date TEXT NOT NULL,
            win_count INTEGER DEFAULT 0,
            loss_count INTEGER DEFAULT 0,
            total_trades INTEGER DEFAULT 0,
            avg_return REAL DEFAULT 0.0,
            sharpe_ratio REAL DEFAULT 0.0,
            information_ratio REAL DEFAULT 0.0,
            max_drawdown REAL DEFAULT 0.0,
            created_at TEXT NOT NULL,
            UNIQUE(factor_id, date)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS factor_edge_status (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            factor_id TEXT NOT NULL UNIQUE,
            edge_status TEXT DEFAULT 'ACTIVE',
            last_checked TEXT,
            current_ir REAL,
            current_winrate REAL,
            override_active INTEGER DEFAULT 0,
            override_reason TEXT,
            updated_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def check_edge(factor_id: str, ir: float, win_rate: float) -> Tuple[str, bool]:
    """
    Check if a factor has NO EDGE based on Information Ratio and Win Rate.

    NO EDGE criteria:
    - Information Ratio < 0.05 OR Win Rate < 50%

    Returns:
        (edge_status: str, is_active: bool)
        - edge_status: "ACTIVE" if has edge, "NO_EDGE" if doesn't have edge
        - is_active: True if signal should be active, False if should be suppressed
    """
    has_ir_edge = ir >= EDGE_THRESHOLD_IR
    has_winrate_edge = win_rate >= EDGE_THRESHOLD_WINRATE

    if has_ir_edge and has_winrate_edge:
        edge_status = "ACTIVE"
        is_active = True
    else:
        edge_status = "NO_EDGE"
        is_active = False

    update_edge_status(factor_id, edge_status, ir, win_rate)
    return edge_status, is_active


def track_record_override(factor_id: str) -> Tuple[bool, Optional[Dict]]:
    """
    Check if a factor has a strong historical track record that overrides NO EDGE.

    Override criteria (if ANY is met):
    - > 100 successful trades in the last 12 months
    - Sharpe Ratio > 1.2 over the last 12 months

    Returns:
        (override_active: bool, track_record_data: dict or None)
        - override_active: True if override is triggered
        - track_record_data: Historical performance metrics or None
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cutoff_date = (datetime.now() - timedelta(days=TRACK_RECORD_LOOKBACK_DAYS)).isoformat()

    cursor.execute("""
        SELECT
            SUM(win_count) as total_wins,
            SUM(loss_count) as total_losses,
            SUM(total_trades) as total_trades,
            AVG(sharpe_ratio) as avg_sharpe,
            MAX(date) as latest_date
        FROM factor_track_record
        WHERE factor_id = ? AND date >= ?
    """, (factor_id, cutoff_date))

    result = cursor.fetchone()
    conn.close()

    if not result or not result[2]:  # no trades recorded
        return False, None

    total_wins = result[0] or 0
    total_losses = result[1] or 0
    total_trades = result[2] or 0
    avg_sharpe = result[3] or 0.0
    latest_date = result[4]

    track_record_data = {
        "total_wins": total_wins,
        "total_losses": total_losses,
        "total_trades": total_trades,
        "win_rate": total_wins / total_trades if total_trades > 0 else 0.0,
        "avg_sharpe": avg_sharpe,
        "lookback_days": TRACK_RECORD_LOOKBACK_DAYS,
        "latest_date": latest_date
    }

    override_active = (
        total_trades >= TRACK_RECORD_MIN_TRADES or
        avg_sharpe >= TRACK_RECORD_MIN_SHARPE
    )

    if override_active:
        update_override_status(factor_id, True, track_record_data)

    return override_active, track_record_data


def update_edge_status(factor_id: str, edge_status: str, ir: float, win_rate: float):
    """Update the factor's edge status in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO factor_edge_status
        (factor_id, edge_status, last_checked, current_ir, current_winrate, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (factor_id, edge_status, datetime.now().isoformat(), ir, win_rate, datetime.now().isoformat()))

    conn.commit()
    conn.close()


def update_override_status(factor_id: str, override_active: bool, track_record: Dict):
    """Update the factor's override status in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    reason = None
    if override_active:
        if track_record["total_trades"] >= TRACK_RECORD_MIN_TRADES:
            reason = f"{track_record['total_trades']} trades (threshold: {TRACK_RECORD_MIN_TRADES})"
        if track_record["avg_sharpe"] >= TRACK_RECORD_MIN_SHARPE:
            reason = f"Sharpe {track_record['avg_sharpe']:.2f} (threshold: {TRACK_RECORD_MIN_SHARPE})"

    cursor.execute("""
        UPDATE factor_edge_status
        SET override_active = ?, override_reason = ?, updated_at = ?
        WHERE factor_id = ?
    """, (1 if override_active else 0, reason, datetime.now().isoformat(), factor_id))

    conn.commit()
    conn.close()


def log_factor_performance(factor_id: str, win_count: int, loss_count: int,
                          avg_return: float, sharpe_ratio: float,
                          information_ratio: float, max_drawdown: float):
    """Log factor performance metrics for track record."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    today = datetime.now().strftime("%Y-%m-%d")
    total_trades = win_count + loss_count

    cursor.execute("""
        INSERT OR REPLACE INTO factor_track_record
        (factor_id, date, win_count, loss_count, total_trades, avg_return,
         sharpe_ratio, information_ratio, max_drawdown, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (factor_id, today, win_count, loss_count, total_trades, avg_return,
          sharpe_ratio, information_ratio, max_drawdown, datetime.now().isoformat()))

    conn.commit()
    conn.close()


def get_edge_status(factor_id: str) -> Optional[Dict]:
    """Retrieve the current edge status for a factor."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT edge_status, override_active, override_reason, current_ir,
               current_winrate, last_checked
        FROM factor_edge_status
        WHERE factor_id = ?
    """, (factor_id,))

    result = cursor.fetchone()
    conn.close()

    if not result:
        return None

    return {
        "edge_status": result[0],
        "override_active": bool(result[1]),
        "override_reason": result[2],
        "current_ir": result[3],
        "current_winrate": result[4],
        "last_checked": result[5]
    }


# Initialize database on module load
init_db()

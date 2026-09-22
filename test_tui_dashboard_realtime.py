#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest unit tests for tui_dashboard.py's real-time polling (adaptive backoff,
stale-snapshot protection) and split-pane factor-score rendering logic.

Pure-function tests only — no running Textual App/event loop, no network.
Run with:  python3 -m pytest test_tui_dashboard_realtime.py -v
"""
import datetime as dt

import pandas as pd
import pytest

import tui_dashboard as td


# --------------------------------------------------------------- backoff ---
def test_poll_backoff_seconds_no_failures_returns_base():
    assert td.poll_backoff_seconds(30.0, 0) == 30.0


def test_poll_backoff_seconds_doubles_per_failure():
    assert td.poll_backoff_seconds(30.0, 1) == 60.0
    assert td.poll_backoff_seconds(30.0, 2) == 120.0


def test_poll_backoff_seconds_caps_at_max_multiplier():
    assert td.poll_backoff_seconds(30.0, 10, max_multiplier=6.0) == 180.0


def test_poll_backoff_seconds_negative_failures_treated_as_zero():
    assert td.poll_backoff_seconds(30.0, -3) == 30.0


# ------------------------------------------------------------ poll state ---
def test_advance_poll_state_success_resets_failures():
    now = dt.datetime(2026, 1, 1, 12, 0, 0)
    prev = td.PollState(consecutive_failures=3, last_success=None, ok=False)
    nxt = td.advance_poll_state(prev, ok=True, now=now)
    assert nxt.consecutive_failures == 0
    assert nxt.last_success == now
    assert nxt.ok is True


def test_advance_poll_state_failure_increments_and_keeps_last_success():
    t0 = dt.datetime(2026, 1, 1, 12, 0, 0)
    prev = td.PollState(consecutive_failures=1, last_success=t0, ok=True)
    nxt = td.advance_poll_state(prev, ok=False, now=t0 + dt.timedelta(seconds=30))
    assert nxt.consecutive_failures == 2
    assert nxt.last_success == t0
    assert nxt.ok is False


# ------------------------------------------------------------- staleness ---
def test_format_staleness_waiting_for_first_refresh():
    text = td.format_staleness(None, dt.datetime(2026, 1, 1), True)
    assert "connecting" in text.lower()


def test_format_staleness_live_shows_age():
    t0 = dt.datetime(2026, 1, 1, 12, 0, 0)
    text = td.format_staleness(t0, t0 + dt.timedelta(seconds=5), True)
    assert "LIVE" in text
    assert "5s" in text


def test_format_staleness_stale_marks_error():
    t0 = dt.datetime(2026, 1, 1, 12, 0, 0)
    text = td.format_staleness(t0, t0 + dt.timedelta(seconds=90), False)
    assert "STALE" in text
    assert "90s" in text


# ----------------------------------------------------------- merge_refresh ---
def _fake_res(ticker):
    return {"ticker": ticker, "last": 100.0, "chg": 1.0, "score": 42.0,
            "verdict": {"tone": "good", "label": "BUY"}, "whale_activity": None}


def test_merge_refresh_empty_fetch_keeps_previous_snapshot():
    prev = td.RefreshOutcome(
        rows=[{"ticker": "NVDA", "last": 1.0, "chg": 0.0, "score": 1,
               "tone": "good", "verdict": "BUY", "whale": ""}],
        res_by_ticker={"NVDA": _fake_res("NVDA")}, ok=True)
    merged = td.merge_refresh(prev, ["NVDA"], {})
    assert merged.rows == prev.rows
    assert merged.res_by_ticker == prev.res_by_ticker
    assert merged.ok is False, "a total fetch outage must be reported as a failed poll"


def test_merge_refresh_first_attempt_empty_fetch_is_not_a_failure():
    prev = td.RefreshOutcome(rows=[], res_by_ticker={}, ok=True)
    merged = td.merge_refresh(prev, ["NVDA"], {})
    assert merged.rows == []
    assert merged.ok is True, "nothing to preserve yet on the very first poll"


def test_merge_refresh_successful_fetch_overwrites_snapshot(monkeypatch):
    prev = td.RefreshOutcome(rows=[{"ticker": "OLD"}], res_by_ticker={"OLD": {}}, ok=True)

    def fake_score(tickers, data):
        return [{"ticker": "NVDA"}], {"NVDA": _fake_res("NVDA")}

    monkeypatch.setattr(td, "score_watchlist", fake_score)
    merged = td.merge_refresh(prev, ["NVDA"], {"NVDA": object()})
    assert merged.rows == [{"ticker": "NVDA"}]
    assert merged.ok is True


# ------------------------------------------------------------ factor bars ---
def test_factor_bar_glyph_positive_value_fills_right_half():
    bar = td.factor_bar_glyph(1.0, width=10)
    left, _, right = bar.partition("│")
    assert set(left) <= {" "}
    assert right.count("█") == 5


def test_factor_bar_glyph_negative_value_fills_left_half():
    bar = td.factor_bar_glyph(-1.0, width=10)
    left, _, right = bar.partition("│")
    assert left.count("█") == 5
    assert set(right) <= {" "}


def test_factor_bar_glyph_zero_value_is_empty():
    bar = td.factor_bar_glyph(0.0, width=10)
    assert "█" not in bar


def test_factor_bar_glyph_clips_out_of_range_values():
    assert td.factor_bar_glyph(5.0, width=10) == td.factor_bar_glyph(1.0, width=10)
    assert td.factor_bar_glyph(-5.0, width=10) == td.factor_bar_glyph(-1.0, width=10)


# ------------------------------------------------------- factor_score_rows ---
def test_factor_score_rows_missing_data_returns_empty():
    assert td.factor_score_rows({}) == []
    assert td.factor_score_rows({"F": pd.DataFrame()}) == []


def test_factor_score_rows_sorts_by_absolute_value_desc():
    F = pd.DataFrame({"Direction": [0.1, 0.2], "Momentum": [0.0, -0.9],
                       "Volume": [0.0, 0.1], "MeanRev": [0.0, 0.05]})
    res = {"F": F, "information_ratio": {"Direction": 0.3, "Momentum": 0.5,
                                          "Volume": 0.1, "MeanRev": 0.05}}
    rows = td.factor_score_rows(res)
    assert [r["name"] for r in rows] == ["Momentum", "Direction", "Volume", "MeanRev"]
    assert rows[0]["value"] == pytest.approx(-0.9)
    assert rows[0]["ir"] == pytest.approx(0.5)


def test_factor_score_rows_missing_ir_defaults_to_zero():
    F = pd.DataFrame({"Direction": [0.4], "Momentum": [0.0],
                       "Volume": [0.0], "MeanRev": [0.0]})
    rows = td.factor_score_rows({"F": F, "information_ratio": None})
    assert rows[0]["name"] == "Direction"
    assert rows[0]["ir"] == 0.0

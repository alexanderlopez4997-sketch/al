#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MERIDIAN TUI DASHBOARD — Textual terminal front-end for the quant engine.

A third, additional interface alongside the Tkinter desktop GUI (quant_gui.py)
and the browser dashboard (web_server.py) — neither is touched or replaced.
The engine (quant_engine.py / quant_gui.py / morning.py) is treated as a pure
data service; this module only renders it and never embeds analysis logic of
its own.

Three zones, always visible:
  TOP    (KPI bar)     — market session, account health, top ticker
  MIDDLE (primary view)— per-ticker chart + verdict, or the ranked watchlist
                          table, switched with [T]/[P]
  BOTTOM (context)     — open positions + catalyst alerts for the active name

    python3 tui_dashboard.py --demo     # offline synthetic data, no API keys
    python3 tui_dashboard.py            # live watchlist.txt tickers

Not financial advice.
"""
import argparse
import asyncio
import os
import datetime as dt

import quant_engine as qe
import leaderboard as lb
import morning as mb

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Grid
from textual.widgets import Header, Footer, Static, DataTable, Sparkline, ContentSwitcher

REFRESH_SEC = 30           # floor matches quant_gui.MIN_REFRESH_SEC
CHART_ROWS = 10            # recent bars shown in the ticker DataTable
DEFAULT_WATCHLIST = "NVDA, AMD, AAPL, MSFT, TSLA, SOFI, PLTR, AMZN"
WATCHLIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchlist.txt")

# quant_gui.py has the GUI's own load_saved_watchlist()/market_session() with
# identical logic, but it unconditionally `import tkinter` at module level —
# pulling it in here would force a GUI-toolkit dependency onto a terminal
# tool whose whole point is running in restrictive, headless environments.
# These are small, tkinter-free local copies instead. Bulk OHLCV fetching
# reuses leaderboard.fetch_batch(), which is already a "no Tk imports" backend.


def parse_watchlist(raw):
    """Comma-joined watchlist string -> list of uppercased tickers."""
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def load_saved_watchlist():
    """Read watchlist.txt next to this file if present, else the default list."""
    try:
        toks = []
        with open(WATCHLIST_PATH) as f:
            for line in f:
                line = line.split("#")[0].strip()
                if line:
                    toks.append(line.upper())
        if toks:
            return toks
    except Exception:
        pass
    return parse_watchlist(DEFAULT_WATCHLIST)


def market_session():
    """Current US market session by ET clock: 'pre', 'open', 'post', or 'closed'."""
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        now = datetime.now()
    if now.weekday() >= 5:
        return "closed"
    mins = now.hour * 60 + now.minute
    if 240 <= mins < 570:   return "pre"      # 4:00-9:30
    if 570 <= mins < 960:   return "open"     # 9:30-16:00
    if 960 <= mins < 1200:  return "post"     # 16:00-20:00
    return "closed"


def bulk_fetch(tickers, demo):
    """{ticker: OHLCV DataFrame}, mirroring web_server.py's _watchlist/_morning_html dispatch."""
    if demo:
        return {t: qe.demo_data(t) for t in tickers}
    return lb.fetch_batch(tickers, "6mo", "1d")


def score_watchlist(tickers, data):
    """-> (rows, res_by_ticker). rows is the _watchlist()-shaped list for the
    KPI bar / Portfolio table; res_by_ticker lets TickerView reuse the same
    analyze() result with no extra fetch/compute."""
    rows, res_by_ticker = [], {}
    for t in tickers:
        df = data.get(t)
        if df is None or len(df) < 60:
            continue
        try:
            r = qe.analyze(t, df, "1d", None)
        except Exception:
            continue
        res_by_ticker[t] = r
        w = r.get("whale_activity")
        rows.append({"ticker": t, "last": round(r["last"], 2), "chg": round(r["chg"], 2),
                     "score": round(r["score"]), "tone": r["verdict"]["tone"],
                     "verdict": r["verdict"]["label"],
                     "whale": ("↑" if w and w["whale"] and w["direction"] == "accumulation"
                               else "↓" if w and w["whale"] and w["direction"] == "distribution" else "")})
    rows.sort(key=lambda x: -x["score"])
    return rows, res_by_ticker


def demo_event_annotations(ticker):
    """TODO: replace with a real earnings/FOMC calendar source (e.g. edgar.py
    filing dates or a finnhub/AV calendar endpoint) — no such source exists in
    this repo yet. Dates below are synthetic, not this ticker's real calendar."""
    today = dt.date.today()
    return [("Earnings (demo)", today - dt.timedelta(days=12)),
            ("FOMC (demo)", today + dt.timedelta(days=9))]


def catalyst_reasons(res):
    """analyze() result -> morning.catalyst_score() reasons, demo-safe (no
    insider/filing/sentiment lookups — those need live API keys)."""
    b = mb.catalyst_score(res["score"], 0.0, None, [], None, res.get("whale_activity"))
    return b["reasons"]


def positions_table_rows():
    """quant_engine.load_positions() -> flat rows for the exec-zone DataTable."""
    data = qe.load_positions()
    return data.get("positions", [])


def account_health(rows):
    """Average open-position pnl_pct, or None if there are no open positions."""
    if not rows:
        return None
    return sum(p.get("pnl_pct", 0.0) for p in rows) / len(rows)


class TickerView(Static):
    """Sparkline + recent-bars table + verdict + demo event annotations for one ticker."""

    def compose(self) -> ComposeResult:
        yield Sparkline([], id="ticker-spark")
        yield DataTable(id="ticker-table")
        yield Static("", id="ticker-verdict")
        yield Static("", id="ticker-events")

    def on_mount(self) -> None:
        table = self.query_one("#ticker-table", DataTable)
        table.add_columns("Close", "RSI", "MACD", "ATR", "RelVol")

    def render_ticker(self, ticker, res):
        d = res["d"]
        self.query_one("#ticker-spark", Sparkline).data = d["Close"].tail(60).tolist()
        table = self.query_one("#ticker-table", DataTable)
        table.clear()
        for _, row in d.tail(CHART_ROWS).iterrows():
            table.add_row(f"{row['Close']:.2f}", f"{row['rsi']:.1f}",
                          f"{row['macd']:.2f}", f"{row['atr']:.2f}", f"{row['relvol']:.2f}")
        v = res["verdict"]
        self.query_one("#ticker-verdict", Static).update(
            f"{ticker}  {v['label']}  (score {res['score']:.0f}, conviction {res['conviction']}%)")
        events = demo_event_annotations(ticker)
        self.query_one("#ticker-events", Static).update(
            "   ".join(f"{label} ~{d.strftime('%b %d')}" for label, d in events))


class PortfolioView(Static):
    """Ranked watchlist table — ticker/last/chg/score/tone/verdict/whale."""

    def compose(self) -> ComposeResult:
        yield DataTable(id="portfolio-table")

    def on_mount(self) -> None:
        table = self.query_one("#portfolio-table", DataTable)
        table.add_columns("Ticker", "Last", "Chg%", "Score", "Verdict", "Whale")

    def load_rows(self, rows):
        table = self.query_one("#portfolio-table", DataTable)
        table.clear()
        for r in rows:
            table.add_row(r["ticker"], f"{r['last']:.2f}", f"{r['chg']:+.2f}",
                          str(r["score"]), r["verdict"], r["whale"])


class PositionsTable(Static):
    """Open positions from positions.json."""

    def compose(self) -> ComposeResult:
        yield DataTable(id="positions-table")

    def on_mount(self) -> None:
        table = self.query_one("#positions-table", DataTable)
        table.add_columns("Ticker", "Entry", "Current", "P&L%", "Signal")

    def load_rows(self, rows):
        table = self.query_one("#positions-table", DataTable)
        table.clear()
        for p in rows:
            table.add_row(p.get("ticker", ""), f"{p.get('entry_price', 0):.2f}",
                          f"{p.get('current_price', 0):.2f}", f"{p.get('pnl_pct', 0):+.2f}",
                          p.get("signal", ""))


class AlertsPanel(Static):
    """Catalyst-score reasons for the active ticker."""

    def render_alerts(self, ticker, reasons):
        if not reasons:
            self.update(f"{ticker}: no active catalysts")
            return
        lines = [f"{'▲' if d > 0 else '▼' if d < 0 else '•'} {text}" for text, d in reasons[:6]]
        self.update(f"{ticker} catalysts:\n" + "\n".join(lines))


class MeridianDashboard(App):
    """Bloomberg-style keyboard-driven TUI over the Meridian quant engine."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 1 3;
        grid-rows: 3 1fr 14;
    }
    #kpi-bar {
        layout: grid;
        grid-size: 3 1;
        height: 3;
    }
    #exec-zone {
        layout: grid;
        grid-size: 2 1;
        height: 14;
    }
    """

    BINDINGS = [
        ("t", "show_ticker", "Ticker view"),
        ("p", "show_portfolio", "Portfolio view"),
        ("n", "next_ticker", "Next ticker"),
        ("r", "refresh_now", "Refresh"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, demo=False):
        super().__init__()
        self.demo = demo
        self.watchlist = load_saved_watchlist()
        self.active_ticker = self.watchlist[0] if self.watchlist else None
        self._res_by_ticker = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with Grid(id="kpi-bar"):
            yield Static("Market: —", id="kpi-market")
            yield Static("Account: —", id="kpi-account")
            yield Static("Top: —", id="kpi-top")
        with ContentSwitcher(initial="ticker-view", id="main-view"):
            yield TickerView(id="ticker-view")
            yield PortfolioView(id="portfolio-view")
        with Grid(id="exec-zone"):
            yield PositionsTable(id="positions-panel")
            yield AlertsPanel(id="alerts-panel")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_all()
        self.set_interval(REFRESH_SEC, self.refresh_all)

    @work(exclusive=True, group="refresh")
    async def refresh_all(self) -> None:
        if not self.watchlist:
            return
        raw = await asyncio.to_thread(bulk_fetch, self.watchlist, self.demo)
        scored, res_by_ticker = await asyncio.to_thread(score_watchlist, self.watchlist, raw)
        self._res_by_ticker = res_by_ticker
        self.update_kpi_bar(scored)
        self.query_one(PortfolioView).load_rows(scored)
        self.render_ticker_view(self.active_ticker)
        self.render_exec_zone()

    def update_kpi_bar(self, scored):
        self.query_one("#kpi-market", Static).update(f"Market: {market_session()}")
        rows = positions_table_rows()
        health = account_health(rows)
        health_str = f"{health:+.2f}%" if health is not None else "no open positions"
        self.query_one("#kpi-account", Static).update(f"Account: {health_str}")
        top = scored[0] if scored else None
        top_str = f"{top['ticker']} ({top['score']:+d})" if top else "—"
        self.query_one("#kpi-top", Static).update(f"Top: {top_str}")

    def render_ticker_view(self, ticker):
        res = self._res_by_ticker.get(ticker)
        if not res:
            return
        self.query_one(TickerView).render_ticker(ticker, res)

    def render_exec_zone(self):
        self.query_one(PositionsTable).load_rows(positions_table_rows())
        res = self._res_by_ticker.get(self.active_ticker)
        reasons = catalyst_reasons(res) if res else []
        self.query_one(AlertsPanel).render_alerts(self.active_ticker or "—", reasons)

    def action_show_ticker(self) -> None:
        self.query_one(ContentSwitcher).current = "ticker-view"

    def action_show_portfolio(self) -> None:
        self.query_one(ContentSwitcher).current = "portfolio-view"

    def action_next_ticker(self) -> None:
        if not self.watchlist:
            return
        i = self.watchlist.index(self.active_ticker)
        self.active_ticker = self.watchlist[(i + 1) % len(self.watchlist)]
        self.render_ticker_view(self.active_ticker)
        self.render_exec_zone()

    def action_refresh_now(self) -> None:
        self.refresh_all()


def main():
    ap = argparse.ArgumentParser(description="Meridian TUI dashboard.")
    ap.add_argument("--demo", action="store_true", help="synthetic data, offline")
    args = ap.parse_args()
    MeridianDashboard(demo=args.demo).run()


if __name__ == "__main__":
    main()

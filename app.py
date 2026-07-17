#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUANT DASHBOARD — Textual TUI controller for quant_engine.py.

Controller-Service split:
  * quant_engine.Engine is the service layer — UI-agnostic, returns plain
    dicts (see quant_engine.summarize_analysis). It knows nothing about
    Textual.
  * QuantDashboard (this file) is the controller — it owns widgets, reads
    user input, calls the engine in a background worker so the UI thread
    never blocks on network/pandas work, then renders the returned dict.

USAGE
  python3 app.py                 # live data via yfinance
  python3 app.py --demo          # synthetic data, no internet (for dev/testing)
"""
import argparse

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import DataTable, Footer, Header, Input, Log

from quant_engine import Engine

WATCHLIST = ["NVDA", "AMD", "AAPL", "MSFT", "TSLA", "SOFI", "PLTR", "AMZN"]

SUMMARY_COLUMNS = [
    ("ticker", "Ticker"), ("price", "Price"), ("chg_pct", "Chg %"),
    ("score", "Score"), ("verdict_label", "Verdict"), ("conviction_pct", "Conv %"),
    ("regime", "Regime"), ("backtest_sharpe", "Sharpe"),
    ("backtest_winrate_pct", "Win %"), ("edge_status", "Edge"),
]


class QuantDashboard(App):
    """Type a ticker + Enter to analyze it. Press ctrl+s to scan the watchlist."""

    CSS = """
    #ticker_input { dock: top; }
    #results { height: 40%; border: solid $accent; }
    #activity_log { height: 1fr; border: solid $panel; }
    """
    # ctrl+ combos (not plain letters) so they still fire while the ticker
    # Input has focus and is capturing normal keystrokes as typed text.
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+s", "scan_watchlist", "Scan watchlist"),
    ]

    status = reactive("idle")

    def __init__(self, demo: bool = False):
        super().__init__()
        self.engine = Engine(demo=demo)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="Enter ticker (e.g. AAPL) and press Enter...", id="ticker_input")
        yield Vertical(
            DataTable(id="results"),
            Log(id="activity_log", auto_scroll=True),
        )
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#results", DataTable)
        table.add_columns(*(label for _key, label in SUMMARY_COLUMNS))
        self.query_one("#activity_log", Log).write_line(
            "Ready. Enter a ticker, or press ctrl+s to scan the watchlist.")

    def watch_status(self, status: str) -> None:
        self.sub_title = status

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        ticker = event.value.strip()
        event.input.value = ""
        if not ticker:
            return
        self.run_worker(self.process_ticker(ticker), exclusive=True)

    def action_scan_watchlist(self) -> None:
        self.run_worker(self.process_scan(WATCHLIST), exclusive=True)

    async def process_ticker(self, ticker: str) -> None:
        log = self.query_one("#activity_log", Log)
        self.status = f"analyzing {ticker}..."
        log.write_line(f"Analyzing {ticker}...")
        try:
            summary = await self.engine.analyze(ticker)
        except Exception as exc:
            log.write_line(f"  ERROR: {exc}")
            self.status = "idle"
            return
        self._add_row(summary)
        log.write_line(f"  {summary['ticker']}: {summary['verdict_label']} "
                        f"(score {summary['score']:+.0f}, conviction {summary['conviction_pct']}%)")
        self.status = "idle"

    async def process_scan(self, tickers: list[str]) -> None:
        log = self.query_one("#activity_log", Log)
        self.status = f"scanning {len(tickers)} tickers..."
        log.write_line(f"Scanning watchlist: {', '.join(tickers)}...")
        summaries = await self.engine.scan(tickers)
        for summary in summaries:
            if "error" in summary:
                log.write_line(f"  {summary['ticker']}: ERROR — {summary['error']}")
                continue
            self._add_row(summary)
            log.write_line(f"  {summary['ticker']}: {summary['verdict_label']} "
                            f"(score {summary['score']:+.0f})")
        self.status = "idle"

    def _add_row(self, summary: dict) -> None:
        table = self.query_one("#results", DataTable)
        table.add_row(*(summary.get(key, "") for key, _label in SUMMARY_COLUMNS))


def main() -> None:
    ap = argparse.ArgumentParser(description="Quant Dashboard TUI")
    ap.add_argument("--demo", action="store_true", help="synthetic data, no internet")
    args = ap.parse_args()
    QuantDashboard(demo=args.demo).run()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSocket client for real-time aggregate minute data using massive.com API.
Subscribes to live market data and maintains rolling windows for diagnostics.
"""
import json
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
import queue

try:
    from massive import WebSocketClient
    from massive.websocket.models import Feed, Market
    HAS_MASSIVE = True
except ImportError:
    HAS_MASSIVE = False


class AggregateBuffer:
    """Maintains rolling window of aggregate minute data for a symbol."""

    def __init__(self, max_size=60):
        self.max_size = max_size
        self.data = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def add(self, agg_data):
        """Add aggregate OHLCV data: {open, high, low, close, volume, timestamp}"""
        with self.lock:
            self.data.append(agg_data)

    def get_prices(self):
        """Return list of closes for current window."""
        with self.lock:
            return [d.get("close", 0) for d in self.data]

    def get_ohlcv(self):
        """Return list of OHLCV tuples."""
        with self.lock:
            return [(d.get("open"), d.get("high"), d.get("low"),
                    d.get("close"), d.get("volume")) for d in self.data]

    def size(self):
        with self.lock:
            return len(self.data)


class DiagnosticsWebSocketClient:
    """
    Connects to massive.com real-time market data and maintains rolling buffers.
    Calculates strategy health, factor IRs, and correlations.
    """

    def __init__(self, symbols=None, api_key=None, max_buffer_size=60):
        """
        Initialize client.

        Args:
            symbols: List of tickers to track (e.g., ["NVDA", "AMD", "AAPL"])
            api_key: API key for massive.com
            max_buffer_size: How many minute bars to keep in rolling window
        """
        self.symbols = symbols or ["NVDA", "AMD", "AAPL", "MSFT", "TSLA"]
        self.api_key = api_key
        self.max_buffer_size = max_buffer_size
        self.buffers = {sym: AggregateBuffer(max_buffer_size) for sym in self.symbols}
        self.client = None
        self.connected = False
        self.running = False
        self.thread = None
        self.last_update = {}
        self.lock = threading.Lock()

    def _handle_messages(self, messages):
        """Handle incoming messages from massive WebSocket."""
        for msg in messages:
            try:
                sym = getattr(msg, 'symbol', '').upper() if hasattr(msg, 'symbol') else ''
                if sym in self.buffers:
                    agg = {
                        "open": getattr(msg, 'open', None),
                        "high": getattr(msg, 'high', None),
                        "low": getattr(msg, 'low', None),
                        "close": getattr(msg, 'close', None),
                        "volume": getattr(msg, 'volume', None),
                        "timestamp": getattr(msg, 'timestamp', None)
                    }
                    self.buffers[sym].add(agg)
                    with self.lock:
                        self.last_update[sym] = datetime.now().isoformat()
            except Exception as e:
                pass

    def _connect_loop(self):
        """Main WebSocket connection loop using massive library."""
        try:
            self.client = WebSocketClient(
                api_key=self.api_key,
                feed=Feed.Delayed,
                market=Market.Stocks
            )

            # Subscribe to aggregate minute data for all symbols
            for sym in self.symbols:
                self.client.subscribe(f"AM.{sym}")

            self.connected = True

            # Run the WebSocket client
            self.client.run(self._handle_messages)
        except Exception as e:
            print(f"WebSocket connection failed: {e}")
            self.connected = False

    def _start_demo_mode(self):
        """Simulate aggregate data for demo/testing."""
        import random

        def generate_demo_data():
            while self.running:
                for sym in self.symbols:
                    # Generate realistic OHLCV data
                    close = random.uniform(100, 500)
                    agg = {
                        "open": close + random.uniform(-5, 5),
                        "high": close + random.uniform(0, 10),
                        "low": close - random.uniform(0, 10),
                        "close": close,
                        "volume": random.randint(1000000, 10000000),
                        "timestamp": int(time.time() * 1000)
                    }
                    self.buffers[sym].add(agg)
                    with self.lock:
                        self.last_update[sym] = datetime.now().isoformat()

                time.sleep(1)  # Simulate 1-minute aggregates

        self.thread = threading.Thread(target=generate_demo_data, daemon=True)
        self.thread.start()
        self.connected = True

    def connect(self, use_demo=True):
        """
        Connect to WebSocket (massive.com or demo mode).

        Args:
            use_demo: If True, simulate data. If False, use actual massive.com API.
        """
        self.running = True

        if use_demo or not HAS_MASSIVE or not self.api_key:
            self._start_demo_mode()
        else:
            self.thread = threading.Thread(target=self._connect_loop, daemon=True)
            self.thread.start()

    def disconnect(self):
        """Disconnect from WebSocket."""
        self.running = False
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
        if self.thread:
            self.thread.join(timeout=2)

    def get_diagnostics(self):
        """
        Return current diagnostics snapshot:
        - health_status: Overall strategy score
        - factor_irs: Information ratios for each symbol
        - correlation_matrix: Correlations between symbols
        """
        with self.lock:
            # Check buffer sizes
            ready_symbols = {sym for sym, buf in self.buffers.items() if buf.size() >= 20}

            if not ready_symbols:
                return {
                    "status": "warming_up",
                    "buffers": {sym: buf.size() for sym, buf in self.buffers.items()},
                    "health_status": {"regime": "neutral", "score": 0},
                    "factor_irs": {},
                    "correlation_matrix": {}
                }

            # Calculate metrics
            prices_by_sym = {sym: self.buffers[sym].get_prices() for sym in ready_symbols}

            # Trend strength for each symbol (simple moving average crossover)
            factor_irs = {}
            for sym in ready_symbols:
                prices = prices_by_sym[sym]
                if len(prices) >= 10:
                    short_ma = sum(prices[-5:]) / 5
                    long_ma = sum(prices[-10:]) / 10
                    trend = (short_ma - long_ma) / long_ma * 100 if long_ma else 0
                    volatility = (max(prices[-10:]) - min(prices[-10:])) / (sum(prices[-10:]) / 10) * 100
                    ir = trend / max(volatility, 1) if volatility else 0
                    factor_irs[sym] = round(ir, 2)

            # Correlation matrix (simple Pearson)
            corr_matrix = {}
            symbols_list = sorted(ready_symbols)
            for i, sym1 in enumerate(symbols_list):
                for sym2 in symbols_list[i:]:
                    key = f"{sym1}-{sym2}"
                    if sym1 == sym2:
                        corr_matrix[key] = 1.0
                    else:
                        p1, p2 = prices_by_sym[sym1], prices_by_sym[sym2]
                        # Simplified correlation: just check if moves together
                        changes1 = [(p2 - p1) / p1 for p1, p2 in zip(p1[:-1], p1[1:])]
                        changes2 = [(p2 - p1) / p1 for p1, p2 in zip(p2[:-1], p2[1:])]
                        if changes1 and changes2:
                            avg1, avg2 = sum(changes1) / len(changes1), sum(changes2) / len(changes2)
                            cov = sum((c1 - avg1) * (c2 - avg2) for c1, c2 in zip(changes1, changes2)) / len(changes1)
                            std1 = (sum((c - avg1) ** 2 for c in changes1) / len(changes1)) ** 0.5
                            std2 = (sum((c - avg2) ** 2 for c in changes2) / len(changes2)) ** 0.5
                            corr = cov / (std1 * std2) if std1 and std2 else 0
                            corr_matrix[key] = round(corr, 2)
                        else:
                            corr_matrix[key] = 0.0

            # Health status based on factor IRs
            avg_ir = sum(factor_irs.values()) / len(factor_irs) if factor_irs else 0
            if avg_ir > 1.5:
                regime = "bullish"
            elif avg_ir < -1.5:
                regime = "bearish"
            else:
                regime = "neutral"

            health_score = round(avg_ir * 10)

            return {
                "status": "ready",
                "timestamp": datetime.now().isoformat(),
                "buffers": {sym: self.buffers[sym].size() for sym in self.symbols},
                "health_status": {
                    "regime": regime,
                    "score": health_score,
                    "avg_ir": round(avg_ir, 2)
                },
                "factor_irs": factor_irs,
                "correlation_matrix": corr_matrix
            }


# Global client instance
_global_client = None


def get_diagnostics_client(symbols=None, api_key=None):
    """Get or create global diagnostics client."""
    global _global_client
    if _global_client is None:
        _global_client = DiagnosticsWebSocketClient(symbols, api_key)
    return _global_client


def start_diagnostics(symbols=None, api_key=None, use_demo=True):
    """Start diagnostics WebSocket client."""
    client = get_diagnostics_client(symbols, api_key)
    if use_demo or not api_key:
        client.connect()  # Demo mode
    else:
        client.connect()
    return client

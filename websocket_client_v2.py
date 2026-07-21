#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production-grade WebSocket client for massive.com aggregate minute data.
Strict separation: connection/streaming only (calculations delegated to quant_engine).
"""
import os
import logging
import threading
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import numpy as np

import sale_conditions
import exchanges

# Optional: load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from massive import WebSocketClient
    from massive.websocket.models import Feed, Market
    try:
        from massive.websocket.models import WebSocketMessage
    except ImportError:
        WebSocketMessage = Any  # Fallback for older versions
    HAS_MASSIVE = True
except ImportError:
    HAS_MASSIVE = False
    WebSocketMessage = Any  # Type stub when massive not installed

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class AggregateBar:
    """Single OHLCV bar from WebSocket."""
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: int

    @classmethod
    def from_message(cls, msg: Any) -> Optional['AggregateBar']:
        """Parse WebSocket message to AggregateBar."""
        try:
            return cls(
                open=float(getattr(msg, 'o', 0)),
                high=float(getattr(msg, 'h', 0)),
                low=float(getattr(msg, 'l', 0)),
                close=float(getattr(msg, 'c', 0)),
                volume=int(getattr(msg, 'v', 0)),
                timestamp=int(getattr(msg, 't', 0))
            )
        except (ValueError, AttributeError, TypeError) as e:
            logger.warning(f"Failed to parse message: {e}")
            return None


@dataclass
class Trade:
    """Single tick-level trade from WebSocket, with its raw sale-condition ids."""
    symbol: str
    price: float
    size: int
    conditions: List[int]
    timestamp: int
    exchange: Optional[int] = None

    @classmethod
    def from_message(cls, msg: Any) -> Optional['Trade']:
        """Parse a raw trade ('T.*') WebSocket message to a Trade."""
        try:
            symbol = getattr(msg, 'symbol', None) or getattr(msg, 'sym', '')
            conditions = getattr(msg, 'conditions', None)
            if conditions is None:
                conditions = getattr(msg, 'c', None) or []
            exchange = getattr(msg, 'exchange', None)
            if exchange is None:
                exchange = getattr(msg, 'x', None)
            return cls(
                symbol=str(symbol).upper(),
                price=float(getattr(msg, 'price', getattr(msg, 'p', 0))),
                size=int(getattr(msg, 'size', getattr(msg, 's', 0))),
                conditions=[int(c) for c in conditions],
                timestamp=int(getattr(msg, 'timestamp', getattr(msg, 't', 0))),
                exchange=int(exchange) if exchange is not None else None,
            )
        except (ValueError, AttributeError, TypeError) as e:
            logger.warning(f"Failed to parse trade message: {e}")
            return None

    @property
    def exchange_name(self) -> Optional[str]:
        """Reporting venue name, resolved from the raw numeric exchange id."""
        if self.exchange is None:
            return None
        ex = exchanges.get_exchange_by_id(self.exchange)
        return ex.name if ex else None


@dataclass
class DiagnosticsSnapshot:
    """Immutable snapshot of diagnostics metrics."""
    status: str  # 'warming_up' | 'ready'
    timestamp: str
    regime: str  # 'bullish' | 'bearish' | 'neutral'
    health_score: int
    factor_irs: Dict[str, float]
    correlation_matrix: Dict[str, float]
    buffer_sizes: Dict[str, int]
    last_exchange: Dict[str, Optional[str]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# BUFFER MANAGEMENT
# ============================================================================

class AggregateBuffer:
    """
    Thread-safe rolling window for OHLCV data.
    Minimum 20 bars required for metrics calculation.
    """

    MIN_BARS = 20

    def __init__(self, max_size: int = 60):
        self.max_size = max_size
        self.data: deque = deque(maxlen=max_size)
        self.lock = threading.RLock()
        self.last_update: Optional[datetime] = None
        self._bar_open = False

    def update_from_trade(self, trade: Trade, rules: Dict[str, bool]) -> None:
        """Fold one trade into the in-progress bar, gated by its sale-condition
        update rules (see sale_conditions.classify_trade_by_id). Excluded trades
        (e.g. average-price, cash sale) still count toward volume by default but
        never move high/low/open/close."""
        if trade.price <= 0 or trade.size < 0:
            logger.warning(f"Invalid trade data: {trade}")
            return

        with self.lock:
            if not self._bar_open:
                bar = AggregateBar(
                    open=trade.price, high=trade.price, low=trade.price,
                    close=trade.price, volume=0, timestamp=trade.timestamp
                )
                self.data.append(bar)
                self._bar_open = True
            else:
                bar = self.data[-1]

            if rules.get("updates_high_low", True):
                bar.high = max(bar.high, trade.price)
                bar.low = min(bar.low, trade.price)
            if rules.get("updates_open_close", True):
                bar.close = trade.price
            if rules.get("updates_volume", True):
                bar.volume += trade.size

            self.last_update = datetime.now()

    def close_bar(self) -> None:
        """Seal the in-progress trade-built bar; the next trade starts a new one."""
        with self.lock:
            self._bar_open = False

    def add(self, bar: AggregateBar) -> None:
        """Add bar with validation and locking."""
        if not isinstance(bar, AggregateBar):
            logger.error(f"Expected AggregateBar, got {type(bar)}")
            return

        if bar.close <= 0 or bar.volume < 0:
            logger.warning(f"Invalid bar data: {bar}")
            return

        with self.lock:
            self.data.append(bar)
            self._bar_open = False  # server-finalized bar; next trade starts fresh
            self.last_update = datetime.now()

    def is_ready(self) -> bool:
        """Check if buffer has minimum bars for analysis."""
        with self.lock:
            return len(self.data) >= self.MIN_BARS

    def get_prices(self) -> np.ndarray:
        """Return close prices as numpy array."""
        with self.lock:
            if not self.data:
                return np.array([])
            return np.array([bar.close for bar in self.data])

    def get_ohlcv(self) -> np.ndarray:
        """Return OHLCV as (n, 5) numpy array."""
        with self.lock:
            if not self.data:
                return np.empty((0, 5))
            data = [[bar.open, bar.high, bar.low, bar.close, bar.volume]
                    for bar in self.data]
            return np.array(data)

    def size(self) -> int:
        """Thread-safe size check."""
        with self.lock:
            return len(self.data)

    def clear(self) -> None:
        """Clear buffer (for testing/reset)."""
        with self.lock:
            self.data.clear()
            self.last_update = None
            self._bar_open = False


# ============================================================================
# METRICS CALCULATION (Vectorized)
# ============================================================================

class DiagnosticsCalculator:
    """Calculate strategy metrics using vectorized numpy operations."""

    def __init__(self, short_window: int = 5, long_window: int = 10):
        self.short_window = short_window
        self.long_window = long_window

    def calculate_information_ratio(self, prices: np.ndarray) -> float:
        """
        IR = trend / volatility
        Simplified metric (complements quant_engine.information_ratio for full analysis).
        """
        if len(prices) < self.long_window:
            return 0.0

        try:
            # Vectorized MAs
            short_ma = np.mean(prices[-self.short_window:])
            long_ma = np.mean(prices[-self.long_window:])

            # Trend as percentage
            trend = (short_ma - long_ma) / long_ma * 100 if long_ma else 0

            # Volatility
            volatility = np.std(prices[-self.long_window:]) / long_ma * 100 if long_ma else 1

            # IR
            ir = trend / max(volatility, 1e-6)
            return float(ir)
        except Exception as e:
            logger.error(f"Error calculating IR: {e}")
            return 0.0

    def calculate_correlations(self, price_matrix: np.ndarray) -> Dict[str, float]:
        """
        Vectorized Pearson correlation for price movements.

        Args:
            price_matrix: (n_symbols, n_bars)

        Returns:
            Dict of correlation pairs: {"SYM1-SYM2": correlation}
        """
        if price_matrix.shape[0] < 2:
            return {}

        try:
            # Convert prices to returns
            returns = np.diff(price_matrix, axis=1) / price_matrix[:, :-1]

            # Vectorized correlation matrix (numpy 2.0 compatible)
            corr_matrix = np.corrcoef(np.asarray(returns))

            # Extract pairs
            symbols = list(range(price_matrix.shape[0]))
            result = {}
            for i, sym1 in enumerate(symbols):
                for j, sym2 in enumerate(symbols[i:], start=i):
                    key = f"SYM{sym1}-SYM{sym2}"
                    result[key] = float(np.nan_to_num(corr_matrix[i, j], 0.0))

            return result
        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
            return {}

    def detect_regime(self, factor_irs: Dict[str, float]) -> str:
        """Classify regime based on average IR."""
        if not factor_irs:
            return "neutral"

        avg_ir = np.mean(list(factor_irs.values()))

        if avg_ir > 1.5:
            return "bullish"
        elif avg_ir < -1.5:
            return "bearish"
        else:
            return "neutral"


# ============================================================================
# WEBSOCKET CLIENT
# ============================================================================

class DiagnosticsWebSocketClient:
    """
    Manages WebSocket connection and buffer aggregation.
    Delegates metrics calculation to DiagnosticsCalculator.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        max_buffer_size: int = 60,
        message_handler: Optional[Callable] = None
    ):
        """
        Initialize WebSocket client.

        Args:
            symbols: List of tickers to track
            api_key: Massive.com API key (from environment if not provided)
            max_buffer_size: Rolling window size per symbol
            message_handler: Optional callback for raw messages
        """
        self.symbols = symbols or ["NVDA", "AMD", "AAPL", "MSFT", "TSLA"]
        self.api_key = api_key or os.environ.get("MASSIVE_API_KEY")
        self.max_buffer_size = max_buffer_size
        self.message_handler = message_handler

        # Buffers and state
        self.buffers: Dict[str, AggregateBuffer] = {
            sym: AggregateBuffer(max_buffer_size) for sym in self.symbols
        }
        self.last_exchange: Dict[str, Optional[str]] = {sym: None for sym in self.symbols}
        self.calculator = DiagnosticsCalculator()

        # Connection state
        self.client: Optional[WebSocketClient] = None
        self.connected = False
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()

        logger.info(f"Initialized client for symbols: {self.symbols}")

    def _validate_symbol(self, symbol: str) -> bool:
        """Validate symbol before processing."""
        if not symbol or not isinstance(symbol, str):
            return False

        symbol = symbol.upper().strip()

        if symbol not in self.buffers:
            logger.debug(f"Symbol not in subscription list: {symbol}")
            return False

        # Basic format check: 1-5 uppercase letters
        if not symbol.isalpha() or len(symbol) > 5:
            logger.warning(f"Invalid symbol format: {symbol}")
            return False

        return True

    def _handle_messages(self, messages: List[WebSocketMessage]) -> None:
        """Process WebSocket messages with error handling. Routes trades ('T')
        through sale-condition-aware bar building and aggregate bars ('AM')
        straight into the buffer, as before."""
        if not messages:
            return

        for msg in messages:
            try:
                symbol = getattr(msg, 'symbol', None) or getattr(msg, 'sym', '')
                symbol = str(symbol).upper() if symbol else ''
                if not self._validate_symbol(symbol):
                    continue

                event = getattr(msg, 'event_type', None) or getattr(msg, 'ev', None)

                if event == 'T':
                    trade = Trade.from_message(msg)
                    if trade is not None:
                        self._process_trade(symbol, trade)
                else:
                    self._process_aggregate(symbol, msg)

            except AttributeError as e:
                logger.warning(f"Malformed message structure: {e}")
            except KeyError as e:
                logger.error(f"Missing required field: {e}")
            except Exception as e:
                logger.error(f"Unexpected error in message handler: {e}", exc_info=True)

    def _process_aggregate(self, symbol: str, msg: Any) -> None:
        """Parse and store a server-finalized aggregate-minute ('AM') bar."""
        bar = AggregateBar.from_message(msg)
        if bar is None:
            return
        self.buffers[symbol].add(bar)
        if self.message_handler:
            self.message_handler(symbol, bar)
        logger.debug(f"Processed: {symbol} @ {bar.close}")

    def _process_trade(self, symbol: str, trade: Trade) -> None:
        """Classify a trade's sale conditions and fold it into the buffer's
        in-progress bar accordingly. Suppressed conditions (average-price,
        cash sale, etc.) still count toward volume but never move high/low/close.
        Also resolves and records the reporting venue for diagnostics."""
        rules = sale_conditions.classify_trade_by_id(trade.conditions)
        self.buffers[symbol].update_from_trade(trade, rules)

        with self.lock:
            self.last_exchange[symbol] = trade.exchange_name

        if self.message_handler:
            self.message_handler(symbol, trade)

        logger.debug(f"Trade: {symbol} @ {trade.price} conditions={trade.conditions} rules={rules}")

    def _connect_loop(self) -> None:
        """Main WebSocket connection loop."""
        try:
            if not HAS_MASSIVE:
                logger.error("massive library not installed")
                return

            if not self.api_key:
                logger.error("MASSIVE_API_KEY not configured")
                return

            logger.info("Connecting to massive.com WebSocket...")

            self.client = WebSocketClient(
                api_key=self.api_key,
                feed=Feed.Delayed,
                market=Market.Stocks
            )

            # Subscribe to aggregate minute data and raw trades (sale-condition classified)
            for sym in self.symbols:
                self.client.subscribe(f"AM.{sym}")
                self.client.subscribe(f"T.{sym}")
                logger.info(f"Subscribed to AM.{sym}, T.{sym}")

            self.connected = True
            logger.info("WebSocket connected and subscribed")

            # Run blocking event loop
            self.client.run(self._handle_messages)

        except ConnectionError as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False
        except TimeoutError:
            logger.error("WebSocket timeout")
            self.connected = False
        except Exception as e:
            logger.error(f"WebSocket error: {e}", exc_info=True)
            self.connected = False

    def _start_demo_mode(self) -> None:
        """Generate realistic synthetic data for testing."""
        import random

        # A handful of trades per synthetic bar go through the real sale-condition
        # pipeline: mostly plain trades (update everything), occasionally an
        # excluded condition (e.g. average-price, cash sale) that must move
        # volume but never high/low/open/close.
        NORMAL_CONDITIONS = [[], [9], [3]]        # none / Cross Trade / Automatic Execution
        SUPPRESSING_CONDITIONS = [[2], [7]]       # Average Price Trade / Cash Sale
        DEMO_EXCHANGE_IDS = [e.id for e in exchanges.DEFAULT_EXCHANGES if e.participant_id]

        def generate_data():
            logger.info("Starting demo mode with synthetic trades")

            # Initialize with realistic prices
            prices = {sym: random.uniform(100, 500) for sym in self.symbols}

            while self.running:
                for sym in self.symbols:
                    for _ in range(random.randint(5, 12)):
                        prices[sym] *= (1 + random.uniform(-0.004, 0.004))
                        conditions = random.choice(SUPPRESSING_CONDITIONS) if random.random() < 0.15 \
                            else random.choice(NORMAL_CONDITIONS)
                        trade = Trade(
                            symbol=sym,
                            price=round(prices[sym], 4),
                            size=random.randint(100, 5000),
                            conditions=conditions,
                            timestamp=int(time.time() * 1000),
                            exchange=random.choice(DEMO_EXCHANGE_IDS),
                        )
                        self._process_trade(sym, trade)

                    self.buffers[sym].close_bar()
                    logger.debug(f"Demo: {sym} @ {prices[sym]:.2f}")

                time.sleep(1)  # Simulate 1-minute aggregates

        self.thread = threading.Thread(target=generate_data, daemon=True, name="DemoDataGenerator")
        self.thread.start()
        self.connected = True

    def connect(self, use_demo: bool = False) -> None:
        """
        Connect to WebSocket or demo mode.

        Args:
            use_demo: Force demo mode (useful for development/testing)
        """
        if self.running:
            logger.warning("Already connected")
            return

        self.running = True

        if use_demo or not HAS_MASSIVE or not self.api_key:
            logger.info("Using demo mode")
            self._start_demo_mode()
        else:
            logger.info("Connecting to massive.com...")
            self.thread = threading.Thread(
                target=self._connect_loop,
                daemon=True,
                name="WebSocketConnection"
            )
            self.thread.start()

    def disconnect(self) -> None:
        """Graceful shutdown."""
        logger.info("Disconnecting...")
        self.running = False

        if self.client:
            try:
                self.client.close()
            except Exception as e:
                logger.warning(f"Error closing client: {e}")

        # Wait for thread
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning(f"Thread {self.thread.name} did not terminate gracefully")

        self.connected = False
        logger.info("Disconnected")

    def get_diagnostics(self) -> DiagnosticsSnapshot:
        """
        Get current diagnostics snapshot.

        Returns:
            Immutable snapshot with status, metrics, and buffer state.
        """
        with self.lock:
            # Check readiness
            ready_symbols = [
                sym for sym, buf in self.buffers.items() if buf.is_ready()
            ]

            buffer_sizes = {sym: buf.size() for sym, buf in self.buffers.items()}

            if not ready_symbols:
                logger.debug(f"Not ready: {buffer_sizes}")
                return DiagnosticsSnapshot(
                    status="warming_up",
                    timestamp=datetime.now().isoformat(),
                    regime="neutral",
                    health_score=0,
                    factor_irs={},
                    correlation_matrix={},
                    buffer_sizes=buffer_sizes,
                    last_exchange=dict(self.last_exchange)
                )

            # Calculate metrics
            factor_irs = {}
            price_matrix_data = []

            for sym in ready_symbols:
                prices = self.buffers[sym].get_prices()
                ir = self.calculator.calculate_information_ratio(prices)
                factor_irs[sym] = ir
                price_matrix_data.append(prices)

            # Vectorized correlation
            price_matrix = np.array(price_matrix_data)
            # Rename keys for compatibility
            raw_corr = self.calculator.calculate_correlations(price_matrix)
            correlation_matrix = {
                f"{ready_symbols[int(k.split('-')[0][3:])]}-{ready_symbols[int(k.split('-')[1][3:])]}"
                if k.count('-') == 1 else k: v
                for k, v in raw_corr.items()
            }

            # Regime detection
            regime = self.calculator.detect_regime(factor_irs)
            health_score = int(np.mean(list(factor_irs.values())) * 10) if factor_irs else 0

            return DiagnosticsSnapshot(
                status="ready",
                timestamp=datetime.now().isoformat(),
                regime=regime,
                health_score=health_score,
                factor_irs=factor_irs,
                correlation_matrix=correlation_matrix,
                buffer_sizes=buffer_sizes,
                last_exchange=dict(self.last_exchange)
            )

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        try:
            self.disconnect()
        except Exception as e:
            logger.warning(f"Error in __del__: {e}")


# ============================================================================
# SINGLETON MANAGER
# ============================================================================

_global_client: Optional[DiagnosticsWebSocketClient] = None
_client_lock = threading.Lock()


def get_client(
    symbols: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    reset: bool = False
) -> DiagnosticsWebSocketClient:
    """
    Get or create global client instance (thread-safe singleton).

    Args:
        symbols: List of tickers
        api_key: Massive API key
        reset: Force new instance
    """
    global _global_client

    with _client_lock:
        if reset and _global_client:
            _global_client.disconnect()
            _global_client = None

        if _global_client is None:
            _global_client = DiagnosticsWebSocketClient(symbols, api_key)

        return _global_client


def start_diagnostics(
    symbols: Optional[List[str]] = None,
    use_demo: bool = False
) -> DiagnosticsWebSocketClient:
    """
    Start diagnostics WebSocket client.

    Args:
        symbols: List of tickers to track
        use_demo: Force demo mode

    Returns:
        Client instance
    """
    client = get_client(symbols)
    client.connect(use_demo=use_demo)
    return client


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    client = start_diagnostics(use_demo=True)

    try:
        for _ in range(10):
            time.sleep(3)
            diag = client.get_diagnostics()
            print(f"\n{diag.timestamp}")
            print(f"Status: {diag.status}")
            print(f"Regime: {diag.regime} (score: {diag.health_score})")
            print(f"IRs: {diag.factor_irs}")
            print(f"Buffers: {diag.buffer_sizes}")
            print(f"Last exchange: {diag.last_exchange}")
    finally:
        client.disconnect()

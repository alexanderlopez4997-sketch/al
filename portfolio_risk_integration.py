#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Risk Integration Layer

Bridges portfolio_risk_manager with quant_engine, stops, and web_server.

Provides clean integration points for:
- Signal vetting against correlation and sector gates
- Dynamic stop management (replaces fixed ATR stops)
- Real-time risk dashboard metrics
- Circuit breaker enforcement

Usage:
    from portfolio_risk_integration import RiskEngine

    engine = RiskEngine(initial_equity=100000)
    engine.register_sector("AAPL", "Technology")

    # Evaluate signal before entry
    approval = engine.evaluate_signal(
        ticker="AAPL",
        proposed_shares=100,
        entry_price=150.0,
    )

    if approval.approved:
        # Execute trade and track position
        engine.add_position(
            ticker="AAPL",
            shares=approval.max_shares,
            entry_price=150.0,
            current_price=150.0,
            atr=2.5,
        )

    # Update positions (typically on each bar or hourly)
    metrics = engine.update(price_data, timestamp)
    if metrics.circuit_breaker_tripped:
        reject_new_signals()
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

from portfolio_risk_manager import (
    PortfolioRiskManager,
    CorrelationGateDecision,
    PortfolioMetrics,
    EngineState,
)

logger = logging.getLogger("RiskEngine")


class RiskEngine:
    """
    High-level risk management orchestrator for trading engine integration.

    Wraps PortfolioRiskManager with convenience methods for:
    - Signal evaluation before entry
    - Position tracking and updates
    - Real-time metrics and circuit breaker status
    - Stop adjustment automation
    """

    def __init__(
        self,
        initial_equity: float,
        max_sector_exposure: float = 3.0,
        intraday_drawdown_limit: float = -3.0,
        weekly_drawdown_limit: float = -5.0,
    ):
        """
        Initialize risk engine.

        Args:
            initial_equity: Starting account equity
            max_sector_exposure: Max % of portfolio per sector
            intraday_drawdown_limit: Intraday drawdown threshold (e.g., -3.0)
            weekly_drawdown_limit: Weekly drawdown threshold (e.g., -5.0)
        """
        self.prm = PortfolioRiskManager(
            initial_equity=initial_equity,
            max_sector_exposure=max_sector_exposure,
            intraday_drawdown_limit=intraday_drawdown_limit,
            weekly_drawdown_limit=weekly_drawdown_limit,
        )
        self._sector_map: Dict[str, str] = {}
        logger.info(
            f"RiskEngine initialized | equity=${initial_equity:,.2f} | "
            f"max_sector={max_sector_exposure}%"
        )

    def register_sector(self, ticker: str, sector: str) -> None:
        """Register a ticker and its sector classification."""
        self._sector_map[ticker] = sector
        self.prm.correlation_gate.register_ticker(ticker, sector)

    def register_universe(self, universe: Dict[str, str]) -> None:
        """Register multiple tickers at once."""
        for ticker, sector in universe.items():
            self.register_sector(ticker, sector)

    def evaluate_signal(
        self,
        ticker: str,
        proposed_shares: float,
        entry_price: float,
    ) -> CorrelationGateDecision:
        """
        Evaluate if a new signal passes all risk gates.

        Args:
            ticker: Stock symbol
            proposed_shares: Intended position size
            entry_price: Entry price

        Returns:
            CorrelationGateDecision with approval status and max_shares

        Example:
            result = engine.evaluate_signal("AAPL", 100, 150.0)
            if result.approved:
                execute_trade("AAPL", result.max_shares, 150.0)
            else:
                logger.warning(f"Signal vetoed: {result.limiting_factor}")
                for warning in result.warnings:
                    logger.warning(f"  - {warning}")
        """
        return self.prm.evaluate_new_signal(
            ticker=ticker,
            proposed_shares=proposed_shares,
            entry_price=entry_price,
            portfolio_equity=self.prm.current_equity,
        )

    def add_position(
        self,
        ticker: str,
        shares: float,
        entry_price: float,
        current_price: float,
        atr: float,
        structural_floor: float = None,
    ) -> None:
        """
        Register a new position for risk tracking.

        Args:
            ticker: Stock symbol
            shares: Position size
            entry_price: Entry price
            current_price: Current market price
            atr: Current ATR value
            structural_floor: EMA20, Supertrend, or support level (optional)
        """
        if structural_floor is None:
            structural_floor = entry_price - (2.0 * atr)

        stop_price = entry_price - (2.0 * atr)
        sector = self._sector_map.get(ticker, "unknown")

        self.prm.add_position(
            ticker=ticker,
            shares=shares,
            entry_price=entry_price,
            stop_price=stop_price,
            structural_floor=structural_floor,
            current_price=current_price,
            sector=sector,
            atr=atr,
        )
        logger.info(
            f"Position tracked | {ticker} | "
            f"shares={shares:.0f} | entry=${entry_price:.2f} | stop=${stop_price:.2f}"
        )

    def update_position(
        self,
        ticker: str,
        current_price: float,
        atr: float,
        structural_floor: float,
        timestamp: datetime = None,
    ) -> Tuple[bool, Optional[float], Optional[str]]:
        """
        Update position with current market data.

        Checks for stop hits and adjusts trailing stops.

        Args:
            ticker: Stock symbol
            current_price: Current price
            atr: Current ATR
            structural_floor: Current structural floor (EMA20, Supertrend, etc.)
            timestamp: Current timestamp (defaults to now)

        Returns:
            (position_valid, new_stop_price, adjustment_reason)
            - position_valid: False if stop was hit
            - new_stop_price: Updated stop if adjusted, else None
            - adjustment_reason: "break_even", "trail", "volatility_spike"

        Example:
            valid, new_stop, reason = engine.update_position(
                "AAPL", 152.0, 2.6, 151.0
            )
            if not valid:
                close_position("AAPL")
            elif new_stop:
                update_broker_stop(new_stop)
        """
        if timestamp is None:
            timestamp = datetime.now()

        valid, event = self.prm.update_position_price(
            ticker=ticker,
            current_price=current_price,
            atr=atr,
            structural_floor=structural_floor,
            timestamp=timestamp,
        )

        new_stop = event.new_stop if event else None
        reason = event.reason.value if event else None

        if not valid:
            logger.warning(
                f"Stop hit | {ticker} | price=${current_price:.2f} | "
                f"stop=${self.prm.positions[ticker].stop_price:.2f}"
            )
        elif new_stop:
            logger.info(
                f"Stop adjusted | {ticker} | "
                f"${self.prm.positions[ticker].stop_price:.2f} -> ${new_stop:.2f} | "
                f"reason={reason}"
            )

        return valid, new_stop, reason

    def update(
        self,
        price_data: Dict[str, float],
        timestamp: datetime = None,
    ) -> Dict:
        """
        Update all positions and compute metrics.

        Args:
            price_data: Dict of {ticker: current_price}
            timestamp: Current timestamp (defaults to now)

        Returns:
            Dict with portfolio status:
            - equity: Current portfolio equity
            - drawdown: Intraday drawdown %
            - positions: Count of open positions
            - circuit_breaker_tripped: Boolean
            - engine_state: "ACTIVE", "CIRCUIT_BREAKER", etc.

        Example:
            metrics = engine.update(
                {"AAPL": 152.0, "MSFT": 305.0},
                datetime.now()
            )
            if metrics['circuit_breaker_tripped']:
                pause_trading()
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Update correlation matrix
        self.prm.update_prices(price_data, timestamp)

        # Compute metrics
        metrics = self.prm.update_metrics(timestamp)

        return {
            "equity": metrics.total_equity,
            "unrealized_pnl": metrics.total_unrealized_pnl,
            "intraday_drawdown": metrics.intraday_drawdown_pct,
            "weekly_drawdown": metrics.rolling_weekly_drawdown_pct,
            "positions": metrics.open_positions_count,
            "max_sector_exposure": metrics.max_sector_exposure_pct,
            "engine_state": metrics.engine_state.value,
            "circuit_breaker_tripped": metrics.circuit_breaker_reason is not None,
            "circuit_breaker_reason": (
                metrics.circuit_breaker_reason.value
                if metrics.circuit_breaker_reason
                else None
            ),
        }

    def close_position(self, ticker: str) -> None:
        """Close out a position (on exit or stop hit)."""
        self.prm.close_position(ticker)
        logger.info(f"Position closed | {ticker}")

    def get_status(self) -> Dict:
        """Get current portfolio status snapshot."""
        return self.prm.get_risk_report()

    def is_circuit_breaker_active(self) -> bool:
        """Check if circuit breaker is locked (new signals blocked)."""
        return self.prm.circuit_breaker.is_locked()

    def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker (requires explicit action)."""
        self.prm.circuit_breaker.manual_reset()
        logger.warning("Circuit breaker manually reset")

    def get_stop_price(self, ticker: str) -> Optional[float]:
        """Get current stop price for a position."""
        if ticker not in self.prm.positions:
            return None
        return self.prm.positions[ticker].stop_price

    def get_positions(self) -> Dict:
        """Get all active positions."""
        return {
            ticker: {
                "shares": pos.shares,
                "entry": pos.entry_price,
                "current": pos.current_price,
                "stop": pos.stop_price,
                "pnl": pos.unrealized_pnl,
                "pnl_pct": pos.unrealized_pnl_pct,
            }
            for ticker, pos in self.prm.positions.items()
        }


if __name__ == "__main__":
    # Quick integration test
    print("=" * 70)
    print("Portfolio Risk Integration — Test")
    print("=" * 70)

    engine = RiskEngine(initial_equity=100000)

    # Register universe
    universe = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "JPM": "Financials",
    }
    engine.register_universe(universe)
    print("\n✓ Universe registered")

    # Evaluate signal
    result = engine.evaluate_signal("AAPL", 20, 150.0)
    print(f"✓ Signal evaluation: approved={result.approved}, max={result.max_shares:.0f}")

    # Add position
    engine.add_position("AAPL", 20, 150.0, 150.0, 2.5, 146.0)
    print("✓ Position added")

    # Update prices
    metrics = engine.update({"AAPL": 152.0, "MSFT": 305.0})
    print(f"✓ Update: equity=${metrics['equity']:,.2f}, drawdown={metrics['intraday_drawdown']:.2f}%")

    # Get status
    status = engine.get_status()
    print(f"✓ Status: {status['positions_count']} positions, state={status['engine_state']}")

    print("\n" + "=" * 70)
    print("✓ Integration test passed")
    print("=" * 70)

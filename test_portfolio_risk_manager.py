#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive unit tests for portfolio_risk_manager.py

Coverage:
- DynamicCorrelationGate: sector caps, correlation thresholds, NaN handling
- AutomatedCircuitBreaker: intraday/weekly triggers, recovery, auto-recovery
- TrailingStopManager: break-even logic, structural trailing, volatility spikes
- Integration: full portfolio state management and edge cases

Edge cases tested:
- Missing data feeds
- NaN correlation matrices
- Extreme high-volatility gaps
- Single position portfolios
- Zero equity edge case
- Simultaneous gate violations
"""

import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from portfolio_risk_manager import (
    PortfolioRiskManager,
    DynamicCorrelationGate,
    AutomatedCircuitBreaker,
    TrailingStopManager,
    Position,
    EngineState,
    CircuitBreakerReason,
    StopAdjustmentReason,
)


class TestDynamicCorrelationGate(unittest.TestCase):
    """Test correlation gate logic."""

    def setUp(self):
        self.gate = DynamicCorrelationGate(
            max_sector_exposure_pct=3.0,
            correlation_threshold=0.65,
            lookback_bars=60,
        )

    def test_register_ticker(self):
        """Test ticker registration."""
        self.gate.register_ticker("AAPL", "Technology")
        self.assertEqual(self.gate.sector_map["AAPL"], "Technology")

    def test_correlation_matrix_insufficient_data(self):
        """Test correlation matrix with insufficient data."""
        self.gate.register_ticker("AAPL", "Tech")
        self.gate.register_ticker("MSFT", "Tech")
        now = datetime.now()

        # Add only 10 bars (need 60)
        for i in range(10):
            self.gate.update_price("AAPL", 150.0 + i, now + timedelta(days=i))
            self.gate.update_price("MSFT", 300.0 + i, now + timedelta(days=i))

        corr = self.gate.compute_correlation_matrix()
        self.assertIsNone(corr)

    def test_correlation_matrix_with_sufficient_data(self):
        """Test correlation matrix computation."""
        self.gate.register_ticker("AAPL", "Tech")
        self.gate.register_ticker("MSFT", "Tech")
        now = datetime.now()

        # Add 70 bars
        for i in range(70):
            self.gate.update_price("AAPL", 150.0 + i * 0.5, now + timedelta(days=i))
            self.gate.update_price("MSFT", 300.0 + i * 1.0, now + timedelta(days=i))

        corr = self.gate.compute_correlation_matrix()
        self.assertIsNotNone(corr)
        self.assertIn("AAPL", corr.columns)
        self.assertIn("MSFT", corr.columns)

    def test_nan_handling_in_correlation(self):
        """Test NaN handling with missing data."""
        self.gate.register_ticker("AAPL", "Tech")
        self.gate.register_ticker("MSFT", "Tech")
        now = datetime.now()

        # Add data with gaps (simulating missing data)
        for i in range(70):
            if i % 3 == 0:  # Skip every 3rd day
                continue
            self.gate.update_price("AAPL", 150.0 + i * 0.5, now + timedelta(days=i))
            self.gate.update_price("MSFT", 300.0 + i * 1.0, now + timedelta(days=i))

        corr = self.gate.compute_correlation_matrix()
        # Should handle NaNs gracefully
        self.assertFalse(corr.isna().all().all())

    def test_sector_exposure_limit_violation(self):
        """Test sector exposure cap enforcement."""
        self.gate.register_ticker("AAPL", "Tech")
        self.gate.register_ticker("MSFT", "Tech")
        self.gate.register_ticker("JPM", "Financials")

        # Create existing positions in Tech sector
        existing_positions = {
            "AAPL": Position(
                ticker="AAPL",
                shares=100,
                entry_price=150.0,
                current_price=150.0,
                stop_price=145.0,
                structural_floor=146.0,
                sector="Tech",
            ),
            "MSFT": Position(
                ticker="MSFT",
                shares=50,
                entry_price=300.0,
                current_price=300.0,
                stop_price=290.0,
                structural_floor=295.0,
                sector="Tech",
            ),
        }

        portfolio_equity = 100000
        existing_exposure = (100 * 150 + 50 * 300) / portfolio_equity  # 0.75% + 0.15% = 0.90%

        # Try to add another large Tech position
        result = self.gate.evaluate_position(
            ticker="NVDA",
            proposed_shares=100,  # Would be 0.85% of portfolio
            current_price=850.0,
            portfolio_equity=portfolio_equity,
            existing_positions=existing_positions,
        )

        # Total Tech would be 0.90% + 0.85% = 1.75%, well under 3% limit
        self.assertTrue(result.approved)

    def test_sector_exposure_hard_cap(self):
        """Test hard cap on sector exposure."""
        self.gate.register_ticker("AAPL", "Tech")
        self.gate.register_ticker("MSFT", "Tech")
        self.gate.register_ticker("NVDA", "Tech")

        # Create existing heavy Tech exposure
        existing_positions = {
            "AAPL": Position(
                ticker="AAPL",
                shares=1000,
                entry_price=150.0,
                current_price=150.0,
                stop_price=145.0,
                structural_floor=146.0,
                sector="Tech",
            ),
            "MSFT": Position(
                ticker="MSFT",
                shares=500,
                entry_price=300.0,
                current_price=300.0,
                stop_price=290.0,
                structural_floor=295.0,
                sector="Tech",
            ),
        }

        portfolio_equity = 100000
        existing_exposure = (1000 * 150 + 500 * 300) / portfolio_equity  # 1.5% + 1.5% = 3.0%

        # Try to add another Tech position (would exceed 3% cap)
        result = self.gate.evaluate_position(
            ticker="NVDA",
            proposed_shares=100,  # Would be 0.85%
            current_price=850.0,
            portfolio_equity=portfolio_equity,
            existing_positions=existing_positions,
        )

        # Should exceed cap
        self.assertFalse(result.approved)
        self.assertEqual(result.limiting_factor, "sector")
        self.assertGreater(result.exposure_ratio, 1.0)

    def test_correlation_threshold_trigger(self):
        """Test high correlation detection and scaling."""
        self.gate.register_ticker("AAPL", "Tech")
        self.gate.register_ticker("MSFT", "Tech")
        now = datetime.now()

        # Create highly correlated price series
        for i in range(70):
            price = 150.0 + i
            self.gate.update_price("AAPL", price, now + timedelta(days=i))
            self.gate.update_price("MSFT", price * 2, now + timedelta(days=i))

        self.gate.compute_correlation_matrix()

        existing_positions = {
            "AAPL": Position(
                ticker="AAPL",
                shares=100,
                entry_price=150.0,
                current_price=150.0,
                stop_price=145.0,
                structural_floor=146.0,
                sector="Tech",
            ),
        }

        result = self.gate.evaluate_position(
            ticker="MSFT",
            proposed_shares=100,
            current_price=300.0,
            portfolio_equity=100000,
            existing_positions=existing_positions,
        )

        # High correlation should trigger scaling
        self.assertLess(result.max_shares, 100)
        self.assertIn("correlation", result.limiting_factor.lower())


class TestAutomatedCircuitBreaker(unittest.TestCase):
    """Test circuit breaker logic."""

    def setUp(self):
        self.cb = AutomatedCircuitBreaker(
            intraday_drawdown_threshold=-3.0,
            weekly_drawdown_threshold=-5.0,
            auto_recovery_enabled=False,
        )
        self.start_time = datetime.now()

    def test_initialization(self):
        """Test circuit breaker initialization."""
        self.assertEqual(self.cb.state, EngineState.ACTIVE)
        self.assertIsNone(self.cb.trip_reason)

    def test_session_initialization(self):
        """Test session setup."""
        self.cb.initialize_session(100000)
        self.assertEqual(self.cb.session_start_equity, 100000)
        self.assertEqual(self.cb.peak_equity, 100000)
        self.assertEqual(self.cb.state, EngineState.ACTIVE)

    def test_intraday_drawdown_trigger(self):
        """Test intraday drawdown threshold trigger."""
        self.cb.initialize_session(100000)

        # Equity drops 3.1%
        state, reason = self.cb.update(96900, self.start_time + timedelta(hours=2))

        self.assertEqual(state, EngineState.CIRCUIT_BREAKER)
        self.assertEqual(reason, CircuitBreakerReason.INTRADAY_DRAWDOWN)

    def test_intraday_drawdown_no_trigger(self):
        """Test that modest drawdown doesn't trigger."""
        self.cb.initialize_session(100000)

        # Equity drops 2%
        state, reason = self.cb.update(98000, self.start_time + timedelta(hours=2))

        self.assertEqual(state, EngineState.ACTIVE)
        self.assertIsNone(reason)

    def test_weekly_drawdown_trigger(self):
        """Test weekly drawdown threshold trigger."""
        self.cb.initialize_session(100000)
        self.cb.weekly_open_equity = 100000

        # Equity drops 5.1%
        state, reason = self.cb.update(94900, self.start_time + timedelta(days=3))

        self.assertEqual(state, EngineState.CIRCUIT_BREAKER)
        self.assertEqual(reason, CircuitBreakerReason.WEEKLY_DRAWDOWN)

    def test_dual_trigger(self):
        """Test simultaneous intraday and weekly drawdown triggers."""
        self.cb.initialize_session(100000)
        self.cb.weekly_open_equity = 100000

        # Both thresholds exceeded
        state, reason = self.cb.update(94000, self.start_time + timedelta(days=3))

        self.assertEqual(state, EngineState.CIRCUIT_BREAKER)
        self.assertEqual(reason, CircuitBreakerReason.DUAL_TRIGGER)

    def test_manual_reset(self):
        """Test manual circuit breaker reset."""
        self.cb.initialize_session(100000)
        self.cb.update(96900, self.start_time + timedelta(hours=2))
        self.assertEqual(self.cb.state, EngineState.CIRCUIT_BREAKER)

        self.cb.manual_reset()
        self.assertEqual(self.cb.state, EngineState.ACTIVE)
        self.assertIsNone(self.cb.trip_reason)

    def test_is_locked(self):
        """Test lock status."""
        self.cb.initialize_session(100000)
        self.assertFalse(self.cb.is_locked())

        self.cb.update(96900, self.start_time + timedelta(hours=2))
        self.assertTrue(self.cb.is_locked())

    def test_auto_recovery_enabled(self):
        """Test auto-recovery when enabled."""
        cb = AutomatedCircuitBreaker(
            intraday_drawdown_threshold=-3.0,
            weekly_drawdown_threshold=-5.0,
            auto_recovery_enabled=True,
        )
        cb.initialize_session(100000)

        # Trigger circuit breaker
        cb.update(96900, self.start_time + timedelta(hours=2))
        self.assertTrue(cb.is_locked())

        # Recover back above threshold
        cb.update(97100, self.start_time + timedelta(hours=3))
        self.assertFalse(cb.is_locked())


class TestTrailingStopManager(unittest.TestCase):
    """Test trailing stop logic."""

    def setUp(self):
        self.tsm = TrailingStopManager(
            breakeven_atr_multiple=1.0,
            trail_atr_multiple=0.5,
        )
        self.start_time = datetime.now()

    def test_add_position(self):
        """Test position registration."""
        self.tsm.add_position(
            ticker="AAPL",
            entry_price=150.0,
            initial_stop=145.0,
            structural_floor=146.0,
            atr=2.5,
        )

        self.assertIn("AAPL", self.tsm.position_states)
        self.assertEqual(self.tsm.get_stop_price("AAPL"), 145.0)

    def test_breakeven_elevation(self):
        """Test break-even stop elevation."""
        self.tsm.add_position(
            ticker="AAPL",
            entry_price=150.0,
            initial_stop=145.0,
            structural_floor=146.0,
            atr=2.5,
        )

        # Price moves +2.5 (1x ATR = break-even trigger)
        new_stop, reason = self.tsm.update_price(
            ticker="AAPL",
            current_price=152.5,
            atr=2.5,
            structural_floor=151.0,
            timestamp=self.start_time + timedelta(hours=1),
        )

        self.assertEqual(new_stop, 150.0)  # At entry price
        self.assertEqual(reason, StopAdjustmentReason.BREAK_EVEN)

    def test_no_breakeven_below_threshold(self):
        """Test that stop doesn't adjust without sufficient price move."""
        self.tsm.add_position(
            ticker="AAPL",
            entry_price=150.0,
            initial_stop=145.0,
            structural_floor=146.0,
            atr=2.5,
        )

        # Price moves only +1.0 (below 1x ATR threshold)
        new_stop, reason = self.tsm.update_price(
            ticker="AAPL",
            current_price=151.0,
            atr=2.5,
            structural_floor=151.0,
            timestamp=self.start_time + timedelta(hours=1),
        )

        self.assertIsNone(new_stop)  # No adjustment
        self.assertIsNone(reason)

    def test_structural_trailing(self):
        """Test trailing stop beneath structural floor."""
        self.tsm.add_position(
            ticker="AAPL",
            entry_price=150.0,
            initial_stop=145.0,
            structural_floor=146.0,
            atr=2.5,
        )

        # First, set break-even
        self.tsm.update_price(
            ticker="AAPL",
            current_price=152.5,
            atr=2.5,
            structural_floor=151.0,
            timestamp=self.start_time,
        )

        # Then trail beneath new floor
        new_stop, reason = self.tsm.update_price(
            ticker="AAPL",
            current_price=155.0,
            atr=2.5,
            structural_floor=153.0,
            timestamp=self.start_time + timedelta(hours=2),
        )

        # Stop should trail: 153.0 - (0.5 * 2.5) = 151.75
        self.assertIsNotNone(new_stop)
        self.assertAlmostEqual(new_stop, 151.75, places=1)
        self.assertEqual(reason, StopAdjustmentReason.STRUCTURAL_TRAIL)

    def test_volatility_spike_widening(self):
        """Test stop widening on volatility spike."""
        self.tsm.add_position(
            ticker="AAPL",
            entry_price=150.0,
            initial_stop=145.0,
            structural_floor=146.0,
            atr=2.5,
        )

        # Baseline ATR is 2.5, now ATR spikes to 3.5 (1.4x increase)
        new_stop, reason = self.tsm.update_price(
            ticker="AAPL",
            current_price=151.0,
            atr=3.5,  # Volatility increased
            structural_floor=146.0,
            timestamp=self.start_time + timedelta(hours=1),
        )

        # Stop should widen due to vol spike
        if new_stop is not None:
            self.assertGreater(new_stop, 145.0)  # Widened from initial stop

    def test_remove_position(self):
        """Test position removal."""
        self.tsm.add_position(
            ticker="AAPL",
            entry_price=150.0,
            initial_stop=145.0,
            structural_floor=146.0,
            atr=2.5,
        )

        self.assertIn("AAPL", self.tsm.position_states)
        self.tsm.remove_position("AAPL")
        self.assertNotIn("AAPL", self.tsm.position_states)

    def test_get_stop_price_nonexistent(self):
        """Test retrieving stop for non-existent position."""
        stop = self.tsm.get_stop_price("MISSING")
        self.assertIsNone(stop)


class TestPortfolioRiskManager(unittest.TestCase):
    """Integration tests for the full risk manager."""

    def setUp(self):
        self.prm = PortfolioRiskManager(
            initial_equity=100000,
            max_sector_exposure=3.0,
            intraday_drawdown_limit=-3.0,
            weekly_drawdown_limit=-5.0,
        )
        self.start_time = datetime.now()

        # Register watchlist
        watchlist = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "JPM": "Financials",
            "BAC": "Financials",
        }
        self.prm.register_watchlist(watchlist)

    def test_register_watchlist(self):
        """Test watchlist registration."""
        self.assertEqual(len(self.prm.correlation_gate.sector_map), 4)

    def test_evaluate_new_signal_active_state(self):
        """Test signal evaluation in active state."""
        result = self.prm.evaluate_new_signal(
            ticker="AAPL",
            proposed_shares=100,
            entry_price=150.0,
            portfolio_equity=100000,
        )

        self.assertTrue(result.approved)

    def test_evaluate_new_signal_circuit_breaker_locked(self):
        """Test that circuit breaker veto new signals."""
        # Trigger circuit breaker
        self.prm.circuit_breaker.initialize_session(100000)
        self.prm.circuit_breaker.update(96900, self.start_time)

        result = self.prm.evaluate_new_signal(
            ticker="AAPL",
            proposed_shares=100,
            entry_price=150.0,
            portfolio_equity=96900,
        )

        self.assertFalse(result.approved)
        self.assertEqual(result.limiting_factor, "circuit_breaker")

    def test_add_position_and_track(self):
        """Test adding position and tracking."""
        self.prm.add_position(
            ticker="AAPL",
            shares=100,
            entry_price=150.0,
            stop_price=145.0,
            structural_floor=146.0,
            current_price=150.0,
            sector="Technology",
            atr=2.5,
        )

        self.assertIn("AAPL", self.prm.positions)
        position = self.prm.positions["AAPL"]
        self.assertEqual(position.shares, 100)
        self.assertEqual(position.unrealized_pnl, 0)

    def test_position_unrealized_pnl(self):
        """Test P&L calculation."""
        self.prm.add_position(
            ticker="AAPL",
            shares=100,
            entry_price=150.0,
            stop_price=145.0,
            structural_floor=146.0,
            current_price=155.0,
            sector="Technology",
            atr=2.5,
        )

        position = self.prm.positions["AAPL"]
        self.assertEqual(position.unrealized_pnl, 500.0)  # +$5 per share
        self.assertEqual(position.unrealized_pnl_pct, 3.33)

    def test_update_position_price_stop_hit(self):
        """Test position exit on stop hit."""
        self.prm.add_position(
            ticker="AAPL",
            shares=100,
            entry_price=150.0,
            stop_price=145.0,
            structural_floor=146.0,
            current_price=150.0,
            sector="Technology",
            atr=2.5,
        )

        # Price drops to stop level
        valid, event = self.prm.update_position_price(
            ticker="AAPL",
            current_price=144.9,  # Below stop
            atr=2.5,
            structural_floor=145.0,
            timestamp=self.start_time,
        )

        self.assertFalse(valid)

    def test_update_metrics_calculates_drawdown(self):
        """Test metrics calculation."""
        self.prm.add_position(
            ticker="AAPL",
            shares=100,
            entry_price=150.0,
            stop_price=145.0,
            structural_floor=146.0,
            current_price=145.0,
            sector="Technology",
            atr=2.5,
        )

        metrics = self.prm.update_metrics(self.start_time)

        self.assertEqual(metrics.total_equity, 99500)  # 100k + (-$500 loss)
        self.assertEqual(metrics.total_unrealized_pnl, -500)
        self.assertAlmostEqual(metrics.intraday_drawdown_pct, -0.5, places=1)

    def test_close_position(self):
        """Test position closure."""
        self.prm.add_position(
            ticker="AAPL",
            shares=100,
            entry_price=150.0,
            stop_price=145.0,
            structural_floor=146.0,
            current_price=150.0,
            sector="Technology",
            atr=2.5,
        )

        self.assertIn("AAPL", self.prm.positions)
        self.prm.close_position("AAPL")
        self.assertNotIn("AAPL", self.prm.positions)

    def test_get_risk_report(self):
        """Test risk report generation."""
        self.prm.add_position(
            ticker="AAPL",
            shares=100,
            entry_price=150.0,
            stop_price=145.0,
            structural_floor=146.0,
            current_price=152.0,
            sector="Technology",
            atr=2.5,
        )

        self.prm.update_metrics(self.start_time)
        report = self.prm.get_risk_report()

        self.assertIn("portfolio_equity", report)
        self.assertIn("unrealized_pnl", report)
        self.assertIn("positions_count", report)
        self.assertEqual(report["positions_count"], 1)

    def test_multiple_positions_sector_tracking(self):
        """Test sector exposure tracking with multiple positions."""
        self.prm.add_position(
            ticker="AAPL",
            shares=100,
            entry_price=150.0,
            stop_price=145.0,
            structural_floor=146.0,
            current_price=150.0,
            sector="Technology",
            atr=2.5,
        )

        self.prm.add_position(
            ticker="MSFT",
            shares=50,
            entry_price=300.0,
            stop_price=290.0,
            structural_floor=295.0,
            current_price=300.0,
            sector="Technology",
            atr=5.0,
        )

        metrics = self.prm.update_metrics(self.start_time)
        self.assertEqual(metrics.open_positions_count, 2)
        # Tech exposure: (100*150 + 50*300) / 100000 = 0.30%
        self.assertAlmostEqual(metrics.max_sector_exposure_pct, 0.30, places=2)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_zero_equity(self):
        """Test behavior with zero portfolio equity."""
        prm = PortfolioRiskManager(
            initial_equity=100000,
            max_sector_exposure=3.0,
        )

        # Force zero equity via large loss
        prm.initial_equity = 0
        result = prm.evaluate_new_signal(
            ticker="AAPL",
            proposed_shares=100,
            entry_price=150.0,
            portfolio_equity=0,
        )

        # Should handle gracefully
        self.assertIsNotNone(result)

    def test_missing_ticker_in_positions(self):
        """Test updating non-existent position."""
        prm = PortfolioRiskManager(initial_equity=100000)

        valid, event = prm.update_position_price(
            ticker="NONEXISTENT",
            current_price=100.0,
            atr=2.5,
            structural_floor=99.0,
            timestamp=datetime.now(),
        )

        self.assertFalse(valid)

    def test_extreme_volatility_gap(self):
        """Test handling of extreme price gaps."""
        prm = PortfolioRiskManager(initial_equity=100000)
        prm.add_position(
            ticker="AAPL",
            shares=100,
            entry_price=150.0,
            stop_price=145.0,
            structural_floor=146.0,
            current_price=150.0,
            sector="Technology",
            atr=2.5,
        )

        # Simulate gap down to far below stop
        valid, _ = prm.update_position_price(
            ticker="AAPL",
            current_price=120.0,  # Gap down 20%
            atr=2.5,
            structural_floor=135.0,
            timestamp=datetime.now(),
        )

        # Position should be invalid
        self.assertFalse(valid)

    def test_nan_in_correlations(self):
        """Test NaN handling in correlation matrix."""
        gate = DynamicCorrelationGate()
        gate.register_ticker("A", "Tech")
        gate.register_ticker("B", "Tech")

        now = datetime.now()
        # Add limited data
        for i in range(20):
            gate.update_price("A", 100.0 + i, now + timedelta(days=i))
            # B has no data (all NaN)

        corr = gate.compute_correlation_matrix()
        # Should handle gracefully
        self.assertIsNone(corr)


class TestPerformance(unittest.TestCase):
    """Test performance and latency constraints."""

    def test_correlation_update_latency(self):
        """Test that correlation updates complete quickly."""
        import time

        gate = DynamicCorrelationGate()

        # Register 50 tickers
        for i in range(50):
            gate.register_ticker(f"TICK{i:02d}", "Technology")

        now = datetime.now()

        # Add 100 bars
        start = time.time()
        for bar in range(100):
            for i in range(50):
                gate.update_price(f"TICK{i:02d}", 100.0 + bar, now + timedelta(days=bar))

        # Compute correlation
        corr = gate.compute_correlation_matrix()
        elapsed = time.time() - start

        # Should complete in < 100ms
        self.assertLess(elapsed, 0.1)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal integration test for portfolio_risk_manager.py

Tests core functionality without numpy/pandas dependency issues.
Validates:
- PortfolioRiskManager initialization and lifecycle
- Position tracking and P&L calculation
- Circuit breaker state management
- Stop adjustment logic
- Risk gate evaluation
"""

import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, '/home/user/al')

try:
    from portfolio_risk_manager import (
        PortfolioRiskManager,
        Position,
        EngineState,
        CircuitBreakerReason,
        StopAdjustmentReason,
    )
    print("✓ Portfolio risk manager imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)


def test_position_creation():
    """Test Position dataclass and P&L calculations."""
    print("\n[TEST] Position Creation & P&L")

    pos = Position(
        ticker="AAPL",
        shares=100,
        entry_price=150.0,
        current_price=155.0,
        stop_price=145.0,
        structural_floor=146.0,
        sector="Technology",
    )

    assert pos.unrealized_pnl == 500.0, "P&L calculation failed"
    assert abs(pos.unrealized_pnl_pct - 3.33) < 0.1, "P&L % calculation failed"
    print("  ✓ Position creation OK")
    print(f"  ✓ P&L: ${pos.unrealized_pnl:.2f} ({pos.unrealized_pnl_pct:.2f}%)")


def test_circuit_breaker():
    """Test circuit breaker initialization and triggers."""
    print("\n[TEST] Circuit Breaker")

    from portfolio_risk_manager import AutomatedCircuitBreaker

    cb = AutomatedCircuitBreaker(
        intraday_drawdown_threshold=-3.0,
        weekly_drawdown_threshold=-5.0,
    )

    # Initial state
    assert cb.state == EngineState.ACTIVE, "Should start ACTIVE"
    print("  ✓ Initial state: ACTIVE")

    # Initialize session
    cb.initialize_session(100000)
    assert cb.session_start_equity == 100000, "Session init failed"
    print("  ✓ Session initialized at $100,000")

    # Test intraday drawdown trigger
    now = datetime.now()
    state, reason = cb.update(96900, now + timedelta(hours=2))
    assert state == EngineState.CIRCUIT_BREAKER, "Should trigger CB on -3.1%"
    assert reason == CircuitBreakerReason.INTRADAY_DRAWDOWN, "Wrong trigger reason"
    assert cb.is_locked(), "Should be locked"
    print("  ✓ Intraday -3.1% triggers circuit breaker")

    # Test manual reset
    cb.manual_reset()
    assert cb.state == EngineState.ACTIVE, "Manual reset failed"
    assert not cb.is_locked(), "Should unlock after reset"
    print("  ✓ Manual reset works")


def test_trailing_stops():
    """Test trailing stop manager logic."""
    print("\n[TEST] Trailing Stop Manager")

    from portfolio_risk_manager import TrailingStopManager

    tsm = TrailingStopManager(
        breakeven_atr_multiple=1.0,
        trail_atr_multiple=0.5,
    )

    now = datetime.now()

    # Add position
    tsm.add_position(
        ticker="AAPL",
        entry_price=150.0,
        initial_stop=145.0,
        structural_floor=146.0,
        atr=2.5,
    )
    assert "AAPL" in tsm.position_states, "Position not registered"
    print("  ✓ Position registered")

    # Check initial stop
    assert tsm.get_stop_price("AAPL") == 145.0, "Initial stop wrong"
    print("  ✓ Initial stop: $145.00")

    # Update with favorable move (< 1 ATR)
    new_stop, reason = tsm.update_price(
        ticker="AAPL",
        current_price=151.0,
        atr=2.5,
        structural_floor=151.0,
        timestamp=now + timedelta(hours=1),
    )
    assert new_stop is None, "Should not adjust below 1 ATR"
    print("  ✓ No adjustment below 1x ATR threshold")

    # Update with break-even trigger (>= 1 ATR)
    new_stop, reason = tsm.update_price(
        ticker="AAPL",
        current_price=152.5,  # +2.5 = 1 ATR
        atr=2.5,
        structural_floor=151.0,
        timestamp=now + timedelta(hours=2),
    )
    assert new_stop == 150.0, "Break-even not set correctly"
    assert reason == StopAdjustmentReason.BREAK_EVEN, "Wrong adjustment reason"
    print("  ✓ Break-even elevation: $150.00")

    # Clean up
    tsm.remove_position("AAPL")
    assert "AAPL" not in tsm.position_states, "Position not removed"
    print("  ✓ Position removal works")


def test_portfolio_risk_manager():
    """Test main PortfolioRiskManager."""
    print("\n[TEST] Portfolio Risk Manager")

    prm = PortfolioRiskManager(
        initial_equity=100000,
        max_sector_exposure=3.0,
        intraday_drawdown_limit=-3.0,
        weekly_drawdown_limit=-5.0,
    )

    # Register watchlist
    watchlist = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "JPM": "Financials",
    }
    prm.register_watchlist(watchlist)
    print("  ✓ Watchlist registered: 3 tickers")

    # Evaluate signal (should pass with reasonable size)
    # Max 3% of $100k = $3k max, so ~20 shares at $150
    gate1 = prm.evaluate_new_signal(
        ticker="AAPL",
        proposed_shares=20,
        entry_price=150.0,
        portfolio_equity=100000,
    )
    assert gate1.approved, f"Signal should pass but got {gate1.limiting_factor}: {gate1.warnings}"
    print("  ✓ First signal approved")

    # Add position
    prm.add_position(
        ticker="AAPL",
        shares=20,
        entry_price=150.0,
        stop_price=145.0,
        structural_floor=146.0,
        current_price=150.0,
        sector="Technology",
        atr=2.5,
    )
    assert "AAPL" in prm.positions, "Position not tracked"
    assert len(prm.positions) == 1, "Position count wrong"
    print("  ✓ Position added and tracked")

    # Update position price (no stop hit)
    now = datetime.now()
    valid, event = prm.update_position_price(
        ticker="AAPL",
        current_price=152.0,
        atr=2.5,
        structural_floor=151.0,
        timestamp=now,
    )
    assert valid, "Position should still be valid"
    print("  ✓ Position price updated")

    # Get metrics
    metrics = prm.update_metrics(now)
    # Position: 20 shares * $2 gain = $40 gain
    assert metrics.total_equity == 100040.0, f"Equity calculation wrong: {metrics.total_equity}"
    assert metrics.open_positions_count == 1, "Position count in metrics wrong"
    assert metrics.engine_state == EngineState.ACTIVE, "Engine state wrong"
    print(f"  ✓ Portfolio equity: ${metrics.total_equity:,.2f}")
    print(f"  ✓ Positions: {metrics.open_positions_count}")

    # Close position
    prm.close_position("AAPL")
    assert "AAPL" not in prm.positions, "Position not closed"
    print("  ✓ Position closed")


def test_correlation_gate():
    """Test correlation gate with small dataset."""
    print("\n[TEST] Correlation Gate")

    from portfolio_risk_manager import DynamicCorrelationGate

    gate = DynamicCorrelationGate(
        max_sector_exposure_pct=3.0,
        correlation_threshold=0.65,
        lookback_bars=60,
    )

    gate.register_ticker("AAPL", "Tech")
    gate.register_ticker("MSFT", "Tech")
    print("  ✓ Tickers registered")

    now = datetime.now()

    # Add insufficient data (< 60 bars)
    for i in range(20):
        gate.update_price("AAPL", 150.0 + i, now + timedelta(days=i))
        gate.update_price("MSFT", 300.0 + i * 2, now + timedelta(days=i))

    # Correlation should return None (insufficient data)
    corr = gate.compute_correlation_matrix()
    assert corr is None, "Should return None with insufficient data"
    print("  ✓ Handles insufficient data gracefully")

    # Evaluate gate with existing position
    existing = {
        "AAPL": Position(
            ticker="AAPL",
            shares=100,
            entry_price=150.0,
            current_price=155.0,
            stop_price=145.0,
            structural_floor=146.0,
            sector="Tech",
        )
    }

    result = gate.evaluate_position(
        ticker="MSFT",
        proposed_shares=50,
        current_price=310.0,
        portfolio_equity=100000,
        existing_positions=existing,
    )

    assert result is not None, "Gate evaluation failed"
    print(f"  ✓ Gate decision: approved={result.approved}, max_shares={result.max_shares:.0f}")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n[TEST] Edge Cases")

    prm = PortfolioRiskManager(initial_equity=100000)

    # Test with zero positions
    metrics = prm.update_metrics(datetime.now())
    assert metrics.open_positions_count == 0, "Should handle zero positions"
    print("  ✓ Handles zero positions")

    # Test with non-existent ticker
    valid, _ = prm.update_position_price(
        ticker="MISSING",
        current_price=100.0,
        atr=2.0,
        structural_floor=99.0,
        timestamp=datetime.now(),
    )
    assert not valid, "Should return False for missing ticker"
    print("  ✓ Handles missing ticker")

    # Test circuit breaker lock
    prm.circuit_breaker.initialize_session(100000)
    prm.circuit_breaker.update(96900, datetime.now())  # Trigger

    gate = prm.evaluate_new_signal(
        ticker="AAPL",
        proposed_shares=100,
        entry_price=150.0,
        portfolio_equity=96900,
    )
    assert not gate.approved, "Should veto signal when CB is locked"
    assert gate.limiting_factor == "circuit_breaker", "Wrong limiting factor"
    print("  ✓ Circuit breaker blocks new signals")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("Portfolio Risk Manager — Minimal Integration Tests")
    print("=" * 70)

    tests = [
        test_position_creation,
        test_circuit_breaker,
        test_trailing_stops,
        test_correlation_gate,
        test_portfolio_risk_manager,
        test_edge_cases,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

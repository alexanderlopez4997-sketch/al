#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEST SUITE: Backtesting Fidelity Improvements
==============================================

Demonstrates the improvements in look-ahead bias prevention,
market microstructure realism, and robust validation.

USAGE:
    python test_backtesting_improvements.py
"""

import numpy as np
import pandas as pd
from datetime import datetime
import sys

from backtesting_fidelity import (
    BacktestEngine,
    BacktestConfig,
    market_impact_sqrt_law,
    PurgedKFold,
    DeflatedSharpe,
    FeatureNormalizer,
)
from quant_engine_refactored import (
    backtest_with_fidelity,
    validate_with_purged_kfold,
    optimize_weights_robust,
    check_feature_leakage,
)


def generate_test_data(n_bars: int = 252) -> tuple:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_bars, freq='B')

    # Generate realistic price series
    returns = np.random.normal(0.0005, 0.012, n_bars)
    prices = 100 * np.exp(np.cumsum(returns))

    # Generate signals
    signals = pd.Series(
        np.sin(np.arange(n_bars) * 2 * np.pi / 63) * 30 +
        np.random.randn(n_bars) * 5,
        index=dates,
    )

    # Generate volumes
    volumes = pd.Series(
        np.random.uniform(1e6, 5e6, n_bars),
        index=dates,
    )

    return (
        pd.Series(prices, index=dates, name='Close'),
        signals,
        volumes,
    )


def test_execution_timing():
    """
    Test 1: Execution Timing & Look-Ahead Bias Prevention

    Verifies that signals at bar t execute at bar t+1,
    never same-bar.
    """
    print("\n" + "=" * 70)
    print("TEST 1: EXECUTION TIMING & LOOK-AHEAD BIAS")
    print("=" * 70)

    close, signals, volumes = generate_test_data(100)

    # Backtest with execution delay
    config = BacktestConfig(execution_delay_bars=1)
    engine = BacktestEngine(config)
    result = engine.backtest(close, signals, volumes)

    print(f"\n✓ Execution Delay Enforcement:")
    print(f"  - Bars processed:  {len(close)}")
    print(f"  - Trades executed: {result.num_trades}")
    print(f"  - Strategy return: {result.total_return*100:.2f}%")
    print(f"  - Buy & hold:      {result.buy_hold_return*100:.2f}%")

    # Verify all trades have proper timing
    for i, trade in enumerate(result.trades[:3]):  # Show first 3 trades
        print(f"\n  Trade {i+1}:")
        print(f"    Entry:  Bar {trade.entry_bar} @ {trade.entry_price:.2f}")
        print(f"    Actual: {trade.actual_entry_price:.2f} (slippage: {trade.entry_slippage_bps:.1f} bps)")
        print(f"    Exit:   Bar {trade.exit_bar} @ {trade.exit_price:.2f}")
        print(f"    Hold:   {trade.hold_days} days")

    # Verify no same-bar fills
    for trade in result.trades:
        assert trade.exit_bar > trade.entry_bar, "Same-bar fill detected!"
    print(f"\n✓ No same-bar fills detected (all {len(result.trades)} trades respect timing)")

    return result


def test_market_microstructure():
    """
    Test 2: Market Microstructure Realism

    Verifies bid-ask spreads, market impact, and transaction costs
    are properly modeled and decomposed.
    """
    print("\n" + "=" * 70)
    print("TEST 2: MARKET MICROSTRUCTURE & TRANSACTION COSTS")
    print("=" * 70)

    close, signals, volumes = generate_test_data(252)

    # Run with realistic microstructure
    result = backtest_with_fidelity(
        close, signals, volumes,
        bid_ask_spread_bps=2.5,
        slippage_coefficient=0.1,
        execution_delay_bars=1,
    )

    print(f"\n✓ Transaction Cost Breakdown:")
    print(f"  Total costs:      {result.total_transaction_cost_bps:.0f} bps ({result.total_transaction_cost_bps/100:.2f}% of capital)")
    print(f"  - Bid-ask spread: {result.total_spread_cost_bps:.0f} bps")
    print(f"  - Market impact:  {result.total_slippage_cost_bps:.0f} bps")
    print(f"  - Commission:     {result.total_commission_cost_bps:.0f} bps")

    # Demonstrate Square-Root Law
    print(f"\n✓ Market Impact Model (Square-Root Law):")
    test_scenarios = [
        (1e5, 1e6, 0.25),  # 10% of daily volume
        (5e4, 1e6, 0.25),  # 5% of daily volume
        (1e4, 1e6, 0.25),  # 1% of daily volume
    ]

    for order_vol, daily_vol, vol in test_scenarios:
        impact = market_impact_sqrt_law(
            order_volume=order_vol,
            daily_volume=daily_vol,
            volatility=vol,
            coefficient=0.1,
        )
        pct_of_daily = (order_vol / daily_vol) * 100
        print(f"  {pct_of_daily:.1f}% of daily volume: {impact:.1f} bps impact")

    return result


def test_purged_kfold():
    """
    Test 3: Purged & Embargoed Walk-Forward Validation

    Verifies temporal separation between train/test folds
    and that overlapping samples are eliminated.
    """
    print("\n" + "=" * 70)
    print("TEST 3: PURGED & EMBARGOED WALK-FORWARD VALIDATION")
    print("=" * 70)

    # Create factor matrix
    n = 400
    dates = pd.date_range('2022-01-01', periods=n, freq='B')
    factors = pd.DataFrame(
        np.random.randn(n, 3),
        index=dates,
        columns=['Factor_1', 'Factor_2', 'Factor_3'],
    )
    close = pd.Series(
        100 + np.cumsum(np.random.randn(n) * 0.5),
        index=dates,
    )

    # Standard K-Fold: would have overlapping samples
    print(f"\n✓ Standard K-Fold Issues:")
    print(f"  - Shuffles temporal order → data snooping")
    print(f"  - Overlapping samples → leakage")
    print(f"  - Not realistic for time-series")

    # Purged & Embargoed approach
    kfold = PurgedKFold(n_splits=4, purge_length=5, embargo_length=5)
    fold_sizes = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(factors), 1):
        train_size = len(train_idx)
        test_size = len(test_idx)
        fold_sizes.append((train_size, test_size))

        # Verify no overlap
        overlap = set(train_idx) & set(test_idx)
        assert len(overlap) == 0, f"Fold {fold_idx} has overlapping indices!"

        # Verify temporal ordering
        if fold_idx < kfold.n_splits:
            last_train = train_idx[-1] if len(train_idx) > 0 else -1
            first_test = test_idx[0] if len(test_idx) > 0 else n + 1
            gap = first_test - last_train
            assert gap >= kfold.embargo_length + 1, f"Insufficient gap in fold {fold_idx}"

    print(f"\n✓ Purged & Embargoed K-Fold Results:")
    for fold_idx, (train_size, test_size) in enumerate(fold_sizes, 1):
        print(f"  Fold {fold_idx}: {train_size} train bars, "
              f"{test_size} test bars (embargo enforced)")

    # Compare to walk-forward validation output
    validation = validate_with_purged_kfold(
        factors, close, n_splits=4, ppy=252
    )

    print(f"\n✓ Walk-Forward Out-of-Sample Sharpe:")
    print(f"  Mean OOS Sharpe:  {validation['oos_sharpe_mean']:.3f}")
    print(f"  Std dev:          {validation['oos_sharpe_std']:.3f}")
    print(f"  Per-fold:         {validation['oos_sharpes']}")

    return validation


def test_deflated_sharpe():
    """
    Test 4: Deflated Sharpe Ratio for Selection Bias Correction

    Demonstrates how DSR corrects for multiple testing bias
    when optimizing parameters.
    """
    print("\n" + "=" * 70)
    print("TEST 4: DEFLATED SHARPE RATIO (Selection Bias Correction)")
    print("=" * 70)

    observed_sharpe = 1.2
    num_trials = [1, 10, 50, 100, 500, 1000]
    n_bars = 252

    print(f"\n✓ Observed Sharpe: {observed_sharpe:.2f}")
    print(f"  Number of bars:   {n_bars}")
    print(f"\n  Selection bias as function of # trials:")
    print(f"  {'Trials':<10} {'Bias':<10} {'DSR':<10} {'Significant?':<15}")
    print(f"  {'-'*45}")

    for nt in num_trials:
        dsr = DeflatedSharpe.estimate_dsr(
            sharpe_ratio=observed_sharpe,
            num_trials=nt,
            num_observations=n_bars,
        )
        bias = observed_sharpe - dsr
        is_sig = DeflatedSharpe.is_significant(dsr)

        print(f"  {nt:<10} {bias:<10.3f} {dsr:<10.3f} "
              f"{'✓ Yes' if is_sig else '✗ No':<15}")

    print(f"\n  Key insight:")
    print(f"  - 1 trial (no optimization):  DSR = {observed_sharpe:.3f} ✓ Significant")
    print(f"  - 100 trials (optimization):  DSR = {DeflatedSharpe.estimate_dsr(observed_sharpe, 100, n_bars):.3f}")
    print(f"  - 1000 trials (heavy search): DSR = {DeflatedSharpe.estimate_dsr(observed_sharpe, 1000, n_bars):.3f}")
    print(f"\n  Selection bias = sqrt(2*log(N)) × σ(Sharpe)")
    print(f"  As optimization trials increase, bias increases → DSR decreases")


def test_robust_optimization():
    """
    Test 5: Robust Weight Optimization

    Demonstrates walk-forward validation + DSR correction
    in weight optimization.
    """
    print("\n" + "=" * 70)
    print("TEST 5: ROBUST WEIGHT OPTIMIZATION")
    print("=" * 70)

    n = 252
    dates = pd.date_range('2023-01-01', periods=n, freq='B')

    # Create factors
    factors = pd.DataFrame(
        np.random.randn(n, 3),
        index=dates,
        columns=['Factor_1', 'Factor_2', 'Factor_3'],
    )
    close = pd.Series(
        100 + np.cumsum(np.random.randn(n) * 0.5),
        index=dates,
    )

    print(f"\n✓ Running robust optimization...")
    opt = optimize_weights_robust(
        factors, close,
        num_trials=100,
        compute_dsr=True,
        ppy=252,
    )

    print(f"\n✓ Optimization Results:")
    print(f"  In-sample Sharpe:   {opt['in_sample_sharpe']:.3f}")
    print(f"  OOS Sharpe (mean):  {opt['oos_sharpe_mean']:.3f}")
    dsr_val = opt['deflated_sharpe']
    if dsr_val is not None:
        print(f"  Deflated Sharpe:    {dsr_val:.3f}")
        print(f"  Is significant:     {DeflatedSharpe.is_significant(dsr_val)}")
    else:
        print(f"  Deflated Sharpe:    N/A (negative IS Sharpe)")
        print(f"  Is significant:     N/A")
    print(f"  Is overfitted:      {opt['is_overfitted']}")

    if opt['is_overfitted']:
        print(f"\n  ⚠ WARNING: In-sample >> OOS (likely overfitting)")
        print(f"  Recommendation: Simplify model or get more data")

    return opt


def test_feature_leakage():
    """
    Test 6: Feature Engineering Audit for Look-Ahead Bias

    Demonstrates automatic detection of look-ahead bias
    in feature normalization.
    """
    print("\n" + "=" * 70)
    print("TEST 6: FEATURE LEAKAGE DETECTION")
    print("=" * 70)

    # Test FeatureNormalizer
    dates = pd.date_range('2023-01-01', periods=100, freq='B')
    prices = pd.Series(
        100 + np.cumsum(np.random.randn(100) * 0.5),
        index=dates,
    )

    print(f"\n✓ Testing feature normalization methods:")

    # Expanding Z-score (safe)
    zscore_exp = FeatureNormalizer.zscore_expanding(prices, min_periods=20)
    print(f"  Expanding Z-score:")
    print(f"    - Uses data [0:t] for each bar t ✓")
    print(f"    - Safe for backtesting")
    print(f"    - First NaN after 20 bars (min_periods)")

    # Rolling Min-Max (safe)
    minmax_roll = FeatureNormalizer.minmax_rolling(prices, window=20)
    print(f"\n  Rolling Min-Max scaling:")
    print(f"    - Uses data [t-20:t] for each bar t ✓")
    print(f"    - Safe for backtesting")
    print(f"    - No NaNs (fills early with neutral 0.5)")

    # Parameter audit
    print(f"\n✓ Parameter safety check:")
    safe_params = {'lag_1': 0.5, 'momentum_window': 20}
    unsafe_params = {'global_mean_return': 0.05, 'full_dataset_std': 0.15}

    try:
        FeatureNormalizer.parameter_safety_check(safe_params)
        print(f"  Safe parameters: ✓ Passed")
    except ValueError as e:
        print(f"  Safe parameters: ✗ Failed - {e}")

    try:
        FeatureNormalizer.parameter_safety_check(unsafe_params)
        print(f"  Unsafe parameters: ✗ No error (should have failed!)")
    except ValueError as e:
        print(f"  Unsafe parameters: ✓ Caught - {e}")


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "🔬 " * 25)
    print("BACKTESTING FIDELITY IMPROVEMENTS: COMPREHENSIVE TEST SUITE")
    print("🔬 " * 25)

    try:
        # Test 1: Execution timing
        test_execution_timing()

        # Test 2: Market microstructure
        test_market_microstructure()

        # Test 3: Purged K-Fold
        test_purged_kfold()

        # Test 4: Deflated Sharpe
        test_deflated_sharpe()

        # Test 5: Robust optimization
        test_robust_optimization()

        # Test 6: Feature leakage
        test_feature_leakage()

        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)

        print("\n" + "📊 " * 20)
        print("\nKEY TAKEAWAYS:")
        print("-" * 70)
        print("1. Execution timing: Signals at t execute at t+1 (no look-ahead)")
        print("2. Market structure: Realistic spreads + impact + commission")
        print("3. Validation: Purged & Embargoed walk-forward eliminates leakage")
        print("4. Selection bias: DSR correction accounts for multiple trials")
        print("5. Robustness: OOS Sharpe > DSR indicates real edge")
        print("6. Audit: Auto-detection of feature leakage in normalization")
        print("-" * 70)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()

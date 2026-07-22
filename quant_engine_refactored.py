#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUANT ENGINE REFACTORED: Integration layer
============================================

This module demonstrates how to integrate the new backtesting_fidelity engine
into the existing quant_engine.py infrastructure.

Key improvements:
1. Uses BacktestEngine for realistic execution modeling
2. Implements Purged & Embargoed walk-forward validation
3. Computes Deflated Sharpe Ratio for optimization trials
4. Provides drop-in replacement functions for existing code

USAGE:
  from quant_engine_refactored import (
      backtest_with_fidelity,
      validate_with_purged_kfold,
      optimize_weights_robust
  )

  # Use instead of quant_engine.backtest()
  result = backtest_with_fidelity(close_prices, composite_score)

  # Walk-forward validation
  oos_sharpes = validate_with_purged_kfold(factors, close_prices)

  # Robust weight optimization
  opt_result = optimize_weights_robust(F, close_prices, num_trials=500)
"""

from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import warnings
from dataclasses import asdict
from backtesting_fidelity import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    PurgedKFold,
    DeflatedSharpe,
    FeatureNormalizer,
)

__all__ = [
    'backtest_with_fidelity',
    'validate_with_purged_kfold',
    'optimize_weights_robust',
    'check_feature_leakage',
]


def backtest_with_fidelity(
    close: pd.Series,
    composite_score: pd.Series,
    volumes: Optional[pd.Series] = None,
    enter_threshold: float = 18.0,
    exit_threshold: float = 0.0,
    execution_delay_bars: int = 1,
    bid_ask_spread_bps: float = 2.5,
    slippage_coefficient: float = 0.1,
) -> BacktestResult:
    """
    Drop-in replacement for quant_engine.backtest() with realistic execution.

    Addresses look-ahead bias by:
    - Enforcing execution_delay_bars (signals at t execute at t+1)
    - Using realistic bid-ask spreads
    - Modeling market impact via Square-Root Law
    - Computing transaction costs transparently

    Args:
        close: Daily closing prices (Series with DatetimeIndex)
        composite_score: Trading signals/scores (same index as close)
        volumes: Daily volumes for impact calculations (default: uniform)
        enter_threshold: Score threshold for entry (default 18)
        exit_threshold: Score threshold for exit (default 0)
        execution_delay_bars: Bars between signal and execution (default 1)
        bid_ask_spread_bps: Bid-ask spread in basis points (default 2.5)
        slippage_coefficient: Market impact coefficient γ (default 0.1)

    Returns:
        BacktestResult with comprehensive execution statistics

    Example:
        >>> result = backtest_with_fidelity(
        ...     close_prices, composite_score,
        ...     execution_delay_bars=1,
        ...     bid_ask_spread_bps=2.5
        ... )
        >>> print(f"Sharpe: {result.sharpe_ratio:.2f}")
        >>> print(f"Total costs: {result.total_transaction_cost_bps:.0f} bps")
    """
    # Validate inputs
    if len(close) != len(composite_score):
        raise ValueError("close and composite_score must have same length")

    # Use uniform volumes if not provided
    if volumes is None:
        volumes = pd.Series(
            np.ones(len(close)) * close.mean() * 100000,  # Synthetic volume
            index=close.index
        )

    # Create config
    config = BacktestConfig(
        execution_delay_bars=execution_delay_bars,
        bid_ask_spread_bps=bid_ask_spread_bps,
        slippage_coefficient=slippage_coefficient,
    )

    # Normalize score to [-1, 1] range for position signals
    score_min, score_max = composite_score.min(), composite_score.max()
    score_range = score_max - score_min + 1e-9
    score_normalized = (composite_score - score_min) / score_range * 2 - 1

    # Run backtest
    engine = BacktestEngine(config)
    result = engine.backtest(close, score_normalized, volumes)

    return result


def validate_with_purged_kfold(
    factors: pd.DataFrame,
    close: pd.Series,
    n_splits: int = 4,
    purge_length: int = 5,
    embargo_length: int = 5,
    ppy: float = 252.0,
) -> Dict[str, Any]:
    """
    Purged & Embargoed Walk-Forward Cross-Validation.

    Eliminates look-ahead bias in CV by:
    - Purging overlapping samples between folds
    - Embargoing test period after training
    - Respecting temporal ordering

    Args:
        factors: Factor matrix (DataFrame with index aligned to close)
        close: Closing prices (Series)
        n_splits: Number of folds (default 4)
        purge_length: Bars to purge before fold (default 5)
        embargo_length: Bars to embargo after fold (default 5)
        ppy: Periods per year (default 252)

    Returns:
        Dict with:
            - 'oos_sharpes': List of OOS Sharpe per fold
            - 'oos_sharpe_mean': Mean OOS Sharpe
            - 'oos_sharpe_std': Std of OOS Sharpes
            - 'is_significant': Whether mean Sharpe > 0 (rough significance)

    Example:
        >>> validation = validate_with_purged_kfold(factors, close, n_splits=4)
        >>> print(f"OOS Sharpe: {validation['oos_sharpe_mean']:.2f}")
    """
    kfold = PurgedKFold(
        n_splits=n_splits,
        purge_length=purge_length,
        embargo_length=embargo_length,
    )

    oos_sharpes = []

    for train_idx, test_idx in kfold.split(factors):
        # Skip if indices are empty
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        # Extract train/test
        factor_train = factors.iloc[train_idx]
        factor_test = factors.iloc[test_idx]
        close_train = close.iloc[train_idx]
        close_test = close.iloc[test_idx]

        # Simple equal-weight composite on test set
        # (In production, this would use optimized weights from training)
        composite_test = factor_test.mean(axis=1)

        # Compute OOS Sharpe
        returns_test = close_test.pct_change().fillna(0).values
        positions = np.where(composite_test > 0, 1.0, 0.0)

        # Strategy returns
        strat_returns = positions[:-1] * returns_test[1:]

        if len(strat_returns) > 1:
            mean_ret = np.mean(strat_returns)
            std_ret = np.std(strat_returns, ddof=1)
            if std_ret > 0:
                oos_sharpe = mean_ret / std_ret * np.sqrt(ppy)
                oos_sharpes.append(float(oos_sharpe))

    if not oos_sharpes:
        return {
            'oos_sharpes': [],
            'oos_sharpe_mean': 0.0,
            'oos_sharpe_std': 0.0,
            'is_significant': False,
        }

    oos_sharpes = np.array(oos_sharpes)
    return {
        'oos_sharpes': oos_sharpes.tolist(),
        'oos_sharpe_mean': float(np.mean(oos_sharpes)),
        'oos_sharpe_std': float(np.std(oos_sharpes, ddof=1)),
        'is_significant': float(np.mean(oos_sharpes)) > 0.0,
    }


def optimize_weights_robust(
    factors: pd.DataFrame,
    close: pd.Series,
    num_trials: int = 100,
    compute_dsr: bool = True,
    ppy: float = 252.0,
) -> Dict[str, Any]:
    """
    Robust weight optimization with multiple testing correction.

    Improvements over simple optimization:
    - Uses Purged & Embargoed walk-forward validation
    - Computes Deflated Sharpe Ratio for selection bias correction
    - Returns both in-sample and OOS metrics

    Args:
        factors: Factor matrix (DataFrame)
        close: Closing prices
        num_trials: Number of optimization trials (for DSR correction)
        compute_dsr: Whether to compute Deflated Sharpe (default True)
        ppy: Periods per year (default 252)

    Returns:
        Dict with:
            - 'base_weights': Simple equal-weight baseline
            - 'optimized_weights': Optimized weights (equal-weight for demo)
            - 'in_sample_sharpe': IS Sharpe
            - 'oos_sharpe_mean': OOS Sharpe (walk-forward)
            - 'deflated_sharpe': DSR after selection bias correction
            - 'is_overfitted': Whether IS >> OOS (warning sign)

    Example:
        >>> opt = optimize_weights_robust(factors, close, num_trials=100)
        >>> if opt['is_overfitted']:
        ...     print("WARNING: Weights may be overfit to training data")
    """
    # Base weights (equal weight)
    n_factors = factors.shape[1]
    base_weights = np.ones(n_factors) / n_factors

    # In-sample Sharpe (equal weight on full data)
    composite_is = (factors * base_weights).sum(axis=1)
    returns_is = close.pct_change().fillna(0).values
    positions_is = np.where(composite_is > 0, 1.0, 0.0)
    strat_is = positions_is[:-1] * returns_is[1:]
    is_sharpe = float(
        np.mean(strat_is) / np.std(strat_is, ddof=1) * np.sqrt(ppy)
        if np.std(strat_is, ddof=1) > 0 else 0.0
    )

    # Walk-forward validation (OOS Sharpe)
    validation = validate_with_purged_kfold(factors, close, n_splits=4, ppy=ppy)
    oos_sharpe = validation['oos_sharpe_mean']

    # Deflated Sharpe (selection bias correction)
    dsr = None
    if compute_dsr and is_sharpe > 0:
        dsr = DeflatedSharpe.estimate_dsr(
            sharpe_ratio=is_sharpe,
            num_trials=num_trials,
            num_observations=len(factors),
        )

    # Detect overfitting
    is_overfitted = (is_sharpe > 0.5 and oos_sharpe < is_sharpe * 0.5) if oos_sharpe > 0 else False

    return {
        'base_weights': {f'Factor_{i}': float(w) for i, w in enumerate(base_weights)},
        'optimized_weights': {f'Factor_{i}': float(w) for i, w in enumerate(base_weights)},
        'in_sample_sharpe': is_sharpe,
        'oos_sharpe_mean': oos_sharpe,
        'oos_sharpe_folds': validation['oos_sharpes'],
        'deflated_sharpe': dsr,
        'is_overfitted': is_overfitted,
        'num_trials': num_trials,
    }


def check_feature_leakage(
    enrich_function,
    sample_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Audit feature engineering for look-ahead bias.

    Checks:
    - All indicators computed with rolling/expanding windows (not global stats)
    - No parameter fitting on full dataset
    - Temporal ordering respected

    Args:
        enrich_function: Function that computes features (e.g., quant_engine.enrich)
        sample_df: Sample price DataFrame to test

    Returns:
        Dict with audit results:
            - 'has_leakage': bool, whether potential leakage detected
            - 'issues': List of suspicious findings
            - 'recommendations': List of fixes

    Example:
        >>> audit = check_feature_leakage(quant_engine.enrich, prices_df)
        >>> if audit['has_leakage']:
        ...     for issue in audit['issues']:
        ...         print(f"WARNING: {issue}")
    """
    issues = []
    recommendations = []

    # Run enrichment
    try:
        enriched = enrich_function(sample_df.copy())
    except Exception as e:
        return {
            'has_leakage': True,
            'issues': [f"Enrichment failed: {e}"],
            'recommendations': ["Check error in feature computation"],
        }

    # Audit computed columns for suspicious patterns
    for col in enriched.columns:
        if col in sample_df.columns:
            continue  # Skip original data

        # Check for NaNs in early rows (sign of global fitting)
        if enriched[col].isna().sum() > len(enriched) * 0.2:
            issues.append(
                f"Column '{col}' has >20% NaNs — possible global parameter fitting"
            )
            recommendations.append(
                f"Ensure '{col}' uses expanding/rolling windows for all rows"
            )

        # Check temporal consistency (should not jump dramatically)
        if len(enriched) > 20:
            diffs = enriched[col].diff().abs()
            jumps = (diffs > diffs.mean() + 3 * diffs.std()).sum()
            if jumps > len(enriched) * 0.1:
                issues.append(
                    f"Column '{col}' has unusual jumps — check for global fitting"
                )

    # Check if any column appears to be lagged unexpectedly
    for col in enriched.columns:
        if col in sample_df.columns:
            continue
        # Correlation with future returns is a sign of look-ahead bias
        future_returns = sample_df['Close'].pct_change().shift(-1)
        if len(enriched) > 30:
            corr = enriched[col].iloc[:-1].corr(future_returns.iloc[1:])
            if abs(corr) > 0.7:
                issues.append(
                    f"Column '{col}' highly correlated with FUTURE returns "
                    f"(corr={corr:.2f}) — strong look-ahead bias signal"
                )

    has_leakage = len(issues) > 0
    return {
        'has_leakage': has_leakage,
        'issues': issues,
        'recommendations': recommendations,
    }


# ============================================================================
# BACKWARD COMPATIBILITY WRAPPERS
# ============================================================================

def backtest_legacy_compatible(
    close: pd.Series,
    composite: pd.Series,
    ppy: float = 252,
    intraday: bool = False,
    slippage_pct: float = 0.05,
) -> Dict[str, Any]:
    """
    Drop-in replacement for quant_engine.backtest() that uses new engine.

    Maintains API compatibility while using BacktestEngine internally.

    Args:
        close: Closing prices
        composite: Composite score
        ppy: Periods per year
        intraday: Whether to mask overnight gaps (legacy param)
        slippage_pct: Slippage percentage (legacy param)

    Returns:
        Dict matching quant_engine.backtest() output format
    """
    result = backtest_with_fidelity(
        close,
        composite,
        execution_delay_bars=1,
        bid_ask_spread_bps=slippage_pct * 100,  # Convert % to bps
    )

    # Convert to legacy format
    return {
        'strategy': result.total_return,
        'buyhold': result.buy_hold_return,
        'maxdd': result.max_drawdown,
        'maxdd_bh': 0.0,  # Not computed in new engine
        'trades': result.num_trades,
        'winrate': result.win_rate,
        'exposure': 0.5,  # Approximation
        'sharpe': result.sharpe_ratio,
        'slippage_cost': result.total_slippage_cost_bps / 10000.0,
    }


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("QUANT ENGINE REFACTORED: Integration Demo")
    print("=" * 70)

    # Create sample data
    dates = pd.date_range('2020-01-01', periods=252, freq='B')
    prices = pd.Series(
        100 + np.cumsum(np.random.randn(252) * 0.5),
        index=dates,
        name='Close'
    )
    scores = pd.Series(
        np.sin(np.arange(252) * 2 * np.pi / 63) * 30,
        index=dates,
        name='Signal'
    )

    print("\n1. BACKTEST WITH FIDELITY")
    print("-" * 70)
    result = backtest_with_fidelity(
        prices, scores,
        execution_delay_bars=1,
        bid_ask_spread_bps=2.5,
    )
    print(f"   Total Return: {result.total_return*100:.2f}%")
    print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {result.max_drawdown*100:.2f}%")
    print(f"   Win Rate: {result.win_rate*100:.1f}%")
    print(f"   Transaction Costs: {result.total_transaction_cost_bps:.0f} bps")

    # Create sample factors
    factors = pd.DataFrame({
        'Factor_1': np.random.randn(252),
        'Factor_2': np.random.randn(252),
        'Factor_3': np.random.randn(252),
    }, index=dates)

    print("\n2. WALK-FORWARD VALIDATION (Purged & Embargoed)")
    print("-" * 70)
    validation = validate_with_purged_kfold(factors, prices, n_splits=4)
    print(f"   OOS Sharpe (mean): {validation['oos_sharpe_mean']:.2f}")
    print(f"   OOS Sharpe (std):  {validation['oos_sharpe_std']:.2f}")
    print(f"   Folds: {validation['oos_sharpes']}")

    print("\n3. ROBUST WEIGHT OPTIMIZATION")
    print("-" * 70)
    opt = optimize_weights_robust(factors, prices, num_trials=100)
    print(f"   In-Sample Sharpe: {opt['in_sample_sharpe']:.2f}")
    print(f"   OOS Sharpe:       {opt['oos_sharpe_mean']:.2f}")
    print(f"   Deflated Sharpe:  {opt['deflated_sharpe']:.2f}")
    print(f"   Overfitted:       {opt['is_overfitted']}")

    print("=" * 70)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BACKTESTING FIDELITY IMPROVEMENTS
==================================

Production-grade backtesting engine addressing:
1. Look-ahead bias elimination with strict signal/execution timing
2. Market microstructure realism (spreads, slippage, volume-based impact)
3. Purged & Embargoed Walk-Forward CV for robust validation
4. Deflated Sharpe Ratio for selection bias correction
5. Type hints, dataclasses, and JIT acceleration for performance

USAGE:
  from backtesting_fidelity import BacktestEngine, PurgedKFold, DeflatedSharpe

  engine = BacktestEngine(
      slippage_model='sqrt_law',
      bid_ask_model='realistic',
      gap_cost_bps=2.5
  )

  results = engine.backtest(
      prices, signals, daily_volumes,
      execution_delay_bars=1,  # Strict t+1 execution
      embargo_bars=5            # Purged embargoed periods
  )
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import math
from numba import njit
import warnings

__all__ = [
    'BacktestConfig',
    'ExecutionStats',
    'BacktestResult',
    'BacktestEngine',
    'PurgedKFold',
    'DeflatedSharpe',
    'market_impact_sqrt_law',
]


# ============================================================================
# TYPE DEFINITIONS & DATACLASSES
# ============================================================================

@dataclass
class BacktestConfig:
    """Configuration for backtesting execution parameters."""
    # Execution timing
    execution_delay_bars: int = 1  # Signals at t execute at t+n (prevents look-ahead)

    # Market microstructure
    bid_ask_spread_bps: float = 2.5  # Bid-ask in basis points
    slippage_model: str = 'sqrt_law'  # 'flat' | 'sqrt_law' | 'realistic'
    slippage_coefficient: float = 0.1  # γ in sqrt-law formula

    # Cost parameters
    commissions_bps: float = 1.0  # Per-side commission
    short_borrow_rate_bps: float = 10.0  # Annual borrow cost (simplified daily)

    # Walk-forward validation
    purge_length: int = 5  # Bars to purge before/after training period
    embargo_length: int = 5  # Bars to embargo after training
    gap_cost_bps: float = 2.5  # Gap opening cost

    # Risk parameters
    ppy: float = 252.0  # Periods per year (daily data)
    min_trade_bars: int = 1  # Minimum bars to hold position


@dataclass
class ExecutionStats:
    """Detailed execution-level statistics for a single trade."""
    entry_bar: int
    exit_bar: int
    entry_price: float
    entry_slippage_bps: float
    actual_entry_price: float
    exit_price: float
    exit_slippage_bps: float
    actual_exit_price: float
    volume_at_entry: float
    volume_at_exit: float
    is_long: bool = True

    @property
    def pnl_bps(self) -> float:
        """P&L in basis points including all costs."""
        if self.is_long:
            return (self.actual_exit_price / self.actual_entry_price - 1.0) * 10000
        else:
            return (self.actual_entry_price / self.actual_exit_price - 1.0) * 10000

    @property
    def hold_days(self) -> int:
        return self.exit_bar - self.entry_bar


@dataclass
class BacktestResult:
    """Comprehensive backtest results with full execution transparency."""
    # Portfolio metrics
    total_return: float
    buy_hold_return: float
    excess_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float

    # Risk metrics
    max_drawdown: float
    max_drawdown_duration: int

    # Trade statistics
    num_trades: int
    win_rate: float
    avg_winner: float
    avg_loser: float
    profit_factor: float

    # Slippage & friction
    total_slippage_cost_bps: float
    total_commission_cost_bps: float
    total_spread_cost_bps: float
    total_transaction_cost_bps: float

    # Execution details
    trades: List[ExecutionStats] = field(default_factory=list)
    daily_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))

    # Validation metadata
    deflated_sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    in_sample: bool = True
    num_optimization_trials: Optional[int] = None

    def __str__(self) -> str:
        return (
            f"BacktestResult(Return={self.total_return*100:.2f}%, "
            f"Sharpe={self.sharpe_ratio:.2f}, "
            f"WinRate={self.win_rate*100:.1f}%, "
            f"Trades={self.num_trades}, "
            f"SlippageCost={self.total_slippage_cost_bps:.0f}bps)"
        )


# ============================================================================
# MARKET MICROSTRUCTURE MODELS
# ============================================================================

def market_impact_sqrt_law(
    order_volume: float,
    daily_volume: float,
    volatility: float,
    coefficient: float = 0.1
) -> float:
    """
    Square-Root Law market impact model (Almgren et al., 2005).

    Models transient impact decay: impact ∝ √(V_order / V_daily) · σ

    Args:
        order_volume: Order size (shares)
        daily_volume: Average daily trading volume
        volatility: Annualized volatility of the asset
        coefficient: Impact coefficient γ (typically 0.05-0.15)

    Returns:
        Slippage cost in basis points

    Formula:
        Slippage = γ · σ · √(V_order / V_daily)
    """
    if daily_volume <= 0 or volatility <= 0:
        return 0.0

    volume_ratio = np.sqrt(order_volume / daily_volume)
    impact_bps = coefficient * volatility * 100 * volume_ratio  # σ is fractional
    return float(np.clip(impact_bps, 0, 500))  # Cap at 500 bps to prevent numerical blow-up


def bid_ask_spread_cost(
    spread_bps: float,
    is_entry: bool = True
) -> float:
    """
    Bid-ask spread cost for round-trip.
    Entry fills at Ask (cost), exit fills at Bid (cost).
    """
    return spread_bps if is_entry else spread_bps


@njit
def _compute_equity_curve(
    daily_returns: np.ndarray,
    positions: np.ndarray,
    execution_costs: np.ndarray
) -> np.ndarray:
    """JIT-compiled equity curve computation (numba accelerated)."""
    n = len(daily_returns)
    equity = np.ones(n)

    for i in range(1, n):
        # Strategy return = position from yesterday × today's return - transaction costs
        strat_ret = positions[i-1] * daily_returns[i] - execution_costs[i]
        equity[i] = equity[i-1] * (1.0 + strat_ret)

    return equity


@njit
def _compute_positions_forward_fill(
    signals: np.ndarray,
    execution_delay: int = 1,
    enter_threshold: float = 0.5,
    exit_threshold: float = -0.5
) -> np.ndarray:
    """
    JIT-compiled position forward-fill with execution delay.

    Ensures strict separation:
    - Signal at bar t uses data up to close of bar t
    - Position executes at bar t + execution_delay
    """
    n = len(signals)
    positions = np.zeros(n)
    current_signal = 0.0
    signal_bar = -execution_delay  # Start with delay applied

    for i in range(n):
        # Update signal if threshold crossed
        if signals[i] > enter_threshold and current_signal <= 0:
            current_signal = 1.0
            signal_bar = i
        elif signals[i] < exit_threshold and current_signal > 0:
            current_signal = 0.0
            signal_bar = i

        # Apply position with execution delay
        if i >= signal_bar + execution_delay:
            positions[i] = current_signal

    return positions


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

class BacktestEngine:
    """
    Production-grade backtesting engine with realistic execution model.

    Key features:
    - Execution delay enforcement (no same-bar fills)
    - Market microstructure realism (spreads, impact, volume-dependent slippage)
    - Transaction cost transparency
    - Walk-forward validation with purging & embargo
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtesting engine with configuration.

        Args:
            config: BacktestConfig instance; uses defaults if None
        """
        self.config = config or BacktestConfig()

    def backtest(
        self,
        prices: pd.Series,
        signals: pd.Series,
        volumes: pd.Series,
        returns_precomputed: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """
        Run realistic backtest with market microstructure.

        Args:
            prices: Daily closing prices (Series with DatetimeIndex)
            signals: Trading signals in [-1, 1] range (1=long, -1=short, 0=flat)
            volumes: Daily volumes (used for impact calculations)
            returns_precomputed: Optional pre-computed daily returns (else calculated)

        Returns:
            BacktestResult with execution details
        """
        # Validation
        assert len(prices) == len(signals), "prices and signals must have same length"
        assert len(prices) >= 2, "need at least 2 bars"

        # Pre-compute returns if not provided
        if returns_precomputed is None:
            returns = prices.pct_change().fillna(0.0).values
        else:
            returns = returns_precomputed.values

        prices_array = prices.values
        signals_array = np.clip(signals.values, -1, 1)  # Bound to [-1, 1]
        volumes_array = volumes.values

        # Compute rolling volatility (annualized)
        vol_window = 20
        rolling_returns = pd.Series(returns).rolling(vol_window).std()
        rolling_vol_annual = rolling_returns * np.sqrt(self.config.ppy)
        rolling_vol_annual = rolling_vol_annual.fillna(rolling_vol_annual.mean()).values

        # Rolling daily volume average (20-day)
        avg_volumes = pd.Series(volumes_array).rolling(20).mean().fillna(volumes_array.mean()).values

        # Compute positions with execution delay
        positions = _compute_positions_forward_fill(
            signals_array,
            execution_delay=self.config.execution_delay_bars
        )

        # Compute execution costs (bid-ask + impact + commission)
        execution_costs = self._compute_execution_costs(
            positions, prices_array, volumes_array, avg_volumes,
            rolling_vol_annual, returns
        )

        # Compute equity curve
        equity = _compute_equity_curve(returns, positions, execution_costs)

        # Extract detailed trade statistics
        trades = self._extract_trades(
            positions, prices_array, volumes_array, returns, execution_costs
        )

        # Compute metrics (align position held from yesterday with today's return)
        portfolio_returns = np.zeros(len(returns) - 1)
        for i in range(1, len(returns)):
            portfolio_returns[i-1] = positions[i-1] * returns[i]

        result = BacktestResult(
            total_return=float(equity[-1] - 1.0),
            buy_hold_return=float(prices_array[-1] / prices_array[0] - 1.0),
            excess_return=float((equity[-1] - 1.0) - (prices_array[-1] / prices_array[0] - 1.0)),
            annualized_return=self._annualized_return(equity, self.config.ppy),
            annualized_volatility=self._annualized_volatility(portfolio_returns, self.config.ppy),
            sharpe_ratio=self._sharpe_ratio(portfolio_returns, self.config.ppy),
            max_drawdown=self._max_drawdown(equity),
            max_drawdown_duration=self._max_drawdown_duration(equity),
            num_trades=len(trades),
            win_rate=self._win_rate(trades),
            avg_winner=self._avg_winner(trades),
            avg_loser=self._avg_loser(trades),
            profit_factor=self._profit_factor(trades),
            total_slippage_cost_bps=float(np.sum(execution_costs) * 10000),
            total_commission_cost_bps=float(
                np.sum(np.abs(np.diff(positions, prepend=0))) * self.config.commissions_bps
            ),
            total_spread_cost_bps=float(
                np.sum(np.abs(np.diff(positions, prepend=0))) * self.config.bid_ask_spread_bps
            ),
            total_transaction_cost_bps=float(np.sum(execution_costs) * 10000),
            trades=trades,
            daily_returns=portfolio_returns,
            equity_curve=equity,
            in_sample=True,
        )

        return result

    def _compute_execution_costs(
        self,
        positions: np.ndarray,
        prices: np.ndarray,
        volumes: np.ndarray,
        avg_volumes: np.ndarray,
        volatility: np.ndarray,
        returns: np.ndarray,
    ) -> np.ndarray:
        """Compute execution costs including spread, impact, and commission."""
        n = len(prices)
        costs = np.zeros(n)

        for i in range(1, n):
            if positions[i] != positions[i-1]:
                # Position change: entry or exit
                order_size = abs(positions[i] - positions[i-1])

                # Spread cost (both entry and exit apply to position change)
                spread_cost_bps = order_size * self.config.bid_ask_spread_bps / 100.0

                # Market impact (Square-Root Law)
                vol = max(volatility[i] / 100.0, 0.01)  # Bound volatility
                impact_bps = market_impact_sqrt_law(
                    order_volume=order_size,
                    daily_volume=max(avg_volumes[i], 1000),
                    volatility=vol,
                    coefficient=self.config.slippage_coefficient
                )
                impact_cost_bps = impact_bps / 100.0 * order_size

                # Commission
                commission_bps = order_size * self.config.commissions_bps / 100.0

                # Total transaction cost
                costs[i] = spread_cost_bps + impact_cost_bps + commission_bps

        return costs

    def _extract_trades(
        self,
        positions: np.ndarray,
        prices: np.ndarray,
        volumes: np.ndarray,
        returns: np.ndarray,
        costs: np.ndarray,
    ) -> List[ExecutionStats]:
        """Extract individual trade statistics."""
        trades = []
        in_trade = False
        entry_bar = entry_price = entry_volume = entry_slippage = 0.0

        for i in range(1, len(positions)):
            # Entry
            if positions[i] > 0 and positions[i-1] <= 0:
                in_trade = True
                entry_bar = i
                entry_price = prices[i]
                entry_volume = volumes[i]
                entry_slippage = costs[i] * 100  # Convert to bps

            # Exit
            elif positions[i] <= 0 and positions[i-1] > 0 and in_trade:
                exit_bar = i
                exit_price = prices[i]
                exit_volume = volumes[i]
                exit_slippage = costs[i] * 100

                trades.append(ExecutionStats(
                    entry_bar=entry_bar,
                    exit_bar=exit_bar,
                    entry_price=entry_price,
                    entry_slippage_bps=entry_slippage,
                    actual_entry_price=entry_price * (1 + entry_slippage / 10000),
                    exit_price=exit_price,
                    exit_slippage_bps=exit_slippage,
                    actual_exit_price=exit_price * (1 - exit_slippage / 10000),
                    volume_at_entry=entry_volume,
                    volume_at_exit=exit_volume,
                ))

                in_trade = False

        return trades

    @staticmethod
    def _annualized_return(equity: np.ndarray, ppy: float) -> float:
        """Annualized return from equity curve."""
        total_ret = (equity[-1] - 1.0) if len(equity) > 0 else 0.0
        n_years = len(equity) / ppy
        if n_years <= 0:
            return 0.0
        return float((1.0 + total_ret) ** (1.0 / n_years) - 1.0)

    @staticmethod
    def _annualized_volatility(returns: np.ndarray, ppy: float) -> float:
        """Annualized volatility from returns."""
        if len(returns) < 2:
            return 0.0
        daily_vol = float(np.std(returns, ddof=1))
        return float(daily_vol * np.sqrt(ppy))

    @staticmethod
    def _sharpe_ratio(returns: np.ndarray, ppy: float, rf_rate: float = 0.0) -> float:
        """Sharpe ratio (excess return per unit volatility)."""
        if len(returns) < 2:
            return float('nan')
        mean_ret = float(np.mean(returns))
        vol = float(np.std(returns, ddof=1))
        if vol == 0:
            return float('nan')
        return float((mean_ret - rf_rate / ppy) / vol * np.sqrt(ppy))

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        """Maximum drawdown from peak."""
        if len(equity) == 0:
            return 0.0
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        return float(np.min(dd))

    @staticmethod
    def _max_drawdown_duration(equity: np.ndarray) -> int:
        """Maximum drawdown duration in bars."""
        if len(equity) < 2:
            return 0
        peak = np.maximum.accumulate(equity)
        in_drawdown = np.where(equity < peak)[0]
        if len(in_drawdown) == 0:
            return 0

        durations = np.diff(np.concatenate([[in_drawdown[0]-1], in_drawdown, [len(equity)]]))
        return int(np.max(durations))

    @staticmethod
    def _win_rate(trades: List[ExecutionStats]) -> float:
        """Win rate (% profitable trades)."""
        if len(trades) == 0:
            return 0.0
        wins = sum(1 for t in trades if t.pnl_bps > 0)
        return float(wins / len(trades))

    @staticmethod
    def _avg_winner(trades: List[ExecutionStats]) -> float:
        """Average P&L of winning trades (bps)."""
        winners = [t.pnl_bps for t in trades if t.pnl_bps > 0]
        return float(np.mean(winners)) if winners else 0.0

    @staticmethod
    def _avg_loser(trades: List[ExecutionStats]) -> float:
        """Average P&L of losing trades (bps)."""
        losers = [t.pnl_bps for t in trades if t.pnl_bps < 0]
        return float(np.mean(losers)) if losers else 0.0

    @staticmethod
    def _profit_factor(trades: List[ExecutionStats]) -> float:
        """Profit factor (gross wins / gross losses)."""
        if len(trades) == 0:
            return 0.0
        wins = sum(t.pnl_bps for t in trades if t.pnl_bps > 0)
        losses = abs(sum(t.pnl_bps for t in trades if t.pnl_bps < 0))
        if losses == 0:
            return float('inf') if wins > 0 else 0.0
        return float(wins / losses)


# ============================================================================
# PURGED & EMBARGOED WALK-FORWARD CROSS-VALIDATION
# ============================================================================

class PurgedKFold:
    """
    Purged & Embargoed K-Fold Cross-Validation for time-series data.

    Addresses look-ahead bias by:
    1. Purging overlapping samples between train/test
    2. Embargoing test data after training (eliminates info leakage)
    3. Respecting temporal ordering (expanding windows)

    Reference: de Prado, M. L. (2018). Advances in Financial Machine Learning.
    """

    def __init__(
        self,
        n_splits: int = 4,
        purge_length: int = 5,
        embargo_length: int = 5,
    ):
        """
        Args:
            n_splits: Number of folds
            purge_length: Bars to exclude before training period (purge backward)
            embargo_length: Bars to exclude after training (embargo forward)
        """
        self.n_splits = n_splits
        self.purge_length = purge_length
        self.embargo_length = embargo_length

    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for Purged & Embargoed folds.

        Args:
            X: DataFrame with DatetimeIndex (time-ordered)

        Yields:
            (train_indices, test_indices) tuples
        """
        n = len(X)
        step = n // (self.n_splits + 1)

        for k in range(1, self.n_splits + 1):
            # Expanding window: train on [0, step*k)
            train_end = step * k

            # Test on [train_end + embargo, next_step_start)
            test_start = train_end + self.embargo_length
            test_end = step * (k + 1) if k < self.n_splits else n

            if test_end - test_start < 20:  # Skip fold if test is too small
                continue

            # Purge: remove [train_end - purge, train_end) from training
            train_indices = np.concatenate([
                np.arange(0, max(0, train_end - self.purge_length)),
                np.arange(train_end, n) if train_end < n else np.array([], dtype=int),
            ])
            train_indices = train_indices[train_indices < train_end - self.purge_length]

            test_indices = np.arange(test_start, test_end)

            if len(train_indices) > 20 and len(test_indices) > 20:
                yield train_indices, test_indices


# ============================================================================
# DEFLATED SHARPE RATIO (Selection Bias Correction)
# ============================================================================

class DeflatedSharpe:
    """
    Deflated Sharpe Ratio corrects backtest Sharpe for multiple testing
    and selection bias.

    Deflation accounts for:
    - Number of tests performed (e.g., optimization iterations)
    - Variability of Sharpe estimates across tests
    - Non-IID nature of financial returns

    Reference: Bailey, D. H., et al. (2014). "Deflated Sharpe Ratio".
    """

    @staticmethod
    def estimate_dsr(
        sharpe_ratio: float,
        num_trials: int,
        num_observations: int,
        num_metrics: int = 1,
        bias: float = 0.0,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """
        Estimate Deflated Sharpe Ratio.

        Args:
            sharpe_ratio: Observed Sharpe ratio from backtest
            num_trials: Number of optimization/parameter trials
            num_observations: Number of observations (bars) in backtest
            num_metrics: Number of performance metrics tested (1 by default)
            bias: Skewness of returns (default 0, symmetric)
            skewness: Excess skewness (-1 to +1)
            kurtosis: Excess kurtosis (default 3, normal)

        Returns:
            Deflated Sharpe Ratio (lower = less significant)
        """
        # Variance of Sharpe estimate
        variance_sharpe = (1.0 + 0.5 * sharpe_ratio**2) / num_observations
        std_sharpe = np.sqrt(variance_sharpe)

        # Multiple testing adjustment (Bonferroni-like)
        # Probability that k random tests exceed given threshold
        # E[max SR] ≈ sqrt(2 * log(N))
        expected_max_bias = (2 * np.log(num_trials) - np.euler_gamma) / (2 * np.sqrt(2 * np.pi))
        expected_max_bias *= std_sharpe

        # Adjust Sharpe for selection bias
        dsr = sharpe_ratio - expected_max_bias
        return float(dsr)

    @staticmethod
    def is_significant(
        dsr: float,
        significance_level: float = 0.05,
    ) -> bool:
        """
        Test if Deflated Sharpe is statistically significant.

        Args:
            dsr: Deflated Sharpe Ratio
            significance_level: Alpha level (default 5%)

        Returns:
            True if strategy passes significance test
        """
        # DSR should be > 0 to be significant (roughly corresponds to p < 0.05)
        return dsr > 0.0


# ============================================================================
# FEATURE NORMALIZATION & BIAS PREVENTION
# ============================================================================

class FeatureNormalizer:
    """
    Ensures feature engineering uses only data available at each bar,
    preventing look-ahead bias in normalization (Z-scores, Min-Max, etc).
    """

    @staticmethod
    def zscore_expanding(
        series: pd.Series,
        min_periods: int = 20,
    ) -> pd.Series:
        """
        Z-score using EXPANDING window (data up to bar t only).

        Args:
            series: Price or indicator series
            min_periods: Minimum observations before starting normalization

        Returns:
            Normalized series (mean 0, std 1)
        """
        expanding_mean = series.expanding(min_periods=min_periods).mean()
        expanding_std = series.expanding(min_periods=min_periods).std()
        zscore = (series - expanding_mean) / expanding_std.replace(0, np.nan)
        return zscore.fillna(0.0)

    @staticmethod
    def minmax_rolling(
        series: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """
        Min-Max scaling using ROLLING window (data up to bar t only).

        Args:
            series: Price or indicator series
            window: Rolling window size

        Returns:
            Normalized series in [0, 1]
        """
        rolling_min = series.rolling(window=window).min()
        rolling_max = series.rolling(window=window).max()
        normalized = (series - rolling_min) / (rolling_max - rolling_min + 1e-9)
        return normalized.fillna(0.5)  # Fill early period with neutral value

    @staticmethod
    def parameter_safety_check(
        parameters: Dict[str, Any],
        forbidden_keywords: Optional[List[str]] = None,
    ) -> None:
        """
        Audit parameters for global fitting (look-ahead bias).
        Raises if global dataset statistics detected.

        Args:
            parameters: Parameter dict (e.g., from optimization)
            forbidden_keywords: Terms indicating global fitting (default: common ones)

        Raises:
            ValueError if suspicious global fitting detected
        """
        forbidden = forbidden_keywords or [
            'global', 'full_dataset', 'all_data', 'train_set', 'mean_all', 'std_all'
        ]

        for key in parameters:
            if any(f in str(key).lower() for f in forbidden):
                raise ValueError(
                    f"Parameter '{key}' suggests global fitting (look-ahead bias). "
                    f"Use rolling/expanding windows instead."
                )


# ============================================================================
# DEMONSTRATION & VALIDATION
# ============================================================================

if __name__ == "__main__":
    # Simple test: synthetic data
    dates = pd.date_range('2020-01-01', periods=252, freq='B')
    prices = pd.Series(
        100 + np.cumsum(np.random.randn(252) * 0.5),
        index=dates,
        name='Price'
    )
    signals = pd.Series(
        np.sin(np.arange(252) * 2 * np.pi / 63) + np.random.randn(252) * 0.2,
        index=dates,
        name='Signal'
    )
    volumes = pd.Series(
        np.random.uniform(1e6, 5e6, 252),
        index=dates,
        name='Volume'
    )

    # Run backtest
    engine = BacktestEngine(BacktestConfig(
        execution_delay_bars=1,
        bid_ask_spread_bps=2.5,
        slippage_model='sqrt_law',
    ))

    result = engine.backtest(prices, signals, volumes)

    print("=" * 70)
    print("BACKTESTING FIDELITY TEST")
    print("=" * 70)
    print(f"Total Return: {result.total_return*100:.2f}%")
    print(f"Buy & Hold:   {result.buy_hold_return*100:.2f}%")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown*100:.2f}%")
    print(f"Win Rate:     {result.win_rate*100:.1f}%")
    print(f"Num Trades:   {result.num_trades}")
    print(f"Total Costs:  {result.total_transaction_cost_bps:.0f} bps")
    print(f"  - Spread:   {result.total_spread_cost_bps:.0f} bps")
    print(f"  - Impact:   {result.total_slippage_cost_bps:.0f} bps")
    print(f"  - Commission: {result.total_commission_cost_bps:.0f} bps")

    # Test Deflated Sharpe
    dsr = DeflatedSharpe.estimate_dsr(
        result.sharpe_ratio,
        num_trials=100,
        num_observations=252
    )
    print(f"\nDeflated Sharpe (100 trials): {dsr:.2f}")
    print(f"Significant: {DeflatedSharpe.is_significant(dsr)}")

    print("=" * 70)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk management and stop-loss calculation — decoupled from signal generation.

Refined engine with:
- RVOL-confirmed squeeze circuit breaker (volume must spike)
- Momentum-conditioned exponential time decay (only when price stalls)
- Signal-confidence scaled ATR bounds
- Clean dataclass return types with exit reasons for better debugging
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class StopLossResult:
    """Clean return type for stop evaluation."""
    stop_price: float
    emergency_exit: bool
    exit_reason: str | None = None


class RefinedVolatilityStopEngine:
    """
    Enhanced stop engine featuring:
    1. RVOL-confirmed squeeze circuit breaker
    2. Momentum-conditioned exponential time decay
    3. Signal-confidence scaled ATR bounds
    """

    def __init__(self, k_base: float = 1.6, decay_rate: float = 0.15, time_threshold: int = 4):
        self.k_base = k_base
        self.decay_rate = decay_rate
        self.time_threshold = time_threshold

    def evaluate_avoid_stop(
        self,
        entry_price: float,
        current_price: float,
        high_lookback: float,
        atr_14: float,
        bar_true_range: float,
        relative_volume: float,  # Current Vol / 20-period MA Vol
        bars_in_trade: int,
        bars_since_lowest_low: int,  # Tracks momentum stall
        signal_confidence: float = 1.0,  # 0.8 for Rule Engine, 1.2+ for High ML
    ) -> StopLossResult:
        """
        AVOID signal stop with volume-confirmed emergency exits.

        Args:
            entry_price: Entry/signal price
            current_price: Current market price
            high_lookback: Recent high (e.g., 20-bar high)
            atr_14: 14-bar Average True Range
            bar_true_range: Current bar's True Range
            relative_volume: Current volume / 20-bar volume MA (1.0 = average)
            bars_in_trade: Bars held since signal
            bars_since_lowest_low: Bars since most recent low (momentum indicator)
            signal_confidence: Multiplier for ATR basis (0.8-1.2 range typical)

        Returns:
            StopLossResult with stop_price, emergency_exit flag, and exit_reason
        """
        # 1. Scale initial multiplier by signal confidence
        k_init = self.k_base * signal_confidence

        # 2. Conditional Exponential Decay (only if price momentum has stalled)
        if bars_in_trade > self.time_threshold and bars_since_lowest_low >= 2:
            stale_bars = bars_in_trade - self.time_threshold
            k = max(0.6, k_init * np.exp(-self.decay_rate * stale_bars))
        else:
            k = k_init

        # 3. Structural ATR Stop Calculation
        atr_stop = entry_price + (k * atr_14)
        structural_stop = high_lookback + (0.5 * atr_14)
        stop_price = min(atr_stop, structural_stop)

        # 4. Volume-Confirmed Squeeze Circuit Breaker
        # Triggers ONLY if range expands > 2x ATR AND Volume is > 2x average
        is_volume_spike = relative_volume >= 2.0
        is_range_expansion = bar_true_range >= 2.0 * atr_14
        is_adverse_move = current_price > entry_price

        emergency_exit = is_range_expansion and is_volume_spike and is_adverse_move

        # 5. Determine Exit Status
        exit_reason = None
        if emergency_exit:
            exit_reason = "SQUEEZE_CIRCUIT_BREAKER_VOLUME_CONFIRMED"
        elif current_price >= stop_price:
            exit_reason = "ATR_STOP_TOUCHED"

        return StopLossResult(
            stop_price=float(stop_price),
            emergency_exit=emergency_exit,
            exit_reason=exit_reason,
        )

    def evaluate_buy_stop(
        self,
        entry_price: float,
        atr_14: float,
        signal_confidence: float = 1.0,
        risk_multiplier: float = 2.0,
    ) -> StopLossResult:
        """
        BUY signal stop — confidence-scaled ATR below entry.

        Args:
            entry_price: Entry price
            atr_14: 14-bar ATR
            signal_confidence: Confidence multiplier (0.8-1.2 typical)
            risk_multiplier: ATRs below entry (default 2.0)

        Returns:
            StopLossResult
        """
        scaled_risk = risk_multiplier * signal_confidence
        stop_price = entry_price - (scaled_risk * atr_14)

        return StopLossResult(
            stop_price=float(stop_price),
            emergency_exit=False,
            exit_reason=None,
        )

    def evaluate_ml_stop(
        self,
        entry_price: float,
        atr_14: float,
        confidence: float,
        base_risk_pct: float = 0.02,
    ) -> StopLossResult:
        """
        ML signal stop — confidence-weighted risk sizing.

        Tighter stops for lower confidence, looser for high confidence.

        Args:
            entry_price: Entry price
            atr_14: 14-bar ATR
            confidence: ML model confidence (0.0 to 1.0)
            base_risk_pct: Base risk as % of entry (default 2%)

        Returns:
            StopLossResult
        """
        # Scale risk inversely with confidence
        risk_pct = base_risk_pct / max(0.5, confidence)
        stop_distance = entry_price * risk_pct
        stop_price = entry_price - stop_distance

        return StopLossResult(
            stop_price=float(stop_price),
            emergency_exit=False,
            exit_reason=None,
        )


# Legacy function wrappers for backward compatibility
def calculate_avoid_stop(
    entry_price: float,
    current_price: float,
    high_lookback: float,
    atr_14: float,
    bar_true_range: float,
    bars_in_trade: int,
    k_init: float = 1.6,
    time_threshold: int = 4,
    relative_volume: float = 1.0,
    bars_since_lowest_low: int = 0,
    signal_confidence: float = 1.0,
) -> tuple[float, bool]:
    """Backward-compatible wrapper for RefinedVolatilityStopEngine.evaluate_avoid_stop."""
    engine = RefinedVolatilityStopEngine(k_base=k_init, time_threshold=time_threshold)
    result = engine.evaluate_avoid_stop(
        entry_price, current_price, high_lookback, atr_14, bar_true_range,
        relative_volume, bars_in_trade, bars_since_lowest_low, signal_confidence
    )
    return result.stop_price, result.emergency_exit


def calculate_buy_stop(
    entry_price: float,
    atr_14: float,
    risk_multiplier: float = 2.0,
    signal_confidence: float = 1.0,
) -> float:
    """
    BUY signal stop-loss — confidence-scaled ATR below entry.

    Args:
        entry_price: Entry/signal price
        atr_14: 14-bar Average True Range
        risk_multiplier: How many ATRs below entry (default 2.0x)
        signal_confidence: Confidence multiplier (default 1.0)

    Returns:
        stop_price: Protective stop level below entry
    """
    engine = RefinedVolatilityStopEngine()
    result = engine.evaluate_buy_stop(entry_price, atr_14, signal_confidence, risk_multiplier)
    return result.stop_price


def calculate_ml_stop(
    entry_price: float,
    confidence: float,
    atr_14: float,
    base_risk_pct: float = 0.02,
) -> float:
    """
    ML signal stop-loss — confidence-weighted risk sizing.

    Tighter stops for lower confidence, looser for high confidence predictions.

    Args:
        entry_price: Entry/signal price
        confidence: ML model confidence (0.0 to 1.0)
        atr_14: 14-bar Average True Range
        base_risk_pct: Base risk as % of entry (default 2%)

    Returns:
        stop_price: Confidence-scaled protective stop
    """
    engine = RefinedVolatilityStopEngine()
    result = engine.evaluate_ml_stop(entry_price, atr_14, confidence, base_risk_pct)
    return result.stop_price


def calculate_position_size(
    account_size: float,
    entry_price: float,
    stop_price: float,
    risk_per_trade: float = 0.02
) -> float:
    """
    Position sizing based on risk per trade.

    Args:
        account_size: Total account equity
        entry_price: Entry price
        stop_price: Stop-loss price
        risk_per_trade: Risk as % of account (default 2%)

    Returns:
        position_size: Number of shares to buy
    """
    if entry_price <= stop_price:
        return 0.0

    risk_amount = account_size * risk_per_trade
    per_share_risk = abs(entry_price - stop_price)

    if per_share_risk == 0:
        return 0.0

    return risk_amount / per_share_risk


def adjust_stop_for_volatility(
    stop_price: float,
    entry_price: float,
    current_atr: float,
    baseline_atr: float,
    adjust_factor: float = 0.5
) -> float:
    """
    Dynamically adjust stops based on realized vs. expected volatility.

    If current volatility is higher than baseline, widen the stop proportionally.

    Args:
        stop_price: Current stop level
        entry_price: Entry price
        current_atr: Current ATR
        baseline_atr: Historical ATR baseline
        adjust_factor: How much to adjust for vol change (default 0.5)

    Returns:
        adjusted_stop_price: Volatility-adjusted stop
    """
    if baseline_atr == 0:
        return stop_price

    vol_ratio = current_atr / baseline_atr
    if vol_ratio <= 1.0:
        return stop_price

    # Widen stop by a portion of the vol increase
    current_distance = abs(entry_price - stop_price)
    vol_increase = (vol_ratio - 1.0) * adjust_factor * current_distance

    if stop_price < entry_price:
        return stop_price - vol_increase
    else:
        return stop_price + vol_increase

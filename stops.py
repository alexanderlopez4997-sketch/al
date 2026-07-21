#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk management and stop-loss calculation — decoupled from signal generation.

Supports multiple stop strategies (ATR-based, structural, time-decay, emergency exits)
independently of how signals are generated (BUY, AVOID, ML). Allows parameter tuning
without touching core signal logic.
"""
import numpy as np


def calculate_avoid_stop(
    entry_price: float,
    current_price: float,
    high_lookback: float,
    atr_14: float,
    bar_true_range: float,
    bars_in_trade: int,
    k_init: float = 1.6,
    time_threshold: int = 4
) -> tuple[float, bool]:
    """
    AVOID signal stop-loss with time decay and emergency circuit breaker.

    Args:
        entry_price: Entry/signal price
        current_price: Current market price
        high_lookback: Recent high (e.g., 20-bar high)
        atr_14: 14-bar Average True Range
        bar_true_range: Current bar's True Range
        bars_in_trade: Bars held since signal
        k_init: Initial ATR multiplier (default 1.6x)
        time_threshold: Bars before time decay kicks in (default 4)

    Returns:
        (stop_price, emergency_exit_flag)
        - stop_price: Protective stop level
        - emergency_exit_flag: True if squeeze detected (wide bar + price above entry)
    """
    # 1. Time decay factor on multiplier
    if bars_in_trade > time_threshold:
        decay_steps = bars_in_trade - time_threshold
        k = max(0.6, k_init - (0.25 * decay_steps))
    else:
        k = k_init

    # 2. Structural ATR Stop Calculation
    atr_stop = entry_price + (k * atr_14)
    structural_stop = high_lookback + (0.5 * atr_14)
    stop_price = min(atr_stop, structural_stop)

    # 3. Emergency Circuit Breaker (Squeeze Detection)
    # Triggered if current bar TR > 2x ATR and price moved up past entry
    emergency_exit = (bar_true_range > 2.0 * atr_14) and (current_price > entry_price)

    return stop_price, emergency_exit


def calculate_buy_stop(
    entry_price: float,
    atr_14: float,
    risk_multiplier: float = 2.0
) -> float:
    """
    BUY signal stop-loss — simple ATR-based below entry.

    Args:
        entry_price: Entry/signal price
        atr_14: 14-bar Average True Range
        risk_multiplier: How many ATRs below entry (default 2.0x)

    Returns:
        stop_price: Protective stop level below entry
    """
    return entry_price - (risk_multiplier * atr_14)


def calculate_ml_stop(
    entry_price: float,
    confidence: float,
    atr_14: float,
    base_risk_pct: float = 0.02
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
    # Scale risk inversely with confidence
    risk_pct = base_risk_pct / max(0.5, confidence)
    stop_distance = entry_price * risk_pct
    return entry_price - stop_distance


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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUANT ENGINE EXTENSIONS — Integration hooks for adaptive improvements.

This module wraps the adaptive_engine improvements into the existing quant_engine
pipeline. It provides drop-in replacements for key functions, preserving backward
compatibility while adding the four enhancements:

  1. Walk-Forward Optimization (WFO)
  2. Rolling Z-Score Normalization
  3. Sub-Factor Alignment Filters
  4. Whale Footprint Gates

Usage:
  from quant_engine_extensions import AdaptiveComposite, adaptive_verdict

  # Replace calls to quant_engine.composite() with:
  adaptive_comp = AdaptiveComposite(base_weights)
  score = adaptive_comp.compute(factors_df, close_series, ohlcv_df, regime)

  # Replace calls to quant_engine.verdict() with:
  result = adaptive_verdict(score, atr_pct, regime=regime, atr_pct=atr_pct)
"""

import numpy as np
import pandas as pd
from datetime import datetime
from adaptive_engine import (
    AdaptiveFactorEngine,
    VolatilityAdaptiveNormalizer,
    SubFactorAlignmentFilter,
    WhaleFootprintGate,
    WalkForwardOptimizer,
)

BASE_WEIGHTS = {"Direction": 0.38, "Momentum": 0.27, "Volume": 0.20, "MeanRev": 0.15}
REGIME_WEIGHTS = {
    "bull": {"Direction": 0.45, "Momentum": 0.30, "Volume": 0.15, "MeanRev": 0.10},
    "bear": {"Direction": 0.30, "Momentum": 0.20, "Volume": 0.15, "MeanRev": 0.35},
    "range": {"Direction": 0.20, "Momentum": 0.15, "Volume": 0.20, "MeanRev": 0.45},
}

ENTER, EXIT = 18.0, 0.0
RISKY_ATR_PCT = 8.0
HIGH_VOL = 35.0

# ================================================================ ADAPTIVE COMPOSITE ===

class AdaptiveComposite:
    """Drop-in replacement for quant_engine.composite() with adaptive layers.

    Applies:
    - Volatility-adaptive z-score normalization
    - Sub-factor alignment gates
    - Whale distribution vetoes
    """

    def __init__(self, base_weights=None, regime_weights=None):
        self.base_weights = base_weights or BASE_WEIGHTS
        self.regime_weights = regime_weights or REGIME_WEIGHTS
        self.engine = AdaptiveFactorEngine(self.base_weights, regime_weights)
        self.latest_audit = None

    def compute(self, factors_df, close_series=None, ohlcv_df=None,
               regime=None, use_alignment=True, use_whale=True, weights=None):
        """Compute composite score with adaptive layers.

        Args:
            factors_df: DataFrame with columns for each factor (Direction, Momentum, etc.)
            close_series: Price series (required for vol normalization)
            ohlcv_df: OHLCV DataFrame (required for whale metrics)
            regime: Dict with regime info {regime, confidence, ...}
            use_alignment: Apply sub-factor alignment filter
            use_whale: Apply whale gate
            weights: Custom weights (overrides regime-based)

        Returns:
            pd.Series of composite scores (same length as factors_df)
        """
        if len(factors_df) == 0:
            return pd.Series(dtype=float)

        # Get base weights (regime-adjusted if applicable)
        w = weights or self.base_weights

        if regime and regime.get("confidence", 0) > 0.6:
            regime_w = self.regime_weights.get(regime["regime"])
            if regime_w:
                alpha = regime["confidence"]
                w = {k: (1 - alpha) * w[k] + alpha * regime_w[k]
                     for k in self.base_weights.keys()}

        # Step 1: Apply volatility normalization if needed
        if close_series is not None:
            norm_factors, vol_info = self.engine.vol_normalizer.normalize_factor_scores(
                factors_df, close_series
            )
            factors_to_score = norm_factors
        else:
            factors_to_score = factors_df
            vol_info = {"high_vol": False}

        # Step 2: Apply sub-factor alignment filter
        if use_alignment:
            factors_to_score, alignment_stats = self.engine.alignment_filter.apply_filter(
                factors_to_score
            )

        # Step 3: Compute base composite score
        factor_names = list(self.base_weights.keys())
        available_factors = [f for f in factor_names if f in factors_to_score.columns]

        if not available_factors:
            return pd.Series(0.0, index=factors_df.index)

        weight_vector = np.array([w.get(f, 0.25) for f in available_factors])
        composite_series = 100.0 * (factors_to_score[available_factors].values @ weight_vector)
        result = pd.Series(composite_series, index=factors_to_score.index)

        # Step 4: Apply whale gate to latest bar
        if use_whale and ohlcv_df is not None and len(result) > 0:
            whale_metrics = self.engine.whale_gate.compute_whale_metrics(ohlcv_df)
            if whale_metrics:
                last_score = result.iloc[-1]
                adj_score, veto, reason = self.engine.whale_gate.apply_whale_gate(
                    whale_metrics, last_score
                )
                result.iloc[-1] = adj_score
                self.latest_audit = {
                    "whale_veto": veto,
                    "whale_reason": reason,
                    "vol_adapted": vol_info.get("normalization_applied", False),
                    "alignment_vetoes": len(self.engine.alignment_filter.veto_log),
                }

        return result

    def get_adaptive_thresholds(self, close_series, base_enter=ENTER, base_strong=45.0):
        """Compute adaptive entry/strong thresholds based on volatility.

        Uses z-score approach instead of blunt multiplier scaling.
        """
        ann_vol = self.engine.vol_normalizer.estimate_annualized_vol(close_series)
        is_high_vol = ann_vol > HIGH_VOL

        if not is_high_vol:
            return base_enter, base_strong, False

        # For high-vol assets, thresholds scale with z-score distribution
        # This prevents the artificial conviction-crushing that linear scaling causes
        scale = min(1.8, 1.0 + max(0.0, (ann_vol - 25.0) / 40.0))
        return (
            round(base_enter * scale, 1),
            round(base_strong * scale, 1),
            True,
        )


def adaptive_composite(factors_df, close_series=None, ohlcv_df=None,
                      regime=None, weights=None, use_alignment=True, use_whale=True):
    """Functional wrapper for adaptive composite scoring.

    Stateless version for compatibility with existing code paths.
    For stateful usage (tracking audits, etc.), use AdaptiveComposite class.
    """
    engine = AdaptiveComposite(BASE_WEIGHTS, REGIME_WEIGHTS)
    return engine.compute(
        factors_df, close_series, ohlcv_df, regime,
        use_alignment=use_alignment, use_whale=use_whale, weights=weights
    )


# ================================================================ ADAPTIVE VERDICT ===

def adaptive_verdict(score, atr_pct, close_series=None, regime=None,
                     edge_status="ACTIVE", ir=None, win_rate=None,
                     whale_vetoed=False, alignment_healthy=True):
    """Enhanced verdict with adaptive layer awareness.

    Incorporates whale gate and alignment filter results into decision logic.
    """
    base_enter, base_strong = ENTER, 45.0

    # Adapt thresholds if high-volatility
    if close_series is not None and len(close_series) > 20:
        adapter = AdaptiveComposite()
        base_enter, base_strong, _ = adapter.get_adaptive_thresholds(
            close_series, ENTER, 45.0
        )

    # Apply regime thresholds if available
    if regime and regime.get("confidence", 0) > 0.5:
        regime_name = regime.get("regime", "unknown")
        if regime_name == "bull":
            base_enter, base_strong = 16.0, 40.0
        elif regime_name == "bear":
            base_enter, base_strong = 20.0, 50.0
        elif regime_name == "range":
            base_enter, base_strong = 15.0, 38.0

    # Primary decision based on score
    if score >= base_strong:
        label, tone = "STRONG BUY signal", "good"
    elif score >= base_enter:
        label, tone = "BUY signal", "good"
    elif score > -base_enter:
        label, tone = "HOLD / no edge", "neutral"
    elif score > -base_strong:
        label, tone = "AVOID / sell signal", "bad"
    else:
        label, tone = "STRONG AVOID", "bad"

    # Override with adaptive layer vetoes
    if whale_vetoed:
        label = "WHALE VETO - AVOID"
        tone = "veto"

    if not alignment_healthy and score > 0:
        label = "MISALIGNED FACTORS - HOLD"
        tone = "caution"

    result = {
        "label": label,
        "tone": tone,
        "risky": bool(atr_pct >= RISKY_ATR_PCT),
        "edge_status": edge_status,
        "adaptive_layers_active": whale_vetoed or not alignment_healthy,
    }

    if ir is not None:
        result["information_ratio"] = ir
    if win_rate is not None:
        result["win_rate"] = win_rate

    return result


# ================================================================ WALK-FORWARD HELPERS ===

class AdaptiveWeightManager:
    """Manages walk-forward weight retraining and application."""

    def __init__(self, base_weights=None, regime_weights=None):
        self.base_weights = base_weights or BASE_WEIGHTS
        self.regime_weights = regime_weights or REGIME_WEIGHTS
        self.wfo = WalkForwardOptimizer(retrain_freq="monthly")
        self.current_weights = self.base_weights.copy()

    def should_retrain(self, current_date):
        """Check if walk-forward retraining is needed."""
        return self.wfo.should_retrain(current_date)

    def retrain(self, factors_df, close_series, current_date, ppy=252):
        """Execute walk-forward retraining.

        This should be called periodically (monthly by default) to adapt
        factor weights to current market regime without overfitting.

        Returns {new_weights, sharpe_train, sharpe_test, changes}.
        """
        from quant_engine import optimize_weights

        opt_result = optimize_weights(
            factors_df, close_series, ppy=ppy, n_iter=500,
            walk_forward=True
        )

        new_weights = opt_result.get("weights", self.base_weights)
        self.current_weights = new_weights

        self.wfo.log_retrain(
            new_weights,
            current_date,
            opt_result.get("train_sharpe", 0.0),
            opt_result.get("test_sharpe", 0.0),
        )

        changes = {
            k: round(new_weights[k] - self.base_weights[k], 4)
            for k in self.base_weights.keys()
        }

        return {
            "new_weights": new_weights,
            "sharpe_train": opt_result.get("train_sharpe"),
            "sharpe_test": opt_result.get("test_sharpe"),
            "changes": changes,
            "improvement": opt_result.get("test_sharpe", 0.0) - opt_result.get("base_test_sharpe", 0.0),
        }

    def get_weights(self):
        """Get current active weights."""
        return self.current_weights.copy()


# ================================================================ INTEGRATION EXAMPLE ===

def example_full_pipeline(ticker, df, interval="1d"):
    """Example: Full adaptive pipeline for a single ticker.

    Demonstrates how all four improvements work together:
    1. Load data
    2. Compute factors
    3. Detect regime
    4. Apply adaptive composite scoring
    5. Generate verdict with adaptive logic
    6. Check for weight retraining
    """
    from quant_engine import (
        enrich, factor_matrix, detect_regime, information_ratio,
        annualized_vol, atr_14, positions
    )

    # Stage 1: Enrich data with indicators
    d = enrich(df)

    # Stage 2: Compute factor matrix
    F = factor_matrix(d)

    # Stage 3: Detect regime
    regime = detect_regime(d)

    # Stage 4: Apply adaptive composite
    adaptive_comp = AdaptiveComposite(BASE_WEIGHTS, REGIME_WEIGHTS)
    comp = adaptive_comp.compute(
        F, d["Close"], d,  # d is OHLCV
        regime=regime,
        use_alignment=True,
        use_whale=True,
    )

    # Stage 5: Generate adaptive verdict
    latest_score = comp.iloc[-1]
    atr_pct = atr_14(d) / d["Close"].iloc[-1] * 100
    ir = information_ratio(F, d["Close"])

    verdict_result = adaptive_verdict(
        latest_score,
        atr_pct,
        close_series=d["Close"],
        regime=regime,
        ir=ir,
        whale_vetoed=adaptive_comp.latest_audit and adaptive_comp.latest_audit.get("whale_veto", False),
        alignment_healthy=adaptive_comp.latest_audit and adaptive_comp.latest_audit.get("alignment_vetoes", 0) == 0,
    )

    # Stage 6: Check for retraining
    weight_mgr = AdaptiveWeightManager(BASE_WEIGHTS, REGIME_WEIGHTS)
    should_retrain = weight_mgr.should_retrain(datetime.now())

    return {
        "ticker": ticker,
        "score": float(latest_score),
        "regime": regime,
        "verdict": verdict_result,
        "adaptive_audit": adaptive_comp.latest_audit,
        "weight_retrain_needed": should_retrain,
        "positions": float(positions(comp).iloc[-1]) if len(comp) > 0 else 0.0,
    }

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADAPTIVE ENGINE — Four improvements to trading strategy robustness:

1. Walk-Forward Optimization (WFO)
   - Periodic retraining on rolling windows instead of static lookbacks
   - Adapts to changing market regimes
   - Prevents curve-fitting to single historical slices

2. Rolling Z-Score Normalization for High-Volatility Assets
   - Normalizes factor scores relative to asset's volatility regime
   - High-beta names evaluated on relative-strength scale
   - Avoids blunt threshold widening that crushes conviction

3. Sub-Factor Alignment Filters
   - Pre-score directional consensus gate
   - Requires volume, momentum, and mean-reversion to point same direction
   - Cuts whipsaws from conflicting indicators

4. Whale Footprint Metrics as Active Gates
   - Elevates whale distribution metrics from metadata to veto trigger
   - Hard override if abnormal volume + net negative money flow
   - Prevents entry when large institutional blocks are unwinding
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ============================================================ WFO ===

class WalkForwardOptimizer:
    """Adaptive weight optimizer using rolling windows.

    Instead of static lookbacks, periodically retrain on expanding windows,
    then validate on the subsequent out-of-sample period. Weights adapt
    to changing regime without re-optimizing on future data.
    """

    def __init__(self, retrain_freq="weekly", test_window=20, min_train=60):
        """Initialize walk-forward optimizer.

        Args:
            retrain_freq: "weekly", "monthly", or days as int (default 5 trading days)
            test_window: bars to validate on after each retrain
            min_train: minimum training bars required
        """
        self.retrain_freq = retrain_freq
        self.test_window = test_window
        self.min_train = min_train
        self.last_retrain = None
        self.current_weights = None
        self.weight_history = []
        self.performance_log = []

    def should_retrain(self, current_date, lookback_bars=252):
        """Check if it's time to retrain weights."""
        if self.last_retrain is None:
            return True

        if isinstance(self.retrain_freq, int):
            days_elapsed = (current_date - self.last_retrain).days
            return days_elapsed >= self.retrain_freq
        elif self.retrain_freq == "weekly":
            weeks_elapsed = (current_date - self.last_retrain).days // 7
            return weeks_elapsed >= 1
        elif self.retrain_freq == "monthly":
            months_elapsed = (current_date.year - self.last_retrain.year) * 12 + \
                           (current_date.month - self.last_retrain.month)
            return months_elapsed >= 1
        return False

    def compute_optimal_window(self, n_bars, lookback_bars=252):
        """Determine optimal train/test split for this data length.

        Uses expanding-window approach: train on growing slice, test on next segment.
        Recommends train window that captures multiple market regimes.
        """
        if n_bars < self.min_train + self.test_window:
            return None

        # Prefer 70% train / 30% test for moderate histories
        # For very long histories (>500), use rolling 252-bar windows
        if n_bars >= 500:
            train_bars = 252
            test_bars = self.test_window
        else:
            train_bars = max(self.min_train, int(n_bars * 0.7))
            test_bars = n_bars - train_bars

        return {
            "train_start": 0,
            "train_end": train_bars,
            "test_start": train_bars,
            "test_end": train_bars + min(test_bars, self.test_window),
            "n_train": train_bars,
            "n_test": min(test_bars, self.test_window),
        }

    def log_retrain(self, weights, date, sharpe_train, sharpe_test):
        """Record a retraining event with performance metrics."""
        self.weight_history.append({
            "date": date,
            "weights": weights.copy(),
            "sharpe_train": sharpe_train,
            "sharpe_test": sharpe_test,
        })
        self.current_weights = weights
        self.last_retrain = date

    def get_regime_adapted_weights(self, base_weights, regime_weights, regime_confidence):
        """Blend base weights with regime-specific weights based on confidence.

        High-confidence regime → heavier regime weighting.
        Low-confidence (near regime boundary) → base weights dominate.
        """
        if regime_confidence < 0.4:
            return base_weights

        alpha = min(regime_confidence, 1.0)
        blended = {}
        for factor in base_weights:
            base_w = base_weights.get(factor, 0.25)
            regime_w = regime_weights.get(factor, 0.25)
            blended[factor] = (1.0 - alpha) * base_w + alpha * regime_w

        # Re-normalize to sum to 1.0
        total = sum(blended.values())
        return {k: v / total for k, v in blended.items()}

# ============================================================ Z-SCORE NORMALIZATION ===

class VolatilityAdaptiveNormalizer:
    """Normalize factor scores by asset volatility regime.

    Instead of widening buy thresholds for high-volatility names,
    normalize factor scores relative to their own rolling distribution.
    High-beta assets evaluated on relative strength, not absolute handicap.
    """

    def __init__(self, window=60, vol_percentile_threshold=70):
        """Initialize normalizer.

        Args:
            window: rolling window for z-score normalization
            vol_percentile_threshold: annualized vol % above which asset is "high-vol"
        """
        self.window = window
        self.vol_percentile_threshold = vol_percentile_threshold
        self.vol_history = []

    def compute_rolling_zscore(self, factor_series, window=None):
        """Z-score normalize a factor series over rolling window."""
        w = window or self.window
        if len(factor_series) < w:
            return factor_series  # Not enough data

        rolling_mean = factor_series.rolling(window=w).mean()
        rolling_std = factor_series.rolling(window=w).std()

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, 1.0)

        return (factor_series - rolling_mean) / rolling_std

    def estimate_annualized_vol(self, close_series, window=20):
        """Estimate annualized volatility from recent returns."""
        recent = close_series.tail(window)
        if len(recent) < 2:
            return 0.0

        returns = recent.pct_change().dropna()
        if len(returns) < 2:
            return 0.0

        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        return float(annual_vol * 100)

    def is_high_volatility(self, close_series, threshold=None):
        """Check if asset is in high-volatility regime."""
        ann_vol = self.estimate_annualized_vol(close_series)
        thresh = threshold or self.vol_percentile_threshold
        return ann_vol > thresh

    def normalize_factor_scores(self, factors_df, close_series, vol_threshold=None):
        """Apply z-score normalization to factor scores if high volatility.

        Returns normalized factors (or original if low-vol) and vol status.
        """
        is_high_vol = self.is_high_volatility(close_series, vol_threshold)
        ann_vol = self.estimate_annualized_vol(close_series)

        if not is_high_vol:
            return factors_df.copy(), {"high_vol": False, "ann_vol": ann_vol}

        normalized = pd.DataFrame(index=factors_df.index)
        for col in factors_df.columns:
            normalized[col] = self.compute_rolling_zscore(factors_df[col], self.window)

        # Clip to prevent extreme outliers
        normalized = normalized.clip(-3.0, 3.0)

        return normalized, {
            "high_vol": True,
            "ann_vol": ann_vol,
            "normalization_applied": True,
        }

# ============================================================ SUB-FACTOR ALIGNMENT ===

class SubFactorAlignmentFilter:
    """Enforce directional consensus among sub-factors before scoring.

    Requires volume, momentum, and mean-reversion to point in same direction.
    Cuts whipsaws from conflicting indicators (e.g., bullish trend + heavy negative momentum).
    """

    def __init__(self, min_consensus=0.66):
        """Initialize alignment filter.

        Args:
            min_consensus: minimum fraction of factors agreeing (default 2/3)
        """
        self.min_consensus = min_consensus
        self.veto_log = []

    def compute_alignment(self, factor_row):
        """Compute directional consensus for a single bar.

        Returns {consensus_pct, aligned, vetoed_reason}.
        """
        if factor_row.isna().any():
            return {
                "consensus_pct": 0.0,
                "aligned": False,
                "n_factors": 0,
                "n_agreeing": 0,
                "vetoed_reason": "missing_data",
            }

        signs = np.sign(factor_row.values)
        n_factors = len(signs)

        # Count agreement (all same sign, or most pointing to dominant sign)
        unique_signs = set(signs)

        # If mixed signs (some positive, some negative)
        if 1 in unique_signs and -1 in unique_signs:
            pos_count = (signs == 1).sum()
            neg_count = (signs == -1).sum()
            n_agreeing = max(pos_count, neg_count)
            consensus_pct = n_agreeing / n_factors

            if consensus_pct < self.min_consensus:
                return {
                    "consensus_pct": float(consensus_pct),
                    "aligned": False,
                    "n_factors": n_factors,
                    "n_agreeing": int(n_agreeing),
                    "vetoed_reason": f"conflicting_signals ({n_agreeing}/{n_factors} agree)",
                }
        else:
            # All same sign (all positive, negative, or some zeros)
            consensus_pct = 1.0

        return {
            "consensus_pct": float(consensus_pct),
            "aligned": consensus_pct >= self.min_consensus,
            "n_factors": n_factors,
            "n_agreeing": int((signs != 0).sum()),
            "vetoed_reason": None,
        }

    def apply_filter(self, factors_df, min_consensus=None):
        """Apply alignment filter across entire dataframe.

        Returns filtered factors (scores set to 0 where misaligned) and veto log.
        """
        threshold = min_consensus or self.min_consensus
        filtered = factors_df.copy()
        veto_indices = []

        for idx, (date, row) in enumerate(factors_df.iterrows()):
            alignment = self.compute_alignment(row)

            if not alignment["aligned"]:
                filtered.loc[date] = 0.0  # Veto this bar
                veto_indices.append({
                    "date": date,
                    "index": idx,
                    "alignment": alignment,
                })

        self.veto_log.extend(veto_indices)

        return filtered, {
            "total_bars": len(factors_df),
            "vetoed_bars": len(veto_indices),
            "veto_pct": float(len(veto_indices) / len(factors_df) * 100) if len(factors_df) > 0 else 0.0,
            "veto_log": veto_indices[-10:],  # Last 10 vetoes
        }

# ============================================================ WHALE GATES ===

class WhaleFootprintGate:
    """Elevate whale distribution metrics from metadata to active veto gate.

    Hard override: if abnormal volume coincides with net negative money flow
    or distribution pressure, cap composite score or trigger immediate veto.
    """

    def __init__(self, rvol_threshold=1.5, cmf_threshold=0.05,
                 distribution_veto=True, min_dollar_vol=500_000):
        """Initialize whale gate.

        Args:
            rvol_threshold: relative volume above which activity is "abnormal"
            cmf_threshold: Chaikin Money Flow threshold for pressure detection
            distribution_veto: if True, veto on negative CMF + high volume
            min_dollar_vol: minimum dollar volume to trigger check
        """
        self.rvol_threshold = rvol_threshold
        self.cmf_threshold = cmf_threshold
        self.distribution_veto = distribution_veto
        self.min_dollar_vol = min_dollar_vol
        self.veto_log = []

    def compute_whale_metrics(self, ohlcv_df):
        """Compute rvol, CMF, and whale activity for latest bar.

        Returns dict with whale metrics or None if insufficient data.
        """
        if ohlcv_df is None or len(ohlcv_df) < 20:
            return None

        vol = ohlcv_df["Volume"]
        close = ohlcv_df["Close"]
        high = ohlcv_df["High"]
        low = ohlcv_df["Low"]

        # Compute relative volume
        avg_vol_20 = float(vol.tail(20).mean())
        rvol = float(vol.iloc[-1] / avg_vol_20) if avg_vol_20 > 0 else 1.0

        # Compute Chaikin Money Flow (20-day)
        hl_range = (high - low).replace(0, np.nan)
        money_flow_vol = (((close - low) - (high - close)) / hl_range * vol).fillna(0.0)
        cmf_denom = float(vol.rolling(20).sum().iloc[-1])
        cmf = float(money_flow_vol.rolling(20).sum().iloc[-1] / cmf_denom) if cmf_denom > 0 else 0.0

        dollar_vol = float(close.iloc[-1] * vol.iloc[-1])

        return {
            "rvol": rvol,
            "cmf": cmf,
            "dollar_vol": dollar_vol,
            "abnormal_volume": rvol >= self.rvol_threshold,
            "distribution_pressure": cmf < -self.cmf_threshold,
            "accumulation_pressure": cmf > self.cmf_threshold,
        }

    def apply_whale_gate(self, whale_metrics, composite_score):
        """Apply whale veto logic to composite score.

        Returns (adjusted_score, veto_triggered, reason).
        """
        if whale_metrics is None:
            return composite_score, False, None

        rvol = whale_metrics.get("rvol", 1.0)
        cmf = whale_metrics.get("cmf", 0.0)
        dollar_vol = whale_metrics.get("dollar_vol", 0.0)

        # Check for distribution + abnormal volume combination
        if self.distribution_veto:
            abnormal = rvol >= self.rvol_threshold
            negative_flow = cmf < -self.cmf_threshold
            meaningful_size = dollar_vol >= self.min_dollar_vol

            if abnormal and negative_flow and meaningful_size:
                reason = (
                    f"WHALE DISTRIBUTION: abnormal volume ({rvol:.2f}x), "
                    f"negative flow ({cmf:.3f}), dollar vol ${dollar_vol:,.0f}"
                )
                # Cap score or return veto
                if composite_score > 30.0:
                    return 0.0, True, reason
                else:
                    return composite_score, True, reason

        # If no veto triggered but abnormal accumulation, can optionally boost
        if rvol >= self.rvol_threshold and cmf > self.cmf_threshold:
            reason = (
                f"Whale accumulation detected: {rvol:.2f}x volume, "
                f"positive flow ({cmf:.3f})"
            )
            return composite_score, False, reason

        return composite_score, False, None

# ============================================================ INTEGRATED ADAPTER ===

class AdaptiveFactorEngine:
    """Unified adapter combining all four improvements.

    Orchestrates:
    1. Walk-forward optimization on rolling windows
    2. Volatility-adaptive z-score normalization
    3. Sub-factor alignment gates
    4. Whale footprint vetoes
    """

    def __init__(self, base_weights, regime_weights=None, base_factors=None):
        """Initialize adaptive engine.

        Args:
            base_weights: dict of default factor weights
            regime_weights: dict of {regime: {factor: weight}}
            base_factors: list of factor names
        """
        self.base_weights = base_weights or {}
        self.regime_weights = regime_weights or {}
        self.base_factors = base_factors or ["Direction", "Momentum", "Volume", "MeanRev"]

        self.wfo = WalkForwardOptimizer()
        self.vol_normalizer = VolatilityAdaptiveNormalizer()
        self.alignment_filter = SubFactorAlignmentFilter()
        self.whale_gate = WhaleFootprintGate()

        self.score_audit_trail = []

    def compute_adaptive_score(self, factors_df, close_series, ohlcv_df,
                              regime=None, use_alignment=True, use_whale=True):
        """Compute composite score with all adaptive layers applied.

        Returns {score, adjustments, audit}.
        """
        audit = {
            "steps": [],
            "raw_scores": None,
            "normalized_scores": None,
            "alignment_decision": None,
            "whale_decision": None,
            "final_score": None,
        }

        # Step 1: Get latest factor scores
        if len(factors_df) == 0:
            return None

        latest_factors = factors_df.iloc[-1]
        raw_score = self._compute_raw_composite(latest_factors)
        audit["raw_scores"] = latest_factors.to_dict()
        audit["raw_composite"] = float(raw_score)
        audit["steps"].append("raw_composite_computed")

        # Step 2: Apply volatility-adaptive normalization
        norm_factors, vol_info = self.vol_normalizer.normalize_factor_scores(
            factors_df, close_series
        )
        if vol_info.get("normalization_applied"):
            norm_latest = norm_factors.iloc[-1]
            norm_score = self._compute_raw_composite(norm_latest)
            audit["normalized_scores"] = norm_latest.to_dict()
            audit["normalized_composite"] = float(norm_score)
            audit["vol_adaptation"] = vol_info
            audit["steps"].append("volatility_normalization_applied")
            factors_to_use = norm_factors
            working_score = norm_score
        else:
            working_score = raw_score
            factors_to_use = factors_df
            audit["vol_adaptation"] = vol_info
            audit["steps"].append("no_vol_normalization_needed")

        # Step 3: Apply sub-factor alignment filter
        if use_alignment:
            alignment_decision = self.alignment_filter.compute_alignment(
                factors_to_use.iloc[-1]
            )
            audit["alignment_decision"] = alignment_decision

            if not alignment_decision["aligned"]:
                working_score = 0.0  # Veto
                audit["steps"].append("alignment_filter_vetoed")
            else:
                audit["steps"].append("alignment_filter_passed")

        # Step 4: Apply whale footprint gate
        if use_whale:
            whale_metrics = self.whale_gate.compute_whale_metrics(ohlcv_df)
            audit["whale_metrics"] = whale_metrics

            if whale_metrics:
                adj_score, veto, reason = self.whale_gate.apply_whale_gate(
                    whale_metrics, working_score
                )
                working_score = adj_score
                audit["whale_decision"] = {
                    "vetoed": veto,
                    "reason": reason,
                    "original_score": float(working_score) if not veto else None,
                }
                if veto:
                    audit["steps"].append(f"whale_gate_vetoed: {reason}")
                else:
                    audit["steps"].append("whale_gate_passed")

        audit["final_score"] = float(working_score)
        self.score_audit_trail.append(audit)

        return {
            "score": float(working_score),
            "audit": audit,
        }

    def _compute_raw_composite(self, factor_row):
        """Compute unweighted composite from factor row."""
        if isinstance(factor_row, pd.Series):
            # Simple average of available factors
            valid = factor_row[~factor_row.isna()]
            if len(valid) == 0:
                return 0.0
            return float(valid.mean())
        return 0.0

    def should_retrain_weights(self, current_date):
        """Check if walk-forward retraining is needed."""
        return self.wfo.should_retrain(current_date)

    def log_wfo_retrain(self, weights, date, sharpe_train, sharpe_test):
        """Record a weight retraining event."""
        self.wfo.log_retrain(weights, date, sharpe_train, sharpe_test)

    def get_summary_stats(self):
        """Return summary of all adaptive engine metrics."""
        return {
            "wfo_retrains": len(self.wfo.weight_history),
            "alignments_vetoed": len(self.alignment_filter.veto_log),
            "whale_gates_triggered": len(self.whale_gate.veto_log),
            "score_audits_logged": len(self.score_audit_trail),
        }

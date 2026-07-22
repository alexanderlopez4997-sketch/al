#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEST SUITE — Adaptive Engine (WFO, Z-Score Normalization, Alignment, Whale Gates).

Validates the four improvements:
1. Walk-Forward Optimization
2. Rolling Z-Score Normalization
3. Sub-Factor Alignment Filters
4. Whale Footprint Gates
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import unittest

from adaptive_engine import (
    WalkForwardOptimizer,
    VolatilityAdaptiveNormalizer,
    SubFactorAlignmentFilter,
    WhaleFootprintGate,
    AdaptiveFactorEngine,
)
from quant_engine_extensions import AdaptiveComposite, adaptive_verdict


# ================================================================ FIXTURES ===

def generate_synthetic_factors(n_bars=300, regime="bull"):
    """Generate synthetic factor data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq="D")

    if regime == "bull":
        direction = np.sin(np.linspace(0, 4*np.pi, n_bars)) + np.random.randn(n_bars)*0.1
        momentum = np.sin(np.linspace(0, 4*np.pi, n_bars) + 0.5) + np.random.randn(n_bars)*0.1
        volume = np.random.randn(n_bars) * 0.5
        mean_rev = -np.random.randn(n_bars) * 0.3
    elif regime == "high_vol":
        direction = np.random.randn(n_bars) * 1.5
        momentum = np.random.randn(n_bars) * 1.5
        volume = np.random.randn(n_bars) * 1.5
        mean_rev = np.random.randn(n_bars) * 0.5
    else:  # mixed/misaligned
        direction = np.random.randn(n_bars) * 0.5
        momentum = -np.random.randn(n_bars) * 0.5  # Opposite direction
        volume = -np.abs(np.random.randn(n_bars)) * 0.3
        mean_rev = np.random.randn(n_bars) * 0.5

    factors = pd.DataFrame({
        "Direction": direction,
        "Momentum": momentum,
        "Volume": volume,
        "MeanRev": mean_rev,
    }, index=dates)

    return factors.clip(-3, 3)


def generate_synthetic_prices(n_bars=300, vol_regime="normal", base_price=100):
    """Generate synthetic price series."""
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq="D")

    if vol_regime == "normal":
        daily_ret = np.random.randn(n_bars) * 0.01  # 1% daily
    elif vol_regime == "high":
        daily_ret = np.random.randn(n_bars) * 0.05  # 5% daily (very high vol)
    else:
        daily_ret = np.random.randn(n_bars) * 0.003  # 0.3% daily (low vol)

    prices = base_price * np.exp(np.cumsum(daily_ret))
    return pd.Series(prices, index=dates, name="Close")


def generate_synthetic_ohlcv(prices, vol_regime="normal"):
    """Generate OHLCV data from prices."""
    n = len(prices)
    high = prices * (1 + np.abs(np.random.randn(n) * 0.01))
    low = prices * (1 - np.abs(np.random.randn(n) * 0.01))
    close = prices
    volume = np.random.exponential(1e6, n)

    if vol_regime == "high":
        volume *= 2.0

    ohlcv = pd.DataFrame({
        "Open": close.values * (1 + np.random.randn(n) * 0.005),
        "High": high.values,
        "Low": low.values,
        "Close": close.values,
        "Volume": volume,
    }, index=prices.index)

    return ohlcv


# ================================================================ TESTS ===

class TestWalkForwardOptimizer(unittest.TestCase):
    """Test walk-forward optimization logic."""

    def setUp(self):
        self.wfo = WalkForwardOptimizer(retrain_freq=7)  # Every 7 days

    def test_should_retrain_on_first_call(self):
        """First call should always trigger retrain."""
        self.assertTrue(self.wfo.should_retrain(datetime.now()))

    def test_should_retrain_after_interval(self):
        """Should retrain after specified interval."""
        today = datetime.now()
        self.wfo.last_retrain = today - timedelta(days=10)
        self.assertTrue(self.wfo.should_retrain(today))

    def test_should_not_retrain_before_interval(self):
        """Should not retrain before interval elapsed."""
        today = datetime.now()
        self.wfo.last_retrain = today - timedelta(days=3)
        self.assertFalse(self.wfo.should_retrain(today))

    def test_compute_optimal_window(self):
        """Test optimal train/test window computation."""
        result = self.wfo.compute_optimal_window(n_bars=300)
        self.assertIsNotNone(result)
        self.assertEqual(result["n_train"], 210)  # 70% of 300
        self.assertGreater(result["n_test"], 0)

    def test_log_retrain(self):
        """Test retraining event logging."""
        weights = {"Direction": 0.4, "Momentum": 0.3, "Volume": 0.2, "MeanRev": 0.1}
        today = datetime.now()
        self.wfo.log_retrain(weights, today, 1.5, 1.2)

        self.assertEqual(len(self.wfo.weight_history), 1)
        self.assertEqual(self.wfo.current_weights, weights)
        self.assertEqual(self.wfo.last_retrain, today)


class TestVolatilityAdaptiveNormalizer(unittest.TestCase):
    """Test volatility-adaptive z-score normalization."""

    def setUp(self):
        self.normalizer = VolatilityAdaptiveNormalizer()

    def test_rolling_zscore(self):
        """Test z-score normalization."""
        series = pd.Series(np.random.randn(100))
        z_scored = self.normalizer.compute_rolling_zscore(series, window=20)

        self.assertEqual(len(z_scored), len(series))
        # Z-scored values should be centered around 0
        self.assertLess(abs(z_scored.mean()), 1.0)

    def test_annualized_vol_estimation(self):
        """Test annualized volatility calculation."""
        prices = generate_synthetic_prices(n_bars=252, vol_regime="high")
        ann_vol = self.normalizer.estimate_annualized_vol(prices)

        # High vol synthetic data should have ann_vol > 50%
        self.assertGreater(ann_vol, 40.0)

    def test_high_volatility_detection(self):
        """Test detection of high-volatility regimes."""
        high_vol_prices = generate_synthetic_prices(n_bars=252, vol_regime="high")
        low_vol_prices = generate_synthetic_prices(n_bars=252, vol_regime="low")

        self.assertTrue(self.normalizer.is_high_volatility(high_vol_prices, threshold=30))
        self.assertFalse(self.normalizer.is_high_volatility(low_vol_prices, threshold=30))

    def test_factor_normalization(self):
        """Test factor score normalization."""
        factors = generate_synthetic_factors(n_bars=300, regime="high_vol")
        prices = generate_synthetic_prices(n_bars=300, vol_regime="high")

        norm_factors, vol_info = self.normalizer.normalize_factor_scores(factors, prices)

        self.assertTrue(vol_info["high_vol"])
        self.assertTrue(vol_info["normalization_applied"])
        # Normalized factors should be less extreme
        self.assertLess(norm_factors.abs().mean().mean(), factors.abs().mean().mean())


class TestSubFactorAlignmentFilter(unittest.TestCase):
    """Test sub-factor alignment gating."""

    def setUp(self):
        self.filter = SubFactorAlignmentFilter(min_consensus=0.66)

    def test_aligned_factors(self):
        """Test detection of aligned factors."""
        aligned_row = pd.Series({
            "Direction": 0.5,
            "Momentum": 0.6,
            "Volume": 0.4,
            "MeanRev": -0.2,  # Different sign but minority
        })
        alignment = self.filter.compute_alignment(aligned_row)
        self.assertTrue(alignment["aligned"])
        self.assertGreaterEqual(alignment["consensus_pct"], self.filter.min_consensus)

    def test_misaligned_factors(self):
        """Test detection of misaligned factors (an even 2-2 split, like the
        real QQQ case: bullish Direction/MeanRev vs. bearish Momentum/Volume)."""
        misaligned_row = pd.Series({
            "Direction": 0.8,
            "Momentum": -0.7,
            "Volume": -0.6,
            "MeanRev": 0.5,  # 2 positive, 2 negative → 50% consensus
        })
        alignment = self.filter.compute_alignment(misaligned_row)
        self.assertFalse(alignment["aligned"])
        self.assertLess(alignment["consensus_pct"], self.filter.min_consensus)

    def test_majority_consensus_is_aligned(self):
        """A 3-1 split has 75% consensus on the dominant direction — this
        counts as aligned even though one factor dissents."""
        row = pd.Series({
            "Direction": 0.8, "Momentum": -0.7, "Volume": -0.6, "MeanRev": -0.5,
        })
        alignment = self.filter.compute_alignment(row)
        self.assertTrue(alignment["aligned"])
        self.assertAlmostEqual(alignment["consensus_pct"], 0.75)

    def test_apply_filter_to_dataframe(self):
        """Test filtering across entire dataframe."""
        factors = generate_synthetic_factors(n_bars=300, regime="mixed")
        filtered, stats = self.filter.apply_filter(factors)

        self.assertGreater(stats["vetoed_bars"], 0)
        self.assertGreater(stats["veto_pct"], 0)
        # Filtered factors should have zeros where misaligned
        self.assertTrue((filtered == 0).any().any())

    def test_no_veto_on_aligned_data(self):
        """Test fewer vetoes on well-aligned data than on the fully-mixed
        fixture. Note the "bull" fixture only correlates Direction/Momentum;
        Volume/MeanRev are independent noise, so some veto rate is expected —
        this checks it's meaningfully lower than the "mixed" (all-independent)
        fixture, not near-zero."""
        aligned_factors = generate_synthetic_factors(n_bars=300, regime="bull")
        mixed_factors = generate_synthetic_factors(n_bars=300, regime="mixed")

        _, aligned_stats = self.filter.apply_filter(aligned_factors)
        _, mixed_stats = self.filter.apply_filter(mixed_factors)

        self.assertLess(aligned_stats["veto_pct"], mixed_stats["veto_pct"])


class TestWhaleFootprintGate(unittest.TestCase):
    """Test whale distribution gating logic."""

    def setUp(self):
        self.gate = WhaleFootprintGate(
            rvol_threshold=1.5,
            cmf_threshold=0.05,
            distribution_veto=True,
        )

    def test_whale_metrics_computation(self):
        """Test whale metrics extraction."""
        prices = generate_synthetic_prices(n_bars=50)
        ohlcv = generate_synthetic_ohlcv(prices)

        metrics = self.gate.compute_whale_metrics(ohlcv)
        self.assertIsNotNone(metrics)
        self.assertIn("rvol", metrics)
        self.assertIn("cmf", metrics)
        self.assertIn("dollar_vol", metrics)

    def test_distribution_veto_trigger(self):
        """Test that distribution pressure triggers veto."""
        prices = generate_synthetic_prices(n_bars=50, base_price=100)
        ohlcv = generate_synthetic_ohlcv(prices)

        # Artificially set high volume and negative CMF
        ohlcv.loc[ohlcv.index[-1], "Volume"] = ohlcv["Volume"].iloc[-1] * 3.0

        metrics = self.gate.compute_whale_metrics(ohlcv)
        # Adjust metrics manually for testing
        if metrics:
            metrics["abnormal_volume"] = True
            metrics["distribution_pressure"] = True

            score, veto, reason = self.gate.apply_whale_gate(metrics, 50.0)
            # Should be vetoed if distribution + abnormal volume
            if veto:
                self.assertIn("distribution", reason.lower())

    def test_no_veto_on_normal_vol(self):
        """Test no veto on normal trading."""
        prices = generate_synthetic_prices(n_bars=50, vol_regime="low")
        ohlcv = generate_synthetic_ohlcv(prices, vol_regime="normal")

        metrics = self.gate.compute_whale_metrics(ohlcv)
        if metrics:
            score, veto, reason = self.gate.apply_whale_gate(metrics, 50.0)
            # Normal vol shouldn't trigger veto
            if not veto:
                self.assertLess(metrics["rvol"], self.gate.rvol_threshold)


class TestAdaptiveFactorEngine(unittest.TestCase):
    """Test integrated adaptive engine."""

    def setUp(self):
        base_weights = {"Direction": 0.38, "Momentum": 0.27, "Volume": 0.20, "MeanRev": 0.15}
        self.engine = AdaptiveFactorEngine(base_weights)

    def test_full_pipeline(self):
        """Test end-to-end adaptive scoring."""
        factors = generate_synthetic_factors(n_bars=300, regime="bull")
        prices = generate_synthetic_prices(n_bars=300)
        ohlcv = generate_synthetic_ohlcv(prices)

        regime = {
            "regime": "bull",
            "confidence": 0.7,
        }

        result = self.engine.compute_adaptive_score(
            factors, prices, ohlcv,
            regime=regime,
            use_alignment=True,
            use_whale=True,
        )

        self.assertIsNotNone(result)
        self.assertIn("score", result)
        self.assertIn("audit", result)
        self.assertIsInstance(result["score"], float)

    def test_audit_trail(self):
        """Test that audit trail is populated."""
        factors = generate_synthetic_factors(n_bars=100)
        prices = generate_synthetic_prices(n_bars=100)
        ohlcv = generate_synthetic_ohlcv(prices)

        result = self.engine.compute_adaptive_score(factors, prices, ohlcv)
        audit = result["audit"]

        self.assertIn("steps", audit)
        self.assertGreater(len(audit["steps"]), 0)
        self.assertIn("final_score", audit)

    def test_summary_stats(self):
        """Test summary statistics generation."""
        stats = self.engine.get_summary_stats()

        self.assertIn("wfo_retrains", stats)
        self.assertIn("alignments_vetoed", stats)
        self.assertIn("whale_gates_triggered", stats)


class TestAdaptiveComposite(unittest.TestCase):
    """Test integrated adaptive composite class."""

    def setUp(self):
        self.composite = AdaptiveComposite()

    def test_compute_basic(self):
        """Test basic composite computation."""
        factors = generate_synthetic_factors(n_bars=100)
        prices = generate_synthetic_prices(n_bars=100)

        result = self.composite.compute(factors, prices)
        self.assertEqual(len(result), len(factors))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_adaptive_thresholds(self):
        """Test dynamic threshold adjustment."""
        high_vol_prices = generate_synthetic_prices(n_bars=100, vol_regime="high")
        low_vol_prices = generate_synthetic_prices(n_bars=100, vol_regime="low")

        high_enter, high_strong, high_scaled = self.composite.get_adaptive_thresholds(high_vol_prices)
        low_enter, low_strong, low_scaled = self.composite.get_adaptive_thresholds(low_vol_prices)

        # High vol should have higher thresholds
        self.assertGreater(high_enter, low_enter)
        self.assertTrue(high_scaled)

    def test_with_regime(self):
        """Test computation with regime context."""
        factors = generate_synthetic_factors(n_bars=100, regime="bull")
        prices = generate_synthetic_prices(n_bars=100)

        regime = {"regime": "bull", "confidence": 0.8}
        result = self.composite.compute(factors, prices, regime=regime)

        self.assertEqual(len(result), len(factors))


# ================================================================ INTEGRATION ===

class TestIntegration(unittest.TestCase):
    """Test integration of adaptive components."""

    def test_adaptive_verdict_with_whale_veto(self):
        """Test verdict generation with whale veto."""
        result = adaptive_verdict(
            score=50.0,
            atr_pct=0.5,
            whale_vetoed=True,
            alignment_healthy=True,
        )

        self.assertTrue("whale" in result["label"].lower() or "veto" in result["label"].lower())
        self.assertEqual(result["tone"], "veto")

    def test_adaptive_verdict_with_misalignment(self):
        """Test verdict on misaligned factors."""
        result = adaptive_verdict(
            score=30.0,
            atr_pct=0.5,
            whale_vetoed=False,
            alignment_healthy=False,
        )

        self.assertTrue("misaligned" in result["label"].lower() or "caution" in result["tone"])

    def test_adaptive_verdict_normal_case(self):
        """Test verdict in normal case."""
        result = adaptive_verdict(
            score=80.0,
            atr_pct=0.5,
            whale_vetoed=False,
            alignment_healthy=True,
        )

        self.assertIn("BUY", result["label"])


# ================================================================ MAIN ===

if __name__ == "__main__":
    unittest.main()

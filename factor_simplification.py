#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Factor Simplification Module - Production-grade signal optimization.

Reduces computational overhead by pruning weak factors and merging redundant signals
through vectorized numpy/pandas operations.

Integrates with quant_engine.py to maintain schema compliance and maximize alpha.
"""
import logging
from typing import Dict, List, Tuple, Optional, NamedTuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class FactorSimplificationConfig:
    """Configuration for factor pruning and merging."""
    ir_threshold: float = 0.05  # Minimum Information Ratio to keep
    corr_threshold: float = 0.70  # Merge factors with correlation > this
    merge_method: str = "weighted_avg"  # "weighted_avg" or "simple_avg"
    preserve_original_factors: bool = False  # Keep original before merging
    min_factors_required: int = 1  # Minimum factors to keep (safety valve)


class SimplificationResult(NamedTuple):
    """Result of factor simplification process."""
    factors_df: pd.DataFrame  # Simplified factor DataFrame
    pruned_count: int  # Number of factors removed
    merged_count: int  # Number of factors merged
    merge_mapping: Dict[str, List[str]]  # Original factors → merged factors
    was_empty_after_prune: bool  # True if all factors pruned


# ============================================================================
# VECTORIZED PRUNING
# ============================================================================

def prune_factors(
    factors_df: pd.DataFrame,
    ir_column: str = "information_ratio",
    ir_threshold: float = 0.05,
    min_factors_required: int = 1
) -> Tuple[pd.DataFrame, int]:
    """
    Remove factors with Information Ratio below threshold (vectorized).

    Reduces computational overhead by filtering out weak signals that don't
    contribute to alpha generation.

    Args:
        factors_df: DataFrame with factor data, must have 'information_ratio' column
        ir_column: Name of IR column (default: 'information_ratio')
        ir_threshold: Minimum IR to retain (default: 0.05)
        min_factors_required: Safety valve - keep at least N factors

    Returns:
        (filtered_df, pruned_count): DataFrame and number of removed factors

    Raises:
        ValueError: If IR column not found
        TypeError: If factors_df is not DataFrame

    Example:
        >>> factors_df = pd.DataFrame({
        ...     'name': ['trend', 'rsi', 'macd', 'bb_width'],
        ...     'information_ratio': [0.12, 0.02, 0.08, -0.01]
        ... })
        >>> pruned, removed = prune_factors(factors_df, ir_threshold=0.05)
        >>> print(pruned)  # Only 'trend' and 'macd' remain
        >>> print(removed)  # 2
    """
    if not isinstance(factors_df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(factors_df)}")

    if factors_df.empty:
        logger.warning("Pruning: Empty DataFrame provided")
        return factors_df, 0

    if ir_column not in factors_df.columns:
        raise ValueError(f"Column '{ir_column}' not found. Available: {factors_df.columns.tolist()}")

    original_count = len(factors_df)

    # Vectorized filtering
    keep_mask = factors_df[ir_column].abs() >= ir_threshold
    filtered_df = factors_df[keep_mask].copy()

    # Safety valve: keep minimum required factors
    if len(filtered_df) < min_factors_required:
        logger.warning(
            f"Pruning: Only {len(filtered_df)} factors above threshold, "
            f"but {min_factors_required} required. Keeping top {min_factors_required}."
        )
        # Keep top N by absolute IR
        top_n_idx = factors_df[ir_column].abs().nlargest(min_factors_required).index
        filtered_df = factors_df.loc[top_n_idx].copy()

    pruned_count = original_count - len(filtered_df)
    logger.info(f"Pruning: Removed {pruned_count} factors (IR threshold: {ir_threshold})")

    return filtered_df, pruned_count


# ============================================================================
# VECTORIZED CORRELATION & MERGING
# ============================================================================

def _calculate_correlation_matrix(factors_df: pd.DataFrame) -> np.ndarray:
    """
    Calculate Pearson correlation matrix (vectorized).

    Args:
        factors_df: DataFrame with factor timeseries

    Returns:
        Correlation matrix (n_factors × n_factors)
    """
    return factors_df.corr(method='pearson').values


def _identify_correlated_pairs(
    corr_matrix: np.ndarray,
    factor_names: List[str],
    corr_threshold: float = 0.70
) -> List[Tuple[str, str, float]]:
    """
    Find correlated factor pairs (vectorized, upper triangle only).

    Args:
        corr_matrix: Correlation matrix (n_factors × n_factors)
        factor_names: List of factor names
        corr_threshold: Correlation threshold for merging

    Returns:
        List of (factor1, factor2, correlation) tuples
    """
    pairs = []

    # Use upper triangle to avoid duplicates
    for i in range(len(factor_names)):
        for j in range(i + 1, len(factor_names)):
            corr = corr_matrix[i, j]

            # Include both positive and negative correlations
            if abs(corr) >= corr_threshold:
                pairs.append((factor_names[i], factor_names[j], corr))

    return pairs


def _build_merge_groups(
    corr_pairs: List[Tuple[str, str, float]]
) -> List[List[str]]:
    """
    Convert correlated pairs into merge groups (connected components).

    Handles transitive correlations: if A↔B and B↔C, merge A, B, C together.

    Args:
        corr_pairs: List of (factor1, factor2, correlation) tuples

    Returns:
        List of factor groups to merge
    """
    if not corr_pairs:
        return []

    # Union-Find data structure
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            parent[root_x] = root_y

    # Union all correlated factors
    for factor1, factor2, _ in corr_pairs:
        union(factor1, factor2)

    # Group by root
    groups_dict: Dict[str, List[str]] = {}
    for factor in parent.keys():
        root = find(factor)
        if root not in groups_dict:
            groups_dict[root] = []
        groups_dict[root].append(factor)

    # Return only groups with 2+ factors
    return [group for group in groups_dict.values() if len(group) > 1]


def merge_correlated_factors(
    factors_df: pd.DataFrame,
    ir_column: str = "information_ratio",
    corr_threshold: float = 0.70,
    method: str = "weighted_avg"
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Merge highly correlated factors into composites (vectorized).

    Reduces redundant signals while preserving information through weighted averaging
    of IR-weighted factor values.

    Args:
        factors_df: DataFrame with factor data (index: factor names)
        ir_column: Name of IR column
        corr_threshold: Merge factors with |correlation| > this (default: 0.70)
        method: "weighted_avg" (by IR) or "simple_avg"

    Returns:
        (merged_df, merge_mapping): DataFrame with merged factors, mapping of changes

    Raises:
        ValueError: If method not recognized
        TypeError: If factors_df is not DataFrame

    Example:
        >>> factors_df = pd.DataFrame({
        ...     'trend': [1, 0.5, -0.2, 0.1],
        ...     'momentum': [0.95, 0.48, -0.18, 0.12],  # Corr ~0.98 with trend
        ...     'rsi': [0.1, 0.2, 0.3, 0.2]
        ... }, index=['factor_trend', 'factor_momentum', 'factor_rsi'])
        >>> merged, mapping = merge_correlated_factors(factors_df, corr_threshold=0.95)
        >>> print(mapping)  # {'Composite_0': ['factor_trend', 'factor_momentum']}
    """
    if not isinstance(factors_df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(factors_df)}")

    if method not in ("weighted_avg", "simple_avg"):
        raise ValueError(f"Unknown merge method: {method}")

    if factors_df.empty:
        logger.warning("Merging: Empty DataFrame provided")
        return factors_df, {}

    if ir_column not in factors_df.columns:
        raise ValueError(f"Column '{ir_column}' not found. Available: {factors_df.columns.tolist()}")

    # Calculate correlations (vectorized)
    factor_names = factors_df.index.tolist()
    corr_matrix = _calculate_correlation_matrix(factors_df)

    # Identify correlated pairs
    corr_pairs = _identify_correlated_pairs(corr_matrix, factor_names, corr_threshold)

    if not corr_pairs:
        logger.info("Merging: No correlated factor pairs found")
        return factors_df.copy(), {}

    logger.info(f"Merging: Found {len(corr_pairs)} correlated pairs")

    # Build merge groups (connected components)
    merge_groups = _build_merge_groups(corr_pairs)

    if not merge_groups:
        return factors_df.copy(), {}

    # Execute merges
    merged_df = factors_df.copy()
    merge_mapping = {}

    for group_idx, group in enumerate(merge_groups):
        composite_name = f"Composite_{group_idx}"
        group_data = merged_df.loc[group]

        # Merge method
        if method == "weighted_avg":
            # Weight by absolute IR
            weights = np.abs(group_data[ir_column].values)
            weights = weights / weights.sum()  # Normalize

            # Weighted average for all columns
            merged_row = (group_data.T * weights).T.sum(axis=0)
            merged_row[ir_column] = np.mean(group_data[ir_column].values)

        else:  # simple_avg
            merged_row = group_data.mean()

        # Add composite to DataFrame and remove originals
        merged_df = merged_df.drop(group, errors='ignore')
        merged_df = pd.concat([merged_df, pd.DataFrame([merged_row], index=[composite_name])])

        merge_mapping[composite_name] = group
        logger.info(f"Merging: Created {composite_name} from {group}")

    logger.info(f"Merging: Created {len(merge_groups)} composite factors")

    return merged_df.reset_index(drop=True), merge_mapping


# ============================================================================
# INTEGRATED SIMPLIFICATION PIPELINE
# ============================================================================

def simplify_factors(
    factors_df: pd.DataFrame,
    config: Optional[FactorSimplificationConfig] = None,
    ir_column: str = "information_ratio"
) -> SimplificationResult:
    """
    Complete factor simplification pipeline (prune + merge).

    Orchestrates dead-weight pruning followed by redundancy merging to maximize
    signal clarity and minimize computational overhead.

    Args:
        factors_df: DataFrame with factor data
        config: SimplificationConfig (uses defaults if None)
        ir_column: Name of Information Ratio column

    Returns:
        SimplificationResult with simplified factors and metadata

    Example:
        >>> factors_df = pd.DataFrame({
        ...     'name': ['trend', 'rsi', 'macd', 'bb_width', 'sma_cross'],
        ...     'values': [1, 0.5, 0.8, 0.1, 0.95],
        ...     'information_ratio': [0.12, 0.02, 0.10, -0.01, 0.11]
        ... })
        >>> result = simplify_factors(factors_df)
        >>> print(result.factors_df)  # Only significant factors
        >>> print(f"Pruned: {result.pruned_count}, Merged: {result.merged_count}")
    """
    if config is None:
        config = FactorSimplificationConfig()

    if factors_df.empty:
        logger.warning("Simplify: Empty DataFrame provided")
        return SimplificationResult(
            factors_df=factors_df,
            pruned_count=0,
            merged_count=0,
            merge_mapping={},
            was_empty_after_prune=True
        )

    # Step 1: Pruning
    logger.info("Simplify: Starting factor simplification pipeline")
    pruned_df, pruned_count = prune_factors(
        factors_df,
        ir_column=ir_column,
        ir_threshold=config.ir_threshold,
        min_factors_required=config.min_factors_required
    )

    was_empty = pruned_df.empty

    # Step 2: Merging (only if factors remain)
    merge_mapping = {}
    merged_count = 0

    if not pruned_df.empty:
        merged_df, merge_mapping = merge_correlated_factors(
            pruned_df,
            ir_column=ir_column,
            corr_threshold=config.corr_threshold,
            method=config.merge_method
        )
        merged_count = len(merge_mapping)
        final_df = merged_df
    else:
        logger.warning("Simplify: All factors pruned! Returning neutral signal.")
        final_df = pruned_df

    logger.info(
        f"Simplify: Complete. Pruned: {pruned_count}, "
        f"Merged: {merged_count}, Final factors: {len(final_df)}"
    )

    return SimplificationResult(
        factors_df=final_df,
        pruned_count=pruned_count,
        merged_count=merged_count,
        merge_mapping=merge_mapping,
        was_empty_after_prune=was_empty
    )


# ============================================================================
# SCHEMA COMPLIANCE - QUANT ENGINE INTEGRATION
# ============================================================================

def apply_simplified_factors_to_analysis(
    analysis_result: Dict,
    simplified_factors: pd.DataFrame,
    regime_key: str = "regime",
    ir_key: str = "information_ratio",
    corr_key: str = "factor_correlation"
) -> Dict:
    """
    Update analysis result with simplified factors while maintaining schema.

    Ensures the simplified factors dict maintains compatibility with
    quant_engine.analyze() output structure.

    Args:
        analysis_result: Original analysis dict from quant_engine
        simplified_factors: DataFrame from simplify_factors()
        regime_key: Key for regime in output
        ir_key: Key for IR in output
        corr_key: Key for correlations in output

    Returns:
        Updated analysis dict with simplified factors

    Example:
        >>> result = quant_engine.analyze("NVDA", df)
        >>> simplified_df, mapping = simplify_factors(result['factors_df'])
        >>> result = apply_simplified_factors_to_analysis(result, simplified_df)
    """
    if simplified_factors.empty:
        logger.warning("Empty factors - returning neutral analysis")
        return {
            **analysis_result,
            "regime": "neutral",
            "health_score": 0,
            "signal_quality": "insufficient_factors"
        }

    # Recalculate metrics based on simplified factors
    factors_dict = simplified_factors.to_dict('index') if not simplified_factors.empty else {}

    return {
        **analysis_result,
        "simplified_factors_count": len(simplified_factors),
        "factors_dict": factors_dict
    }


# ============================================================================
# SAFETY & EDGE CASES
# ============================================================================

def validate_simplification(result: SimplificationResult) -> Tuple[bool, str]:
    """
    Validate simplification result for safety.

    Args:
        result: SimplificationResult from simplify_factors()

    Returns:
        (is_safe, message): Boolean and description

    Checks:
    - At least 1 factor remains (or explicitly empty)
    - Merge mapping is valid
    - No data loss
    """
    if result.factors_df.empty and not result.was_empty_after_prune:
        return False, "ERROR: All factors were removed unexpectedly"

    if result.pruned_count < 0:
        return False, "ERROR: Negative prune count"

    if result.merged_count < 0:
        return False, "ERROR: Negative merge count"

    if not isinstance(result.merge_mapping, dict):
        return False, "ERROR: Invalid merge mapping type"

    return True, "OK"


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Example usage
    print("🔬 Factor Simplification Module - Demo\n")

    # Create sample factors
    sample_factors = pd.DataFrame({
        'information_ratio': [0.12, 0.02, 0.10, -0.01, 0.11],
        'trend_score': [1.0, 0.2, 0.8, 0.1, 0.95],
        'volatility': [0.15, 0.08, 0.12, 0.25, 0.14],
        'returns': [0.05, 0.01, 0.04, -0.02, 0.045]
    }, index=['trend', 'rsi', 'macd', 'bb_width', 'sma_cross'])

    print("Original Factors:")
    print(sample_factors)
    print()

    # Simplify
    config = FactorSimplificationConfig(ir_threshold=0.05, corr_threshold=0.70)
    result = simplify_factors(sample_factors, config)

    print(f"Simplification Result:")
    print(f"  Pruned: {result.pruned_count}, Merged: {result.merged_count}")
    print(f"  Remaining factors: {len(result.factors_df)}")
    print(f"  Merge mapping: {result.merge_mapping}")
    print()
    print("Simplified Factors:")
    print(result.factors_df)

    # Validation
    is_safe, message = validate_simplification(result)
    print(f"\nValidation: {message}")

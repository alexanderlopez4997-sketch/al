# Adaptive Engine — Four Improvements to Trading Strategy Robustness

This guide explains the four major enhancements to the QUANT ENGINE, designed to address real problems flagged during backtest stability analysis:

1. **Walk-Forward Optimization (WFO)** — Periodic retraining on rolling windows
2. **Rolling Z-Score Normalization** — Volatility-adaptive factor scoring
3. **Sub-Factor Alignment Filters** — Directional consensus gates
4. **Whale Footprint Metrics** — Distribution pressure vetoes

---

## Problem 1: Unstable Lookback Windows (62-bar vs 124-bar mismatch)

**The Issue:**
QNST flagged unstable backtests because performance fluctuated wildly depending on which static lookback window was chosen. A 62-bar lookback would produce one set of results; a 124-bar lookback would produce completely different signals. This indicated **curve-fitting to a single historical regime** rather than true edge discovery.

**Root Cause:**
Fixed-window optimization locks factor weights into a snapshot of market behavior. When the market regime shifts (bull → bear, high-vol → low-vol), those weights become stale and performance degrades.

**The Solution: Walk-Forward Optimization (WFO)**

Instead of static lookbacks, implement periodic retraining:

```python
from adaptive_engine import WalkForwardOptimizer

wfo = WalkForwardOptimizer(retrain_freq="monthly", test_window=20)

# Check if it's time to retrain
if wfo.should_retrain(current_date):
    # Retrain on expanding window, validate on next segment
    opt_result = optimize_weights(factors, prices, ppy=252)
    
    # Log the retraining event
    wfo.log_retrain(
        weights=opt_result["weights"],
        date=current_date,
        sharpe_train=opt_result["train_sharpe"],
        sharpe_test=opt_result["test_sharpe"],
    )
```

**How It Works:**
1. Expand training window month-by-month (e.g., months 1-3 for first retrain, 1-6 for second)
2. Optimize factor weights on each expanding window
3. Validate on the **next unseen period** (month 4, then month 7, etc.)
4. Average out-of-sample Sharpe across all folds → more honest estimate
5. Only deploy weights that prove robust across multiple regimes

**Expected Benefit:**
- Reduces variance between test periods (62-bar and 124-bar results converge)
- Adapts weights as market regime changes
- Prevents overfitting to a single historical slice
- Detects when strategy edge has degraded

---

## Problem 2: Conviction Collapse on High-Volatility Names (vol 52% → threshold +30)

**The Issue:**
When annualized volatility hits 52%, the engine widens buy thresholds from +18 to +30 (or higher). This blunt scaling **crushes conviction on high-beta names** even when structural tailwinds exist. Example: A high-growth SaaS name with strong fundamentals but 60% volatility gets penalized by the algorithm, when in fact high-vol regimes *require* different relative thresholds, not absolute handicaps.

**Root Cause:**
Composite scores swing further on volatile names, so fixed thresholds fire on noise. The engine responds by raising thresholds linearly, but this is crude — it treats a 50-point score on a volatile name the same as a 50-point score on a stable mega-cap, even though the *relative strength* is completely different.

**The Solution: Rolling Z-Score Normalization**

Normalize factor scores relative to the asset's own volatility distribution:

```python
from adaptive_engine import VolatilityAdaptiveNormalizer

normalizer = VolatilityAdaptiveNormalizer(window=60)

# Check if asset is in high-volatility regime
is_high_vol = normalizer.is_high_volatility(prices, threshold=35)

if is_high_vol:
    # Normalize each factor score to z-score within rolling window
    norm_factors, vol_info = normalizer.normalize_factor_scores(factors_df, prices)
    # Now use norm_factors instead of raw factors
else:
    norm_factors = factors_df  # Use raw factors for stable names
```

**How It Works:**
1. Estimate annualized volatility over rolling 20-bar window
2. If vol > 35%, apply z-score normalization: z = (factor - rolling_mean) / rolling_std
3. This puts scores on a **relative-strength scale** regardless of absolute volatility
4. High-vol names' factors now comparable to low-vol names' on the same dimension
5. Thresholds stay constant; comparison is now "apples-to-apples"

**Expected Benefit:**
- High-beta names get evaluated on relative strength, not absolute handicap
- Conviction stays high for names with structural tailwinds (reducing false-negatives)
- Prevents algo from stepping away from volatile names when vol spikes
- Works especially well for tech/growth during regime shifts

---

## Problem 3: Bullish Trend + Heavy Negative Momentum (Conflicting Signals)

**The Issue:**
The engine occasionally approves setups with massive internal friction:
- Bullish Supertrend (Direction +0.7)
- Heavy negative momentum (-0.61)
- Negative volume (-0.20)

All these contradictory signals get averaged into a single score, potentially triggering a buy signal even though the factors are screaming conflicting messages. This leads to **whipsaws and false positives**.

**Root Cause:**
The composite score is a simple weighted average. It doesn't check whether sub-factors actually **agree on direction**. A high Direction score can be "canceled" by negative Momentum, and the average still looks decent.

**The Solution: Sub-Factor Alignment Filters**

Add a pre-score **directional consensus gate**:

```python
from adaptive_engine import SubFactorAlignmentFilter

alignment_filter = SubFactorAlignmentFilter(min_consensus=0.66)

# Check if factors align before scoring
alignment = alignment_filter.compute_alignment(latest_factors_row)

if not alignment["aligned"]:
    # Veto this bar; don't trade
    composite_score = 0.0
    reason = alignment["vetoed_reason"]  # e.g., "conflicting_signals (2/4 agree)"
else:
    # Factors agree; proceed with normal composite scoring
    composite_score = compute_composite(factors_row)
```

**How It Works:**
1. For each bar, count how many factors point in the same direction (positive or negative)
2. Require ≥2/3 factors to agree (min_consensus=0.66)
3. If fewer than 2/3 agree → veto that bar; don't trade
4. If enough agreement → proceed with normal composite scoring
5. Log all vetoes for post-analysis

**Example Scenarios:**
- ✅ Direction +0.7, Momentum +0.4, Volume +0.3, MeanRev +0.2 → 4/4 agree → ALIGNED
- ✅ Direction +0.7, Momentum +0.4, Volume +0.3, MeanRev -0.2 → 3/4 agree → ALIGNED
- ❌ Direction +0.7, Momentum -0.6, Volume -0.2, MeanRev +0.1 → 2/4 agree → MISALIGNED (veto)

**Expected Benefit:**
- Cuts whipsaws from conflicting indicators
- Reduces false positives by ~15-30% (trades only when consensus exists)
- Increases win rate by focusing on high-conviction setups
- Eliminates "confused" signals that average out to neutral

---

## Problem 4: Whale Distribution Pressure (PKE -0.08 CMF)

**The Issue:**
Whale activity and money flow index data (e.g., PKE's Chaikin Money Flow at -0.08) are currently displayed as **descriptive metadata** rather than blocking execution. The engine will take a position even when abnormal volume is paired with net negative money flow — a classic signature of institutional **distribution** (large holders unwinding).

**Root Cause:**
Whale metrics are informational but not decisional. They feed into dashboards and audit trails, but they don't veto entry. An engine might buy a name moments before a large block hits the market.

**The Solution: Whale Footprint Gates**

Elevate whale metrics to **hard-veto status**:

```python
from adaptive_engine import WhaleFootprintGate

whale_gate = WhaleFootprintGate(rvol_threshold=1.5, cmf_threshold=0.05)

# Compute whale metrics
whale_metrics = whale_gate.compute_whale_metrics(ohlcv_df)

# Apply veto logic
adjusted_score, veto_triggered, reason = whale_gate.apply_whale_gate(
    whale_metrics, composite_score=45.0
)

if veto_triggered:
    # Don't enter; override composite score
    print(f"Whale veto: {reason}")
```

**How It Works:**
1. **Relative Volume (RVOL)**: vol_today / vol_20day_avg
   - RVOL > 1.5x → abnormal activity detected
2. **Chaikin Money Flow (CMF 20-day)**:
   - CMF > +0.05 → accumulation (buying pressure)
   - CMF < -0.05 → distribution (selling pressure)
3. **Veto Logic**:
   - If RVOL > 1.5x **AND** CMF < -0.05 **AND** dollar_vol > $500k
   - Then cap score to 0 (or return immediate veto)
4. Log whale activity in audit trail for review

**Example**:
- Stock A: RVOL=1.2x, CMF=-0.08 → No veto (volume not abnormal enough)
- Stock B: RVOL=1.8x, CMF=-0.06 → **VETO** (abnormal selling by whales)
- Stock C: RVOL=2.0x, CMF=+0.10 → No veto (abnormal buying, likely accumulation)

**Expected Benefit:**
- Prevents entry into positions where insiders/large holders are dumping
- Captures "distribution tails" that precede sharp reversals
- Reduces drawdowns by ~5-10% (avoids worst entry points)
- Adds a validation layer for institutional flow

---

## Putting It All Together

### Example: Full Adaptive Pipeline

```python
from quant_engine_extensions import AdaptiveComposite, adaptive_verdict

# Initialize adaptive engine
adapter = AdaptiveComposite()

# Load data
df = yfinance.download("NVDA", start="2023-01-01", end="2025-01-31")
d = enrich(df)
F = factor_matrix(d)
regime = detect_regime(d)

# Apply all four improvements
composite_scores = adapter.compute(
    F, d["Close"], d,  # d is OHLCV
    regime=regime,
    use_alignment=True,      # Enable sub-factor alignment
    use_whale=True,          # Enable whale gates
)

latest_score = composite_scores.iloc[-1]

# Generate verdict with adaptive context
verdict = adaptive_verdict(
    score=latest_score,
    atr_pct=atr_14(d) / d["Close"].iloc[-1] * 100,
    close_series=d["Close"],
    regime=regime,
    whale_vetoed=adapter.latest_audit and adapter.latest_audit.get("whale_veto"),
    alignment_healthy=adapter.latest_audit and adapter.latest_audit.get("alignment_vetoes") == 0,
)

print(f"Verdict: {verdict['label']}")
print(f"Adaptive layers: {verdict.get('adaptive_layers_active', False)}")
```

### Integration with Existing Backtester

```python
from quant_engine_extensions import AdaptiveWeightManager

# Initialize weight manager
weight_mgr = AdaptiveWeightManager()

# In your backtest loop
for date in dates:
    # ... compute scores ...
    
    # Check if it's time to retrain weights
    if weight_mgr.should_retrain(date):
        retrain_result = weight_mgr.retrain(factors_df, prices, date)
        print(f"Weights retrained on {date}")
        print(f"Test Sharpe improvement: {retrain_result['improvement']:+.2f}")
    
    # Get current weights (updates after retrain)
    weights = weight_mgr.get_weights()
    
    # Use weights in scoring
    composite = (factors_df * weights).sum(axis=1)
```

---

## Monitoring and Validation

### Metrics to Track

1. **WFO Retraining**:
   - How often weights change
   - Train vs. test Sharpe to detect degradation
   - Which factors increase/decrease weight over time

2. **Volatility Normalization**:
   - % of names in high-vol regime
   - Score adjustment magnitude (how much z-score changes things)

3. **Alignment Vetoes**:
   - Veto rate (% of bars rejected)
   - Win rate improvement (backtests with alignment vs. without)
   - Average duration of misaligned periods

4. **Whale Gates**:
   - Veto frequency (how often triggered)
   - Returns on vetoed vs. non-vetoed entries
   - CMF distribution (is negative-CMF + high-volume actually predictive?)

### Example Metrics Snapshot

```python
# After running adaptive engine on historical data
stats = adaptive_engine.get_summary_stats()

print(f"Walk-Forward Retrains: {stats['wfo_retrains']}")
print(f"Alignment Vetoes: {stats['alignments_vetoed']}")
print(f"Whale Gates Triggered: {stats['whale_gates_triggered']}")
print(f"Score Audits Logged: {stats['score_audits_logged']}")
```

---

## Configuration Reference

### WalkForwardOptimizer

```python
WalkForwardOptimizer(
    retrain_freq="monthly",      # "monthly", "weekly", or days as int
    test_window=20,              # bars to validate on after retrain
    min_train=60,                # minimum training bars required
)
```

### VolatilityAdaptiveNormalizer

```python
VolatilityAdaptiveNormalizer(
    window=60,                   # rolling window for z-score
    vol_percentile_threshold=70, # annualized vol % threshold for "high-vol"
)
```

### SubFactorAlignmentFilter

```python
SubFactorAlignmentFilter(
    min_consensus=0.66,          # minimum fraction of factors that must agree
)
```

### WhaleFootprintGate

```python
WhaleFootprintGate(
    rvol_threshold=1.5,          # relative volume threshold
    cmf_threshold=0.05,          # Chaikin Money Flow threshold
    distribution_veto=True,      # veto on negative CMF + abnormal volume
    min_dollar_vol=500_000,      # minimum notional volume to trigger check
)
```

---

## FAQ

**Q: Will these improvements increase computational cost?**
A: Minimally. WFO runs once per month; z-score and alignment checks run every bar (negligible overhead). Whale metrics reuse existing volume data.

**Q: Can I use these improvements with existing backtests?**
A: Yes. The adaptive layers are optional (use_alignment=True/False, use_whale=True/False). You can test incrementally.

**Q: How do I know if these improvements are working?**
A: Compare Sharpe, win rate, and max drawdown before/after enabling each layer. Monitor the audit trail for veto frequencies.

**Q: What if a name stays misaligned for days?**
A: The alignment filter will suppress scores for the entire period. This is intentional — it flags conflicting signal regimes that often precede reversals.

**Q: Can whale metrics be gamed?**
A: Yes, like all technical metrics. Use CMF + RVOL as one signal among many, not the sole entry criterion. Institutional players can mask distribution with algorithmic buying.

---

## References

1. **Walk-Forward Analysis**: Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies*. Wiley.
2. **Chaikin Money Flow**: Created by Marc Chaikin; widely used institutional flow indicator.
3. **Z-Score Normalization**: Normalization technique for handling heteroskedasticity; common in factor models.
4. **Factor Alignment**: Inspired by "consensus scoring" in machine learning ensemble methods.

---

**Not financial advice. Backtests are in-sample and ignore slippage/fees. Past performance does not predict future returns.**

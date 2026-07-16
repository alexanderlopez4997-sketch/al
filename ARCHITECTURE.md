# Meridian Quant Engine - Modular Pipeline Architecture

## System Design

The engine is organized as a 5-stage data processing pipeline with explicit data flow:

```
Market Data  →  Regime Detector  →  Factor Analyzer  →  Risk Engine  →  Metrics
(Stage 1)       (Stage 2)           (Stage 3)          (Stage 4)       (Stage 5)
```

Each stage:
- Takes input data
- Performs computation
- Passes output to next stage
- All state is explicit (no hidden side effects)

## Stage Breakdown

### Stage 1: Market Data
Loads historical price data and computes technical indicators.

Input: Ticker + date range
Output: OHLCV data with 30+ indicators (EMA, RSI, MACD, Bollinger Bands, etc.)

Components:
- fetch() - download data from yfinance
- enrich() - add all technical indicators
- PipelineData.d - enriched dataframe

### Stage 2: Regime Detector
Classifies market state into BULL/BEAR/RANGE with confidence score.

Input: Enriched market data
Output: Regime classification + confidence + trend metrics

Components:
- detect_regime() - EMA-based classification
- PipelineData.regime - dict with regime, confidence, metrics

Rules:
- BULL: price above 50-EMA + EMA rising
- BEAR: price below 50-EMA + EMA falling
- RANGE: mixed signals, consolidating

### Stage 3: Factor Analyzer
Computes 4 technical factors and their predictive power.

Input: Enriched data + regime
Output: Factor scores + correlations + information ratios

Components:
- factor_matrix() - compute Direction, Momentum, Volume, MeanRev
- composite() - weighted average (adapts per regime)
- information_ratio() - predictive power per factor
- factor_correlation() - redundancy detection

Factors:
1. Direction - merged trend + structure (trend following)
2. Momentum - MACD + RSI (momentum chasing)
3. Volume - OBV + volume profile (accumulation/distribution)
4. MeanRev - Z-score (mean reversion)

### Stage 4: Risk Engine
Computes position sizing and risk metrics.

Input: Scored data + account parameters
Output: Position size + entry/stop/target + backtests

Components:
- position_size() - Kelly-adjusted shares + stop/target
- backtest() - historical strategy performance
- backtest_by_regime() - separate backtest per regime

Metrics:
- Position size (shares to buy)
- Stop loss (2x ATR below entry)
- Take profit target (2x ATR above entry)
- Risk dollar amount
- Backtest Sharpe ratio
- Slippage costs (0.10% round trip)

### Stage 5: Metrics & Verdict
Computes final verdict and conviction.

Input: All prior stage outputs
Output: BUY/HOLD/AVOID signal + conviction + final metrics

Components:
- verdict() - classify signal strength
- conviction() - multi-factor agreement
- Final output object

## Data Structure

```python
class PipelineData:
    ticker          # Symbol
    df              # Raw OHLCV
    d               # Enriched with indicators
    regime          # Market classification
    F               # Factor matrix
    comp            # Composite score
    account         # Account size
    position        # Position sizing
    backtest        # Historical performance
    backtest_by_regime  # Per-regime performance
```

## Pipeline Execution

```python
data = Pipeline.execute(ticker, df, account=100000, risk_pct=2.0)

# Returns fully populated PipelineData object with all 5 stages complete
```

## Usage

### Normal Report (traditional format)
```bash
python3 quant_engine.py NVDA --period 1y
```

### Pipeline View (modular format)
```bash
python3 quant_engine.py NVDA --period 1y --pipeline
```

Shows explicit stages:
```
STAGE 1: Market Data [501 bars loaded]
STAGE 2: Regime Detector [RANGE at 45% confidence]
STAGE 3: Factor Analyzer [4 factors with IR scores]
STAGE 4: Risk Engine [Position sizing for $100k account]
STAGE 5: Metrics & Verdict [Final BUY/HOLD/AVOID]
```

### With Position Sizing
```bash
python3 quant_engine.py NVDA --pipeline --account 100000 --risk 2
```

Adds to Stage 4:
- Account: $100,000
- Risk: 2.0% per trade
- Shares: [calculated]
- Stop: [2x ATR below entry]
- Target: [2x ATR above entry]

## Key Design Principles

1. **Explicit data flow** - Each stage clearly transforms data
2. **No hidden state** - All computations visible in PipelineData
3. **Modular** - Can swap/modify individual stages
4. **Testable** - Each stage function can be unit tested
5. **Transparent** - CLI shows stage-by-stage breakdown

## Extension Points

To add new analysis:

1. **Create a new stage class:**
```python
class CustomStage:
    @staticmethod
    def execute(data):
        # Transform data
        data.custom_output = compute_something(data)
        return data
```

2. **Insert in pipeline:**
```python
class Pipeline:
    @staticmethod
    def execute(ticker, df):
        data = MarketDataStage.execute(data)
        data = RegimeDetectorStage.execute(data)
        data = CustomStage.execute(data)  # NEW
        # ... rest of pipeline
        return data
```

3. **Add to report:**
```python
def report_pipeline(res, args):
    # ... existing stages
    print(f"CUSTOM: {res['custom_output']}")
```

## Performance

Pipeline overhead is minimal:
- Stage 1: O(n) - indicator computation
- Stage 2: O(n) - EMA calculations
- Stage 3: O(n² ) - correlations (small n=4 factors)
- Stage 4: O(n) - backtest
- Stage 5: O(1) - verdict logic

Total: < 1 second for typical stock analysis

## Future Enhancements

1. **Parallel stages** - Run independent metrics in parallel
2. **Streaming input** - Real-time updates as data arrives
3. **Plugin system** - Load custom stages from files
4. **State persistence** - Save/load pipeline state
5. **Visualization** - Render stage outputs as charts

## Backward Compatibility

Old analyze() function still works unchanged:
```python
res = analyze(ticker, df, interval, weights)
# Returns: dict with all outputs as before
```

New analyze_pipeline() is available for explicit stage access:
```python
res = analyze_pipeline(ticker, df, interval, account, risk)
# Returns: structured dict with stage breakdowns
```

Both can coexist - choose based on use case.

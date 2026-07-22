# Portfolio-Level Risk Management & Execution-Safety System

## Overview

A production-grade portfolio risk management module providing real-time correlation monitoring, circuit-breaker automation, and trailing stop management for algorithmic trading engines.

**Components:**
1. **DynamicCorrelationGate** - 60-day rolling Pearson correlations with sector exposure caps
2. **AutomatedCircuitBreaker** - Intraday (-3%) and rolling weekly (-5%) drawdown triggers
3. **TrailingStopManager** - Break-even elevation and structural floor trailing
4. **PortfolioRiskManager** - Unified orchestrator integrating all components

---

## Core Architecture

### DynamicCorrelationGate

Enforces hard portfolio exposure caps via rolling correlation monitoring.

**Key Features:**
- 60-day rolling Pearson correlation matrix (vectorized Pandas)
- Per-sector exposure tracking with 3% portfolio heat limit
- Pairwise correlation threshold (0.65) triggers position scaling
- Automatic NaN handling with forward/backward fill imputation
- O(n²) correlation computation, < 5ms for 50-ticker universe

**Usage:**
```python
gate = DynamicCorrelationGate(
    max_sector_exposure_pct=3.0,
    correlation_threshold=0.65,
    lookback_bars=60,
)

gate.register_ticker("AAPL", "Technology")
gate.update_price("AAPL", 150.25, datetime.now())

decision = gate.evaluate_position(
    ticker="MSFT",
    proposed_shares=100,
    current_price=300.0,
    portfolio_equity=100000,
    existing_positions={...}
)

if decision.approved:
    # Execute trade with decision.max_shares
    trade_size = decision.max_shares
```

**Decision Output:**
- `approved`: Boolean approval
- `max_shares`: Scaled position size (0 = veto)
- `exposure_ratio`: Current exposure / limit
- `limiting_factor`: "sector", "correlation", "both", or "none"
- `warnings`: List of gate violations

---

### AutomatedCircuitBreaker

Real-time portfolio drawdown monitoring with automatic kill switch.

**Triggers:**
- **Intraday**: Drawdown >= -3.0% from session start
- **Weekly**: Drawdown >= -5.0% from weekly open
- **Dual**: Both conditions simultaneously (escalated)

**Actions on Trip:**
- Cancel all pending entry orders
- Lock out new signal generation
- Transition to RISK-OFF state
- Log critical alert

**State Transitions:**
```
ACTIVE → CIRCUIT_BREAKER (on threshold breach)
         ↓
      RISK-OFF (orders blocked, signals locked)
         ↓
      RECOVERY (manual reset or auto-recovery if enabled)
         ↓
      ACTIVE
```

**Usage:**
```python
cb = AutomatedCircuitBreaker(
    intraday_drawdown_threshold=-3.0,
    weekly_drawdown_threshold=-5.0,
    auto_recovery_enabled=False,  # Requires explicit reset
)

cb.initialize_session(starting_equity=100000)

# On each update (e.g., hourly)
state, trip_reason = cb.update(current_equity, timestamp)

if cb.is_locked():
    # Reject new signals
    return RISK_OFF

# Manual reset (requires user action)
cb.manual_reset()
```

---

### TrailingStopManager

Stateful stop-loss elevation and trailing logic per position.

**Three-Stage Stop Management:**

**Stage 1: Break-Even Elevation**
- Initial stop at 2x ATR below entry
- Once price moves favorably by +1x ATR → elevate to entry (break-even)
- Prevents catastrophic losses on early reversal

**Stage 2: Structural Trailing**
- After break-even set, trail stop beneath structural floor (EMA20, Supertrend)
- Trail offset: 0.5x ATR below floor
- Dynamic floor tracking as price action evolves

**Stage 3: Volatility Adaptation**
- If volatility spikes (current ATR > 1.2x baseline) → widen stop proportionally
- Cap widening at 5% of entry price
- Prevents false stops during gap scenarios

**Usage:**
```python
tsm = TrailingStopManager(
    breakeven_atr_multiple=1.0,
    trail_atr_multiple=0.5,
    max_stop_widening_pct=5.0,
)

# Register position
tsm.add_position(
    ticker="AAPL",
    entry_price=150.0,
    initial_stop=145.0,
    structural_floor=146.0,
    atr=2.5,
)

# On each bar
new_stop, reason = tsm.update_price(
    ticker="AAPL",
    current_price=152.0,
    atr=2.55,
    structural_floor=151.0,
    timestamp=now,
)

if new_stop is not None:
    # Stop was adjusted, update order
    update_stop_order(new_stop)
    log_adjustment(reason)
```

**Adjustment Reasons:**
- `BREAK_EVEN`: Price moved +1x ATR, elevated to entry
- `STRUCTURAL_TRAIL`: Trailing beneath floor
- `VOLATILITY_SPIKE`: Widened for vol increase

---

### PortfolioRiskManager

Unified orchestrator combining all three systems.

**Responsibilities:**
- Position lifecycle management
- Real-time P&L tracking
- Portfolio-wide metrics computation
- Single entry point for all risk checks

**Workflow:**

```python
prm = PortfolioRiskManager(
    initial_equity=100000,
    max_sector_exposure=3.0,
    intraday_drawdown_limit=-3.0,
    weekly_drawdown_limit=-5.0,
)

# Register universe
watchlist = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "JPM": "Financials",
}
prm.register_watchlist(watchlist)

# Evaluate new signal
gate_result = prm.evaluate_new_signal(
    ticker="NVDA",
    proposed_shares=100,
    entry_price=850.0,
    portfolio_equity=100000,
)

if gate_result.approved:
    # Add position with stop management
    prm.add_position(
        ticker="NVDA",
        shares=gate_result.max_shares,
        entry_price=850.0,
        stop_price=840.0,
        structural_floor=842.0,
        current_price=850.0,
        sector="Technology",
        atr=8.5,
    )

# On each update cycle (hourly, daily, etc.)
metrics = prm.update_metrics(datetime.now())

if metrics.circuit_breaker_reason:
    log_critical(f"CB TRIP: {metrics.circuit_breaker_reason.value}")
    reject_new_signals()

# Update active position
valid, stop_event = prm.update_position_price(
    ticker="NVDA",
    current_price=855.0,
    atr=8.6,
    structural_floor=843.0,
    timestamp=datetime.now(),
)

if not valid:
    # Stop was hit, exit position
    close_position("NVDA")
    prm.close_position("NVDA")
elif stop_event:
    # Stop was adjusted
    update_broker_stop_order(stop_event.new_stop)
    log_adjustment(stop_event.reason)

# Get risk snapshot
report = prm.get_risk_report()
print(f"Portfolio: ${report['portfolio_equity']:,.2f}")
print(f"Drawdown: {report['intraday_drawdown']}%")
print(f"Engine State: {report['engine_state']}")
```

---

## Integration with Async Pipeline

### Event Loop Integration

```python
import asyncio
from portfolio_risk_manager import PortfolioRiskManager

class RiskEngine:
    def __init__(self):
        self.prm = PortfolioRiskManager(
            initial_equity=100000,
            max_sector_exposure=3.0,
        )
    
    async def evaluate_signal(self, ticker, shares, price):
        """Evaluate new signal against risk gates."""
        gate_result = self.prm.evaluate_new_signal(
            ticker=ticker,
            proposed_shares=shares,
            entry_price=price,
            portfolio_equity=self.prm.current_equity,
        )
        
        if not gate_result.approved:
            return None  # Signal vetoed
        
        return gate_result.max_shares
    
    async def update_positions(self, price_data):
        """Update all positions and check circuit breaker."""
        for ticker, (price, atr, floor) in price_data.items():
            if ticker in self.prm.positions:
                valid, event = self.prm.update_position_price(
                    ticker=ticker,
                    current_price=price,
                    atr=atr,
                    structural_floor=floor,
                    timestamp=datetime.now(),
                )
                
                if not valid:
                    await self.close_position(ticker)
                elif event:
                    await self.adjust_stop_order(event)
        
        # Update metrics and check circuit breaker
        metrics = self.prm.update_metrics(datetime.now())
        
        if metrics.circuit_breaker_reason:
            await self.handle_circuit_break(metrics)

# Usage in async main
async def main():
    engine = RiskEngine()
    
    while True:
        # Fetch latest prices
        prices = await fetch_market_data()
        
        # Update portfolio risk
        await engine.update_positions(prices)
        
        # Generate and evaluate new signals
        signals = await generate_signals()
        for signal in signals:
            max_shares = await engine.evaluate_signal(
                signal.ticker,
                signal.shares,
                signal.price,
            )
            if max_shares:
                await place_order(signal.ticker, max_shares, signal.price)
        
        await asyncio.sleep(60)  # Update cycle

asyncio.run(main())
```

---

## Type Safety & Logging

### Structured Logging

All state transitions and risk events are logged with context:

```
2024-07-22 10:30:45 | PortfolioRiskManager | INFO | Position added: AAPL | shares=100 | entry=$150.00 | stop=$145.00
2024-07-22 10:35:12 | TrailingStopManager | INFO | Adjusted AAPL | old=145.00 -> new=150.00 | reason=break_even
2024-07-22 11:45:30 | CircuitBreaker | CRITICAL | CIRCUIT BREAKER TRIGGERED (INTRADAY) | dd=-3.15%
2024-07-22 11:45:30 | PortfolioRiskManager | WARNING | Stop hit: MSFT | price=$295.50 | stop=$295.00
```

### Type Hints

Comprehensive type annotations throughout:

```python
def evaluate_position(
    self,
    ticker: str,
    proposed_shares: float,
    current_price: float,
    portfolio_equity: float,
    existing_positions: Dict[str, Position],
) -> CorrelationGateDecision:
    """..."""
```

---

## Edge Case Handling

### Handled Scenarios

1. **Missing Data Feeds**
   - Correlation matrix returns None if insufficient bars
   - Position updates gracefully handle missing tickers
   - Metrics gracefully handle zero positions

2. **NaN Correlation Matrices**
   - Forward/backward fill imputation
   - Pairwise correlation ignores NaN pairs
   - Logs warnings for data quality issues

3. **Extreme High-Volatility Gaps**
   - Circuit breaker captures gap-down stop hits
   - Volatility spike detection prevents false stops
   - Position validity checked before any update

4. **Zero/Negative Equity**
   - Exposure calculations handle zero divisors
   - Drawdown calculations bounded to [0, 100%]
   - Risk gates fail-safe to veto positions

5. **Simultaneous Gate Violations**
   - Sector AND correlation violations both logged
   - Position scaled to strictest constraint
   - Multiple warnings returned

---

## Performance Characteristics

| Operation | Complexity | Latency |
|-----------|-----------|---------|
| Position registration | O(1) | < 1ms |
| Price update | O(1) | < 1ms |
| Circuit breaker check | O(1) | < 1ms |
| Stop adjustment | O(1) | < 1ms |
| Correlation matrix (50 tickers) | O(n²) | < 5ms |
| Portfolio metrics | O(n) | < 2ms |
| Gate evaluation | O(n) | < 3ms |

**Total update cycle for 50-position portfolio:**
< 15ms (per-second updates feasible)

---

## Configuration & Tuning

### Risk Parameters (Configurable)

```python
prm = PortfolioRiskManager(
    initial_equity=500000,
    
    # Sector concentration cap (default 3%)
    max_sector_exposure=2.5,
    
    # Circuit breaker thresholds (default -3% intraday, -5% weekly)
    intraday_drawdown_limit=-2.5,  # More conservative
    weekly_drawdown_limit=-4.0,
)

# Gate tuning
gate = DynamicCorrelationGate(
    max_sector_exposure_pct=3.0,
    correlation_threshold=0.70,  # Stricter correlation gate
    lookback_bars=90,  # Longer history window
)

# Stop management tuning
tsm = TrailingStopManager(
    breakeven_atr_multiple=1.5,  # Require +1.5 ATR before elevation
    trail_atr_multiple=0.3,  # Tighter trailing (0.3x ATR)
    max_stop_widening_pct=3.0,  # Conservative vol widening
)
```

### Recommended Defaults

**Conservative (< $50k account):**
- max_sector_exposure: 2.0%
- intraday_drawdown: -2.5%
- weekly_drawdown: -4.0%

**Moderate ($50k - $500k):**
- max_sector_exposure: 3.0%
- intraday_drawdown: -3.0%
- weekly_drawdown: -5.0%

**Aggressive (> $500k):**
- max_sector_exposure: 3.5%
- intraday_drawdown: -4.0%
- weekly_drawdown: -6.0%

---

## Testing

### Test Coverage

- **Unit Tests**: 40+ test cases covering all components
- **Edge Cases**: Missing data, NaN matrices, zero equity, extreme gaps
- **Integration**: Full portfolio lifecycle from signal to exit
- **Performance**: Latency validation for 50-ticker universes

### Running Tests

```bash
python3.12 test_portfolio_risk_manager.py -v
```

Or with unittest:

```bash
python3.12 -m unittest test_portfolio_risk_manager -v
```

### Test Categories

- `TestDynamicCorrelationGate` - Gate logic and edge cases
- `TestAutomatedCircuitBreaker` - Trigger conditions and recovery
- `TestTrailingStopManager` - Stop elevation and trailing
- `TestPortfolioRiskManager` - Integration and lifecycle
- `TestEdgeCases` - Boundary conditions
- `TestPerformance` - Latency verification

---

## Integration Points

### With Existing Modules

**quant_engine.py:**
```python
import portfolio_risk_manager as prm

# Evaluate signal before entry
prm = prm.PortfolioRiskManager(initial_equity=account_size)
gate = prm.evaluate_new_signal(ticker, shares, entry_price, equity)

if gate.approved:
    execute_trade(ticker, gate.max_shares)
```

**stops.py:**
```python
# Use TrailingStopManager for dynamic stops instead of fixed ATR multipliers
tsm = TrailingStopManager()
tsm.add_position(ticker, entry, initial_stop, floor, atr)

# Update on each bar
new_stop, reason = tsm.update_price(ticker, price, atr, floor, now)
if new_stop:
    update_order(new_stop)
```

**web_server.py / quant_gui.py:**
```python
# Display risk dashboard
metrics = prm.update_metrics(now)
display_metrics({
    'equity': metrics.total_equity,
    'drawdown': metrics.intraday_drawdown_pct,
    'positions': metrics.open_positions_count,
    'engine_state': metrics.engine_state.value,
    'max_exposure': metrics.max_sector_exposure_pct,
})
```

---

## Troubleshooting

### Circuit Breaker Trips Unexpectedly

1. Check portfolio equity calculation (includes unrealized P&L)
2. Verify session initialization: `circuit_breaker.initialize_session()`
3. Review drawdown threshold settings (default -3% intraday)
4. Check timestamp continuity (weekly reset on Monday)

### Correlation Matrix Returns None

1. Insufficient bars: need 60+ for default lookback
2. Insufficient tickers: need 2+ for correlation
3. Check for all-NaN price series (missing data)
4. Verify prices are numeric (not strings)

### Stops Not Adjusting

1. Position must be in TrailingStopManager: `add_position()`
2. Break-even requires +1x ATR favorable move (default)
3. Structural trailing only activates after break-even
4. Check `update_price()` is called each bar

### High Latency on Updates

1. Correlation matrix computed every call: cache if static
2. 50+ tickers may need batched updates
3. Use Numba JIT for large universes (future optimization)
4. Profile with cProfile for bottleneck identification

---

## Future Enhancements

1. **Parallel Risk Checks** - Vectorized gate evaluation for 100+ tickers
2. **Machine Learning** - Learn correlation thresholds from historical data
3. **Options-Aware** - Extend to option positions and Greeks
4. **VaR/CVaR** - Value-at-Risk and Conditional VaR monitoring
5. **Regime Adaptation** - Adjust parameters based on market regime (bull/bear/range)
6. **Graphical Dashboard** - Real-time risk visualization (web or desktop)

---

## References

- **Correlation**: Pearson correlation coefficient, rolling window
- **Circuit Breaker**: Maximum Adverse Excursion (MAE), Drawdown monitoring
- **Trailing Stops**: Average True Range (ATR), Structural Support/Resistance
- **Position Sizing**: Kelly Criterion, Risk-Per-Trade methodology

---

## License & Attribution

Part of Meridian Quant Engine. Production-grade for live trading with strict risk controls.

For questions or issues: Create an issue in the repository.

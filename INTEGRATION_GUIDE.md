# Portfolio Risk Manager — Integration Guide

How to use the portfolio risk management system in your trading workflows.

## Quick Start

### 1. Basic Usage (Standalone)

```python
from portfolio_risk_integration import RiskEngine
from datetime import datetime

# Initialize
engine = RiskEngine(
    initial_equity=100000,
    max_sector_exposure=3.0,
    intraday_drawdown_limit=-3.0,
    weekly_drawdown_limit=-5.0,
)

# Register universe
engine.register_universe({
    "AAPL": "Technology",
    "MSFT": "Technology",
    "JPM": "Financials",
})

# Evaluate signal before entry
result = engine.evaluate_signal(
    ticker="AAPL",
    proposed_shares=100,
    entry_price=150.0,
)

if result.approved:
    # Execute trade
    engine.add_position(
        ticker="AAPL",
        shares=result.max_shares,  # Use scaled size
        entry_price=150.0,
        current_price=150.0,
        atr=2.5,
        structural_floor=146.0,
    )
    print(f"✓ Executed {result.max_shares} shares")
else:
    # Signal rejected
    print(f"✗ Signal rejected: {result.limiting_factor}")
    for warning in result.warnings:
        print(f"  - {warning}")

# Update on each bar (hourly, daily, etc.)
metrics = engine.update(
    price_data={"AAPL": 152.0, "MSFT": 305.0},
    timestamp=datetime.now(),
)

if metrics["circuit_breaker_tripped"]:
    print("⚠ Circuit breaker tripped!")
    print(f"  Reason: {metrics['circuit_breaker_reason']}")
    pause_trading()

# Close position
engine.close_position("AAPL")
```

### 2. Integration with quant_engine.py

```python
import quant_engine as qe
from portfolio_risk_integration import RiskEngine

# Initialize risk engine
risk_engine = RiskEngine(initial_equity=100000)
risk_engine.register_universe({...})

# Analyze stock
res = qe.analyze("AAPL", df, interval="1d", weights=qe.BASE_WEIGHTS)

# Gate the verdict through risk manager
verdict = qe.verdict(
    score=res["score"],
    atr_pct=res.get("atr_pct", 0),
    regime=res.get("regime"),
    risk_engine=risk_engine,
    ticker="AAPL",
    entry_price=res["last"],
    proposed_shares=100,
)

# Check for veto
if verdict.get("risk_gate_veto"):
    print(f"Signal gated: {verdict['limiting_factor']}")
    print(f"Max allowed: {verdict.get('max_shares', 0)} shares")
else:
    print(f"✓ {verdict['label']}")
```

### 3. Integration with Live Trading

```python
import asyncio
from portfolio_risk_integration import RiskEngine
from datetime import datetime

class TradingBot:
    def __init__(self, account_equity=100000):
        self.risk_engine = RiskEngine(initial_equity=account_equity)
        self.risk_engine.register_universe({
            "AAPL": "Technology",
            "MSFT": "Technology",
            # ... more tickers
        })
    
    async def on_signal(self, ticker, score, entry_price, atr):
        """Process new signal with risk gating."""
        
        # Check if circuit breaker is active
        if self.risk_engine.is_circuit_breaker_active():
            print(f"⚠ Trading paused: CB active")
            return
        
        # Evaluate signal
        result = self.risk_engine.evaluate_signal(
            ticker=ticker,
            proposed_shares=100,
            entry_price=entry_price,
        )
        
        if result.approved:
            # Execute trade
            shares = result.max_shares
            await self.broker.place_order(ticker, shares, "BUY", entry_price)
            
            # Track position
            self.risk_engine.add_position(
                ticker=ticker,
                shares=shares,
                entry_price=entry_price,
                current_price=entry_price,
                atr=atr,
            )
        else:
            # Log veto
            print(f"Signal rejected ({result.limiting_factor}):")
            for w in result.warnings:
                print(f"  - {w}")
    
    async def on_bar(self, prices, atr_data):
        """Update portfolio on each bar."""
        
        # Build price dict
        price_dict = {t: p for t, p in prices.items()}
        
        # Update positions
        metrics = self.risk_engine.update(price_dict)
        
        # Handle stops
        for ticker in list(self.risk_engine.get_positions().keys()):
            valid, new_stop, reason = self.risk_engine.update_position(
                ticker=ticker,
                current_price=prices[ticker],
                atr=atr_data[ticker],
                structural_floor=prices[ticker] - atr_data[ticker],
            )
            
            if not valid:
                # Stop hit - close position
                await self.broker.close_position(ticker)
                self.risk_engine.close_position(ticker)
            elif new_stop:
                # Update stop order
                await self.broker.update_stop(ticker, new_stop)
        
        # Check circuit breaker
        if metrics["circuit_breaker_tripped"]:
            print(f"⚠ CIRCUIT BREAKER: {metrics['circuit_breaker_reason']}")
            await self.broker.close_all()
            await self.pause_trading()
    
    async def pause_trading(self):
        """Pause trading and require manual reset."""
        print("⚠ Trading paused. Reason: Circuit breaker active")
        print("Portfolio metrics:")
        status = self.risk_engine.get_status()
        for key, val in status.items():
            print(f"  {key}: {val}")
        
        # Require explicit reset
        input("Press Enter to reset circuit breaker: ")
        self.risk_engine.reset_circuit_breaker()
        print("✓ Circuit breaker reset")

# Usage
async def main():
    bot = TradingBot(account_equity=100000)
    
    # Simulate signals
    await bot.on_signal("AAPL", score=25.0, entry_price=150.0, atr=2.5)
    
    # Simulate bar updates
    await bot.on_bar(
        prices={"AAPL": 152.0, "MSFT": 305.0},
        atr_data={"AAPL": 2.6, "MSFT": 5.1},
    )

asyncio.run(main())
```

---

## Configuration by Account Size

### Conservative (< $50k)

```python
engine = RiskEngine(
    initial_equity=50000,
    max_sector_exposure=2.0,      # Tight sector cap
    intraday_drawdown_limit=-2.5, # Aggressive CB trigger
    weekly_drawdown_limit=-4.0,
)
```

**Best for:** New accounts, risk-averse traders

### Moderate ($50k - $500k)

```python
engine = RiskEngine(
    initial_equity=250000,
    max_sector_exposure=3.0,      # Default
    intraday_drawdown_limit=-3.0,
    weekly_drawdown_limit=-5.0,
)
```

**Best for:** Most traders

### Aggressive (> $500k)

```python
engine = RiskEngine(
    initial_equity=1000000,
    max_sector_exposure=3.5,      # Looser sector cap
    intraday_drawdown_limit=-4.0, # Higher CB threshold
    weekly_drawdown_limit=-6.0,
)
```

**Best for:** Experienced, well-capitalized traders

---

## API Reference

### RiskEngine Methods

#### `evaluate_signal(ticker, proposed_shares, entry_price) → CorrelationGateDecision`

Evaluate if a new signal passes all risk gates.

```python
result = engine.evaluate_signal("AAPL", 100, 150.0)

# result.approved: bool - Signal approval
# result.max_shares: float - Scaled position size
# result.limiting_factor: str - "sector", "correlation", "circuit_breaker", etc.
# result.warnings: List[str] - Gate violation details
```

#### `add_position(ticker, shares, entry_price, current_price, atr, structural_floor=None)`

Register a new position for tracking.

```python
engine.add_position(
    ticker="AAPL",
    shares=100,
    entry_price=150.0,
    current_price=150.0,
    atr=2.5,
    structural_floor=146.0,  # Optional: EMA20, Supertrend, etc.
)
```

#### `update_position(ticker, current_price, atr, structural_floor, timestamp=None) → (valid, new_stop, reason)`

Update position with current price action. Returns stop adjustment if applicable.

```python
valid, new_stop, reason = engine.update_position(
    ticker="AAPL",
    current_price=152.0,
    atr=2.6,
    structural_floor=151.0,
)

if not valid:
    close_position("AAPL")
elif new_stop:
    update_broker_stop(new_stop)
    print(f"Stop adjusted ({reason}): {new_stop}")
```

#### `update(price_data, timestamp=None) → Dict`

Update all positions and compute metrics.

```python
metrics = engine.update(
    {"AAPL": 152.0, "MSFT": 305.0},
    datetime.now(),
)

# Returns:
# {
#   "equity": 101234.56,
#   "unrealized_pnl": 1234.56,
#   "intraday_drawdown": -1.5,
#   "weekly_drawdown": -2.3,
#   "positions": 2,
#   "max_sector_exposure": 2.8,
#   "engine_state": "ACTIVE",
#   "circuit_breaker_tripped": False,
#   "circuit_breaker_reason": None,
# }
```

#### `close_position(ticker)`

Close out a position.

```python
engine.close_position("AAPL")
```

#### `is_circuit_breaker_active() → bool`

Check if circuit breaker is locked (blocks new signals).

```python
if engine.is_circuit_breaker_active():
    print("Trading paused")
```

#### `reset_circuit_breaker()`

Manually reset circuit breaker (requires explicit action).

```python
engine.reset_circuit_breaker()
```

#### `get_status() → Dict`

Get portfolio status snapshot.

```python
status = engine.get_status()
print(f"Equity: ${status['portfolio_equity']}")
print(f"State: {status['engine_state']}")
```

#### `get_positions() → Dict`

Get all active positions with current P&L.

```python
positions = engine.get_positions()
# {
#   "AAPL": {
#     "shares": 100,
#     "entry": 150.0,
#     "current": 152.0,
#     "stop": 145.0,
#     "pnl": 200.0,
#     "pnl_pct": 1.33,
#   },
#   ...
# }
```

#### `get_stop_price(ticker) → Optional[float]`

Get current stop price for a position.

```python
stop = engine.get_stop_price("AAPL")
print(f"Stop: ${stop}")
```

---

## Event Handling

### Circuit Breaker Trip

```python
metrics = engine.update(prices)

if metrics["circuit_breaker_tripped"]:
    reason = metrics["circuit_breaker_reason"]
    # "intraday_drawdown" - Intraday hit -3%
    # "weekly_drawdown" - Weekly hit -5%
    # "dual_trigger" - Both triggered simultaneously
    
    print(f"Trading halted: {reason}")
    
    # Close all positions or move to defensive mode
    for ticker in engine.get_positions():
        close_position(ticker)
        engine.close_position(ticker)
    
    # Require manual reset
    engine.reset_circuit_breaker()
```

### Stop Adjustment

```python
valid, new_stop, reason = engine.update_position(ticker, price, atr, floor)

if new_stop:
    # reason: "break_even", "trail", "volatility_spike"
    log_event({
        "event": "stop_adjusted",
        "ticker": ticker,
        "old_stop": position.stop_price,
        "new_stop": new_stop,
        "reason": reason,
    })
    
    # Update broker
    broker.update_stop_order(ticker, new_stop)
```

### Signal Gate Veto

```python
# In quant_engine verdict:
verdict = qe.verdict(..., risk_engine=engine, ...)

if verdict.get("risk_gate_veto"):
    print(f"Signal gated: {verdict['limiting_factor']}")
    for warning in verdict.get("gate_warnings", []):
        print(f"  - {warning}")
    
    # Log for analysis
    log_event({
        "event": "signal_gated",
        "ticker": ticker,
        "limiting_factor": verdict["limiting_factor"],
        "max_allowed": verdict["max_shares"],
    })
```

---

## Monitoring & Dashboards

### Real-Time Metrics

```python
def display_dashboard(engine):
    status = engine.get_status()
    positions = engine.get_positions()
    
    print("=" * 70)
    print(f"Portfolio: ${status['portfolio_equity']:,.2f}")
    print(f"P&L: ${status['unrealized_pnl']:+,.2f} ({status['win_rate']:.1%})")
    print(f"Drawdown: {status['intraday_drawdown']:.2f}% | {status['weekly_drawdown']:.2f}%")
    print(f"State: {status['engine_state']} | Positions: {len(positions)}")
    print("=" * 70)
    
    for ticker, pos in positions.items():
        print(f"{ticker:6s} | {pos['shares']:5.0f}@ | "
              f"Entry ${pos['entry']:7.2f} | Now ${pos['current']:7.2f} | "
              f"Stop ${pos['stop']:7.2f} | P&L ${pos['pnl']:+7.2f} ({pos['pnl_pct']:+5.2f}%)")
```

---

## Troubleshooting

### Circuit Breaker Triggers Too Often

**Issue:** Portfolio keeps hitting -3% drawdown threshold.

**Solution:** Adjust thresholds for account size/risk tolerance:

```python
engine = RiskEngine(
    initial_equity=100000,
    intraday_drawdown_limit=-5.0,  # Raise to -5%
    weekly_drawdown_limit=-7.0,
)
```

### Signals Being Rejected by Gate

**Issue:** Good setups are being vetoed by correlation gate.

**Solution:** Check sector concentration and correlations:

```python
result = engine.evaluate_signal("AAPL", 100, 150.0)
print(f"Limiting factor: {result.limiting_factor}")
for warning in result.warnings:
    print(f"  {warning}")

# Reduce position size or wait for correlation to drop
result2 = engine.evaluate_signal("AAPL", 50, 150.0)  # Smaller size
```

### Stops Not Adjusting

**Issue:** Trailing stops not moving even with favorable price action.

**Solution:** Ensure `update_position()` is called on each bar:

```python
# ✓ Correct: Called for every bar
for bar in price_data:
    valid, new_stop, reason = engine.update_position(ticker, bar.close, bar.atr, bar.floor)

# ✗ Wrong: Only called on close
if is_market_close:
    engine.update_position(...)
```

---

## Performance Tips

1. **Batch Updates**: Update all positions at once vs. individually
   ```python
   # ✓ Good: Single update call
   metrics = engine.update(all_prices)
   
   # ✗ Wasteful: N update calls
   for ticker in tickers:
       engine.update_position(ticker, ...)
   ```

2. **Cache Metrics**: Don't recompute frequently
   ```python
   metrics = engine.update(prices)
   # Use metrics for multiple checks
   if metrics["circuit_breaker_tripped"]:
       ...
   if metrics["equity"] < threshold:
       ...
   ```

3. **Lazy Correlation**: Skip correlation matrix if numpy unavailable
   ```python
   # Automatically skipped if numpy/pandas not installed
   # Gate still works via sector concentration checks
   ```

---

## See Also

- `PORTFOLIO_RISK_SYSTEM.md` - Complete system documentation
- `portfolio_risk_manager.py` - Core implementation
- `portfolio_risk_integration.py` - Integration layer
- `test_portfolio_risk_minimal.py` - Test examples

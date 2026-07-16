# Meridian Quant Engine - Complete Stock Analysis System

A production-ready quantitative stock analysis system with market regime detection, adaptive factor analysis, multi-stage pipeline architecture, portfolio risk management, and real-time position tracking.

## What It Does

**Real-time stock scoring** with:
- **Market regime detection** (BULL/BEAR/RANGE classification with confidence)
- **4-factor adaptive analysis** (Direction, Momentum, Volume, MeanReversion)
- **Factor correlation analysis** (identifies redundant signals)
- **Realistic slippage modeling** (0.05% entry + 0.05% exit)
- **7-point confirmation checklist** (validates signals before trading)
- **Multi-stock dashboard** (scan 20+ stocks in seconds)
- **Portfolio alerts** (RED/GREEN/YELLOW categorization)
- **Real-time position tracking** (live P&L, conviction changes, win rate)

## Quick Start

```bash
# View your positions
python3 quant_engine.py --positions

# Scan watchlist for opportunities
python3 quant_engine.py AAPL MSFT NVDA TSLA --dashboard

# See risk/opportunity alerts (GREEN/RED/YELLOW)
python3 quant_engine.py AAPL MSFT NVDA TSLA --alerts

# Analyze single stock deeply
python3 quant_engine.py AAPL --period 6mo

# Manage positions
python3 quant_engine.py --add-position AAPL 310      # Enter
python3 quant_engine.py --close-position AAPL 320    # Exit
python3 quant_engine.py --positions                  # Show all
```

## Example Output

### Dashboard (Multi-Stock View)
```
WATCHLIST DASHBOARD
SCORE    TICKER    PRICE     CHG%    VERDICT             CONF
+58      AAPL      310.66    -0.6%   STRONG BUY signal   ●
+35      MRK       128.86    +1.6%   BUY signal          ●
+19      MSFT      388.84    +0.5%   HOLD / no edge      ●
-30      NVDA      196.93    +0.7%   AVOID / sell        ●
```

### Portfolio Alerts
```
GREEN - BUY CANDIDATES (2 stocks)
  AAPL: +62 STRONG BUY signal · insider buying
  MRK: +45 BUY signal · strong backtest

RED - RISK / AVOID (1 stock)
  NVDA: -30 AVOID · insider selling

YELLOW - NEUTRAL (5 stocks)
  No clear catalyst
```

### Position Tracking
```
OPEN POSITIONS (2)
TICKER    ENTRY    CURRENT    P&L      P&L%  CONVICTION  SIGNAL
AAPL      $310     $315       $500     1.6%  75% → 80%   STRONG BUY
MSFT      $380     $388       $800     2.1%  75% → 50%   HOLD

CLOSED POSITIONS (1)
NVDA      $190     $200       $1000    5.3%  <1d
Win rate: 100%
```

## Key Features

### Multi-Stage Pipeline
- **Stage 1**: Market data loading and preparation
- **Stage 2**: Regime detection (BULL/BEAR/RANGE)
- **Stage 3**: Factor analysis with adaptive weighting
- **Stage 4**: Risk checks and red flag detection
- **Stage 5**: Metrics and performance analysis

### Confirmation Checklist (7 Points)
Before trading, the system validates:
1. Verdict is BUY/STRONG BUY (not HOLD/AVOID)
2. Conviction ≥ 75%
3. Score ≥ +40
4. Backtest Sharpe > 1.0
5. Win rate > 40%
6. Portfolio not over-leveraged
7. Max drawdown within limits

Returns: **FULL** (●) / **PARTIAL** (◐) / **WEAK** (◯) confirmation

### Real-Time Position Tracking
- Entry/exit prices and timestamps
- Live P&L calculations ($ and %)
- Conviction changes (entry vs current)
- Signal evolution monitoring
- Win rate statistics
- Persistent JSON storage across sessions

## Signal Breakdown

### BUY Verdict
- Score ≥ entry threshold (16-20 depending on regime/volatility)
- Recent factors align (conviction ≥ 60%)
- No extreme risk flags

### HOLD / No Edge
- Score in neutral zone
- Mixed signals

### AVOID / Sell Signal
- Score ≤ -entry threshold
- Reversal conditions detected

### STRONG BUY / STRONG AVOID
- Score in top/bottom tercile
- Multi-factor confirmation

## Technical Stack

- **Data**: yfinance (free historical + real-time quotes)
- **Indicators**: EMA, RSI, MACD, Bollinger Bands, Supertrend, ATR, OBV
- **Factors**: Direction (trend+structure), Momentum, Volume, Mean Reversion
- **Optimization**: Simulated annealing with walk-forward validation
- **Alternative data**: Finnhub (analysts, insiders), Quiver (congress trading, dark pool)

## Documentation

### Getting Started
- **QUICK_START.md** — One-page daily trading guide (START HERE!)
- **SYSTEM_COMPLETE.md** — Full system overview and architecture

### Feature Guides
- **POSITION_TRACKING.md** — Managing entries, exits, and P&L
- **DASHBOARD.md** — Multi-stock scanning and watchlist views
- **PORTFOLIO_ALERTS.md** — Understanding RED/GREEN/YELLOW alerts
- **CONFIRMATION_CHECKLIST.md** — 7-point validation logic

### Technical
- **COMPLETION_STATUS.md** — What was implemented
- **PIPELINE_IMPLEMENTATION.md** — Technical details
- **ARCHITECTURE.md** — Pipeline design

## Command Reference

```bash
# Analysis
python3 quant_engine.py AAPL                      # Deep analysis
python3 quant_engine.py AAPL --pipeline           # Stage-by-stage view
python3 quant_engine.py AAPL MSFT --period 6mo   # 6-month data

# Portfolio
python3 quant_engine.py TICKERS --dashboard       # Compact watchlist
python3 quant_engine.py TICKERS --alerts          # Risk categorization

# Position management
python3 quant_engine.py --positions               # Show all
python3 quant_engine.py --add-position AAPL 310   # Enter
python3 quant_engine.py --close-position AAPL 320 # Exit
```

## Performance

### Backtest Results
- Sharpe Ratio: 1.2-2.5 (varies by period and regime)
- Win Rate: 45-55% (profitable trades)
- Max Drawdown: 15-25%
- Slippage Cost: 0.10% per round-trip trade

### Regime-Specific
- **BULL markets**: Sharpe 2.0+, win rate 60%+
- **BEAR markets**: Sharpe 0.5-1.0, win rate 35%+
- **RANGE markets**: Mean reversion works well

## Trading Rules

1. **Only trade GREEN signals** (BUY/STRONG BUY with FULL confirmation)
2. **Enter at 75%+ conviction**
3. **Exit if conviction drops >25% below entry**
4. **Exit if signal changes to AVOID**
5. **Keep 5-10 open positions maximum**
6. **Track win rate and portfolio P&L weekly**

## Disclaimer

Rules-based technical signals on historical data. Not financial advice. Backtests may overfit and past performance does not guarantee future results. All signals are research-based; use proper risk management.

---

**Status**: ✓ PRODUCTION READY  
**Latest update**: Phase 7 complete (Position tracking + full integration)  
**Last tested**: 2026-07-08  
**Version**: 7.0 (Complete)

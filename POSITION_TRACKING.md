# Position Tracking - Live P&L & Performance Metrics

## Overview

Real-time position tracking with live P&L, conviction changes, and signal performance. Track entry/exit prices, hold times, and portfolio performance.

## Features

- Track open and closed positions with live prices
- Real-time P&L calculations (absolute $ and %)
- Conviction at entry vs current conviction
- Signal performance changes
- Win rate and hold time statistics
- Persistent JSON storage

## Commands

### Show Positions

```bash
python3 quant_engine.py --positions
```

Displays:
- All open positions with live P&L
- Closed positions with final P&L
- Total portfolio P&L
- Win rate statistics

### Add Position

```bash
python3 quant_engine.py --add-position TICKER ENTRY_PRICE
```

Example:
```bash
python3 quant_engine.py --add-position AAPL 300.00
```

Creates position with:
- Ticker
- Entry price
- Entry time (timestamp)
- Entry conviction (default 75%)
- 100 shares
- Open status

### Close Position

```bash
python3 quant_engine.py --close-position TICKER EXIT_PRICE
```

Example:
```bash
python3 quant_engine.py --close-position NVDA 200.00
```

Calculates:
- Exit price
- P&L ($)
- P&L (%)
- Hold time
- Moves to closed positions

## Output Format

### Open Positions

```
OPEN POSITIONS (3)
--------------------------------------------------
TICKER    ENTRY    CURRENT    P&L      P&L%  CONV  CHG  SIGNAL
--------------------------------------------------
AAPL      $300.00  $310.66    $1066    3.6%  75%   +0%  STRONG BUY
MSFT      $380.00  $388.84    $884     2.3%  75%   -25% HOLD
NVDA      $190.00  $196.93    $693     3.6%  75%   +0%  AVOID
--------------------------------------------------
TOTAL                          $2643
```

### Closed Positions

```
CLOSED POSITIONS (5)
--------------------------------------------------
TICKER  ENTRY    EXIT     P&L     P&L%  HOLD TIME
--------------------------------------------------
NVDA    $190.00  $200.00  $1000   5.3%  <1d
TSLA    $400.00  $390.00  -$1000  -2.5% 5d
(top 10 shown)
--------------------------------------------------
Win rate: 80% (4 winners, 1 loser)
```

## Data Structure (positions.json)

```json
{
  "positions": [
    {
      "ticker": "AAPL",
      "entry_price": 300.0,
      "entry_time": "2026-07-08T09:19:46.944663",
      "entry_conviction": 75,
      "shares": 100,
      "current_price": 310.66,
      "current_conviction": 75,
      "pnl": 1066,
      "pnl_pct": 3.6,
      "conviction_change": 0,
      "signal": "STRONG BUY signal",
      "status": "open"
    }
  ],
  "closed": [
    {
      "ticker": "NVDA",
      "entry_price": 190.0,
      "exit_price": 200.0,
      "entry_time": "2026-07-08T09:19:00",
      "exit_time": "2026-07-08T09:20:00",
      "pnl": 1000,
      "pnl_pct": 5.3,
      "status": "closed",
      "shares": 100
    }
  ]
}
```

## Workflow

### Morning Trading Setup

```bash
# View all positions
python3 quant_engine.py --positions

# Check conviction changes
# AAPL: conviction still 75% (no change)
# MSFT: conviction dropped to 50% (watch for exit)

# Enter new positions based on alerts
python3 quant_engine.py AAPL MRK GS AMZN TSLA --period 3mo --alerts
# → GREEN: AAPL, MRK
python3 quant_engine.py --add-position AAPL 310
python3 quant_engine.py --add-position MRK 129

# Check updated portfolio
python3 quant_engine.py --positions
```

### During Market Hours

```bash
# Track live P&L (run after market analysis)
python3 quant_engine.py AAPL MRK MSFT --period 3mo
# Updates positions.json with live prices

# View updated P&L
python3 quant_engine.py --positions
```

### End of Day / Exit Trades

```bash
# Close winning trades
python3 quant_engine.py --close-position AAPL 320
python3 quant_engine.py --close-position MRK 130

# Close losing trades
python3 quant_engine.py --close-position MSFT 375

# Review closed positions
python3 quant_engine.py --positions
```

## Metrics Tracked

### Per Position

**Entry Metrics:**
- Entry price
- Entry time
- Entry conviction (0-100%)
- Signal at entry

**Live Metrics:**
- Current price
- Current conviction
- P&L ($)
- P&L (%)
- Conviction change (% points)
- Current signal

**Exit Metrics:**
- Exit price
- Exit time
- Hold duration
- Final P&L

### Portfolio Level

- Total open P&L
- Total closed P&L
- Total portfolio P&L
- Win rate (% of closed trades profitable)
- Number of open/closed positions

## Analysis Insights

### Conviction Tracking

Check whether conviction is improving or declining:

```
AAPL: 75% → 80% (conviction improving, hold)
MSFT: 75% → 50% (conviction declining, consider exit)
NVDA: 75% → 75% (conviction stable)
```

### Signal Evolution

Watch how signals change post-entry:

```
Entry Signal: STRONG BUY (75% conviction)
Current Signal: BUY (80% conviction) → Holding, improving
Current Signal: HOLD (50% conviction) → Weakening, watch for exit
Current Signal: AVOID (20% conviction) → Exit recommended
```

### Win Rate Tracking

Monitor strategy profitability:

```
Win rate: 80% (4 winners / 5 trades)
Avg winner: $2,000
Avg loser: $-500
Profit factor: 4.0 (4K profit / 1K loss)
```

## Integration with Dashboard

When running analysis on tracked stocks:

```bash
python3 quant_engine.py AAPL MRK MSFT --dashboard
```

The dashboard shows current signals, and positions.json is automatically updated with live prices and conviction changes.

## Storage & Persistence

- Positions stored in `positions.json` in working directory
- JSON persists between sessions
- Automatic updates when running analysis on tracked tickers
- Manual add/close via CLI commands

## Tips

1. **Entry discipline**: Only add positions from BUY CANDIDATES (green alerts)
2. **Conviction as stop**: Exit if conviction drops below entry conviction by >25%
3. **Signal evolution**: STRONG BUY → BUY → HOLD → AVOID is natural; exit on AVOID
4. **Review weekly**: Check win rate, average holds, and P&L per position
5. **Portfolio limits**: Keep 5-10 open positions max to avoid dilution

## Example Session

```bash
# Morning briefing
$ python3 quant_engine.py AAPL MSFT NVDA TSLA --alerts
GREEN - BUY CANDIDATES: AAPL, MSFT
RED - RISK / AVOID: NVDA
YELLOW - NEUTRAL: TSLA

# Enter positions
$ python3 quant_engine.py --add-position AAPL 310
$ python3 quant_engine.py --add-position MSFT 388

# End of day review
$ python3 quant_engine.py --positions
OPEN POSITIONS (2)
AAPL    $310.00  $312.50  $250   0.8%  75%  +5%  STRONG BUY
MSFT    $388.00  $390.00  $200   0.5%  75%  +0%  BUY
TOTAL                       $450

# Close winner
$ python3 quant_engine.py --close-position AAPL 320
Closed position: AAPL
  Entry: $310.00, Exit: $320.00
  P&L: $1000.00 (+3.2%)

# Check final state
$ python3 quant_engine.py --positions
OPEN POSITIONS (1): MSFT
CLOSED POSITIONS (1): AAPL
Win rate: 100%
```

## Files

- `positions.json` - Position tracking storage
- `quant_engine.py` - Position tracking functions (add_position, close_position, show_positions, update_positions_live)

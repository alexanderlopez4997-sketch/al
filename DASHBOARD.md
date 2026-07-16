# Portfolio Dashboard - Watchlist View

## Overview

Compact, scannable view of multiple stocks with scores, prices, and verdicts. Perfect for portfolio monitoring, pre-market briefings, and watchlist updates.

## Features

- Single-screen view of up to 20+ stocks
- Sorted by score (strongest signals first)
- Color-coded verdicts (green = BUY, yellow = HOLD, red = AVOID)
- Confirmation level indicator
- Real-time prices and % change
- Updated timestamp

## Usage

```bash
# Dashboard view of multiple stocks
python3 quant_engine.py AAPL MRK GS AMZN TSLA MSFT AMD NVDA --dashboard

# With custom period
python3 quant_engine.py AAPL MSFT NVDA --period 6mo --dashboard

# Load from watchlist file
python3 quant_engine.py $(cat watchlist.txt) --dashboard
```

## Output Format

```
===============================================================================================
WATCHLIST DASHBOARD
===============================================================================================
SCORE    TICKER          PRICE     CHG%              VERDICT   CONF
-----------------------------------------------------------------------------------------------
+58      AAPL           310.66    -0.6%    STRONG BUY signal      ●
+35      MRK            128.86    +1.6%           BUY signal      ●
+26      AMZN           245.98    +0.7%           BUY signal      ●
+19      MSFT           388.84    +0.5%       HOLD / no edge      ●
+14      TSLA           402.90    -4.0%       HOLD / no edge      ●
+10      GS            1042.98    -1.2%       HOLD / no edge      ●
-14      AMD            516.11    -6.5%       HOLD / no edge      ●
-30      NVDA           196.93    +0.7%   AVOID / sell signa      ●
===============================================================================================
Updated 09:14:00 AM
```

## Column Breakdown

### SCORE
- Color coded: green (+40+), yellow (0-39), red (-∞ to -1)
- Higher = stronger signal
- Range: -100 to +100

### TICKER
- Stock symbol
- 8 character width for alignment

### PRICE
- Current price (last close or latest trade)
- Right-aligned for easy scanning

### CHG%
- Price change from previous close
- Green if positive, red if negative
- Shows momentum at a glance

### VERDICT
- Buy/Sell signal from the quant model
- Color coded:
  - Green (STRONG BUY, BUY) = buy signals
  - Yellow (HOLD) = neutral
  - Red (AVOID) = sell signals

### CONF
- Confirmation level indicator
- Green dot (●) = FULL confirmation (75%+)
- Yellow dot (●) = PARTIAL confirmation (50-75%)
- Red dot (●) = WEAK confirmation (<50%)

## Reading the Dashboard

### Top Priority (Green, High Score)

```
+58      AAPL           310.66    -0.6%    STRONG BUY signal      ●
+35      MRK            128.86    +1.6%           BUY signal      ●
+26      AMZN           245.98    +0.7%           BUY signal      ●
```

Action: Strongest signals. Full confirmation. These are your primary trade opportunities.

### Caution Signals (Yellow/Red)

```
-14      AMD            516.11    -6.5%       HOLD / no edge      ●
-30      NVDA           196.93    +0.7%   AVOID / sell signa      ●
```

Action: Avoid or consider shorts. Weak signals in mature downtrends.

### Neutral Holds (Yellow)

```
+19      MSFT           388.84    +0.5%       HOLD / no edge      ●
+14      TSLA           402.90    -4.0%       HOLD / no edge      ●
```

Action: No edge. Skip these unless you have other reasons to hold.

## Trading Workflow

1. **Morning**: Run dashboard on watchlist
2. **Identify**: Green signals (STRONG BUY, BUY)
3. **Confirm**: Check CONF level (prefer FULL)
4. **Rank**: Sort by score (highest first)
5. **Size**: Apply position sizing based on account
6. **Execute**: Trade the top 3-5 highest-conviction setups

## Example Watchlist File

Save tickers to file (watchlist.txt):

```
AAPL
MRK
GS
AMZN
TSLA
MSFT
AMD
NVDA
```

Load at any time:

```bash
python3 quant_engine.py $(cat watchlist.txt) --dashboard
```

## Dashboard vs Other Views

### Dashboard (Multi-Stock Scan)
- Best for: Portfolio monitoring, watchlist scanning
- Output: Table of all stocks
- Time: 1-2 seconds for 10 stocks

### Report (Single Stock)
- Best for: Deep analysis of one stock
- Output: Full factor breakdown, backtest, etc.
- Time: 5-10 seconds per stock

### Pipeline View (Single Stock)
- Best for: Understanding stage-by-stage computation
- Output: 5-stage data flow visualization
- Time: 5-10 seconds per stock

## Real-World Example

Morning routine:

```bash
# Update watchlist
python3 quant_engine.py AAPL MSFT NVDA AMZN TSLA GOOGL META NFLX --dashboard

# Results show:
# AAPL: +62 STRONG BUY (FULL) → Top priority
# NVDA: -30 AVOID (WEAK) → Skip
# MSFT: +40 BUY (FULL) → Second priority
# AMZN: +35 BUY (PARTIAL) → Monitor
# Others: neutral

# Run full analysis on top 2 signals
python3 quant_engine.py AAPL
python3 quant_engine.py MSFT
```

## Performance Tips

- Limit to 15-20 stocks for quick scanning
- Use consistent period (e.g., always 1d, 6mo)
- Run before market open to get fresh signals
- Combine with custom watchlists for sectors
- Check CONF level before sizing up

## Dashboard Enhancements (Future)

- Live refresh: `--refresh 5` (update every 5 seconds)
- Filtering: `--min-score 20` (only show strong signals)
- Sector grouping: show by industry
- Performance tracking: P&L per position
- Alerts: notify on signal changes
- Historical: compare to prior day scores

## Integration

Can be called from scripts:

```python
from quant_engine import analyze, dashboard

results = []
for ticker in ["AAPL", "MSFT", "NVDA"]:
    df = fetch(ticker, "3mo", "1d")
    res = analyze(ticker, df, "1d")
    results.append(res)

dashboard(results)  # Display summary
```

Perfect for:
- Morning briefings
- Portfolio reviews
- Watchlist monitoring
- Screening for new opportunities

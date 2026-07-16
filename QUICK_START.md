# Meridian Quant Engine - Quick Start Guide

## One-Page Cheat Sheet for Daily Trading

### Morning Routine (5 minutes)

```bash
# 1. Check your positions (P&L, conviction changes)
python3 quant_engine.py --positions

# 2. Scan watchlist for opportunities
python3 quant_engine.py AAPL MRK GS AMZN TSLA --dashboard

# 3. See alerts (green = buy, red = avoid, yellow = neutral)
python3 quant_engine.py AAPL MRK GS AMZN TSLA --alerts

# 4. Enter position if GREEN alert appears
python3 quant_engine.py --add-position AAPL 310
```

### During Day (continuous monitoring)

```bash
# Update P&L (run after market analysis)
python3 quant_engine.py AAPL MSFT --dashboard

# View positions (auto-updated with latest prices)
python3 quant_engine.py --positions

# Check conviction changes (watch for drops > 25%)
# AAPL: 75% → 80% (improving, hold)
# MSFT: 75% → 50% (declining, watch for exit)
```

### Exit Rules

```bash
# Close winner (conviction still strong)
python3 quant_engine.py --close-position AAPL 320

# Close loser (conviction dropped below entry)
python3 quant_engine.py --close-position MSFT 375

# Review closed positions and win rate
python3 quant_engine.py --positions
```

## Signal Reference

| Signal | When | Action |
|--------|------|--------|
| STRONG BUY | Highest conviction | Go full size |
| BUY | Good confirmation | Normal position |
| HOLD | No edge | Skip, don't hold |
| AVOID | Bearish | Exit or avoid |

## Confirmation Levels

- **●** = FULL (75%+) — Trade it
- **◐** = PARTIAL (50-75%) — Monitor first
- **◯** = WEAK (<50%) — Skip

## Dashboard Color Coding

- **GREEN score** (+40+) = Strong signal
- **YELLOW score** (-20 to +39) = Neutral
- **RED score** (below -20) = Bearish

## Command Cheat Sheet

```bash
# Analysis
python3 quant_engine.py AAPL                    # Full report on AAPL
python3 quant_engine.py AAPL MSFT NVDA          # Multi-stock scan
python3 quant_engine.py AAPL MSFT --dashboard   # Compact dashboard
python3 quant_engine.py AAPL MSFT --alerts      # Risk/opportunity alerts
python3 quant_engine.py AAPL --pipeline         # Stage-by-stage pipeline
python3 quant_engine.py AAPL --period 6mo       # 6-month data

# Positions
python3 quant_engine.py --positions             # Show all positions
python3 quant_engine.py --add-position AAPL 310  # Enter position
python3 quant_engine.py --close-position AAPL 320 # Exit position
```

## Entry Checklist

Before entering a position, verify:

1. ✓ Signal = BUY or STRONG BUY (not HOLD/AVOID)
2. ✓ Confirmation = FULL (●) or PARTIAL (◐) at worst
3. ✓ Score >= +40
4. ✓ Backtest Sharpe > 1.0
5. ✓ No recent insider selling
6. ✓ No securities offerings (424B5)
7. ✓ Portfolio not over-leveraged (<10 positions)

## Exit Checklist

Exit position if any occur:

1. ✗ Signal changed to AVOID
2. ✗ Conviction dropped >25% below entry
3. ✗ Insider selling alert appears
4. ✗ Major bad news or surprise earnings
5. ✗ P&L gain target hit (optional)
6. ✗ Stop loss hit (optional)

## Position Management Tips

### Tracking Conviction

```
Entry: 75%  Current: 80%  → Improving, hold
Entry: 75%  Current: 50%  → Declining 25%, watch for exit
Entry: 75%  Current: 30%  → Major decline, exit soon
```

### Signal Evolution

```
STRONG BUY → BUY → HOLD → AVOID (normal progression)
Exit when it reaches AVOID (no edge left)
```

### Portfolio Limits

- Max 10 open positions
- Min conviction: 75% at entry
- Watch conviction weekly
- Track win rate (target 55%+)

## Watchlist File

Save tickers to `watchlist.txt`:

```
AAPL
MRK
GS
AMZN
TSLA
MSFT
NVDA
GOOGL
```

Load at any time:

```bash
python3 quant_engine.py $(cat watchlist.txt) --dashboard
python3 quant_engine.py $(cat watchlist.txt) --alerts
```

## Performance Monitoring

After 10+ closed trades, check:

```bash
# View positions
python3 quant_engine.py --positions

# Calculate metrics
Win rate = (winners / total trades)  # Target 50%+
Avg winner / Avg loser ratio = Profit factor  # Target 2.0+
Total P&L = sum of all closed P&L
```

## Common Mistakes to Avoid

1. ❌ Trading HOLD signals (wait for BUY)
2. ❌ Ignoring conviction changes (exit on >25% drop)
3. ❌ Too many positions (keep to 5-10 max)
4. ❌ Holding through AVOID signal
5. ❌ Ignoring insider selling or offerings
6. ❌ Over-sizing on low-confirmation signals
7. ❌ Chasing after factor correlations trigger

## Example Trading Day

```bash
# 9:30 AM - Market open
$ python3 quant_engine.py --positions
AAPL: $310 entry, $312.50 now, +$250 P&L (0.8%)

# Check updated signals
$ python3 quant_engine.py AAPL --no-color
Signal: STRONG BUY (conviction 80%)

# 2:00 PM - Mid-day check
$ python3 quant_engine.py AAPL MSFT --dashboard
AAPL: +62 STRONG BUY
MSFT: +40 BUY (PARTIAL confirmation)

# 3:45 PM - End of day
$ python3 quant_engine.py --positions
AAPL: +0.8%, conviction stable, hold
MSFT: +1.2%, conviction down to 60%, watch for exit

# 4:00 PM - Close positions
$ python3 quant_engine.py --close-position AAPL 320
Closed: Entry $310, Exit $320, P&L +$1000 (3.2%)

$ python3 quant_engine.py --positions
Win rate: 100% (1/1 closed trades)
```

## Data Interpretation

### Dashboard Column Meanings

```
SCORE    = Signal strength (-100 to +100)
TICKER   = Stock symbol
PRICE    = Latest closing price
CHG%     = % change from previous close
VERDICT  = Signal (STRONG BUY/BUY/HOLD/AVOID)
CONF     = Confirmation level (● ◐ ◯)
```

### Position Table Column Meanings

```
ENTRY     = Entry price you paid
CURRENT   = Current/latest price
P&L       = Dollar gain/loss
P&L%      = Percentage gain/loss
CONV      = Conviction at entry → current
CHG       = Change in conviction (% points)
SIGNAL    = Current market signal
```

## Help & Questions

For detailed documentation:
- `SYSTEM_COMPLETE.md` — Full system overview
- `POSITION_TRACKING.md` — Position management details
- `DASHBOARD.md` — Multi-stock scanning
- `PORTFOLIO_ALERTS.md` — Alert categorization
- `CONFIRMATION_CHECKLIST.md` — Validation logic

---

**Key Principle:** Only trade when confirmation is FULL (●) and conviction is high (75%+). Exit when conviction drops significantly or signal changes to AVOID. Keep position count to 5-10 stocks.

Good luck trading!

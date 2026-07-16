# Confirmation Checklist - Signal Validation System

## Overview

Multi-criteria validation system that checks signal reliability before trading. Answers: "Is this signal actually trustworthy?"

## The 7 Checks

1. **Verdict is BUY / STRONG BUY** - Does the signal recommend buying?
   - Pass: verdict tone is "good" (BUY or STRONG BUY)
   - Fail: verdict is HOLD or AVOID

2. **Conviction >= 60%** - Do the factors agree?
   - Conviction = % of factors pointing same direction
   - 60%+ = solid multi-factor agreement
   - Shows factor alignment

3. **Score clears threshold** - Is score strong enough?
   - Pass: score >= entry threshold (volatility-adjusted)
   - Threshold adapted per market regime
   - Example: +20 in normal, +16 in bull, +25 in bear

4. **Backtest Sharpe positive** - Does strategy win historically?
   - Pass: Sharpe ratio > 0 (more gains than volatility)
   - Fail: Sharpe < 0 (strategy loses)
   - Shows statistical edge

5. **Winning streak (win rate > 50%)** - Do trades win?
   - Pass: > 50% of trades are profitable
   - Fail: <= 50% win rate (coin flip or worse)
   - Example: 5/10 trades win = 50% (PASS), 2/5 trades win = 40% (FAIL)

6. **Exposure >= 25%** - Is strategy actually trading?
   - Pass: in position >= 25% of the time
   - Fail: strategy is inactive/sitting in cash
   - Shows the strategy is confident enough to take positions

7. **Max drawdown < 40%** - Is risk acceptable?
   - Pass: worst peak-to-trough decline < 40%
   - Fail: drawdown >= 40% (too risky)
   - Shows risk is controlled

## Confirmation Levels

- **FULL** (>= 75% pass) - Strong validation, high confidence
- **PARTIAL** (50-75% pass) - Mixed signals, moderate confidence
- **WEAK** (< 50% pass) - Many failures, low confidence

## Real Examples

### AAPL 3-Month: FULL Confirmation

```
CONFIRMATION  FULL - 7/7 checks pass
  ✓ Verdict is BUY / STRONG BUY                   STRONG BUY signal
  ✓ Conviction >= 60%                             75%
  ✓ Score clears threshold (>=20)                 +58
  ✓ Backtest Sharpe positive                      3.38
  ✓ Winning streak (win rate > 50%)               100%
  ✓ Exposure >= 25% (not sitting idle)            71%
  ✓ Max drawdown < 40%                            -4%
```

Interpretation: All boxes checked. This signal is rock solid. High confidence to trade.

### DEMO: FULL Confirmation

```
CONFIRMATION  FULL - 6/7 checks pass
  ✓ Verdict is BUY / STRONG BUY                   BUY signal
  ✓ Conviction >= 60%                             75%
  ✓ Score clears threshold (>=19)                 +41
  ✓ Backtest Sharpe positive                      2.56
  ✗ Winning streak (win rate > 50%)               27%
  ✓ Exposure >= 25% (not sitting idle)            35%
  ✓ Max drawdown < 40%                            -13%
```

Interpretation: One check fails (low win rate), but overall FULL confirmation (6/7). Strategy wins less often but by enough margin to be profitable. Still trustworthy.

### NVDA 3-Month: PARTIAL Confirmation

```
CONFIRMATION  PARTIAL - 5/7 checks pass
  ✗ Verdict is BUY / STRONG BUY                   AVOID / sell signal
  ✓ Conviction >= 60%                             75%
  ✗ Score clears threshold (>=25)                 -30
  ✓ Backtest Sharpe positive                      1.88
  ✓ Winning streak (win rate > 50%)               100%
  ✓ Exposure >= 25% (not sitting idle)            45%
  ✓ Max drawdown < 40%                            -9%
```

Interpretation: PARTIAL confirmation. Main issues:
- Verdict says AVOID (✗), not BUY
- Score is negative and doesn't clear threshold
- BUT backtest shows edge (Sharpe > 0, 100% win rate, good exposure)

Meaning: The signal is conflicted. Historical data suggests a setup, but current conditions argue against it. Caution warranted.

## How to Use

### Trading Rules Based on Confirmation Level

**FULL (>= 75%)**
- Full position size
- Confident entry
- Hold through noise

**PARTIAL (50-75%)**
- 50-75% position size
- Cautious entry
- Lower conviction
- Tighter stops

**WEAK (< 50%)**
- Skip the trade
- Wait for confirmation to improve
- Signal is too uncertain

### Checklist Interpretation

Read the individual check results to understand *why* confirmation is partial:

- Verdict fails? ➜ Current conditions disagree, despite historical edge
- Conviction low? ➜ Factors conflict, mixed signal
- Score weak? ➜ Not strong enough conviction
- Sharpe negative? ➜ Historical strategy loses (red flag)
- Win rate low? ➜ Trades lose more than they win
- Exposure low? ➜ Strategy is mostly in cash (not confident)
- Drawdown high? ➜ Risk profile concerning

Fix strategy by addressing failures.

## Real-World Usage

### Scenario 1: All-Green Signal (FULL)

```
CONFIRMATION  FULL - 7/7 checks pass
```

Action: Go full size. High confidence. This is the setup you're looking for.

### Scenario 2: Mostly Passing (FULL)

```
CONFIRMATION  FULL - 6/7 checks pass
  ✗ Winning streak (win rate > 50%)               48%
```

Action: Go full size. One borderline check. Strategy sometimes loses but makes it back on larger wins.

### Scenario 3: Mixed Signals (PARTIAL)

```
CONFIRMATION  PARTIAL - 5/7 checks pass
  ✗ Verdict is BUY / STRONG BUY                   HOLD
  ✗ Score clears threshold (>=18)                 +12
```

Action: Size down 50%. Current conditions weak, but historical data shows edge. Hedge your bets.

### Scenario 4: Failing Checks (WEAK)

```
CONFIRMATION  WEAK - 3/7 checks pass
  ✗ Verdict is BUY / STRONG BUY                   AVOID
  ✗ Conviction >= 60%                             35%
  ✗ Backtest Sharpe positive                      -0.8
  ✗ Winning streak (win rate > 50%)               30%
```

Action: Skip. Too many failures. This signal has no edge. Don't trade.

## Code Implementation

```python
def confirmation_checklist(score, verdict, conviction, backtest, fwd_stats, buy_th, strong_th):
    """Multi-criteria validation. Returns {'passed': int, 'total': int, 'level': str, 'checks': list}"""
    checks = [
        {"name": "Verdict is BUY / STRONG BUY", "pass": verdict["tone"] == "good"},
        {"name": "Conviction >= 60%", "pass": conviction >= 60},
        {"name": "Score clears threshold", "pass": score >= buy_th},
        {"name": "Backtest Sharpe positive", "pass": backtest["sharpe"] > 0},
        {"name": "Winning streak (win rate > 50%)", "pass": backtest["winrate"] > 0.50},
        {"name": "Exposure >= 25%", "pass": backtest["exposure"] >= 0.25},
        {"name": "Max drawdown < 40%", "pass": backtest["maxdd"] > -0.40}
    ]
    passed = sum(1 for c in checks if c["pass"])
    total = len(checks)
    level = "FULL" if passed/total >= 0.75 else "PARTIAL" if passed/total >= 0.50 else "WEAK"
    return {"passed": passed, "total": total, "level": level, "checks": checks}
```

## Why This Matters

Confirmation checklist prevents:
- Trading low-conviction signals (saves money)
- Ignoring backtest warnings (avoids surprises)
- Over-trading in unprofitable regimes (protects capital)
- Chasing scores without validation (reduces whipsaws)

It's a quick sanity check: before you trade, does the signal pass the 7-point validation?

## Display in Reports

Shown at the top of stock analysis:

```
CONFIRMATION  FULL - 7/7 checks pass
  ✓ Verdict is BUY / STRONG BUY                   STRONG BUY signal
  ✓ Conviction >= 60%                             75%
  ✓ Score clears threshold (>=20)                 +58
  ✓ Backtest Sharpe positive                      3.38
  ✓ Winning streak (win rate > 50%)               100%
  ✓ Exposure >= 25%                               71%
  ✓ Max drawdown < 40%                            -4%
```

Instantly see whether the signal is trustworthy.

## Future Enhancements

1. **Weighted checks** - Some checks more important than others
2. **Market-regime adaptation** - Different checks in bull vs bear
3. **Time-decay** - Recent backtest data weighted more
4. **Cross-market** - How stock performs vs sector/market
5. **Custom checks** - User-defined validation rules

# Portfolio Alerts - Risk & Opportunity Highlighting

## Overview

Enhanced portfolio dashboard that groups stocks by opportunity/risk and highlights key catalysts and concerns.

## Structure

```
🟢 BUY CANDIDATES (n stocks)
├─ Stock A: +65 STRONG BUY
│  ├─ Catalyst: insider buying
│  ├─ Insider: $2.5M bought / $0.0M sold (90d)
│  └─ Risk: moderate

🔴 RISK / AVOID (n stocks)
├─ Stock B: -45 AVOID
│  ├─ Catalyst: insider selling + offering
│  ├─ Insider: $0.0M bought / $51.4M sold (90d)
│  ├─ Event: 424B5 securities offering
│  └─ Risk: HIGH

🟡 NEUTRAL (n stocks)
├─ Stock C: +12 HOLD
│  └─ No clear catalyst
```

## Real Example: AMZN Risk Alert

```
🔴 RISK / AVOID
AMZN
catalyst -29
chart: HOLD / no edge
▼ $0.0M bought / $51.4M sold (90d) · 37d ago (decay ×0.29)
▼ 424B5 — securities offering — dilution
▲ news tone +0.32
```

**Interpretation:**
- Technical: HOLD / no edge (neutral)
- Insider: Major selling ($51.4M), no buying
- Catalyst: Securities offering (dilution risk)
- Macro: Slight positive news
- Overall: **Avoid** - Insider selling + dilution outweigh positive news

## Alert Types

### Green (BUY CANDIDATES)

Conditions:
- Score >= +40 AND Verdict = BUY/STRONG BUY
- Confirmation = FULL
- No major insider selling
- No securities offerings
- Positive backtest edge

Examples:
- Strong technical score
- Insider accumulation
- Positive analyst momentum
- Good relative strength

### Red (RISK / AVOID)

Conditions:
- Score <= -20 OR Verdict = AVOID
- Multiple risk factors present
- Recent insider selling
- Securities offerings
- Negative earnings surprises

Red flags:
- Insider dumping (high $, recent)
- Form 424B5 (dilution via offering)
- Short interest spike
- Insider departures
- Negative fundamental changes

### Yellow (NEUTRAL)

Everything else:
- Score between -20 and +40
- Mixed signals
- No clear catalyst
- Insufficient information

## Key Risk Indicators

### Insider Trading

🔴 **Red Flag:**
- $X,XXM sold / $0M bought (90d)
- Executives bailing out
- Timing before bad news

🟢 **Green Flag:**
- $X,XXM bought / $0M sold (90d)
- Insiders have conviction
- Recent accumulation

### Securities Offerings

🔴 **Red Flag:**
- Form 424B5 (secondary offering)
- Dilution to existing shareholders
- Stock likely to decline
- Usually follows stock run-up

### News & Macro

🟢 **Green Flag:**
- news tone +0.50+ (strong positive)
- analyst upgrades
- positive catalysts

🔴 **Red Flag:**
- news tone -0.50- (strong negative)
- analyst downgrades
- competitive threats

### Technical

🟢 **Green Flag:**
- Score >= +40
- Backtest Sharpe > 2.0
- Winning trade history

🔴 **Red Flag:**
- Score <= -30
- Backtest Sharpe < -0.5
- Losing pattern

## Construction Rules

```python
def portfolio_alerts(results):
    buy_candidates = []
    risk_avoid = []
    neutral = []
    
    for res in results:
        score = res["score"]
        verdict = res["verdict"]
        insiders = res.get("insiders", {})
        news = res.get("macro_signal", {})
        
        # Categorize
        if score >= 40 and verdict["tone"] == "good" and not has_red_flags(res):
            buy_candidates.append(res)
        elif score <= -20 or verdict["tone"] == "bad" or has_multiple_red_flags(res):
            risk_avoid.append(res)
        else:
            neutral.append(res)
    
    return {
        "buy": buy_candidates,
        "risk": risk_avoid,
        "neutral": neutral
    }

def has_red_flags(res):
    flags = 0
    if res.get("insider_selling_ratio", 0) > 10:  # >10:1 selling
        flags += 1
    if res.get("has_424b5"):  # securities offering
        flags += 1
    if res.get("news_tone", 0) < -0.4:  # negative news
        flags += 1
    return flags >= 2
```

## Display Format

### Compact View (Console)

```
🟢 BUY CANDIDATES (2)
AAPL +62 STRONG BUY · insider buy $2.5M · news +0.45
MSFT +45 BUY · strong backtest · risk-on

🔴 RISK / AVOID (1)
AMZN -29 HOLD · insider sell $51.4M · 424B5 offering

🟡 NEUTRAL (5)
NVDA, GS, TSLA, AMD, GOOGL — no clear edge
```

### Detailed View (HTML/Dashboard)

```
╔════════════════════════════════════════════════════════════╗
║ 🟢 BUY CANDIDATES                                      2   ║
╠════════════════════════════════════════════════════════════╣
║ AAPL +62 STRONG BUY                                        ║
║   ✓ Technical: strong uptrend, conviction 80%             ║
║   ✓ Insider: $2.5M bought, 0 sold (90d)                   ║
║   ✓ News: +0.45 (positive momentum)                       ║
║   ✓ Risk: LOW                                              ║
╠════════════════════════════════════════════════════════════╣
║ 🔴 RISK / AVOID                                        1   ║
╠════════════════════════════════════════════════════════════╣
║ AMZN -29 HOLD / AVOID                                      ║
║   ✗ Technical: no edge, mixed signals                      ║
║   ✗ Insider: $0 bought, $51.4M sold (90d, 37d ago)       ║
║   ✗ Event: Form 424B5 — secondary offering (dilution)     ║
║   ✓ News: +0.32 (slight positive)                         ║
║   ✗ Risk: HIGH (insider dump + offering)                  ║
╚════════════════════════════════════════════════════════════╝
```

## Trading Strategy

### BUY CANDIDATES (🟢)

Action: **Go Full Size**
- Conviction high
- Multiple green flags
- Low risk
- Backtest edge confirmed

Example: AAPL at +62 with insider buying

### RISK / AVOID (🔴)

Action: **Skip or Short**
- Multiple red flags
- Insider selling = bearish conviction
- Offerings = dilution ahead
- Technical breakdown

Example: AMZN at -29 with $51M insider selling + offering

### NEUTRAL (🟡)

Action: **Wait for Clarity**
- Score in middle range
- No clear catalyst
- Wait for technical breakout or insider accumulation
- Set alerts for changes

Example: NVDA at +12 with mixed signals

## Real-World Workflow

### Morning (5 min)

```bash
python3 quant_engine.py $(cat watchlist.txt) --alerts
```

Output:
```
🟢 BUY CANDIDATES: 2 stocks (AAPL, MSFT)
🔴 RISK / AVOID: 1 stock (AMZN)
🟡 NEUTRAL: 5 stocks (no edge currently)
```

Decision: **Trade top 2 buy candidates, skip the rest**

### Pre-Trade (3 min)

Check each buy candidate:
- Full technical report
- Position sizing based on risk
- Set stops and targets

### Throughout Day

Monitor:
- Insider activity changes
- Earnings surprises
- Analyst rating changes
- News catalysts

## Alert Examples

### Alert: New Insider Buying

```
NEW ALERT: NVDA insider buying
$5.2M purchased by CEO · first purchase in 180 days
Score: +15 → consider upgrading to monitor

Action: Watch for technical breakout
```

### Alert: Form 424B5 Filed

```
WARNING: AMZN securities offering filed (424B5)
≈5% dilution expected · stock typically declines 10-15%

Action: Avoid until offering completes
```

### Alert: Analyst Downgrade

```
UPDATE: TSLA analyst downgrade (Goldman Sachs)
Price target: $400 → $250 · maintain AVOID rating

Action: Confirm downtrend continues
```

## Integration

Add to dashboard CLI:

```bash
python3 quant_engine.py AAPL MSFT AMZN TSLA NVDA --alerts
```

Shows:
- BUY candidates (green)
- Risk/avoid (red)
- Key catalysts
- Portfolio summary

Perfect for:
- Daily portfolio reviews
- Risk management
- Opportunity spotting
- Watchlist prioritization

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Data Test for Portfolio Risk Manager

Fetches real market data for a portfolio and tests:
- Signal evaluation against risk gates
- Position tracking with real price movements
- Dynamic stop adjustments
- Circuit breaker triggers
- Real-time metrics

Usage:
    python3 test_risk_manager_live.py
"""

import sys
from datetime import datetime, timedelta
from typing import Dict, Tuple

try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "yfinance"])
    import yfinance as yf

from portfolio_risk_integration import RiskEngine

# Test configuration
INITIAL_EQUITY = 100000
TEST_TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
SECTORS = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "JPM": "Financials",
    "XOM": "Energy",
}

print("=" * 80)
print("PORTFOLIO RISK MANAGER — LIVE DATA TEST")
print("=" * 80)


def fetch_live_data(tickers: list, period: str = "1y") -> Dict:
    """Fetch live OHLCV data from yfinance."""
    print(f"\n📊 Fetching live data for {len(tickers)} tickers ({period})...")

    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, progress=False)
            if len(df) >= 60:  # Need at least 60 bars for correlation
                data[ticker] = df
                print(f"  ✓ {ticker}: {len(df)} bars")
            else:
                print(f"  ⚠ {ticker}: Insufficient data ({len(df)} < 60)")
        except Exception as e:
            print(f"  ✗ {ticker}: {e}")

    return data


def calculate_technical_indicators(df) -> Tuple[float, float, float]:
    """Calculate ATR, EMA20, and current price from OHLCV data."""
    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = [high_low, high_close, low_close]
    true_range = ranges[0]
    for r in ranges[1:]:
        true_range = true_range.combine(r, max)
    atr = true_range.rolling(14).mean().iloc[-1]

    # EMA20 (Exponential Moving Average)
    ema20 = df['Close'].ewm(span=20, adjust=False).mean().iloc[-1]

    # Current price
    current_price = df['Close'].iloc[-1]

    return float(atr), float(ema20), float(current_price)


def simulate_signals(data: Dict, engine: RiskEngine) -> Dict:
    """
    Simulate signal generation based on technical analysis.

    Simple heuristic: Compare current price to EMA20
    - Price > EMA20 + ATR = BUY signal
    """
    print(f"\n🎯 Generating signals...")
    signals = {}

    for ticker, df in data.items():
        atr, ema20, price = calculate_technical_indicators(df)

        # Simple signal: price above EMA20 + 0.5*ATR
        signal_threshold = ema20 + (0.5 * atr)

        if price > signal_threshold:
            # Proposed size: Risk $1000 per position
            risk_amount = 1000
            stop_distance = atr * 2.0
            proposed_shares = max(1, int(risk_amount / stop_distance))

            signals[ticker] = {
                "price": price,
                "atr": atr,
                "ema20": ema20,
                "proposed_shares": proposed_shares,
                "stop_distance": stop_distance,
            }
            print(f"  📈 {ticker}: Price ${price:.2f} > EMA20 ${ema20:.2f} | Signal: {proposed_shares} shares @ ${price:.2f}")
        else:
            print(f"  ➖ {ticker}: No signal (${price:.2f} ≤ ${signal_threshold:.2f})")

    return signals


def evaluate_and_execute(engine: RiskEngine, signals: Dict) -> Dict:
    """Evaluate signals through risk gates and execute approved trades."""
    print(f"\n⚔️  Evaluating {len(signals)} signals through risk gates...")
    executed = {}
    rejected = {}

    for ticker, signal in signals.items():
        # Evaluate signal
        result = engine.evaluate_signal(
            ticker=ticker,
            proposed_shares=signal["proposed_shares"],
            entry_price=signal["price"],
        )

        if result.approved:
            # Execute trade
            engine.add_position(
                ticker=ticker,
                shares=result.max_shares,
                entry_price=signal["price"],
                current_price=signal["price"],
                atr=signal["atr"],
                structural_floor=signal["ema20"],
            )
            executed[ticker] = {
                "shares": result.max_shares,
                "entry": signal["price"],
                "stop": signal["price"] - (2.0 * signal["atr"]),
            }
            print(f"  ✅ {ticker}: Executed {result.max_shares:.0f} shares")
        else:
            rejected[ticker] = {
                "reason": result.limiting_factor,
                "max_allowed": result.max_shares,
                "warnings": result.warnings,
            }
            print(f"  ❌ {ticker}: Rejected ({result.limiting_factor})")
            for w in result.warnings:
                print(f"      - {w}")

    return executed, rejected


def simulate_price_movement(data: Dict, data_offset: int = -5) -> Dict:
    """Simulate price movement by looking back in historical data."""
    print(f"\n📉 Simulating price movement (offset bar: {data_offset})...")
    prices = {}
    indicators = {}

    for ticker, df in data.items():
        try:
            # Use historical data to simulate price movement
            bar = df.iloc[data_offset]
            atr, _, price = calculate_technical_indicators(df.iloc[:data_offset])

            prices[ticker] = float(bar['Close'])
            indicators[ticker] = {
                "atr": atr,
                "ema20": float(df['Close'].ewm(span=20, adjust=False).mean().iloc[data_offset]),
                "price": float(bar['Close']),
                "high": float(bar['High']),
                "low": float(bar['Low']),
            }
            print(f"  {ticker}: ${prices[ticker]:.2f}")
        except Exception as e:
            print(f"  ⚠ {ticker}: {e}")

    return prices, indicators


def update_portfolio(engine: RiskEngine, prices: Dict, indicators: Dict) -> Dict:
    """Update portfolio and check for stop hits and adjustments."""
    print(f"\n📊 Updating portfolio...")

    metrics = engine.update(prices)
    print(f"  Equity: ${metrics['equity']:,.2f}")
    print(f"  Unrealized P&L: ${metrics['unrealized_pnl']:+,.2f}")
    print(f"  Intraday Drawdown: {metrics['intraday_drawdown']:.2f}%")
    print(f"  Positions: {metrics['positions']}")

    if metrics["circuit_breaker_tripped"]:
        print(f"  ⚠️  CIRCUIT BREAKER: {metrics['circuit_breaker_reason']}")
    else:
        print(f"  ✓ State: {metrics['engine_state']}")

    # Update individual positions
    print(f"\n🔄 Updating positions...")
    closed = []
    adjusted = []

    for ticker in list(engine.get_positions().keys()):
        if ticker not in indicators:
            continue

        ind = indicators[ticker]
        valid, new_stop, reason = engine.update_position(
            ticker=ticker,
            current_price=ind["price"],
            atr=ind["atr"],
            structural_floor=ind["ema20"],
        )

        if not valid:
            print(f"  🔴 {ticker}: Stop hit at ${ind['price']:.2f}")
            closed.append(ticker)
        elif new_stop:
            print(f"  🟡 {ticker}: Stop adjusted to ${new_stop:.2f} ({reason})")
            adjusted.append(ticker)
        else:
            print(f"  🟢 {ticker}: Price ${ind['price']:.2f} | Stop ${engine.get_stop_price(ticker):.2f}")

    return metrics, closed, adjusted


def display_portfolio_summary(engine: RiskEngine):
    """Display final portfolio summary."""
    print(f"\n" + "=" * 80)
    print("PORTFOLIO SUMMARY")
    print("=" * 80)

    status = engine.get_status()
    positions = engine.get_positions()

    print(f"\n💰 Account Status:")
    print(f"   Equity: ${status['portfolio_equity']:,.2f}")
    print(f"   P&L: ${status['unrealized_pnl']:+,.2f}")
    print(f"   Intraday Drawdown: {status['intraday_drawdown']:.2f}%")
    print(f"   Weekly Drawdown: {status['weekly_drawdown']:.2f}%")
    print(f"   Engine State: {status['engine_state']}")

    if positions:
        print(f"\n📍 Open Positions ({len(positions)}):")
        print(f"   {'Ticker':<8} {'Shares':<8} {'Entry':<10} {'Current':<10} {'Stop':<10} {'P&L':<12} {'%':<8}")
        print(f"   {'-'*76}")
        for ticker, pos in positions.items():
            print(f"   {ticker:<8} {pos['shares']:<8.0f} ${pos['entry']:<9.2f} ${pos['current']:<9.2f} "
                  f"${pos['stop']:<9.2f} ${pos['pnl']:<11.2f} {pos['pnl_pct']:<7.2f}%")
    else:
        print(f"\n   No open positions")


def run_live_test():
    """Run complete live data test."""

    # Initialize risk engine
    print(f"\n🚀 Initializing RiskEngine with ${INITIAL_EQUITY:,} equity...")
    engine = RiskEngine(
        initial_equity=INITIAL_EQUITY,
        max_sector_exposure=3.0,
        intraday_drawdown_limit=-3.0,
        weekly_drawdown_limit=-5.0,
    )
    engine.register_universe(SECTORS)
    print("   ✓ Engine ready")

    # Fetch live data
    data = fetch_live_data(TEST_TICKERS, period="1y")
    if len(data) < 2:
        print("\n❌ Insufficient data. Cannot run test.")
        return

    # Generate signals
    signals = simulate_signals(data, engine)
    if not signals:
        print("\n⚠️  No signals generated. Trying more aggressive criteria...")
        # Retry with more lenient criteria
        for ticker, df in data.items():
            atr, ema20, price = calculate_technical_indicators(df)
            if price > ema20:  # Just above EMA20
                risk_amount = 1000
                stop_distance = atr * 2.0
                proposed_shares = max(1, int(risk_amount / stop_distance))
                signals[ticker] = {
                    "price": price,
                    "atr": atr,
                    "ema20": ema20,
                    "proposed_shares": proposed_shares,
                    "stop_distance": stop_distance,
                }

    if not signals:
        print("\n⚠️  Still no signals. Test inconclusive.")
        return

    # Evaluate and execute
    executed, rejected = evaluate_and_execute(engine, signals)

    if not executed:
        print("\n⚠️  No signals approved. Testing gate logic...")
        print("   (This is expected if portfolio is already concentrated)")

        # Try with smaller position sizes
        for ticker in list(signals.keys())[:1]:  # Try first ticker with smaller size
            result = engine.evaluate_signal(
                ticker=ticker,
                proposed_shares=10,  # Much smaller
                entry_price=signals[ticker]["price"],
            )
            if result.approved:
                engine.add_position(
                    ticker=ticker,
                    shares=result.max_shares,
                    entry_price=signals[ticker]["price"],
                    current_price=signals[ticker]["price"],
                    atr=signals[ticker]["atr"],
                    structural_floor=signals[ticker]["ema20"],
                )
                executed[ticker] = {
                    "shares": result.max_shares,
                    "entry": signals[ticker]["price"],
                }
                print(f"   ✅ Executed {ticker} with smaller size")

    # Simulate price movement
    if executed:
        prices, indicators = simulate_price_movement(data)

        # Update portfolio
        metrics, closed, adjusted = update_portfolio(engine, prices, indicators)

        # Close positions if stops hit
        for ticker in closed:
            engine.close_position(ticker)

        print(f"\n📈 Summary:")
        print(f"   Executed: {len(executed)}")
        print(f"   Closed (stop): {len(closed)}")
        print(f"   Adjusted stops: {len(adjusted)}")

    # Display final summary
    display_portfolio_summary(engine)

    print(f"\n" + "=" * 80)
    print("✅ LIVE DATA TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    try:
        run_live_test()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

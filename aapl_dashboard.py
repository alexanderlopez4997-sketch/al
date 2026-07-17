#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AAPL Stock Data Visualization Dashboard

Processes and visualizes AAPL stock data from Polygon.io API.
Provides comprehensive analysis including price trends, technical indicators,
and performance metrics. Integrates SMA data from Massive API.
"""

from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import json
import math
import os
import requests


def fetch_sma_data(ticker: str, window: int = 50, timespan: str = "day") -> Optional[Dict[str, Any]]:
    """
    Fetch SMA data from Massive API.

    Args:
        ticker: Stock ticker symbol
        window: SMA window period (default 50)
        timespan: Time period ('day', 'week', 'month')

    Returns:
        JSON response with SMA values, or None if request fails
    """
    try:
        api_key = os.environ.get("MASSIVE_API_KEY")
        if not api_key:
            return None

        base_url = "https://api.massive.com/v1/indicators/sma"
        params = {
            "apiKey": api_key,
            "timespan": timespan,
            "window": window,
            "series_type": "close",
            "order": "desc",
            "limit": 100
        }

        response = requests.get(f"{base_url}/{ticker}", params=params, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


class AAPLDataProcessor:
    """Process AAPL stock data and generate analytics."""

    def __init__(self, data_points: List[Dict[str, Any]]):
        """
        Initialize with raw data points from Polygon.io API.

        Args:
            data_points: List of dicts with 'timestamp' and 'value' keys
        """
        self.raw_data = data_points
        self.processed_data = self._process_data()

    def _process_data(self) -> List[Dict[str, Any]]:
        """Convert raw API data to processed format with dates and prices."""
        processed = []
        for point in self.raw_data:
            ts = point["timestamp"]
            # Convert milliseconds to seconds if needed
            if ts > 10000000000:
                ts = ts // 1000
            dt = datetime.fromtimestamp(ts)
            processed.append({
                "date": dt.strftime("%Y-%m-%d"),
                "price": round(point["value"], 2),
                "timestamp": ts
            })
        # Sort by timestamp ascending for chart display
        return sorted(processed, key=lambda x: x["timestamp"])

    def get_price_range(self) -> Tuple[float, float, float]:
        """Get (min_price, max_price, avg_price) from processed data."""
        if not self.processed_data:
            return 0, 0, 0
        prices = [d["price"] for d in self.processed_data]
        return min(prices), max(prices), sum(prices) / len(prices)

    def get_price_change(self) -> Tuple[float, float]:
        """Get (price_change, percent_change) from first to last price."""
        if len(self.processed_data) < 2:
            return 0, 0
        first = self.processed_data[0]["price"]
        last = self.processed_data[-1]["price"]
        change = last - first
        pct = (change / first * 100) if first != 0 else 0
        return round(change, 2), round(pct, 2)

    def calculate_moving_averages(self, periods: List[int] = None) -> Dict[str, List[Dict]]:
        """Calculate moving averages for specified periods."""
        if periods is None:
            periods = [5, 20, 50]

        prices = [d["price"] for d in self.processed_data]
        mas = {}

        for period in periods:
            ma_values = []
            for i in range(len(prices)):
                if i < period - 1:
                    ma_values.append(None)
                else:
                    avg = sum(prices[i - period + 1:i + 1]) / period
                    ma_values.append(round(avg, 2))

            mas[f"MA{period}"] = [
                {"date": self.processed_data[i]["date"], "value": ma_values[i]}
                for i in range(len(ma_values))
            ]

        return mas

    def calculate_volatility(self) -> Dict[str, float]:
        """Calculate price volatility metrics."""
        prices = [d["price"] for d in self.processed_data]
        if len(prices) < 2:
            return {"volatility": 0, "std_dev": 0}

        # Daily returns
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)

        # Standard deviation of returns
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance)

        return {
            "volatility": round(std_dev * 100, 2),
            "std_dev": round(std_dev, 4)
        }

    def get_price_levels(self) -> Dict[str, float]:
        """Get key price levels (support/resistance)."""
        prices = [d["price"] for d in self.processed_data]
        if not prices:
            return {}

        recent_20 = prices[-20:] if len(prices) >= 20 else prices
        return {
            "current": prices[-1],
            "high_52w": max(prices),
            "low_52w": min(prices),
            "resistance": max(recent_20),
            "support": min(recent_20),
            "avg_price": round(sum(prices) / len(prices), 2)
        }

    def get_chart_data(self) -> List[Dict[str, Any]]:
        """Get formatted data for charting."""
        return self.processed_data

    def generate_summary(self, include_sma: bool = True) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""
        min_p, max_p, avg_p = self.get_price_range()
        change, pct_change = self.get_price_change()
        volatility = self.calculate_volatility()
        levels = self.get_price_levels()
        mas = self.calculate_moving_averages()

        sma_data = None
        if include_sma:
            sma_response = fetch_sma_data("AAPL", window=50, timespan="day")
            if sma_response:
                sma_data = sma_response

        return {
            "ticker": "AAPL",
            "data_points": len(self.processed_data),
            "date_range": {
                "start": self.processed_data[0]["date"] if self.processed_data else None,
                "end": self.processed_data[-1]["date"] if self.processed_data else None
            },
            "price_stats": {
                "current": levels["current"],
                "min": min_p,
                "max": max_p,
                "avg": avg_p,
                "change": change,
                "change_pct": pct_change
            },
            "volatility": volatility,
            "levels": levels,
            "moving_averages": mas,
            "chart_data": self.get_chart_data(),
            "sma_data": sma_data
        }


def render_dashboard_html(summary: Dict[str, Any]) -> str:
    """Render interactive HTML dashboard for AAPL data."""

    chart_data = json.dumps(summary["chart_data"])
    price_stats = summary["price_stats"]
    volatility = summary["volatility"]
    levels = summary["levels"]
    sma_data = summary.get("sma_data")

    # Format change color
    change_color = "#2ECC8F" if price_stats["change"] >= 0 else "#FF5449"
    change_sign = "+" if price_stats["change"] >= 0 else ""

    # SMA info
    sma_info = ""
    if sma_data and sma_data.get("results") and sma_data["results"].get("values"):
        sma_values = sma_data["results"]["values"]
        if sma_values:
            latest_sma = sma_values[0]["value"] if isinstance(sma_values[0], dict) else None
            if latest_sma:
                sma_info = f"<div class='info-item'><div class='info-label'>SMA(50)</div><div class='info-value'>${latest_sma:.2f}</div></div>"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>AAPL Stock Analysis Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: #e8eef5;
                padding: 20px;
                min-height: 100vh;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
            }}
            .header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 2px solid #4F9DE0;
            }}
            .title {{
                font-size: 2.5em;
                font-weight: 700;
                color: #ffffff;
            }}
            .subtitle {{
                font-size: 0.9em;
                color: #6B7E92;
                margin-top: 5px;
            }}
            .metrics {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .metric-card {{
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(79, 157, 224, 0.2);
                border-radius: 12px;
                padding: 20px;
                backdrop-filter: blur(10px);
            }}
            .metric-label {{
                font-size: 0.9em;
                color: #6B7E92;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 8px;
            }}
            .metric-value {{
                font-size: 1.8em;
                font-weight: 700;
                color: #ffffff;
            }}
            .metric-secondary {{
                font-size: 0.85em;
                color: #C9D6E2;
                margin-top: 8px;
            }}
            .change {{
                color: {change_color};
                font-weight: 600;
            }}
            .charts {{
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 20px;
                margin-bottom: 30px;
            }}
            .chart-container {{
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(79, 157, 224, 0.2);
                border-radius: 12px;
                padding: 20px;
                backdrop-filter: blur(10px);
            }}
            .chart-title {{
                font-size: 1.2em;
                font-weight: 600;
                margin-bottom: 15px;
                color: #E8EEF5;
            }}
            #priceChart {{
                max-height: 400px;
            }}
            #distributionChart {{
                max-height: 400px;
            }}
            .levels-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 12px;
            }}
            .level {{
                padding: 12px;
                background: rgba(79, 157, 224, 0.1);
                border-left: 3px solid #4F9DE0;
                border-radius: 6px;
            }}
            .level-label {{
                font-size: 0.8em;
                color: #6B7E92;
                text-transform: uppercase;
                margin-bottom: 4px;
            }}
            .level-value {{
                font-size: 1.4em;
                font-weight: 700;
                color: #2ECC8F;
            }}
            .info-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
            }}
            .info-item {{
                background: rgba(79, 157, 224, 0.08);
                padding: 15px;
                border-radius: 8px;
                border: 1px solid rgba(79, 157, 224, 0.15);
            }}
            .info-label {{
                font-size: 0.85em;
                color: #6B7E92;
                margin-bottom: 6px;
            }}
            .info-value {{
                font-size: 1.3em;
                font-weight: 600;
                color: #E8EEF5;
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid rgba(79, 157, 224, 0.2);
                color: #6B7E92;
                font-size: 0.9em;
            }}
            @media (max-width: 1024px) {{
                .charts {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div>
                    <div class="title">AAPL Stock Analysis</div>
                    <div class="subtitle">Real-time data visualization & technical metrics</div>
                </div>
            </div>

            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-label">Current Price</div>
                    <div class="metric-value">${{price_stats["current"]:.2f}}</div>
                    <div class="metric-secondary"><span class="change">{change_sign}${{price_stats["change"]:.2f}} ({price_stats["change_pct"]:.2f}%)</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">52-Week High</div>
                    <div class="metric-value">${{levels["high_52w"]:.2f}}</div>
                    <div class="metric-secondary">Range: ${{{levels["low_52w"]:.2f}}} - ${{{levels["high_52w"]:.2f}}}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Average Price</div>
                    <div class="metric-value">${{{levels["avg_price"]:.2f}}}</div>
                    <div class="metric-secondary">Volatility: {volatility["volatility"]}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Data Coverage</div>
                    <div class="metric-value">{summary["data_points"]}</div>
                    <div class="metric-secondary">{summary["date_range"]["start"]} to {summary["date_range"]["end"]}</div>
                </div>
            </div>

            <div class="charts">
                <div class="chart-container">
                    <div class="chart-title">Price Trend</div>
                    <canvas id="priceChart"></canvas>
                </div>
                <div class="chart-container">
                    <div class="chart-title">Price Distribution</div>
                    <canvas id="distributionChart"></canvas>
                </div>
            </div>

            <div class="chart-container">
                <div class="chart-title">Key Price Levels</div>
                <div class="levels-grid">
                    <div class="level">
                        <div class="level-label">Resistance</div>
                        <div class="level-value">${{{levels["resistance"]:.2f}}}</div>
                    </div>
                    <div class="level">
                        <div class="level-label">Support</div>
                        <div class="level-value">${{{levels["support"]:.2f}}}</div>
                    </div>
                    <div class="level">
                        <div class="level-label">52W High</div>
                        <div class="level-value">${{{levels["high_52w"]:.2f}}}</div>
                    </div>
                    <div class="level">
                        <div class="level-label">52W Low</div>
                        <div class="level-value">${{{levels["low_52w"]:.2f}}}</div>
                    </div>
                </div>
            </div>

            <div class="chart-container" style="margin-top: 20px;">
                <div class="chart-title">Analysis Metrics</div>
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Volatility (Std Dev)</div>
                        <div class="info-value">{volatility["volatility"]}%</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Price Change</div>
                        <div class="info-value"><span class="change">{change_sign}{price_stats["change"]:.2f}%</span></div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Data Points</div>
                        <div class="info-value">{summary["data_points"]}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Current vs Avg</div>
                        <div class="info-value">{round((price_stats["current"] - levels["avg_price"]) / levels["avg_price"] * 100, 2)}%</div>
                    </div>
                    {sma_info}
                </div>
            </div>

            <div class="footer">
                <p>AAPL Stock Data Dashboard | Real-time analysis powered by Meridian Quant Engine</p>
                <p>Data updated: {summary["date_range"]["end"]}</p>
            </div>
        </div>

        <script>
            const chartData = {chart_data};

            // Price Trend Chart
            const dates = chartData.map(d => d.date);
            const prices = chartData.map(d => d.price);

            const datasets = [{{
                label: 'AAPL Price',
                data: prices,
                borderColor: '#4F9DE0',
                backgroundColor: 'rgba(79, 157, 224, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointBackgroundColor: '#4F9DE0',
                pointBorderColor: '#ffffff',
                pointRadius: 4,
                pointHoverRadius: 6
            }}];

            const priceCtx = document.getElementById('priceChart').getContext('2d');
            new Chart(priceCtx, {{
                type: 'line',
                data: {{
                    labels: dates,
                    datasets: datasets
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        legend: {{
                            labels: {{ color: '#E8EEF5' }}
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: false,
                            ticks: {{ color: '#6B7E92' }},
                            grid: {{ color: 'rgba(79, 157, 224, 0.1)' }}
                        }},
                        x: {{
                            ticks: {{ color: '#6B7E92' }},
                            grid: {{ color: 'rgba(79, 157, 224, 0.1)' }}
                        }}
                    }}
                }}
            }});

            // Price Distribution Chart
            const priceBuckets = {{}};
            prices.forEach(p => {{
                const bucket = Math.floor(p / 10) * 10;
                priceBuckets[bucket] = (priceBuckets[bucket] || 0) + 1;
            }});

            const bucketLabels = Object.keys(priceBuckets).sort((a, b) => a - b);
            const bucketValues = bucketLabels.map(b => priceBuckets[b]);

            const distCtx = document.getElementById('distributionChart').getContext('2d');
            new Chart(distCtx, {{
                type: 'bar',
                data: {{
                    labels: bucketLabels.map(b => `${{b}}`),
                    datasets: [{{
                        label: 'Frequency',
                        data: bucketValues,
                        backgroundColor: '#2ECC8F',
                        borderColor: '#1fa372',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        legend: {{
                            labels: {{ color: '#E8EEF5' }}
                        }}
                    }},
                    scales: {{
                        y: {{
                            ticks: {{ color: '#6B7E92' }},
                            grid: {{ color: 'rgba(79, 157, 224, 0.1)' }}
                        }},
                        x: {{
                            ticks: {{ color: '#6B7E92' }},
                            grid: {{ color: 'rgba(79, 157, 224, 0.1)' }}
                        }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """

    return html

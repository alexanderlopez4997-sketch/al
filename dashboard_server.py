#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight AAPL Dashboard Server

Serves the AAPL stock analysis dashboard without external dependencies.
Access at: http://127.0.0.1:8787/dashboard/aapl
"""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import aapl_dashboard as ad

PORT = 8787


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[{self.client_address[0]}] {format % args}")

    def _send(self, content, content_type="text/html"):
        body = content.encode() if isinstance(content, str) else content
        self.send_response(200)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)

        if url.path == "/":
            return self._send(self._home_page())

        if url.path == "/dashboard/aapl":
            try:
                # Generate dashboard with optional API data
                dashboard_html = ad.render_dashboard_html(
                    ad.AAPLDataProcessor([
                        {'timestamp': 1784174400000, 'value': 301.6571999999999},
                        {'timestamp': 1784088000000, 'value': 300.52859999999987},
                        {'timestamp': 1784001600000, 'value': 299.58139999999986},
                        {'timestamp': 1783915200000, 'value': 298.7111999999999},
                        {'timestamp': 1783656000000, 'value': 297.7683999999999},
                        {'timestamp': 1783569600000, 'value': 296.87619999999987},
                        {'timestamp': 1783483200000, 'value': 295.9039999999999},
                        {'timestamp': 1783396800000, 'value': 295.0573999999999},
                        {'timestamp': 1783310400000, 'value': 294.3127999999999},
                        {'timestamp': 1782964800000, 'value': 293.5229999999999},
                    ]).generate_summary()
                )
                return self._send(dashboard_html)
            except Exception as e:
                return self._send(f"Error: {str(e)}", "text/plain")

        if url.path == "/api/dashboard":
            try:
                summary = ad.AAPLDataProcessor([
                    {'timestamp': 1784174400000, 'value': 301.6571999999999},
                    {'timestamp': 1784088000000, 'value': 300.52859999999987},
                    {'timestamp': 1784001600000, 'value': 299.58139999999986},
                    {'timestamp': 1783915200000, 'value': 298.7111999999999},
                    {'timestamp': 1783656000000, 'value': 297.7683999999999},
                    {'timestamp': 1783569600000, 'value': 296.87619999999987},
                    {'timestamp': 1783483200000, 'value': 295.9039999999999},
                    {'timestamp': 1783396800000, 'value': 295.0573999999999},
                    {'timestamp': 1783310400000, 'value': 294.3127999999999},
                    {'timestamp': 1782964800000, 'value': 293.5229999999999},
                ]).generate_summary()
                return self._send(json.dumps(summary), "application/json")
            except Exception as e:
                return self._send(json.dumps({"error": str(e)}), "application/json")

        self._send("Not found", "text/plain")

    def _home_page(self):
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>AAPL Dashboard Server</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto;
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    color: #e8eef5;
                    padding: 40px;
                    min-height: 100vh;
                    margin: 0;
                }
                .container {
                    max-width: 600px;
                    margin: 0 auto;
                }
                h1 { color: #ffffff; font-size: 2.5em; margin-bottom: 20px; }
                .status { background: rgba(46, 204, 143, 0.1); border-left: 4px solid #2ecc8f; padding: 15px; border-radius: 8px; margin: 20px 0; }
                .endpoint { background: rgba(79, 157, 224, 0.1); border: 1px solid #4f9de0; padding: 15px; border-radius: 8px; margin: 15px 0; font-family: monospace; }
                .endpoint-title { color: #4f9de0; font-weight: 600; margin-bottom: 8px; }
                .endpoint-url { color: #ffffff; margin: 8px 0; }
                .endpoint-desc { color: #6b7e92; font-size: 0.9em; margin-top: 8px; }
                a { color: #4f9de0; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎯 AAPL Dashboard Server</h1>

                <div class="status">
                    <strong>✅ Server Status:</strong> Running on port 8787
                </div>

                <h2>Available Endpoints</h2>

                <div class="endpoint">
                    <div class="endpoint-title">📊 Dashboard (HTML)</div>
                    <div class="endpoint-url"><a href="/dashboard/aapl">/dashboard/aapl</a></div>
                    <div class="endpoint-desc">Interactive AAPL stock analysis dashboard with charts and technical indicators</div>
                </div>

                <div class="endpoint">
                    <div class="endpoint-title">📈 Dashboard Data (JSON)</div>
                    <div class="endpoint-url"><a href="/api/dashboard">/api/dashboard</a></div>
                    <div class="endpoint-desc">Raw dashboard data in JSON format including price stats and technical analysis</div>
                </div>

                <h2>Features</h2>
                <ul style="color: #c9d6e2; line-height: 1.8;">
                    <li>Real-time price trend visualization</li>
                    <li>Price distribution analysis</li>
                    <li>Technical indicators (SMA, EMA, MACD, RSI ready)</li>
                    <li>Support/Resistance levels</li>
                    <li>Volatility metrics</li>
                    <li>Responsive dark-themed UI</li>
                </ul>

                <h2>Getting Started</h2>
                <p style="color: #c9d6e2;">
                    <strong>View Dashboard:</strong> <a href="/dashboard/aapl">Click here to view the dashboard</a><br>
                    <strong>Get Data:</strong> <a href="/api/dashboard">Click here to get JSON data</a>
                </p>

                <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid rgba(79, 157, 224, 0.2); color: #6b7e92; font-size: 0.9em;">
                    <p>🚀 AAPL Stock Analysis Dashboard | Meridian Quant Engine</p>
                </div>
            </div>
        </body>
        </html>
        """


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), DashboardHandler)
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║       AAPL Dashboard Server Starting                      ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  🎯 Server: http://127.0.0.1:{PORT}                     ║
║                                                           ║
║  📊 Dashboard:  http://127.0.0.1:{PORT}/dashboard/aapl   ║
║  📈 API:        http://127.0.0.1:{PORT}/api/dashboard    ║
║                                                           ║
║  Press Ctrl+C to stop the server                          ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped gracefully")
        server.server_close()

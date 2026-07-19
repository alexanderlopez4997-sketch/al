#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meridian Web Terminal — full browser front-end for the quant engine (stdlib only).

Views: live Watchlist dashboard · Analyze (interactive candlestick chart + full
report) · Screener · After-Hours · Morning Brief · Track Record. Backed by the
same Python engine as the desktop app; no new Python dependencies (the only
external asset is the TradingView lightweight-charts lib, loaded from a CDN in
the browser for the interactive candles).

    python3 web_server.py   →   http://127.0.0.1:8787
"""
import html as _html
import json
import os
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import quant_engine as qe
import quant_gui as g
import fundamental_engine as fe
import sentiment_engine as se
import edgar
import orderflow as of
import afterhours as ah
import morning as mb
import trackrecord as tr
import websocket_client as wsc
import aapl_dashboard as ad

PORT = 8787
TAG = {"txt": "#C9D6E2", "dim": "#6B7E92", "buy": "#2ECC8F", "sell": "#FF5449",
       "warn": "#E0A83B", "head": "#E8EEF5", "big": "#FFFFFF", "gold": "#C8A24B",
       "formula": "#E8D9A8", "blue": "#4F9DE0"}

WATCHLIST_CATEGORIES = {
    "high_beta": {
        "name": "High-Beta Momentum",
        "desc": "Volatile stocks where correlations break down — momentum models excel",
        "tickers": ["TSLA", "AMD", "NVDA", "GME", "RIOT", "MARA"]
    },
    "mid_growth": {
        "name": "Mid-Tier Growth",
        "desc": "Established growth with good liquidity — sweet spot for ML edge",
        "tickers": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NFLX"]
    },
    "value_div": {
        "name": "Value & Dividend",
        "desc": "Stable dividend payers with different price drivers",
        "tickers": ["MRK", "JNJ", "KO", "PG", "UNH", "WMT"]
    },
    "fintech": {
        "name": "Fintech & Emerging Growth",
        "desc": "Newer, less efficient pricing — higher noise but potential edges",
        "tickers": ["SOFI", "PLTR", "COIN", "SQ", "HOOD", "UPST"]
    },
    "sectors": {
        "name": "Sector Leaders (ETFs)",
        "desc": "Diversified sector exposure — broad market patterns",
        "tickers": ["XLF", "XLK", "XLV", "XLE", "XLI", "XLY"]
    }
}


def _try(fn, d=None):
    try:
        return fn()
    except Exception:
        return d


def _seg_html(segs):
    out = []
    for text, tag in segs:
        c = TAG.get(tag, "#C9D6E2")
        w = "700" if tag in ("head", "big", "gold") else "400"
        sz = "1.5em" if tag == "big" else "1.08em" if tag in ("head", "gold") else "1em"
        out.append(f'<span style="color:{c};font-weight:{w};font-size:{sz}">{_html.escape(text)}</span>')
    return "".join(out)


# ---------------------------------------------------------------- analyze ---
def _full_analyze(sym, demo, optimize=False):
    res = g.screen_one(sym, demo, "6mo", "1d", optimize, cache=None, realtime_key=qe.FINNHUB_DEFAULT_KEY)
    if not demo and not res.get("ineligible"):
        fkey = qe.FINNHUB_DEFAULT_KEY
        qtok = os.environ.get("QUIVER_API_TOKEN"); avk = os.environ.get("ALPHA_VANTAGE_KEY")
        akey, asec = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_API_SECRET")
        recs = _try(lambda: qe.finnhub_recs(sym, fkey))
        congress = _try(lambda: qe.quiver_congress(sym, qtok))
        insiders = _try(lambda: qe.finnhub_insiders(sym, fkey))
        whale = _try(lambda: qe.whale_signal(qe.options_whale_flow(sym), qe.quiver_darkpool(sym, qtok)))
        market = _try(lambda: qe.market_context(res["d"], qe.fetch("SPY", "6mo", "1d")))
        res["fund"] = _try(lambda: fe.fetch_fundamentals(sym, fkey, avk))
        sen = _try(lambda: se.news_sentiment(sym, fkey, avk))
        res["filings"] = _try(lambda: edgar.recent_filings(sym, days=3), [])
        if akey and asec:
            w0, w1 = of.after_hours_window()
            res["orderflow"] = _try(lambda: of.darkpool_blocks(sym, akey, asec, w0, w1, 200000))
        res["sentiment"] = sen
        try:
            tilt = qe.alt_data_tilt(congress, recs, insiders, whale, se.macro_signal(sen))
            if tilt or market:
                qe.apply_alt_tilt(res, tilt, market)
        except Exception:
            pass
    elif res.get("ineligible"):
        res["alt_skipped"] = True                 # gate skipped alt-data — mark it honestly
    g.log_verdicts([{"ticker": sym, "tone": res["verdict"]["tone"], "label": res["verdict"]["label"],
                     "score": res["score"], "price": res["last"], "tags": g.verdict_tags(res)}], demo)
    segs = g.build_report_segments(res, res.get("opt"), 10000.0, 1.0)
    return {"ticker": sym, "score": round(res["score"]), "verdict": res["verdict"]["label"],
            "tone": res["verdict"]["tone"], "last": round(res["last"], 2), "chg": round(res["chg"], 2),
            "regime": (res.get("regime") or {}).get("regime", "unknown"),
            "edge_status": res.get("verdict", {}).get("edge_status", "ACTIVE"),
            "information_ratio": round(res.get("verdict", {}).get("information_ratio", 0.0), 3),
            "win_rate": round(res.get("verdict", {}).get("win_rate", 0.5), 3),
            "report": _seg_html(segs)}


def _ohlc(sym, demo):
    df = (qe.demo_data(sym) if demo else qe.fetch(sym, "6mo", "1d")).tail(130)
    bars = []
    for ts, row in df.iterrows():
        t = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
        bars.append({"time": t, "open": round(float(row["Open"]), 2), "high": round(float(row["High"]), 2),
                     "low": round(float(row["Low"]), 2), "close": round(float(row["Close"]), 2)})
    return {"ticker": sym, "bars": bars}


# -------------------------------------------------------------- watchlist ---
def _watchlist(tickers, demo):
    data = ({t: qe.demo_data(t) for t in tickers} if demo
            else g.fetch_many_concurrent(tickers, "6mo", "1d"))
    out = []
    for t in tickers:
        df = data.get(t)
        if df is None or len(df) < 60:
            continue
        r = _try(lambda: g.analyze_prefetched(t, df, "1d"))
        if not r:
            continue
        w = r.get("whale_activity")
        out.append({"ticker": t, "last": round(r["last"], 2), "chg": round(r["chg"], 2),
                    "score": round(r["score"]), "tone": r["verdict"]["tone"], "verdict": r["verdict"]["label"],
                    "whale": ("↑" if w and w["whale"] and w["direction"] == "accumulation"
                              else "↓" if w and w["whale"] and w["direction"] == "distribution" else "")})
    out.sort(key=lambda x: -x["score"])
    return out


# ------------------------------------------------------------- afterhours ---
def _afterhours_html(tickers, demo):
    akey, asec = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_API_SECRET")
    data = ({t: qe.demo_data(t) for t in tickers} if demo
            else g.fetch_many_concurrent(tickers, "6mo", "1d"))
    reads = {}

    def one(t):
        df = data.get(t)
        if df is None or len(df) < 2:
            return
        reg, prev = float(df["Close"].iloc[-1]), float(df["Close"].iloc[-2])
        ahpx = qe.alpaca_latest_trade(t, akey, asec) if (akey and asec) else None
        r = (ah.read_one(t, reg, prev, ahpx) if ahpx else
             {"ticker": t, "reg_close": reg, "ah_price": None, "ah_chg": 0.0, "flag": False})
        r["filings"] = [] if demo else _try(lambda: edgar.recent_filings(t, days=2), [])
        r["whale"] = qe.whale_score(df)
        if r["filings"]:
            r["flag"] = True
        reads[t] = r
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(one, tickers))
    flagged = sorted([r for r in reads.values() if r["flag"]], key=lambda r: -abs(r.get("ah_chg") or 0))
    rows = ""
    for r in flagged:
        offer = next((f for f in r["filings"] if f["bias"] < 0), None)
        head = ("⚠ DILUTION/OFFERING" if offer else "🔔 MATERIAL FILING" if r["filings"] else "🌙 AH MOVE")
        px = (f'<b style="color:{"#FF5449" if r["ah_chg"]<0 else "#2ECC8F"}">{r["ah_chg"]:+.1f}%</b> '
              f'→ {r["ah_price"]:.2f} vs {r["reg_close"]:.2f}' if r.get("ah_price") else "—")
        fil = "".join(f'<div class="sub">📂 {f["form"]} — {f["note"]}'
                      f'{" · ⏰ after-hours" if f["after_hours"] else ""} '
                      f'<a href="{f["url"]}" target="_blank">open</a></div>' for f in r["filings"][:3])
        rows += (f'<div class="ohcard"><div class="ohh"><b>{r["ticker"]}</b>'
                 f'<span class="tagpill">{head}</span></div><div>{px}</div>{fil}</div>')
    if not rows:
        rows = '<div class="muted">Nothing moving after hours and no fresh material filings.</div>'
    return {"html": f'<div class="grid3">{rows}</div>'
            + '<div class="muted" style="margin-top:14px">Reports the after-hours move + the SEC filings '
              'that cause moves. It does not infer institutional intent from order flow.</div>'}


# ----------------------------------------------------------- morning brief ---
def _morning_html(tickers, demo):
    akey, asec = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_API_SECRET")
    fkey = qe.FINNHUB_DEFAULT_KEY; avk = os.environ.get("ALPHA_VANTAGE_KEY")
    data = ({t: qe.demo_data(t) for t in tickers} if demo
            else g.fetch_many_concurrent(tickers, "6mo", "1d"))
    briefs = {}

    def one(t):
        df = data.get(t)
        if df is None or len(df) < 60:
            return
        r = _try(lambda: g.analyze_prefetched(t, df, "1d"))
        if not r:
            return
        reg = float(df["Close"].iloc[-1])
        ahpx = qe.alpaca_latest_trade(t, akey, asec) if (akey and asec and not demo) else None
        ahchg = (ahpx / reg - 1) * 100 if ahpx else 0.0
        ins = None if demo else qe.insider_signal(_try(lambda: qe.finnhub_insiders(t, fkey)))
        fil = [] if demo else _try(lambda: edgar.recent_filings(t, days=2), [])
        sen = None if demo else _try(lambda: se.news_sentiment(t, fkey, avk))
        b = mb.catalyst_score(r["score"], ahchg, ins, fil, sen, r.get("whale_activity"))
        b.update({"ticker": t, "ahchg": ahchg, "tech": r["verdict"]["label"]})
        briefs[t] = b
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(one, tickers))
    ranked = sorted(briefs.values(), key=lambda b: -b["score"])

    def block(title, items, color):
        if not items:
            return f'<h3 style="color:{color}">{title}</h3><div class="muted">(none)</div>'
        h = f'<h3 style="color:{color}">{title}</h3>'
        for b in items:
            rs = "".join(f'<div class="sub" style="color:{"#2ECC8F" if d>0 else "#FF5449" if d<0 else "#6B7E92"}">'
                         f'{"▲" if d>0 else "▼" if d<0 else "•"} {_html.escape(x)}</div>' for x, d in b["reasons"][:4])
            h += (f'<div class="ohcard"><div class="ohh"><b>{b["ticker"]}</b>'
                  f'<span class="tagpill">catalyst {b["score"]:+d}</span></div>'
                  f'<div class="sub">chart: {b["tech"]}</div>{rs}</div>')
        return h
    buys = [b for b in ranked if b["verdict"] == "BUY candidate"]
    risks = [b for b in ranked if b["verdict"] == "RISK / avoid"]
    return {"html": '<div class="grid2col">'
            + '<div>' + block("🟢 BUY CANDIDATES", buys, "#2ECC8F") + '</div>'
            + '<div>' + block("🔴 RISK / AVOID", risks, "#FF5449") + '</div></div>'}


# ------------------------------------------------------------- recent news ---
def _recent_news_html(tickers, demo):
    """Fetch recent news for watchlist tickers and format as dashboard cards."""
    if demo:
        # Demo news data showing the layout
        demo_news = [
            {"ticker": "NVDA", "signal": 0.65, "confidence": 0.85, "n": 12, "detail": "12 articles · bullish momentum on AI demand", "defensive_shift": False},
            {"ticker": "AAPL", "signal": 0.35, "confidence": 0.72, "n": 8, "detail": "8 articles · mixed sentiment on services guidance", "defensive_shift": False},
            {"ticker": "MSFT", "signal": 0.55, "confidence": 0.90, "n": 15, "detail": "15 articles · strong cloud growth narrative", "defensive_shift": False},
        ]
        rows = ""
        for item in demo_news:
            sig = item.get("signal", 0)
            sig_color = "#2ECC8F" if sig > 0.2 else "#FF5449" if sig < -0.2 else "#E0A83B"
            tag = "POSITIVE" if sig > 0.2 else "NEGATIVE" if sig < -0.2 else "NEUTRAL"
            defensive = " · ⚠ DEFENSIVE SHIFT" if item.get("defensive_shift") else ""
            detail = item.get("detail", f"{item.get('n', 0)} articles")
            rows += (f'<div class="ohcard"><div class="ohh"><b>{item["ticker"]}</b>'
                    f'<span class="tagpill" style="color:{sig_color};">{tag}</span></div>'
                    f'<div class="sub">{detail}{defensive}</div></div>')
        return {"html": f'<div class="grid3">{rows}</div>'
                + '<div class="muted" style="margin-top:14px">[DEMO] Recent news sentiment across 7 days.</div>'}
    fkey = qe.FINNHUB_DEFAULT_KEY
    avk = os.environ.get("ALPHA_VANTAGE_KEY")
    news_items = []

    def fetch_one(t):
        try:
            sent = se.news_sentiment(t, fkey, avk, days=7)
            if sent and sent.get("n", 0) > 0:
                news_items.append({"ticker": t, **sent})
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(fetch_one, tickers))

    # Sort by confidence * signal magnitude (most impactful first)
    news_items.sort(key=lambda x: -(abs(x.get("signal", 0)) * x.get("confidence", 0)))

    if not news_items:
        return {"html": '<div class="muted">No recent news coverage on watchlist.</div>'}

    rows = ""
    for item in news_items[:8]:  # Show top 8
        sig = item.get("signal", 0)
        conf = item.get("confidence", 0)
        sig_color = "#2ECC8F" if sig > 0.2 else "#FF5449" if sig < -0.2 else "#E0A83B"
        tag = "POSITIVE" if sig > 0.2 else "NEGATIVE" if sig < -0.2 else "NEUTRAL"
        defensive = " · ⚠ DEFENSIVE SHIFT" if item.get("defensive_shift") else ""
        detail = item.get("detail", f"{item.get('n', 0)} articles")
        rows += (f'<div class="ohcard"><div class="ohh"><b>{item["ticker"]}</b>'
                f'<span class="tagpill" style="color:{sig_color};">{tag}</span></div>'
                f'<div class="sub">{detail}{defensive}</div></div>')

    return {"html": f'<div class="grid3">{rows}</div>'
            + '<div class="muted" style="margin-top:14px">Recent news sentiment across 7 days. '
            'Defensive shift indicates recent tone deterioration.</div>'}


# ------------------------------------------------------------ track record ---
def _trackrecord_html():
    tickers = sorted({e["ticker"] for e in tr._load()})
    data = g.fetch_many_concurrent(tickers, "6mo", "1d") if tickers else {}
    summ = tr.summary(tr.score(lambda t: data.get(t), 5), 5)
    h = (f'<div class="stat">{summ["total"]} logged · {summ["graded"]} graded · '
         f'{summ["pending"]} pending</div>')
    if not summ["graded"]:
        return {"html": h + '<div class="muted" style="margin-top:12px">Nothing graded yet — '
                'verdicts age 5 trading days before scoring. Run live scans over a few days.</div>'}
    h += '<h3 style="color:#C8A24B">By verdict</h3>'
    for name, s in summ["by_tone"].items():
        col = "#2ECC8F" if s["hit_rate"] >= .5 else "#FF5449"
        h += (f'<div class="ohcard"><b>{name}</b> — {s["n"]} calls · '
              f'<b style="color:{col}">{s["hit_rate"]*100:.0f}% correct</b> · avg {s["avg_fwd"]*100:+.1f}%</div>')
    if summ["by_tag"]:
        h += '<h3 style="color:#C8A24B">Which signals add edge</h3>'
        for t, s in sorted(summ["by_tag"].items(), key=lambda kv: -kv[1]["hit_rate"]):
            col = "#2ECC8F" if s["hit_rate"] >= .5 else "#FF5449"
            h += (f'<div class="ohcard">{t} — {s["n"]} · '
                  f'<b style="color:{col}">{s["hit_rate"]*100:.0f}%</b> · avg {s["avg_fwd"]*100:+.1f}%</div>')
    return {"html": h}


# ------------------------------------------------------------ diagnostics ---
_diag_client = None

def _init_diagnostics(tickers):
    global _diag_client
    if _diag_client is None:
        api_key = os.environ.get("MASSIVE_API_KEY")
        _diag_client = wsc.get_diagnostics_client(symbols=tickers, api_key=api_key)
        _diag_client.connect(use_demo=not api_key)
    return _diag_client

def _diagnostics_json(tickers):
    client = _init_diagnostics(tickers)
    diag = client.get_diagnostics()
    return {
        "status": diag.get("status", "warming_up"),
        "timestamp": diag.get("timestamp", ""),
        "health_status": diag.get("health_status", {}),
        "factor_irs": diag.get("factor_irs", {}),
        "correlation_matrix": diag.get("correlation_matrix", {}),
        "buffers": diag.get("buffers", {})
    }


# ---------------------------------------------------------------- routing ---
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        g1 = lambda k, d="": q.get(k, [d])[0]
        demo = g1("demo", "0") == "1"
        tks = [t.upper() for t in g1("tickers").replace(",", " ").split() if t] \
            or ["NVDA", "AMD", "AAPL", "MSFT", "TSLA", "GS", "MRK", "AMZN"]
        try:
            if u.path in ("/", "/index.html"):
                return self._send(PAGE, "text/html; charset=utf-8")
            if u.path == "/api/analyze":
                return self._send(json.dumps(_full_analyze((g1("ticker", "NVDA") or "NVDA").upper(), demo,
                                                           g1("opt", "0") == "1")))
            if u.path == "/api/ohlc":
                return self._send(json.dumps(_ohlc((g1("ticker", "NVDA") or "NVDA").upper(), demo)))
            if u.path == "/api/watchlist":
                return self._send(json.dumps(_watchlist(tks, demo)))
            if u.path == "/api/afterhours":
                return self._send(json.dumps(_afterhours_html(tks, demo)))
            if u.path == "/api/morning":
                return self._send(json.dumps(_morning_html(tks, demo)))
            if u.path == "/api/news":
                return self._send(json.dumps(_recent_news_html(tks, demo)))
            if u.path == "/api/trackrecord":
                return self._send(json.dumps(_trackrecord_html()))
            if u.path == "/api/screen":
                data = ({t: qe.demo_data(t) for t in tks} if demo
                        else g.fetch_many_concurrent(tks, "6mo", "1d"))
                results = [r for t in tks if (df := data.get(t)) is not None and len(df) >= 60
                           for r in [_try(lambda: g.analyze_prefetched(t, df, "1d"))] if r]
                return self._send(json.dumps({"html": g.build_screener_html(results, [], "1d", demo, "none")}))
            if u.path == "/api/ml_screen":
                data = ({t: qe.demo_data(t) for t in tks} if demo
                        else g.fetch_many_concurrent(tks, "2y", "1d"))
                ml_results = g.build_ml_screener_data(tks, data)
                return self._send(json.dumps({"html": g.build_ml_screener_html(ml_results, demo)}))
            if u.path == "/api/categories":
                cats = [{k: {"name": v["name"], "desc": v["desc"], "count": len(v["tickers"])} for k, v in WATCHLIST_CATEGORIES.items()}]
                return self._send(json.dumps(cats[0] if cats else {}))
            if u.path == "/api/category_screen":
                cat = g1("category", "mid_growth")
                if cat not in WATCHLIST_CATEGORIES:
                    return self._send(json.dumps({"error": "Invalid category"}))
                tks_cat = WATCHLIST_CATEGORIES[cat]["tickers"]
                data = ({t: qe.demo_data(t) for t in tks_cat} if demo
                        else g.fetch_many_concurrent(tks_cat, "2y", "1d"))
                ml_results = g.build_ml_screener_data(tks_cat, data)
                edges = [r for r in ml_results if r["has_edge"]]
                no_edges = [r for r in ml_results if not r["has_edge"]]
                html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>ML Results</title><style>
body{{margin:0;background:#0A0E15;color:#C9D6E2;font-family:system-ui;padding:20px}}
.cat{{font-size:24px;font-weight:700;margin-bottom:8px}}
.desc{{color:#6B7E92;margin-bottom:16px}}
.stat{{background:#10161F;padding:12px;border-radius:6px;margin-bottom:8px;border-left:3px solid #2ECC8F}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px;margin-top:16px}}
.card{{background:#10161F;border:1px solid #232F3D;border-radius:8px;padding:12px}}
.tkr{{font-weight:700;font-size:16px;margin-bottom:4px}}
.acc{{font-family:monospace;color:#4F9DE0;font-weight:700}}
.edge{{color:#2ECC8F}} .noedge{{color:#FF5449}}
</style></head><body>
<div class="cat">{WATCHLIST_CATEGORIES[cat]["name"]}</div>
<div class="desc">{WATCHLIST_CATEGORIES[cat]["desc"]}</div>
<div class="stat"><span class="edge">✓ {len(edges)} with EDGE</span> · <span class="noedge">{len(no_edges)} no edge</span> · {len(ml_results)} total</div>
"""
                if edges:
                    html += '<h3 style="margin-top:20px;color:#2ECC8F">Models with Edge</h3><div class="grid">'
                    for r in edges:
                        html += f'<div class="card"><div class="tkr">{r["ticker"]}</div><div class="acc">{r["pred_pct"]}% accuracy</div><div style="font-size:12px;color:#6B7E92">vs {r["baseline"]*100:.0f}% baseline</div></div>'
                    html += '</div>'
                if no_edges:
                    html += '<h3 style="margin-top:20px;color:#FF5449">No Edge Detected</h3><div class="grid">'
                    for r in no_edges:
                        html += f'<div class="card"><div class="tkr">{r["ticker"]}</div><div style="font-size:12px;color:#6B7E92">{r["pred_pct"]}% vs {r["baseline"]*100:.0f}% baseline</div></div>'
                    html += '</div>'
                html += '</body></html>'
                return self._send(json.dumps({"html": html}))
        except Exception as e:
            return self._send(json.dumps({"error": str(e)}))
        self._send("not found", "text/plain")


# ----------------------------------------------------------- aapl dashboard ---
def _aapl_dashboard_html(data_points=None):
    """Generate AAPL dashboard HTML with provided data."""
    if data_points is None:
        data_points = [
            {"timestamp": 1784174400000, "value": 301.6571999999999},
            {"timestamp": 1784088000000, "value": 300.52859999999987},
            {"timestamp": 1784001600000, "value": 299.58139999999986},
            {"timestamp": 1783915200000, "value": 298.7111999999999},
            {"timestamp": 1783656000000, "value": 297.7683999999999},
            {"timestamp": 1783569600000, "value": 296.87619999999987},
            {"timestamp": 1783483200000, "value": 295.9039999999999},
            {"timestamp": 1783396800000, "value": 295.0573999999999},
            {"timestamp": 1783310400000, "value": 294.3127999999999},
            {"timestamp": 1782964800000, "value": 293.5229999999999},
        ]
    processor = ad.AAPLDataProcessor(data_points)
    summary = processor.generate_summary()
    html = ad.render_dashboard_html(summary)
    return {"html": html}


def _feeds():
    return {"alpaca": bool(os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_API_SECRET")),
            "quiver": bool(os.environ.get("QUIVER_API_TOKEN")), "av": bool(os.environ.get("ALPHA_VANTAGE_KEY"))}


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"Meridian Web Terminal → {url}")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


_F = _feeds()
_pill = lambda n, on: f'<span class="feed {"on" if on else "off"}">● {n}</span>'
PAGE = ("""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Meridian Terminal</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
 :root{--bg:#0A0E15;--panel:#10161F;--panel2:#161F2B;--line:#232F3D;--txt:#C9D6E2;--dim:#6B7E92;
   --gold:#C8A24B;--buy:#2ECC8F;--sell:#FF5449;--amber:#E0A83B}
 *{box-sizing:border-box} html,body{margin:0;height:100%}
 body{background:var(--bg);color:var(--txt);font-family:-apple-system,"SF Pro Text",Inter,system-ui,sans-serif;
   font-size:14px;display:flex;flex-direction:column;height:100vh}
 .top{display:flex;align-items:center;gap:14px;background:var(--panel);padding:10px 18px;border-bottom:2px solid var(--gold)}
 .diamond{color:var(--gold);font-size:22px}.brand{font-family:Georgia,serif;font-weight:700;font-size:20px;letter-spacing:1px}
 .sublabel{color:var(--gold);font-size:9px;letter-spacing:2px}.spacer{flex:1}
 .clock{font-family:ui-monospace,Menlo,monospace;font-weight:700}.pill{background:var(--panel2);color:var(--dim);padding:3px 10px;border-radius:3px;font-size:11px;letter-spacing:1px}
 .feeds{display:flex;gap:14px;background:var(--panel2);padding:5px 18px;border-bottom:1px solid var(--line);font-size:11px;letter-spacing:1px;align-items:center}
 .feed.on{color:var(--buy)}.feed.off{color:#3A4657}.lbl{color:var(--dim)}.stream{margin-left:auto;color:var(--buy)}
 .ctrl{display:flex;align-items:center;gap:8px;padding:10px 18px;background:var(--panel);border-bottom:1px solid var(--line);flex-wrap:wrap}
 input,button{font-family:inherit;font-size:14px;border-radius:5px;border:1px solid var(--line);outline:none}
 input{background:var(--panel2);color:var(--txt);padding:8px 12px}
 #tk{width:120px;text-transform:uppercase;font-family:ui-monospace,monospace;font-weight:700;letter-spacing:1px}
 #wl{width:340px;font-family:ui-monospace,monospace;font-size:12px}
 button{background:var(--buy);color:#04140c;font-weight:700;padding:8px 18px;cursor:pointer;border:none}
 button:hover{filter:brightness(1.1)}
 .tabs{display:flex;gap:4px}.tab{padding:8px 14px;background:transparent;color:var(--dim);border:none;border-radius:5px;cursor:pointer}
 .tab.active{background:var(--panel2);color:var(--gold)}.toggle{color:var(--dim);display:flex;align-items:center;gap:6px;cursor:pointer}
 .main{flex:1;overflow:auto;padding:18px}
 .hd{display:flex;align-items:baseline;gap:14px;margin-bottom:10px}.tk{font-size:30px;font-weight:800}
 .px{font-family:ui-monospace,monospace;font-size:18px}.badge{margin-left:auto;padding:6px 16px;border-radius:6px;font-weight:800;letter-spacing:1px}
 .good{background:#0f2f22;color:var(--buy);border:1px solid var(--buy)}.neutral{background:#2f2710;color:var(--amber);border:1px solid var(--amber)}.bad{background:#2f1414;color:var(--sell);border:1px solid var(--sell)}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:14px}
 #chart{height:300px}
 .report{white-space:pre-wrap;font-family:ui-monospace,Menlo,monospace;line-height:1.55}
 .muted{color:var(--dim)}.loader{color:var(--gold)}iframe{width:100%;height:80vh;border:0;border-radius:10px;background:#fff}
 .wgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px}
 .tile{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--line);border-radius:9px;padding:14px;cursor:pointer}
 .tile:hover{border-color:var(--gold)}.tile.g{border-left-color:var(--buy)}.tile.b{border-left-color:var(--sell)}.tile.n{border-left-color:var(--amber)}
 .tile .t{font-size:19px;font-weight:800}.tile .p{font-family:ui-monospace,monospace;margin:4px 0}
 .tile .v{font-size:11px;letter-spacing:.5px}.tile .sc{float:right;font-family:ui-monospace,monospace;font-weight:800}
 .grid3{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
 .grid2col{display:grid;grid-template-columns:1fr 1fr;gap:20px}
 .ohcard{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:12px;margin-bottom:10px}
 .ohh{display:flex;justify-content:space-between;margin-bottom:6px}.tagpill{font-size:11px;color:var(--amber)}
 .sub{font-size:12px;color:var(--dim);margin-top:3px}.sub a{color:var(--blue,#4F9DE0)}.stat{font-family:ui-monospace,monospace}
 h3{font-size:13px;letter-spacing:1px;margin:16px 0 8px}
 .diag-panel{display:flex;gap:16px}.diag-col{flex:1}.diag-status{display:flex;align-items:center;gap:10px;padding:14px;border-radius:8px;border:1px solid var(--line);margin-bottom:12px}
 .diag-regime{font-size:18px;font-weight:700}.regime-bullish{color:var(--buy)}.regime-bearish{color:var(--sell)}.regime-neutral{color:var(--amber)}
 .ir-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px}
 .ir-card{background:var(--panel2);border:1px solid var(--line);border-radius:6px;padding:10px;text-align:center}
 .ir-symbol{font-size:12px;font-weight:700;color:var(--gold);margin-bottom:4px}
 .ir-value{font-size:14px;font-family:ui-monospace;font-weight:700}
 .ir-positive{color:var(--buy)}.ir-negative{color:var(--sell)}.ir-neutral{color:var(--amber)}
 .corr-table{font-family:ui-monospace;font-size:11px;line-height:1.6;overflow-x:auto}
 .corr-row{display:flex;gap:8px;margin-bottom:4px}
 .corr-label{width:100px;color:var(--dim)}.corr-value{width:60px;text-align:right}
 .buffer-status{font-size:11px;color:var(--dim);margin-top:10px}
 ::-webkit-scrollbar{width:10px;height:10px}::-webkit-scrollbar-thumb{background:var(--line);border-radius:5px}
</style></head><body>
<div class="top"><span class="diamond">◆</span><div><div class="brand">MERIDIAN</div>
  <div class="sublabel">QUANTITATIVE&nbsp;&nbsp;TRADING&nbsp;&nbsp;TERMINAL</div></div>
  <div class="spacer"></div><span class="pill" id="sess">—</span><span class="clock" id="clock">--:--:-- ET</span></div>
<div class="feeds"><span class="lbl">DATA FEEDS</span>__FEEDS__<span class="stream" id="stream">● STREAMING</span></div>
<div class="ctrl">
  <div class="tabs">
    <button class="tab active" data-v="dash" onclick="view('dash')">Dashboard</button>
    <button class="tab" data-v="analyze" onclick="view('analyze')">Analyze</button>
    <button class="tab" data-v="screen" onclick="view('screen')">Screener</button>
    <button class="tab" data-v="watchlist" onclick="view('watchlist')">Watchlists</button>
    <button class="tab" data-v="mlscreen" onclick="view('mlscreen')">ML Screener</button>
    <button class="tab" data-v="diag" onclick="view('diag')">Diagnostics</button>
    <button class="tab" data-v="ah" onclick="view('ah')">After-Hours</button>
    <button class="tab" data-v="mb" onclick="view('mb')">Morning</button>
    <button class="tab" data-v="tr" onclick="view('tr')">Track Record</button>
  </div>
  <span class="spacer"></span>
  <input id="tk" value="NVDA" onkeydown="if(event.key==='Enter'){view('analyze');go()}">
  <button onclick="view('analyze');go()">Analyze</button>
  <label class="toggle"><input type="checkbox" id="demo" style="width:auto"> Demo</label>
</div>
<div class="ctrl" id="wlrow"><span class="lbl">WATCHLIST</span>
  <input id="wl" value="NVDA,AMD,AAPL,MSFT,TSLA,SOFI,PLTR,AMZN">
  <button onclick="refresh()">Refresh</button><span class="muted" id="wlnote"></span></div>
<div class="main" id="main"></div>
<script>
const $=id=>document.getElementById(id); let V='dash', chart=null, timer=null;
function demo(){return $('demo').checked?1:0} function wl(){return encodeURIComponent($('wl').value)}
function renderVerdict(data){const r=(data.regime||'').toLowerCase();
 const color=r==='bull'?'var(--buy)':r==='bear'?'var(--sell)':'var(--amber)';
 return '<span class="pill" style="background-color:'+color+';color:#04140c">'+(data.regime||'—').toUpperCase()+'</span>';}
function clock(){const n=new Date(new Date().toLocaleString('en-US',{timeZone:'America/New_York'}));
 $('clock').textContent=n.toTimeString().slice(0,8)+' ET';const m=n.getHours()*60+n.getMinutes();let s='CLOSED';
 if(n.getDay()>0&&n.getDay()<6){if(m>=240&&m<570)s='PRE-MARKET';else if(m>=570&&m<960)s='MARKET OPEN';else if(m>=960&&m<1200)s='AFTER HOURS';}
 $('sess').textContent=s;const live=s!=='CLOSED';$('stream').textContent=(live?(n.getSeconds()%2?'○':'●'):'○')+(live?' STREAMING':' IDLE');
 $('stream').style.color=live?'var(--buy)':'var(--dim)';}
setInterval(clock,1000);clock();
function view(v){V=v;document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active',t.dataset.v===v));
 $('wlrow').style.display=(v==='dash'||v==='ah'||v==='mb')?'flex':'none';
 if(timer){clearInterval(timer);timer=null;}
 if(v==='dash'){refresh();timer=setInterval(refresh,30000);}
 else if(v==='screen')screen_(); else if(v==='watchlist')watchlist_(); else if(v==='mlscreen')mlscreen_();
 else if(v==='ah')load('/api/afterhours','after-hours');
 else if(v==='mb')load('/api/morning','morning brief'); else if(v==='tr')load('/api/trackrecord','track record');
 else if(v==='diag'){loadDiagnostics();timer=setInterval(loadDiagnostics,5000);}
 else if(v==='analyze')$('main').innerHTML='<div class="muted">Type a ticker → Analyze.</div>';}
async function watchlist_(){$('main').innerHTML='<div class="loader">Loading watchlist categories…</div>';
 try{const cats=await(await fetch('/api/categories')).json();
  let h='<div style="padding:20px"><h2 style="color:#C8A24B;margin-bottom:16px">Watchlist Categories</h2>';
  h+='<p style="color:#6B7E92;margin-bottom:20px">Select a category to run ML edge detection across all tickers</p>';
  h+='<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px">';
  for(const[k,v]of Object.entries(cats)){
   h+=`<div style="background:#10161F;border:1px solid #232F3D;border-radius:8px;padding:16px;cursor:pointer" onclick="screenCategory('${k}')">
    <div style="font-weight:700;font-size:15px;color:#C9D6E2;margin-bottom:4px">${v.name}</div>
    <div style="font-size:12px;color:#6B7E92;margin-bottom:8px">${v.desc}</div>
    <div style="font-size:11px;color:#4F9DE0">${v.count} tickers</div></div>`;}
  h+='</div></div>';$('main').innerHTML=h;
 }catch(e){$('main').innerHTML='<div class="card" style="color:var(--sell)">'+e+'</div>';}}
async function screenCategory(cat){$('main').innerHTML='<div class="loader">Screening '+cat+' category…</div>';
 try{const d=await(await fetch('/api/category_screen?category='+cat+'&demo='+demo())).json();
  if(d.error){$('main').innerHTML='<div class="card" style="color:var(--sell)">'+d.error+'</div>';return;}
  const f=document.createElement('iframe');f.srcdoc=d.html;$('main').innerHTML='';$('main').appendChild(f);
 }catch(e){$('main').innerHTML='<div class="card" style="color:var(--sell)">'+e+'</div>';}}
async function refresh(){$('wlnote').textContent='updating…';
 try{const [d, cats, news]=await Promise.all([
  fetch('/api/watchlist?demo='+demo()+'&tickers='+wl()).then(r=>r.json()),
  fetch('/api/categories').then(r=>r.json()),
  fetch('/api/news?demo='+demo()+'&tickers='+wl()).then(r=>r.json())]);
  let h='<div style="padding:0"><h2 style="color:var(--gold);margin:0 0 12px;font-size:16px">YOUR WATCHLIST</h2>';
  h+='<div class="wgrid">';for(const r of d){const c=r.tone==='good'?'g':r.tone==='bad'?'b':'n';
   const cc=r.chg>=0?'var(--buy)':'var(--sell)';
   h+=`<div class="tile ${c}" onclick="$('tk').value='${r.ticker}';view('analyze');go()">
     <span class="sc" style="color:${r.tone==='good'?'var(--buy)':r.tone==='bad'?'var(--sell)':'var(--amber)'}">${r.score>0?'+':''}${r.score}</span>
     <div class="t">${r.ticker} <span style="color:var(--amber)">${r.whale}</span></div>
     <div class="p">${r.last} <span style="color:${cc}">${r.chg>=0?'+':''}${r.chg}%</span></div>
     <div class="v" style="color:${r.tone==='good'?'var(--buy)':r.tone==='bad'?'var(--sell)':'var(--amber)'}">${r.verdict}</div></div>`;}
  h+='</div>';
  h+='<h2 style="color:var(--gold);margin:20px 0 12px;font-size:16px">RECENT NEWS & SENTIMENT</h2>';
  if(news.html){h+=news.html;}else{h+='<div class="muted">No news data available.</div>';}
  h+='<h2 style="color:var(--gold);margin:20px 0 12px;font-size:16px">DISCOVER BY STRATEGY</h2>';
  h+='<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px">';
  for(const[k,v]of Object.entries(cats)){
   h+=`<div style="background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px;cursor:pointer;transition:all 0.2s" onmouseover="this.style.borderColor='var(--gold)'" onmouseout="this.style.borderColor='var(--line)'" onclick="screenCategory('${k}')">
    <div style="font-weight:700;font-size:13px;color:var(--gold);margin-bottom:4px;letter-spacing:0.5px">${v.name.toUpperCase()}</div>
    <div style="font-size:11px;color:var(--dim);line-height:1.4;margin-bottom:8px">${v.desc}</div>
    <div style="display:flex;justify-content:space-between;align-items:center">
     <span style="font-size:10px;color:var(--dim)">${v.count} tickers</span>
     <span style="color:var(--blue);font-weight:700">→</span></div></div>`;}
  h+='</div></div>';$('main').innerHTML=h;$('wlnote').textContent='updated '+new Date().toLocaleTimeString();
 }catch(e){$('main').innerHTML='<div class="card" style="color:var(--sell)">'+e+'</div>';}}
async function go(){const t=$('tk').value.trim().toUpperCase()||'NVDA';
 $('main').innerHTML='<div class="loader">Analyzing '+t+'… technicals, fundamentals, alt-data, order flow…</div>';
 try{const [a,o]=await Promise.all([fetch('/api/analyze?demo='+demo()+'&ticker='+t).then(r=>r.json()),
    fetch('/api/ohlc?demo='+demo()+'&ticker='+t).then(r=>r.json())]);
  if(a.error){$('main').innerHTML='<div class="card" style="color:var(--sell)">'+a.error+'</div>';return;}
  const cls=a.tone==='good'?'good':a.tone==='bad'?'bad':'neutral',cc=a.chg>=0?'var(--buy)':'var(--sell)';
  $('main').innerHTML='<div class="hd"><span class="tk">'+a.ticker+'</span>'
   +'<span class="px">'+a.last.toFixed(2)+' <span style="color:'+cc+'">'+(a.chg>=0?'+':'')+a.chg+'%</span></span>'
   +renderVerdict(a)
   +'<span class="badge '+cls+'">'+a.verdict+'</span></div>'
   +'<div class="card"><div id="chart"></div></div><div class="card report">'+a.report+'</div>';
  drawChart(o.bars);
 }catch(e){$('main').innerHTML='<div class="card" style="color:var(--sell)">'+e+'</div>';}}
function drawChart(bars){const el=$('chart');if(!el||!window.LightweightCharts)return;
 chart=LightweightCharts.createChart(el,{autoSize:true,layout:{background:{color:'#10161F'},textColor:'#C9D6E2'},
   grid:{vertLines:{color:'#1b2532'},horzLines:{color:'#1b2532'}},rightPriceScale:{borderColor:'#232F3D'},
   timeScale:{borderColor:'#232F3D'},crosshair:{mode:0}});
 const s=chart.addCandlestickSeries({upColor:'#2ECC8F',downColor:'#FF5449',wickUpColor:'#2ECC8F',wickDownColor:'#FF5449',borderVisible:false});
 s.setData(bars);chart.timeScale().fitContent();}
async function load(url,name){$('main').innerHTML='<div class="loader">Loading '+name+'…</div>';
 try{const d=await(await fetch(url+'?demo='+demo()+'&tickers='+wl())).json();
  $('main').innerHTML=d.error?'<div class="card" style="color:var(--sell)">'+d.error+'</div>':d.html;
 }catch(e){$('main').innerHTML='<div class="card" style="color:var(--sell)">'+e+'</div>';}}
async function screen_(){$('main').innerHTML='<div class="loader">Screening…</div>';
 try{const d=await(await fetch('/api/screen?demo='+demo()+'&tickers='+wl())).json();
  if(d.error){$('main').innerHTML='<div class="card" style="color:var(--sell)">'+d.error+'</div>';return;}
  const f=document.createElement('iframe');f.srcdoc=d.html;$('main').innerHTML='';$('main').appendChild(f);
 }catch(e){$('main').innerHTML='<div class="card" style="color:var(--sell)">'+e+'</div>';}}
async function loadDiagnostics(){
 try{const d=await(await fetch('/api/diagnostics?demo='+demo()+'&tickers='+wl())).json();
  if(d.error){$('main').innerHTML='<div class="card" style="color:var(--sell)">'+d.error+'</div>';return;}
  let h='<div class="diag-panel"><div class="diag-col"><div style="padding:10px 0"><h3>📊 Strategy Health</h3>';
  const h_status=d.health_status||{};const regime=h_status.regime||'neutral';
  const regimeClass='regime-'+regime;h+=`<div class="diag-status"><div class="diag-regime ${regimeClass}">${regime.toUpperCase()}</div>
   <div style="flex:1"><div style="color:var(--dim);font-size:11px">Market Regime</div>
   <div style="font-size:16px;font-weight:700">Score: ${h_status.score||0}</div></div></div>`;
  if(d.status==='warming_up'){h+='<div class="muted">Warming up buffers… '+Object.values(d.buffers||{}).map(v=>v+'/20').join(' | ')+'</div>';}
  h+='</div><div style="padding:10px 0"><h3>💹 Factor Information Ratios</h3><div class="ir-grid">';
  const irs=d.factor_irs||{};for(const[sym,ir] of Object.entries(irs)){
   const irClass=ir>1.5?'ir-positive':ir<-1.5?'ir-negative':'ir-neutral';
   h+=`<div class="ir-card"><div class="ir-symbol">${sym}</div><div class="ir-value ${irClass}">${ir>=0?'+':''}${ir.toFixed(2)}</div></div>`;}
  h+='</div></div></div><div class="diag-col"><div style="padding:10px 0"><h3>🔗 Correlation Matrix</h3>';
  const corr=d.correlation_matrix||{};if(Object.keys(corr).length){
   h+='<div class="corr-table">';for(const[pair,val] of Object.entries(corr)){
    const corrClass=val>0.5?'buy':val<-0.5?'sell':'dim';const c=corrClass==='buy'?'var(--buy)':corrClass==='sell'?'var(--sell)':'var(--dim)';
    h+=`<div class="corr-row"><div class="corr-label">${pair}</div><div class="corr-value" style="color:${c}">${val>=0?'+':''}${val.toFixed(2)}</div></div>`;}
   h+='</div>';}else{h+='<div class="muted">Calculating correlations…</div>';}
  h+='<div class="buffer-status" style="margin-top:14px;padding-top:10px;border-top:1px solid var(--line)">';
  h+='<div style="font-size:11px;margin-bottom:6px">Buffer Status (min 20 bars)</div>';
  for(const[sym,size] of Object.entries(d.buffers||{})){h+=`<div>${sym}: ${size}/20</div>`;}
  h+='</div></div></div></div>';$('main').innerHTML=h;
 }catch(e){$('main').innerHTML='<div class="card" style="color:var(--sell)">'+e+'</div>';}}
view('dash');
</script></body></html>""").replace("__FEEDS__",
    _pill("FINNHUB", True) + _pill("ALPACA·SIP", _F["alpaca"]) + _pill("QUIVER", _F["quiver"])
    + _pill("ALPHA·V", _F["av"]) + _pill("SEC·EDGAR", True))


if __name__ == "__main__":
    main()

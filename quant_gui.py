#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUANT ENGINE — desktop app (window version) with built-in SCREENER.

Two tabs:
  • Analyze   — one stock: full factor report + backtest + sizing + verdict
  • Screener  — rank a whole watchlist, filter out the traps, chart them
                (charts open in your browser via a button)

Requirements: just Python (Tkinter ships with the python.org installer).
Keep this file in the SAME folder as quant_engine.py.

Run once from Terminal to open the window:
    cd ~/Downloads && python3 quant_gui.py
or double-click "Quant Engine.command".
"""
import datetime as dt
import html as _html
import math
import os
import queue
import sys
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import quant_engine as qe
import fundamental_engine as fe
import sentiment_engine as se
import afterhours as ah
import edgar
import morning as mb
import orderflow as of
import trackrecord as tr
import confirmation as cf
import leaderboard as lb
from meridian_cache import MeridianCache

try:
    import tkinter as tk
    from tkinter import ttk, font as tkfont
    from tkinter.scrolledtext import ScrolledText
    HAS_TK = True
except ImportError:
    HAS_TK = False

BG, PANEL, LINE, TXT, DIM = "#0A0E15", "#10161F", "#232F3D", "#C9D6E2", "#6B7E92"
PANEL2 = "#161F2B"
BUY, SELL, AMBER, BLUE = "#2ECC8F", "#FF5449", "#E0A83B", "#4F9DE0"
GOLD = "#C8A24B"
TONE_HEX = {"good": BUY, "neutral": AMBER, "bad": SELL}
DEFAULT_WATCHLIST = "NVDA, AMD, AAPL, MSFT, TSLA, SOFI, PLTR, AMZN"
MIN_REFRESH_SEC = 30


def watchlist_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchlist.txt")


def load_saved_watchlist():
    """Read watchlist.txt next to the app if present, else the default list."""
    try:
        toks = []
        with open(watchlist_path()) as f:
            for line in f:
                line = line.split("#")[0].strip()
                if line:
                    toks.append(line.upper())
        if toks:
            return ", ".join(toks)
    except Exception:
        pass
    return DEFAULT_WATCHLIST


def save_watchlist_file(tickers):
    with open(watchlist_path(), "w") as f:
        f.write("# Quant Engine watchlist — one ticker per line. Edit freely.\n")
        f.write("\n".join(tickers) + "\n")


def period_for(interval, day_trade):
    """Pick a lookback that (a) stays within yfinance's per-interval history limit
    and (b) gives enough bars (>=60) for stable signals. Interval-driven so it's
    robust even if day_trade is toggled with a non-intraday interval."""
    if interval == "1d":
        return "6mo"
    if interval == "1m":
        return "2d"          # Yahoo caps 1m at ~7 days; 2d ≈ 780 bars, recent focus
    if interval == "2m":
        return "3d" if day_trade else "5d"     # 2m allowed up to 60d
    if interval == "5m":
        return "3d" if day_trade else "5d"
    return "3d" if day_trade else "1mo"        # 15m/30m/1h


# Curated universe of liquid US names — lives in quant_engine so headless
# backend scripts (leaderboard.py) can use it without importing the GUI.
UNIVERSE_LIQUID = qe.UNIVERSE_LIQUID

# US market microcaps/movers are what many swing setups come from — the live
# "movers" source surfaces them dynamically. Best-effort; falls back gracefully.


def fetch_many_concurrent(tickers, period, interval, chunk=50, workers=4):
    """Fetch many tickers concurrently for 2-4x faster downloads.
    Uses ThreadPoolExecutor to download chunks in parallel."""
    import yfinance as yf
    import pandas as pd
    
    out = {}
    
    def fetch_chunk(grp):
        """Download a single chunk of tickers."""
        try:
            df = yf.download(grp, period=period, interval=interval, progress=False,
                           auto_adjust=True, group_by="ticker", threads=True)
        except Exception:
            return {}
        
        if df is None or df.empty:
            return {}
        
        cols = ["Open", "High", "Low", "Close", "Volume"]
        chunk_out = {}
        
        if len(grp) == 1:
            t = grp[0]
            sub = df.copy()
            if isinstance(sub.columns, pd.MultiIndex):
                sub.columns = sub.columns.get_level_values(-1)
            try:
                s = sub[cols].dropna()
                if len(s):
                    chunk_out[t] = s
            except Exception:
                pass
        else:
            lvl0 = df.columns.get_level_values(0)
            for t in grp:
                if t not in lvl0:
                    continue
                try:
                    s = df[t][cols].dropna()
                    if len(s):
                        chunk_out[t] = s
                except Exception:
                    continue
        
        return chunk_out
    
    # Split into chunks and download in parallel
    chunks = [tickers[i:i+chunk] for i in range(0, len(tickers), chunk)]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_chunk, grp): grp for grp in chunks}
        for future in as_completed(futures):
            try:
                result = future.result()
                out.update(result)
            except Exception:
                continue

    return out


def fetch_daily_cached(cache, tickers, period, interval):
    """Daily-bar disk cache in front of the concurrent downloader. Only 1d bars
    are cached (immutable history); other intervals pass straight through. Cache
    misses/failures fall back to a live download, so this never breaks a scan."""
    if cache is None or interval != "1d":
        return fetch_many_concurrent(tickers, period, interval, workers=4)
    out, need = {}, []
    for t in tickers:
        try:
            if cache.is_valid(t, period):
                df = cache.get(t, period)
                if len(df) >= 60:
                    out[t] = df
                    continue
        except Exception:
            pass
        need.append(t)
    if need:
        fresh = fetch_many_concurrent(need, period, interval, workers=4)
        for t, df in fresh.items():
            try:
                cache.save(t, df, period)
            except Exception:
                pass
            out[t] = df
    return out


# Intraday bars can't use the daily disk cache (that key is per-day), and they
# go stale within one bar period — so an in-memory short-TTL cache is the right
# tool. TTL ≈ one bar, so back-to-back scans and fast auto-refreshes are instant
# while still re-pulling once a new bar has formed.
_INTRADAY_TTL = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "60m": 3600}


def _check_api_url(url, headers=None, timeout=3):
    """Check if an API endpoint is accessible."""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
            return True
    except Exception:
        return False


def _finnhub_available():
    """Check if Finnhub API is accessible."""
    return _check_api_url("https://finnhub.io/api/v1/quote?symbol=AAPL&token=c9c4iyhr01qq7nfq51c0")


def _sec_available():
    """Check if SEC EDGAR API is accessible."""
    return _check_api_url("https://www.sec.gov/files/company_tickers.json",
                         headers={"User-Agent": "Meridian Research meridian-app contact@example.com"})


class IntradayCache:
    """Short-TTL intraday price cache: in-memory + disk-backed, so reloads AND
    app restarts within one bar period skip the (uncached, flaky) Yahoo download.
    Uses file mtime as the timestamp; prunes files older than a day on startup."""
    def __init__(self, dirpath=None):
        self.store = {}                                  # (ticker,period,interval) -> (df, ts)
        self.dir = dirpath or os.path.join(os.path.expanduser("~/.meridian_cache"), "intraday")
        try:
            os.makedirs(self.dir, exist_ok=True)
            cutoff = time.time() - 86400
            for f in os.listdir(self.dir):               # best-effort prune of stale files
                p = os.path.join(self.dir, f)
                if os.path.isfile(p) and os.path.getmtime(p) < cutoff:
                    os.remove(p)
        except Exception:
            self.dir = None                              # fail open: memory-only

    def _path(self, ticker, period, interval):
        safe = ticker.replace("/", "_").replace(".", "_")
        return os.path.join(self.dir, f"{safe}_{interval}_{period}.pkl")

    def get(self, ticker, period, interval, ttl):
        k = (ticker, period, interval)
        v = self.store.get(k)
        if v and (time.time() - v[1]) < ttl:
            return v[0]
        if self.dir:
            p = self._path(ticker, period, interval)
            try:
                mt = os.path.getmtime(p)
                if time.time() - mt < ttl:
                    import pickle
                    with open(p, "rb") as f:
                        df = pickle.load(f)
                    self.store[k] = (df, mt)
                    return df
            except Exception:
                pass
        return None

    def put(self, ticker, period, interval, df):
        self.store[(ticker, period, interval)] = (df, time.time())
        if self.dir:
            try:
                import pickle
                with open(self._path(ticker, period, interval), "wb") as f:
                    pickle.dump(df, f)
            except Exception:
                pass


# Alpaca real-time intraday bars (free IEX feed). Activated when ALPACA_API_KEY
# and ALPACA_API_SECRET are set; otherwise the app transparently uses Yahoo.
_ALPACA_TF = {"1m": "1Min", "2m": "2Min", "3m": "3Min", "5m": "5Min",
              "15m": "15Min", "30m": "30Min", "1h": "1Hour", "60m": "1Hour"}
_ALPACA_PERIOD_DAYS = {"1d": 1, "2d": 2, "3d": 3, "5d": 5, "1mo": 30,
                       "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}


def alpaca_keys():
    return os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_API_SECRET")


def fetch_alpaca_bars(tickers, interval, period, key, secret):
    """Real-time intraday bars from Alpaca's free IEX feed → {ticker: OHLCV df}.
    Returns {} on any failure so the caller falls back to Yahoo. NOTE: the free
    IEX feed reports only IEX volume (a slice of the consolidated tape), so prices
    are real-time but volume is understated vs Yahoo's all-exchange total."""
    tf = _ALPACA_TF.get(interval)
    if not (key and secret and tf):
        return {}
    import json
    import urllib.parse
    import urllib.request
    import pandas as pd
    days = _ALPACA_PERIOD_DAYS.get(period, 5) + 4                 # pad weekends/holidays
    start = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    hdr = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    cols = {"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}
    out = {}
    for i in range(0, len(tickers), 100):
        grp = tickers[i:i + 100]
        agg, page = {t: [] for t in grp}, None
        for _ in range(20):                                      # bounded pagination
            params = {"symbols": ",".join(grp), "timeframe": tf, "start": start,
                      "limit": 10000, "feed": "iex", "adjustment": "raw"}
            if page:
                params["page_token"] = page
            url = "https://data.alpaca.markets/v2/stocks/bars?" + urllib.parse.urlencode(params)
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=20) as r:
                    data = json.loads(r.read().decode())
            except Exception:
                break
            for t, rows in (data.get("bars") or {}).items():
                agg.setdefault(t, []).extend(rows)
            page = data.get("next_page_token")
            if not page:
                break
        for t, rows in agg.items():
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df.index = pd.to_datetime(df["t"], utc=True)
            sub = df.rename(columns=cols)[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(sub):
                out[t] = sub
    return out


def fetch_prices(daily_cache, intraday_cache, tickers, period, interval):
    """Unified price fetch: daily bars use the persistent disk cache; intraday
    bars come from Alpaca real-time (if keys set) else Yahoo, cached in-memory.
    Only cache MISSES hit the network."""
    if interval == "1d":
        return fetch_daily_cached(daily_cache, tickers, period, interval)
    akey, asec = alpaca_keys()
    alpaca = bool(akey and asec and interval in _ALPACA_TF)
    ttl = 45 if alpaca else _INTRADAY_TTL.get(interval, 300)     # real-time → short TTL
    out, need = {}, []
    for t in tickers:
        df = intraday_cache.get(t, period, interval, ttl) if intraday_cache else None
        if df is not None and len(df) >= 60:
            out[t] = df
        else:
            need.append(t)
    if need:
        fresh = fetch_alpaca_bars(need, interval, period, akey, asec) if alpaca else {}
        still = [t for t in need if t not in fresh]              # Alpaca miss → Yahoo
        if still:
            fresh.update(fetch_many_concurrent(still, period, interval, workers=4))
        for t, df in fresh.items():
            if intraday_cache is not None:
                intraday_cache.put(t, period, interval, df)
            out[t] = df
    return out


def fetch_many(tickers, period, interval, chunk=50):
    """Batch-download many tickers in a few calls (fast, rate-limit friendly).
    Returns {ticker: OHLCV DataFrame}. Missing/failed tickers are skipped."""
    import yfinance as yf
    import pandas as pd
    out = {}
    for i in range(0, len(tickers), chunk):
        grp = tickers[i:i + chunk]
        try:
            df = yf.download(grp, period=period, interval=interval, progress=False,
                             auto_adjust=True, group_by="ticker", threads=True)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        cols = ["Open", "High", "Low", "Close", "Volume"]
        if len(grp) == 1:
            t = grp[0]
            sub = df.copy()
            if isinstance(sub.columns, pd.MultiIndex):
                sub.columns = sub.columns.get_level_values(-1)
            try:
                s = sub[cols].dropna()
                if len(s):
                    out[t] = s
            except Exception:
                pass
        else:
            lvl0 = df.columns.get_level_values(0)
            for t in grp:
                if t not in lvl0:
                    continue
                try:
                    s = df[t][cols].dropna()
                    if len(s):
                        out[t] = s
                except Exception:
                    continue
    return out


class ResultCache:
    """Cache analysis results to skip re-computation on live refresh."""
    def __init__(self, max_age_sec=300):
        self.cache = {}
        self.max_age = max_age_sec
    
    def key(self, ticker, interval, period, demo=False):
        return f"{ticker}:{interval}:{period}:{demo}"

    def get(self, ticker, interval, period, demo=False):
        """Get cached result if available and fresh."""
        k = self.key(ticker, interval, period, demo)
        if k in self.cache:
            cached_res, ts = self.cache[k]
            if time.time() - ts < self.max_age:
                return cached_res
        return None

    def set(self, ticker, interval, period, result, demo=False):
        """Cache result with timestamp."""
        k = self.key(ticker, interval, period, demo)
        self.cache[k] = (result, time.time())
    
    def clear(self):
        """Clear all cached results."""
        self.cache.clear()


def fetch_movers(limit=100):
    """Best-effort list of today's most-active / gapping US stocks via Yahoo.
    Returns a ticker list, or None if the feed isn't available. `count` must be
    passed to yf.screen for predefined queries — it defaults to only 25 (max 250)."""
    try:
        import yfinance as yf
    except Exception:
        return None
    count = max(1, min(int(limit), 250))
    for query in ("most_actives", "day_gainers"):
        try:
            try:
                r = yf.screen(query, count=count)
            except TypeError:            # older yfinance without the count kwarg
                r = yf.screen(query)
            quotes = r.get("quotes") if isinstance(r, dict) else None
            if quotes:
                syms = [q.get("symbol") for q in quotes if q.get("symbol")]
                syms = [s for s in syms if s and "." not in s]
                if syms:
                    return syms[:limit]
        except Exception:
            continue
    return None


def analyze_prefetched(ticker, df, interval):
    res = qe.analyze(ticker, df, interval, None)
    res["dollar_vol"] = dollar_volume(res["d"])
    res["opt"] = None
    return res


def _notify_script(title, message, sound):
    esc = lambda s: str(s).replace("\\", "\\\\").replace('"', '\\"')
    return (f'display notification "{esc(message)}" with title "{esc(title)}" '
            f'sound name "{esc(sound)}"')


def mac_notify(title, message, sound="Glass"):
    """Native macOS banner + sound via osascript. No-op / bell elsewhere."""
    import subprocess
    if sys.platform == "darwin":
        try:
            subprocess.run(["osascript", "-e", _notify_script(title, message, sound)],
                           check=False, timeout=5,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False
    print("\a", end="", flush=True)   # terminal bell fallback
    return False


def detect_new_buys(prev_good, ranked, alert_on):
    """Pure: given last refresh's BUY set and this refresh's ranked results,
    return (newly-turned-BUY tickers, current BUY set). First run (prev None)
    or alerts off => no new signals, just establish the baseline."""
    current_good = {r["ticker"] for r in ranked if r["verdict"]["tone"] == "good"}
    if prev_good is None or not alert_on:
        return set(), current_good
    return current_good - prev_good, current_good


# ==========================================================================
# PURE LOGIC (no Tk) — screening + report building, unit-testable
# ==========================================================================
def dollar_volume(d, n=20):
    v = (d["Close"] * d["Volume"]).tail(n)
    return float(v.mean()) if len(v) else 0.0


_CRYPTO_TF = {"1d": "1Day", "1h": "1Hour", "60m": "1Hour", "30m": "30Min",
              "15m": "15Min", "5m": "5Min", "2m": "2Min", "1m": "1Min"}


def _crypto_pair(ticker):
    """BTC-USD / BTC / btc/usd  ->  BTC/USD (Alpaca's format)."""
    t = ticker.upper().replace("-", "/")
    return t if "/" in t else t + "/USD"


def fetch_alpaca_crypto(tickers, period, interval):
    """Keyless real-time crypto bars from Alpaca → {original_ticker: OHLCV df}.
    No API key and no SDK required (Alpaca's crypto data is fully open). Returns
    {} on any failure so callers fall back to yfinance."""
    tf = _CRYPTO_TF.get(interval)
    if not tf:
        return {}
    import json
    import urllib.parse
    import urllib.request
    import pandas as pd
    days = _ALPACA_PERIOD_DAYS.get(period, 180)
    start = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    pairs = {t: _crypto_pair(t) for t in tickers}
    rows_by_pair, page = {}, None
    for _ in range(20):
        params = {"symbols": ",".join(sorted(set(pairs.values()))),
                  "timeframe": tf, "start": start, "limit": 10000}
        if page:
            params["page_token"] = page
        url = "https://data.alpaca.markets/v1beta3/crypto/us/bars?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                data = json.loads(r.read().decode())
        except Exception:
            break
        for pair, rows in (data.get("bars") or {}).items():
            rows_by_pair.setdefault(pair, []).extend(rows)
        page = data.get("next_page_token")
        if not page:
            break
    cols = {"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}
    out = {}
    for t, pair in pairs.items():
        rows = rows_by_pair.get(pair)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["t"], utc=True)
        sub = df.rename(columns=cols)[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if len(sub):
            out[t] = sub
    return out


def patch_realtime_batch(data, tickers, key, workers=8):
    """Patch each ticker's latest bar with a live Finnhub last price, concurrently.
    Keyless-free real-time for the intraday Screener. Mutates `data` in place;
    each thread writes only its own key, so no locking is needed."""
    if not key:
        return data
    targets = [t for t in tickers if t in data and not qe.is_crypto(t)]
    def one(t):
        try:
            q = qe.finnhub_quote(t, key)
            if q:
                data[t] = qe.patch_realtime(data[t], q)
        except Exception:
            pass
    if targets:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, targets))
    return data


def _fetch_one_cached(cache, ticker, period, interval):
    """Single-ticker fetch. Crypto → Alpaca's keyless real-time feed (fall back to
    yfinance). Equities → disk cache (1d) or yfinance. Fail-open throughout."""
    if qe.is_crypto(ticker):
        got = fetch_alpaca_crypto([ticker], period, interval)
        if ticker in got and len(got[ticker]) >= 60:
            return got[ticker]
        return qe.fetch(ticker, period, interval)      # yfinance BTC-USD fallback
    if cache is not None and interval == "1d":
        try:
            if cache.is_valid(ticker, period):
                df = cache.get(ticker, period)
                if len(df) >= 60:
                    return df
        except Exception:
            pass
        df = qe.fetch(ticker, period, interval)
        try:
            cache.save(ticker, df, period)
        except Exception:
            pass
        return df
    return qe.fetch(ticker, period, interval)


def market_session():
    """Current US market session by ET clock: 'pre', 'open', 'post', or 'closed'.
    Lets the live-price patch label whether it's a regular or aftermarket price."""
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        now = datetime.now()
    if now.weekday() >= 5:
        return "closed"
    mins = now.hour * 60 + now.minute
    if 240 <= mins < 570:   return "pre"      # 4:00–9:30
    if 570 <= mins < 960:   return "open"     # 9:30–16:00
    if 960 <= mins < 1200:  return "post"     # 16:00–20:00
    return "closed"


def screen_one(ticker, demo, period, interval, optimize, cache=None, realtime_key=None):
    df = qe.demo_data(ticker) if demo else _fetch_one_cached(cache, ticker, period, interval)
    quote = None
    if not demo and realtime_key:                   # patch the last bar with a live price
        quote = qe.realtime_quote(ticker, realtime_key,
                                  os.environ.get("ALPACA_API_KEY"),
                                  os.environ.get("ALPACA_API_SECRET"))
        if quote:
            quote["session"] = market_session()
        df = qe.patch_realtime(df, quote)
    d = qe.enrich(df); F = qe.factor_matrix(d)      # compute the heavy steps once
    weights, opt = None, None
    if optimize:
        opt = qe.optimize_weights(F, d["Close"], qe.PPY.get(interval, 252))
        weights = opt["weights"]
    res = qe.analyze(ticker, df, interval, weights, d=d, F=F, calibrate=True)
    res["dollar_vol"] = dollar_volume(res["d"])
    res["opt"] = opt
    res["quote"] = quote
    return res


def verdict_tags(res):
    """Compact list of the signals present on a verdict, for track-record attribution."""
    tags = []
    w = res.get("whale_activity")
    if w and w["whale"]:
        tags.append("whale_accum" if w["direction"] == "accumulation" else "whale_distrib")
    c = res.get("congress")
    if c and (c.get("recent_buys", 0) - c.get("recent_sells", 0)) > 0:
        tags.append("congress_buy")
    if res.get("opt"):
        tags.append("optimized")
    if res.get("fwd_stats") and res["fwd_stats"].get("edge", 0) > 0.05:
        tags.append("pos_edge")
    return tags


def log_verdicts(results, demo):
    """Persist the app's live verdicts so it can grade itself later (skips demo)."""
    if demo or not results:
        return
    try:
        tr.log_verdicts([{"ticker": r["ticker"], "tone": r["verdict"]["tone"],
                          "label": r["verdict"]["label"], "score": r["score"],
                          "price": r["last"], "tags": verdict_tags(r)} for r in results])
    except Exception:
        pass


def passes_filters(res, exclude_risky, buy_only, min_dollar_vol, whale_only=False):
    v = res["verdict"]
    if exclude_risky and v["risky"]:
        return False
    if buy_only and v["tone"] != "good":
        return False
    if min_dollar_vol and res["dollar_vol"] < min_dollar_vol:
        return False
    if whale_only:                       # ride whale ACCUMULATION: big volume + buying pressure
        w = res.get("whale_activity")
        if not (w and w["whale"] and w["direction"] == "accumulation"):
            return False
    return True


def fmt_vol(x):
    if x >= 1e9: return f"${x/1e9:.1f}B"
    if x >= 1e6: return f"${x/1e6:.1f}M"
    if x >= 1e3: return f"${x/1e3:.0f}K"
    return f"${x:.0f}"


def congress_net(summary):
    """Format a ticker's recent congressional net as (label, tag) or None.
    e.g. 3 recent buys / 1 sell -> ('▲3/▼1', 'buy')."""
    if not summary:
        return None
    b, s = summary.get("recent_buys", 0), summary.get("recent_sells", 0)
    if b == 0 and s == 0:
        return None
    tag = "buy" if b > s else "sell" if s > b else "dim"
    return (f"▲{b}/▼{s}", tag)


def _congress_html(summary):
    cn = congress_net(summary)
    if not cn:
        return ""
    color = {"buy": BUY, "sell": SELL, "dim": DIM}[cn[1]]
    return f'<span style="color:{color}">congress {cn[0]}</span>'


def _fund_html(fund):
    s = fe.fmt_fund(fund)
    return f'<span style="color:{DIM}">{s}</span>' if s else ""


def sparkline(d, w=280, h=70, bars=60):
    dd = d.tail(bars)
    close = dd["Close"].to_numpy()
    e20 = dd["e20"].to_numpy() if "e20" in dd else close
    vol = dd["Volume"].to_numpy()
    if len(close) < 2:
        return f'<svg width="{w}" height="{h}"></svg>'
    lo, hi = float(np.nanmin(close)), float(np.nanmax(close))
    if hi == lo: hi, lo = hi + 1, lo - 1
    pad = (hi - lo) * 0.1; lo -= pad; hi += pad
    vh = h * 0.28; ph = h - vh - 4
    X = lambda i: i / (len(close) - 1) * (w - 2) + 1
    Y = lambda v: ph - (v - lo) / (hi - lo) * ph + 2
    vmax = float(np.nanmax(vol)) or 1.0
    bw = max(1.0, (w - 2) / len(vol) * 0.7)
    vbars = "".join(
        f'<rect x="{X(i)-bw/2:.1f}" y="{h-(vv/vmax)*vh:.1f}" width="{bw:.1f}" '
        f'height="{(vv/vmax)*vh:.1f}" fill="{BUY if (i==0 or close[i]>=close[i-1]) else SELL}" '
        f'opacity="0.35"/>' for i, vv in enumerate(vol))
    pts = " ".join(f"{X(i):.1f},{Y(close[i]):.1f}" for i in range(len(close)))
    epts = " ".join(f"{X(i):.1f},{Y(e20[i]):.1f}" for i in range(len(e20))
                    if not math.isnan(e20[i]))
    up = close[-1] >= close[0]; lc = BUY if up else SELL
    area = f'{X(0):.1f},{ph+2:.1f} ' + pts + f' {X(len(close)-1):.1f},{ph+2:.1f}'
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">' + vbars
            + f'<polygon points="{area}" fill="{lc}" opacity="0.08"/>'
            + (f'<polyline points="{epts}" fill="none" stroke="{AMBER}" stroke-width="1" opacity="0.7"/>' if epts else "")
            + f'<polyline points="{pts}" fill="none" stroke="{lc}" stroke-width="1.5"/></svg>')


def factor_bars_html(res):
    out = []
    for k in qe.FACTORS:
        val = float(res["F"][k].iloc[-1])
        pct = abs(val) * 50
        col = BUY if val >= 0 else SELL
        side = "left:50%" if val >= 0 else "right:50%"
        out.append(f'<div class="fbar"><span class="fname">{k}</span>'
                   f'<div class="ftrack"><div class="ffill" style="width:{pct:.0f}%;{side};'
                   f'background:{col}"></div></div><span class="fval" style="color:{col}">{val:+.2f}</span></div>')
    return "".join(out)


def build_screener_html(results, filtered_out, interval, demo, filt_txt):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    ranked = sorted(results, key=lambda r: -r["score"])
    rows = []
    for i, r in enumerate(ranked, 1):
        v = r["verdict"]; col = TONE_HEX[v["tone"]]; bt = r["bt"]
        sh = "—" if math.isnan(bt["sharpe"]) else f"{bt['sharpe']:.2f}"
        risky = ' <span class="chip risky">RISKY</span>' if v["risky"] else ""
        rows.append(f'<tr><td class="num">{i}</td><td class="tkr">{_html.escape(r["ticker"])}</td>'
                    f'<td class="num">{r["last"]:.2f}</td>'
                    f'<td class="num" style="color:{BUY if r["chg"]>=0 else SELL}">{r["chg"]:+.2f}%</td>'
                    f'<td class="num" style="color:{col};font-weight:700">{r["score"]:+.0f}</td>'
                    f'<td style="color:{col}">{v["label"]}{risky}</td>'
                    f'<td class="num">{sh}</td><td class="num">{fmt_vol(r["dollar_vol"])}</td>'
                    f'<td class="num">{r["atr_pct"]:.1f}%</td></tr>')
    cards = []
    for r in ranked:
        v = r["verdict"]; col = TONE_HEX[v["tone"]]; bt = r["bt"]
        sh = "—" if math.isnan(bt["sharpe"]) else f"{bt['sharpe']:.2f}"
        wr = "—" if math.isnan(bt["winrate"]) else f"{bt['winrate']*100:.0f}%"
        risky = '<span class="chip risky">RISKY</span>' if v["risky"] else ""
        cards.append(f'<div class="card"><div class="chd">'
                     f'<span class="ctkr">{_html.escape(r["ticker"])}</span>'
                     f'<span class="cpx">{r["last"]:.2f} <span style="color:{BUY if r["chg"]>=0 else SELL}">{r["chg"]:+.2f}%</span></span>'
                     f'<span class="cv" style="background:{col}22;border:1px solid {col};color:{col}">{v["label"]} {risky}</span></div>'
                     f'<div class="chart">{sparkline(r["d"])}</div>'
                     f'<div class="factors">{factor_bars_html(r)}</div>'
                     f'<div class="cstats"><span>score <b style="color:{col}">{r["score"]:+.0f}</b></span>'
                     f'<span>conv {r["conviction"]}%</span><span>Sharpe {sh}</span><span>win {wr}</span>'
                     f'<span>vol {fmt_vol(r["dollar_vol"])}</span><span>ATR {r["atr_pct"]:.1f}%</span>'
                     f'{_congress_html(r.get("congress"))}'
                     f'{_fund_html(r.get("fund"))}</div></div>')
    dropped = ""
    if filtered_out:
        items = " ".join(f'<span class="drop">{_html.escape(t)}</span>' for t in filtered_out)
        dropped = f'<div class="dropped">Filtered out ({len(filtered_out)}): {items}</div>'
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Quant Screener</title>
<style>
 body{{margin:0;background:{BG};color:{TXT};font-family:-apple-system,Inter,system-ui,sans-serif;padding:20px}}
 .num,.tkr,.ctkr{{font-family:ui-monospace,Menlo,monospace}}
 h1{{font-size:20px;margin:0 0 2px}} .sub{{color:{DIM};font-size:12px;margin-bottom:16px}}
 table{{width:100%;border-collapse:collapse;background:{PANEL};border:1px solid {LINE};border-radius:8px;overflow:hidden;margin-bottom:22px}}
 th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid {LINE};font-size:13px}}
 th{{color:{DIM};font-size:10px;letter-spacing:1px;text-transform:uppercase;cursor:pointer}}
 th:hover{{color:{TXT}}} td.num,th.num{{text-align:right}} .tkr{{font-weight:700}}
 tr:last-child td{{border-bottom:none}} tr:hover td{{background:{LINE}55}}
 .chip{{font-size:9px;padding:1px 5px;border-radius:3px}} .risky{{background:{SELL}22;border:1px solid {SELL};color:{SELL}}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}
 .card{{background:{PANEL};border:1px solid {LINE};border-radius:8px;padding:12px}}
 .chd{{display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap}}
 .ctkr{{font-size:16px;font-weight:700}} .cpx{{font-family:ui-monospace,monospace;font-size:13px}}
 .cv{{margin-left:auto;font-size:10px;padding:2px 8px;border-radius:4px;font-weight:600}}
 .chart{{margin:4px 0 8px}} .factors{{display:flex;flex-direction:column;gap:3px;margin-bottom:8px}}
 .fbar{{display:flex;align-items:center;gap:6px;font-size:10px}} .fname{{width:64px;color:{DIM}}}
 .ftrack{{flex:1;height:8px;background:{BG};border-radius:2px;position:relative;overflow:hidden}}
 .ftrack::after{{content:"";position:absolute;left:50%;top:0;bottom:0;width:1px;background:{LINE}}}
 .ffill{{position:absolute;top:0;bottom:0;border-radius:2px}} .fval{{width:34px;text-align:right;font-family:ui-monospace,monospace}}
 .cstats{{display:flex;flex-wrap:wrap;gap:10px;font-size:11px;color:{DIM};border-top:1px solid {LINE};padding-top:8px}}
 .cstats b{{font-family:ui-monospace,monospace}} .dropped{{color:{DIM};font-size:12px;margin:10px 0 20px}}
 .drop{{font-family:ui-monospace,monospace;background:{PANEL};border:1px solid {LINE};border-radius:3px;padding:1px 6px;margin:0 2px;color:{SELL}}}
 .disc{{color:{DIM};font-size:11px;margin-top:20px;line-height:1.5;border-top:1px solid {LINE};padding-top:14px}}
</style></head><body>
<h1>Quant Screener</h1>
<div class="sub">{now} · {len(results)} passed / {len(results)+len(filtered_out)} screened · interval {interval} · filters: {filt_txt} {'· DEMO DATA' if demo else ''}</div>
<table id="tbl"><thead><tr>
 <th class="num" onclick="sortT(0,1)">#</th><th onclick="sortT(1,0)">Ticker</th>
 <th class="num" onclick="sortT(2,1)">Last</th><th class="num" onclick="sortT(3,1)">Chg</th>
 <th class="num" onclick="sortT(4,1)">Score</th><th onclick="sortT(5,0)">Verdict</th>
 <th class="num" onclick="sortT(6,1)">Sharpe</th><th class="num" onclick="sortT(7,1)">$ Vol</th>
 <th class="num" onclick="sortT(8,1)">ATR%</th></tr></thead>
 <tbody>{''.join(rows) if rows else '<tr><td colspan=9 style="color:'+DIM+'">Nothing passed the filters.</td></tr>'}</tbody></table>
{dropped}<div class="grid">{''.join(cards)}</div>
<div class="disc">Not financial advice. No screener guarantees profit — it ranks setups by fixed rules and flags risk; judgment stays yours. Backtests are in-sample, no fees/slippage. Green bars = up bars, amber line = EMA20.</div>
<script>function sortT(c,n){{var t=document.getElementById('tbl'),b=t.tBodies[0],r=Array.from(b.rows),d=t.getAttribute('data-dir')==='asc'?-1:1;t.setAttribute('data-dir',d===1?'asc':'desc');r.sort(function(a,e){{var x=a.cells[c].innerText.replace(/[^0-9.\\-]/g,''),y=e.cells[c].innerText.replace(/[^0-9.\\-]/g,'');return n?(parseFloat(x)-parseFloat(y))*d:a.cells[c].innerText.localeCompare(e.cells[c].innerText)*d;}});r.forEach(function(x){{b.appendChild(x);}});}}</script>
</body></html>"""


def build_formula_reference():
    """The mathematics behind the model, as (text, tag) segments for the Model tab."""
    seg = []
    add = lambda t, tag="txt": seg.append((t, tag))
    def block(title, formula, desc):
        add("  " + formula + "\n", "formula")
        add("    " + desc + "\n\n", "dim")
    def section(name):
        add("\n" + name + "\n", "gold")
        add("─" * 56 + "\n", "dim")

    add("QUANTITATIVE MODEL — METHODOLOGY\n", "big")
    add("Every number in the Analyze and Screener tabs comes from the formulas\n"
        "below. Nothing is a black box.\n", "dim")

    section("1 · TREND — Exponential Moving Averages")
    block("EMA", "EMAₜ = α·Pₜ + (1−α)·EMAₜ₋₁,   α = 2 / (n+1)",
          "Weights recent price more heavily. We use n = 20 and n = 50.")
    block("trend factor", "f_trend = ⅓·[ sgn(P−EMA₂₀) + sgn(EMA₂₀−EMA₅₀) + Supertrend ]",
          "Three agreeing trend cues averaged to the −1…+1 range.")

    section("2 · MOMENTUM — RSI & MACD")
    block("RSI", "RSI = 100 − 100 / (1 + RS),   RS = AvgGain₁₄ / AvgLoss₁₄",
          "Oscillator 0–100. Above 70 overbought, below 30 oversold.")
    block("MACD", "MACD = EMA₁₂ − EMA₂₆   ·   Signal = EMA₉(MACD)   ·   Hist = MACD − Signal",
          "Histogram > 0 = upward momentum building; rollover warns of fading.")

    section("3 · VOLATILITY & RISK")
    block("True Range", "TR = max( H−L,  |H−Cₚ|,  |L−Cₚ| )",
          "Cₚ = previous close. Captures gaps, unlike H−L alone.")
    block("ATR", "ATR = EMA₁₄(TR)",
          "Average True Range — the volatility unit used for stops and sizing.")
    block("z-score", "σ₂₀ = stdev(Close,20)   ·   z = (Close − SMA₂₀) / σ₂₀",
          "Standard deviations from the 20-bar mean; |z|≥2 is stretched.")
    block("annualized vol", "σₐ = stdev(returns) · √(periods per year)",
          "Daily bars → √252. The RISKY flag trips when ATR ≥ 8% of price.")

    section("4 · TREND REGIME — Supertrend")
    block("bands", "UB = (H+L)/2 + 3·ATR₁₀   ·   LB = (H+L)/2 − 3·ATR₁₀",
          "Direction flips when price closes across the active band (+1 up / −1 down).")

    section("5 · VOLUME & PARTICIPATION")
    block("OBV", "OBV = Σ sgn(ΔClose) · Volume",
          "On-Balance Volume — cumulative signed volume; confirms or diverges from price.")
    block("relative volume", "RelVol = Volume / SMA₂₀(Volume)",
          "≥ 3× flags a 'whale' bar; ≥ 1× means today is busier than usual.")
    block("imbalance", "Imb = (ΣVol↑ − ΣVol↓) / (ΣVol↑ + ΣVol↓)   over 20 bars",
          "Up-bar vs down-bar volume; a proxy for net buying/selling pressure.")

    section("6 · COMPOSITE SCORE")
    block("score", "Score = 100 · Σ wᵢ · fᵢ",
          "Weighted blend of the five factors, each in −1…+1, scaled to −100…+100.")
    add("    weights:  ", "dim")
    add("Trend 0.28 · Momentum 0.27 · Volume 0.15 · MeanRev 0.15 · Structure 0.15\n", "formula")
    add("    thresholds:  ", "dim")
    add("≥45 STRONG BUY · ≥18 BUY · −18…18 HOLD · ≤−18 AVOID\n\n", "formula")

    section("7 · BACKTEST METRICS")
    block("returns", "rₜ = Closeₜ / Closeₜ₋₁ − 1   ·   position = 1 when Score>18 else 0",
          "Long-flat rules; equity compounds the strategy's daily returns.")
    block("Sharpe", "Sharpe = ( mean(r) / stdev(r) ) · √(periods per year)",
          "Risk-adjusted return. Negative = the rules have lost on that name historically.")
    block("max drawdown", "MaxDD = min( Equityₜ / max(Equity₀…ₜ) − 1 )",
          "Worst peak-to-trough loss along the equity curve.")

    section("8 · POSITION SIZING (risk-based)")
    block("shares", "N = ⌊ (Account · Risk%) / (2·ATR) ⌋",
          "Sized so a stop-out loses exactly your chosen % of the account.")
    block("levels", "Stop = Entry − 2·ATR   ·   Target = Entry + 4·ATR   (2R)",
          "Two-to-one reward:risk on the ATR-based stop distance.")

    section("9 · WEIGHT OPTIMIZER — Simulated Annealing")
    block("objective", "L = −Sharpe + 3·MaxDD + 2·Turnover   (minimized)",
          "Rewards risk-adjusted return; penalizes deep drawdowns and signal flip-flopping.")
    block("acceptance", "accept worse weights with probability  P = exp(−ΔL / T)",
          "T (temperature) cools geometrically; escapes local optima early, settles late.")
    block("validation", "walk-forward: expand train window fold by fold, test on the NEXT unseen segment",
          "Averaging several out-of-sample periods exposes weights that only worked on one slice.")
    block("calibration", "BUY/STRONG cutoffs fit to this name's forward returns (Analyze tab)",
          "Replaces the fixed 18/45 with score levels that actually preceded gains here. In-sample.")
    block("market-relative", "adjust the live score by strength vs SPY and the market regime",
          "Outperformance helps; a lone gainer in a falling tape is discounted.")

    add("\n", "txt")
    add("Not investment advice. Formulas describe historical price behavior; they do not\n"
        "predict future returns. Backtests exclude fees, slippage, and financing.\n", "dim")
    return seg


def build_live_math_segments(res):
    """Plug THIS stock's live numbers into the key formulas."""
    import numpy as np
    d = res["d"]; row = d.iloc[-1]
    close = float(row["Close"])
    sma20 = float(d["Close"].rolling(20).mean().iloc[-1])
    sd20 = float(d["Close"].rolling(20).std().iloc[-1])
    seg = []
    add = lambda t, tag="txt": seg.append((t, tag))
    add("\nMATHEMATICS — " + res["ticker"] + " (live values)\n", "gold")
    add("─" * 52 + "\n", "dim")
    add(f"  RSI₁₄  = {row['rsi']:.1f}", "formula")
    add(f"   ({'overbought' if row['rsi']>=70 else 'oversold' if row['rsi']<=30 else 'neutral'})\n", "dim")
    add(f"  MACD   = EMA₁₂ − EMA₂₆ = {row['macd']:+.3f}   ·   Hist = {row['mach']:+.3f}\n", "formula")
    add(f"  ATR₁₄  = {row['atr']:.2f}   = {res['atr_pct']:.1f}% of price\n", "formula")
    add(f"  z      = ({close:.2f} − {sma20:.2f}) / {sd20:.2f} = {row['z']:+.2f}\n", "formula")
    add(f"  σₐ     = {res['ann_vol']:.0f}%  annualized volatility\n", "formula")
    bt = res["bt"]
    sh = "n/a" if math.isnan(bt["sharpe"]) else f"{bt['sharpe']:.2f}"
    add(f"  Sharpe = {sh}   ·   MaxDD = {bt['maxdd']*100:.0f}%\n", "formula")
    add("  Score  = 100 · Σ wᵢfᵢ:\n", "formula")
    total = 0.0
    for k, w in [("Trend", 0.28), ("Momentum", 0.27), ("Volume", 0.15),
                 ("MeanRev", 0.15), ("Structure", 0.15)]:
        fi = float(res["F"][k].iloc[-1]); term = w * fi * 100; total += term
        add(f"     {w:.2f}·({fi:+.2f})", "formula")
        add(f" = {term:+.1f}\n", "dim")
    tech_label = qe.verdict(total, res["atr_pct"])["label"]
    add(f"     Σ = {total:+.0f}  →  {tech_label}\n", "gold")
    extra = 0.0
    if res.get("alt"): extra += res["alt"]["adjustment"]
    if res.get("market"): extra += res["market"]["adjustment"]
    if extra:
        add(f"     + tilt {extra:+.1f} (alt-data & market)  →  final {res['score']:+.0f}"
            f"  {res['verdict']['label']}\n", "gold")
    return seg


def build_report_segments(res, opt, account, risk):
    seg = []
    add = lambda t, tag="txt": seg.append((t, tag))
    d = res["d"]; row = d.iloc[-1]; v = res["verdict"]
    vtag = {"good": "buy", "neutral": "warn", "bad": "sell"}[v["tone"]]
    add(f"{res['ticker']}", "big"); add(f"   last {res['last']:.2f}   ", "txt")
    add(f"{res['chg']:+.2f}%", "buy" if res["chg"] >= 0 else "sell")
    add(f"   ·  {len(d)} bars", "dim")
    q = res.get("quote")
    if q:
        sess = q.get("session", "open")
        label = {"pre": "● PRE-MARKET", "post": "● AFTER-HOURS",
                 "closed": "● last (mkt closed)"}.get(sess, "● LIVE")
        add(f"  ·  {label} px ({q.get('source', 'live')})",
            "warn" if sess in ("pre", "post", "closed") else "buy")
    add("\n", "dim"); add("─" * 52 + "\n", "dim")
    if res.get("interval_fallback"):
        add("↳ recent listing — not enough daily history, so this read is on "
            f"{res['interval_fallback']} bars (shorter horizon than the usual daily swing setup)\n", "warn")
        add("─" * 52 + "\n", "dim")
    # ---- LIMITED HISTORY: young listing, indicators not fully formed ----
    if res.get("limited_history"):
        add("⚠ LIMITED HISTORY  ", "head")
        add(f"— only {res.get('n_bars', '?')} bars (recent listing?). EMA50, calibration, "
            f"and the backtest aren't fully formed — treat the read as low-confidence.\n", "warn")
        add("─" * 52 + "\n", "dim")
    # ---- BACKTEST GATE: flag assets the rules have historically lost on ----
    if res.get("ineligible"):
        bt0 = res["bt"]
        sh0 = "n/a" if math.isnan(bt0["sharpe"]) else f"{bt0['sharpe']:.2f}"
        add("⛔ INELIGIBLE ASSET  ", "head")
        add(f"— the strategy has historically LOST on {res['ticker']} "
            f"(Sharpe {sh0}", "sell")
        if not math.isnan(bt0["winrate"]):
            add(f", win {bt0['winrate']*100:.0f}%", "sell")
        tail = " Alt-data skipped to save API calls." if res.get("alt_skipped") else ""
        add(f"). Score shown for reference — do not treat as a buy.{tail}\n", "sell")
        add("─" * 52 + "\n", "dim")
    # ---- EDGE STATUS: NO EDGE detection and track record override ----
    edge_status = v.get("edge_status", "ACTIVE")
    if edge_status in ("NO_EDGE", "OVERRIDDEN"):
        add("EDGE STATUS  ", "head")
        if edge_status == "NO_EDGE":
            add("🔴 NO EDGE", "sell")
            ir = v.get("information_ratio", 0.0)
            wr = v.get("win_rate", 0.5)
            add(f" — Information Ratio {ir:.3f} < 0.05 or Win Rate {wr*100:.0f}% < 50%", "dim")
            add("\n  No statistical edge detected. Signal suppressed (score set to 0).\n", "warn")
        elif edge_status == "OVERRIDDEN":
            add("🟡 OVERRIDDEN", "warn")
            add(" — Track record override active", "dim")
            add("\n  Historical performance (100+ trades or Sharpe > 1.2) overrides current NO EDGE.\n", "dim")
        add("─" * 52 + "\n", "dim")
    # ---- CONFIRMATION SCORE: auto-run the "verify a green light" checklist ----
    try:
        cs = cf.confirm(res)
    except Exception:
        cs = None
    if cs and res["verdict"]["tone"] == "good":
        htag = ("buy" if cs["headline"].startswith("✅") else
                "sell" if cs["headline"].startswith("🔴") else "warn")
        add("CONFIRMATION  ", "head")
        add(cs["headline"], htag)
        add(f"   · {cs['passed']}/{cs['checkable']} signals agree\n", "dim")
        for label, state, detail in cs["checks"]:
            if state == "na":
                continue
            mark, mtag = ("✓", "buy") if state == "pass" else ("✗", "sell")
            add(f"  {mark} ", mtag)
            add(f"{label}", "txt")
            add(f"  {detail}\n" if detail else "\n", "dim")
        for label, detail in cs["kills"]:
            add(f"  🔴 KILL-SWITCH: {label}", "sell"); add(f" — {detail}\n", "dim")
        add("─" * 52 + "\n", "dim")
    add("FACTORS\n", "head")
    for k in qe.FACTORS:
        val = float(res["F"][k].iloc[-1])
        add(f"  {k:<11}", "txt"); add(f"{val:+.2f}\n", "buy" if val >= 0 else "sell")
    add("\n", "txt"); add("EVIDENCE\n", "head")
    for good, txt in qe.reasons(row):
        add("  ▲ " if good else "  ▼ ", "buy" if good else "sell"); add(txt + "\n", "txt")
    add("\n", "txt"); add("RISK\n", "head")
    add(f"  ATR {res['atr']:.2f} ({res['atr_pct']:.1f}% of price) · ann vol {res['ann_vol']:.0f}% · max drawdown {res['maxdd']:.0f}%\n", "txt")
    if v["risky"]:
        add("  ! RISKY — extreme volatility degrades signal reliability\n", "sell")
    add("\n", "txt")
    w = res.get("whale_activity")
    if w:
        add("WHALE ACTIVITY ", "head"); add("(large-money footprint — size, not identity)\n", "dim")
        dtag = "buy" if w["direction"] == "accumulation" else "sell" if w["direction"] == "distribution" else "dim"
        flag = "  🐋 WHALE BAR" if w["whale"] else ""
        add(f"  volume {w['rvol']:.1f}× avg · money flow ", "txt")
        add(f"{w['cmf']:+.2f} ({w['direction']})", dtag)
        add(flag + "\n", "warn")
        if w["whale"]:
            add(f"  ↳ abnormal volume with {w['direction']} pressure — big money is active "
                f"(who/why is unknowable)\n", "dim")
        add("\n", "txt")
    fl = res.get("orderflow")
    if fl and fl.get("n_blocks"):
        add("DARK-POOL BLOCK FLOW ", "head"); add("(FINRA TRF via SIP · recent session)\n", "dim")
        add(f"  ${fl['block_usd']/1e6:.1f}M across {fl['n_blocks']} block(s)", "warn")
        add(f" · {fl['dp_share']*100:.0f}% of prints off-exchange · {fl.get('sweeps',0)} sweeps\n", "dim")
        if "buy_usd" in fl:
            add("  aggressor: ", "txt")
            add(f"${fl['buy_usd']/1e6:.1f}M buy", "buy"); add(" · ", "dim")
            add(f"${fl['sell_usd']/1e6:.1f}M sell", "sell")
            add(f" · ${fl['mid_usd']/1e6:.1f}M ambiguous at midpoint\n", "dim")
            alert = of.block_alert(fl)
            if alert:
                add(f"  🟢 BUY-INITIATED BLOCKS: {alert['detail']}\n", "buy")
        if fl.get("largest"):
            L = fl["largest"]
            add(f"  largest ${L['usd']/1e6:.1f}M ({L['shares']:,} sh @ {L['price']:.2f})\n", "dim")
        add("  ↳ blocks anonymous; midpoint prints have no determinable side (fact, not story)\n", "dim")
        add("\n", "txt")
    bt = res["bt"]
    wr = "n/a" if math.isnan(bt["winrate"]) else f"{bt['winrate']*100:.0f}%"
    sh = "n/a" if math.isnan(bt["sharpe"]) else f"{bt['sharpe']:.2f}"
    add("BACKTEST ", "head"); add("(long score>18, flat <0 · no fees · in-sample)\n", "dim")
    add("  strategy ", "txt"); add(f"{bt['strategy']*100:+.1f}%", "buy" if bt["strategy"] >= 0 else "sell")
    add(" vs buy&hold ", "txt"); add(f"{bt['buyhold']*100:+.1f}%", "buy" if bt["buyhold"] >= 0 else "sell")
    add(f" · trades {bt['trades']} · win {wr} · Sharpe {sh} · maxDD {bt['maxdd']*100:.0f}% · exposure {bt['exposure']*100:.0f}%\n\n", "txt")
    if account:
        # Stop width scales with volatility: high-beta names get a 3×ATR stop so
        # normal noise doesn't stop them out. Risk-based sizing then AUTO-cuts the
        # share count (shares = risk$ / stop-distance), keeping cash risk constant.
        # A defensive news-tone shift instead TIGHTENS to 1.5×ATR.
        defensive = bool((res.get("sentiment") or {}).get("defensive_shift"))
        high_vol = res.get("ann_vol", 0) >= qe.HIGH_VOL
        smult = 1.5 if defensive else (3.0 if high_vol else 2.0)
        ps = qe.position_size(res["last"], res["atr"], account, risk, smult)
        add("POSITION SIZING ", "head")
        add(f"(risk {risk}% of {account:,.0f}, stop {smult:g}×ATR)\n", "dim")
        if defensive:
            add("  ⚠ news tone turned defensive — stop tightened to 1.5×ATR\n", "warn")
        elif high_vol:
            add(f"  ↔ high volatility ({res['ann_vol']:.0f}%) — widened to 3×ATR, "
                f"share size auto-cut to hold cash risk flat\n", "dim")
        if ps:
            add(f"  {ps['shares']} shares (~{ps['notional']:,.0f}) · entry {ps['entry']:.2f} · ", "txt")
            add(f"stop {ps['stop']:.2f}", "sell"); add(" · ", "txt")
            add(f"target {ps['target']:.2f}", "buy"); add(" · ", "txt")
            add(f"risk ~{ps['risk_dollars']:,.0f}", "sell"); add(" · ", "txt")
            add(f"est. profit ~{ps['reward_dollars']:,.0f}", "buy"); add("\n\n", "txt")
        else:
            add("  position too small to size at this risk level\n\n", "dim")
    if opt:
        f = lambda x: "n/a" if (x is None or x <= -9 or math.isnan(x)) else f"{x:.2f}"
        fl = lambda x: "n/a" if (x is None or x >= 999 or math.isnan(x)) else f"{x:.2f}"
        add("ANNEALED WEIGHTS\n", "head"); add(f"  {opt['weights']}\n", "dim")
        add(f"  Sharpe — train {f(opt['train_sharpe'])} · unseen test {f(opt['test_sharpe'])} · default on same test {f(opt['base_test_sharpe'])}\n", "txt")
        if "train_loss" in opt:
            add(f"  objective (−Sharpe + 3·MaxDD + 2·Turnover) — train {fl(opt['train_loss'])} · test {fl(opt['test_loss'])}\n", "dim")
        if "wf_sharpe" in opt:
            folds = " ".join(f(s) for s in opt["wf_folds"])
            add(f"  walk-forward OOS Sharpe {f(opt['wf_sharpe'])} ", "txt")
            add(f"({opt['wf_pos_frac']*100:.0f}% of {len(opt['wf_folds'])} folds positive)", "txt")
            add(f"  folds [{folds}]\n", "dim")
            if opt["wf_pos_frac"] < 0.5:
                add("  caution: most walk-forward folds are negative — weights don't generalize\n", "warn")
            elif isinstance(opt["test_sharpe"], float) and isinstance(opt["train_sharpe"], float) and opt["test_sharpe"] < opt["train_sharpe"] * 0.4:
                add("  caution: big train→test drop = tuning likely overfit\n", "warn")
        elif isinstance(opt["test_sharpe"], float) and isinstance(opt["train_sharpe"], float) and opt["test_sharpe"] < opt["train_sharpe"] * 0.4:
            add("  caution: big train→test drop = tuning likely overfit\n", "warn")
        add("\n", "txt")
    fund = res.get("fund")
    if fund:
        fv = lambda x, fmt: fmt.format(x) if x is not None else "—"
        gv = fund.get("growth")
        add("FUNDAMENTALS ", "head"); add("(Alpha Vantage)\n", "dim")
        add(f"  P/E {fv(fund.get('pe'), '{:.1f}')} · "
            f"D/E {fv(fund.get('de'), '{:.2f}')} · rev growth ", "txt")
        add(fv(gv, "{:+.1f}%"), "txt" if gv is None else ("buy" if gv >= 0 else "sell"))
        add(f" · current ratio {fv(fund.get('current_ratio'), '{:.2f}')}\n", "txt")
        if fund.get("sector"):
            add(f"  {fund.get('name', '')} · {fund['sector']}\n", "dim")
        add("\n", "txt")
    alt = res.get("alt"); market = res.get("market")
    if alt:
        add("ALT-DATA TILT ", "head")
        add("(adjusts the live rating; backtest above is technical-core only)\n", "dim")
        src_map = {"Congress": "Quiver", "Analyst": "Finnhub",
                   "Insider": "SEC Form 4", "WhaleFlow": "options+dark pool",
                   "Macro": "news sentiment"}
        for k, p in alt["parts"].items():
            src = src_map.get(k, "")
            contrib = p["signal"] * p["confidence"]
            add(f"  {k + ' (' + src + ')':<30}", "txt")
            add(f"{contrib:+.2f}", "buy" if contrib >= 0 else "sell")
            add(f"   {p['detail']}\n", "dim")
    if market:
        add("MARKET CONTEXT ", "head"); add("(vs SPY)\n", "dim")
        add("  relative strength ", "txt")
        add(f"{market['rel']*100:+.1f}%", "buy" if market["rel"] >= 0 else "sell")
        add(f" over {market['window']} bars   ", "dim")
        add(f"(stock {market['stock_ret']*100:+.1f}% vs SPY {market['spy_ret']*100:+.1f}%)\n", "dim")
        add("  regime ", "txt")
        add("RISK-ON" if market["risk_on"] else "RISK-OFF",
            "buy" if market["risk_on"] else "sell")
        add(" (SPY " + ("above" if market["risk_on"] else "below") + " its 50-EMA)\n", "dim")
    if alt or market:
        add("  score: ", "txt"); add(f"technical {res['base_score']:+.0f}", "txt")
        if alt:
            add("  ·  ", "dim"); add(f"alt {alt['adjustment']:+.1f}", "buy" if alt["adjustment"] >= 0 else "sell")
        if market:
            add("  ·  ", "dim"); add(f"market {market['adjustment']:+.1f}", "buy" if market["adjustment"] >= 0 else "sell")
        add("  →  ", "dim"); add(f"final {res['score']:+.0f}\n\n", "txt")
    cal = res.get("calib")
    if cal:
        add("THRESHOLDS ", "head")
        add(f"(calibrated to this name's {cal['horizon']}-bar forward returns, {cal['n']} bars)\n", "dim")
        add(f"  BUY ≥ {cal['buy']:+.0f} · STRONG ≥ {cal['strong']:+.0f}", "txt")
        add(f"   (defaults +18 / +45)   avg fwd return above BUY {cal['fwd_mean']*100:+.1f}%\n", "dim")
    else:
        bth, sth = res.get("buy_th", 18.0), res.get("strong_th", 45.0)
        add("THRESHOLDS ", "head")
        if bth > 18.5:                       # volatility-widened above the defaults
            add(f"volatility-scaled BUY ≥ {bth:+.0f} · STRONG ≥ {sth:+.0f} "
                f"(widened from +18/+45 for {res['ann_vol']:.0f}% ann vol)\n", "dim")
        else:
            add(f"default BUY ≥ {bth:+.0f} · STRONG ≥ {sth:+.0f} "
                "(not enough history for a per-name calibration)\n", "dim")
    fw = res.get("fwd_stats")
    if fw:
        add("SIGNAL TRACK RECORD ", "head")
        add(f"(this name's own history, {fw['n']} similar bars — in-sample)\n", "dim")
        add(f"  when score ≈ {fw['score']:+.0f}, next {fw['horizon']} bars were up ", "txt")
        add(f"{fw['win_rate']*100:.0f}%", "buy" if fw["win_rate"] >= 0.5 else "sell")
        add(f" of the time (avg {fw['mean_fwd']*100:+.1f}%)\n", "txt")
        edge = fw["edge"] * 100
        add(f"  vs baseline {fw['base_win_rate']*100:.0f}% up (avg {fw['base_mean_fwd']*100:+.1f}%) → edge ", "dim")
        add(f"{edge:+.0f} pts", "buy" if edge >= 0 else "sell")
        add("\n", "dim")
    add("─" * 52 + "\n", "dim"); add("VERDICT  ", "head"); add(f"score {res['score']:+.0f}   ", "txt")
    add(v["label"], vtag)
    if v["risky"]: add("  [RISKY]", "sell")
    add(f"   · conviction {res['conviction']}%\n", "dim")
    return seg


def build_ml_screener_data(tickers, data):
    """Train per-ticker ML model to predict next-day direction. Returns list of dicts with
    predictions, validation accuracy, and real track record for web display."""
    results = []
    scored = tr.score(lambda t: data.get(t), horizon=1)
    tr_by_ticker = {e["ticker"]: e for e in scored}
    for ticker in tickers:
        df = data.get(ticker)
        if df is None or len(df) < 60:
            continue
        try:
            d = df[["Close"]].copy()
            if "Volume" in df.columns:
                d["Volume"] = df["Volume"]
            d["ret"] = d["Close"].pct_change()
            d["fwd_ret"] = d["ret"].shift(-1)
            d["label"] = (d["fwd_ret"] > 0).astype(int)
            d["ema12"] = d["Close"].ewm(12, adjust=False).mean()
            d["ema20"] = d["Close"].ewm(20, adjust=False).mean()
            d["ema50"] = d["Close"].ewm(50, adjust=False).mean()
            d["dist20"] = (d["Close"] - d["ema20"]) / d["ema20"]
            d["dist50"] = (d["Close"] - d["ema50"]) / d["ema50"]
            d["rsi"] = qe.rsi(d["Close"], 14)
            macd12 = d["Close"].ewm(12, adjust=False).mean()
            macd26 = d["Close"].ewm(26, adjust=False).mean()
            d["macd"] = macd12 - macd26
            d["macd_signal"] = d["macd"].ewm(9, adjust=False).mean()
            d["macd_hist"] = d["macd"] - d["macd_signal"]
            d["mom"] = d["Close"].pct_change(10)
            d["vol"] = d["Close"].rolling(20).std() / d["Close"]
            if "Volume" in d.columns:
                d["rel_vol"] = d["Volume"] / d["Volume"].rolling(20).mean()
            d["up_down"] = np.sign(d["ret"])
            features = ["dist20", "dist50", "rsi", "macd_hist", "mom", "vol"]
            if "rel_vol" in d.columns:
                features.append("rel_vol")
            X = d[features].bfill().ffill().fillna(0).values
            y = d["label"].values
            if len(X) < 60 or np.isnan(X).any() or np.isnan(y).any():
                continue
            train_size = int(len(X) * 0.65)
            val_start = train_size
            val_end = min(train_size + int(len(X) * 0.2), len(X) - 1)
            X_train, y_train = X[:train_size], y[:train_size]
            X_val, y_val = X[val_start:val_end], y[val_start:val_end]
            if len(X_train) < 20 or len(X_val) < 10:
                continue
            mean_X = X_train.mean(axis=0)
            std_X = X_train.std(axis=0)
            std_X[std_X == 0] = 1.0
            X_train_norm = (X_train - mean_X) / std_X
            X_val_norm = (X_val - mean_X) / std_X
            try:
                w = np.linalg.lstsq(X_train_norm, y_train, rcond=None)[0]
                scores_val = X_val_norm @ w
                pred_val = scores_val > np.median(scores_val)
                acc_val = (pred_val == y_val).mean()
                baseline_val = max(y_val.mean(), 1 - y_val.mean())
            except Exception:
                continue
            tr_graded = [e for e in scored if e["ticker"] == ticker and e.get("status") == "scored" and e["win"] is not None]
            tr_text = ""
            if len(tr_graded) >= 3:
                tr_hit = sum(e["win"] for e in tr_graded) / len(tr_graded)
                tr_ret = np.mean([e.get("fwd_ret", 0) for e in tr_graded])
                tr_text = f'{len(tr_graded)} past calls, {tr_hit*100:.0f}% actually correct — avg {tr_ret*100:+.1f}%'
            cur_px = float(df["Close"].iloc[-1])
            results.append({
                "ticker": ticker,
                "pred_pct": int(acc_val * 100),
                "price": float(cur_px),
                "acc_val": float(acc_val),
                "baseline": float(baseline_val),
                "n_train": int(len(X_train)),
                "n_val": int(len(X_val)),
                "tr_text": tr_text,
                "has_edge": bool(acc_val > baseline_val)
            })
        except Exception:
            continue
    return sorted(results, key=lambda r: -(r["acc_val"] - r["baseline"] if r["has_edge"] else -999))


def build_ml_screener_html(ml_data, demo):
    """Format ML screener results as HTML for web display."""
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    cards = []
    for r in ml_data:
        tr_line = f'<div class="sub">real track record: {r["tr_text"]}</div>' if r["tr_text"] else ""
        edge_badge = "" if r["has_edge"] else f'<div class="sub" style="color:{SELL}">NO EDGE — {r["acc_val"]*100:.0f}% accurate, worse than the {r["baseline"]*100:.0f}% naive baseline — don\'t trust this prediction</div>'
        cards.append(f'''<div class="mlcard">
<div class="mlh"><b>{r["ticker"]}</b><span class="mlpct">{r["pred_pct"]}%</span></div>
<div class="sub">${r["price"]:.2f} · predicted odds of being up in 1 trading days</div>
<div class="sub" style="color:{"#2ECC8F" if r["has_edge"] else "#6B7E92"}">validated: {r["acc_val"]*100:.0f}% accurate vs {r["baseline"]*100:.0f}% naive baseline ({r["n_val"]} held-out samples)</div>
{tr_line}
{edge_badge}
<div class="sub" style="color:{DIM}">trained on {r["n_train"]} bars · validated on {r["n_val"]} unseen bars</div>
</div>''')
    if not cards:
        cards = [f'<div style="color:{DIM}">No tickers with sufficient data for ML modeling.</div>']
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ML Screener</title>
<style>
 body{{margin:0;background:{BG};color:{TXT};font-family:-apple-system,Inter,system-ui,sans-serif;padding:20px}}
 h1{{font-size:20px;margin:0 0 2px}} .sub{{color:{DIM};font-size:12px;margin-top:4px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}}
 .mlcard{{background:{PANEL};border:1px solid {LINE};border-radius:8px;padding:12px}}
 .mlh{{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;gap:8px}}
 .mlpct{{font-family:ui-monospace,monospace;font-size:16px;font-weight:700;color:{BLUE}}}
 .disc{{color:{DIM};font-size:11px;margin-top:20px;line-height:1.5;border-top:1px solid {LINE};padding-top:14px}}
</style></head><body>
<h1>PREDICTIVE SCREENER</h1>
<div class="sub">{len(ml_data)} scanned · 2y history · {now}</div>
<h2 style="margin-top:16px;color:{GOLD};font-size:14px">ML MODEL</h2>
<div class="sub">Trains a simple linear model per ticker on its own factor history to predict the odds it's up in 1 trading day, then checks the model against data it never trained on. A high win probability from a model that does NOT beat a naive majority-class guess (NO EDGE) is noise, not a signal. Once a ticker has 3+ real graded calls in Track Record, that ACTUAL forward-tested history outranks this run's synthetic validation — real evidence over a fresh backtest.</div>
<div class="grid">{''.join(cards)}</div>
<div class="disc">Not financial advice. No model guarantees profit. This trains on past behavior; patterns don't predict the future. Green = validated edge (beats baseline); NO EDGE badge = don't use this signal. Real track record badge shows actual forward-tested calls graded after the verdicts age 5 trading days.</div>
</body></html>"""


# ==========================================================================
# GUI
# ==========================================================================
class App:
    def __init__(self, root):
        self.root = root
        root.title("Meridian — Quantitative Research")
        root.configure(bg=BG)
        root.geometry("900x880")
        root.minsize(700, 660)
        self.mono = tkfont.Font(family="Menlo", size=13)
        self.monob = tkfont.Font(family="Menlo", size=13, weight="bold")
        self.big = tkfont.Font(family="Menlo", size=17, weight="bold")
        self.ui = tkfont.Font(family="SF Pro Text", size=13)
        self.small = tkfont.Font(family="SF Pro Text", size=11)
        self.brand = tkfont.Font(family="Georgia", size=19, weight="bold")
        self.brandsub = tkfont.Font(family="SF Pro Text", size=9)

        style = ttk.Style()
        try: style.theme_use("clam")
        except tk.TclError: pass
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG, foreground=DIM,
                        padding=(22, 9), font=self.ui, borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", PANEL)],
                  foreground=[("selected", GOLD)])
        style.configure("TCombobox", fieldbackground=PANEL2, background=PANEL2,
                        foreground=TXT, arrowcolor=DIM, borderwidth=0, relief="flat")
        style.map("TCombobox", fieldbackground=[("readonly", PANEL2)],
                  foreground=[("readonly", TXT)], selectbackground=[("readonly", PANEL2)])

        # ---------- header ----------
        header = tk.Frame(root, bg=PANEL, height=64); header.pack(fill="x", side="top")
        header.pack_propagate(False)
        left = tk.Frame(header, bg=PANEL); left.pack(side="left", padx=20)
        tk.Label(left, text="◆", bg=PANEL, fg=GOLD,
                 font=tkfont.Font(family="Georgia", size=22)).pack(side="left", padx=(0, 10))
        wm = tk.Frame(left, bg=PANEL); wm.pack(side="left")
        tk.Label(wm, text="MERIDIAN", bg=PANEL, fg=TXT, font=self.brand).pack(anchor="w")
        tk.Label(wm, text="QUANTITATIVE  RESEARCH  TERMINAL", bg=PANEL, fg=GOLD,
                 font=self.brandsub).pack(anchor="w")
        right = tk.Frame(header, bg=PANEL); right.pack(side="right", padx=20)
        self.h_clock = tk.Label(right, text="--:--:-- ET", bg=PANEL, fg=TXT, font=self.monob)
        self.h_clock.pack(side="right", padx=(12, 0))
        self.h_pill = tk.Label(right, text="—", bg=PANEL2, fg=DIM, font=self.small, padx=9, pady=3)
        self.h_pill.pack(side="right")
        tk.Frame(root, bg=GOLD, height=2).pack(fill="x", side="top")

        # ---------- data-feed status strip (trader-terminal chrome) ----------
        alpaca_on = bool(os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_API_SECRET"))
        feedbar = tk.Frame(root, bg=PANEL2, height=26); feedbar.pack(fill="x", side="top")
        feedbar.pack_propagate(False)
        tk.Label(feedbar, text="DATA FEEDS", bg=PANEL2, fg=DIM,
                 font=self.brandsub).pack(side="left", padx=(20, 12))
        self.feed_labels = {}
        feeds_def = [("FINNHUB", _finnhub_available), ("ALPACA·SIP", lambda: alpaca_on),
                     ("QUIVER", lambda: bool(os.environ.get("QUIVER_API_TOKEN"))),
                     ("ALPHA·V", lambda: bool(os.environ.get("ALPHA_VANTAGE_KEY"))),
                     ("SEC·EDGAR", _sec_available)]
        for name, check_fn in feeds_def:
            lbl = tk.Label(feedbar, text=f"● {name}", bg=PANEL2, fg=BUY, font=self.small)
            lbl.pack(side="left", padx=(0, 14))
            self.feed_labels[name] = (lbl, check_fn)
        self.h_live = tk.Label(feedbar, text="● STREAMING", bg=PANEL2, fg=BUY, font=self.small)
        self.h_live.pack(side="right", padx=(0, 20))
        self._update_feeds()
        tk.Frame(root, bg=LINE, height=1).pack(fill="x", side="top")

        # ---------- footer / status bar ----------
        footer = tk.Frame(root, bg=PANEL, height=26); footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        src = ("Alpaca SIP real-time + Yahoo" if alpaca_on else "Yahoo Finance (~15-min delayed)")
        sec_status = " · SEC EDGAR" if _sec_available() else ""
        tk.Label(footer, text=f"⚠ Research output — not investment advice", bg=PANEL, fg=AMBER,
                 font=self.small).pack(side="left", padx=(16, 12))
        tk.Label(footer, text=f"· data: {src}{sec_status}", bg=PANEL, fg=DIM,
                 font=self.small).pack(side="left")
        tk.Label(footer, text="MERIDIAN QUANT  ·  v2.0", bg=PANEL, fg=GOLD,
                 font=self.brandsub).pack(side="right", padx=16)

        # ---------- tabs ----------
        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True, padx=10, pady=(8, 8))
        self.tab_analyze = tk.Frame(nb, bg=BG)
        self.tab_screen = tk.Frame(nb, bg=BG)
        self.tab_lead = tk.Frame(nb, bg=BG)
        self.tab_model = tk.Frame(nb, bg=BG)
        nb.add(self.tab_analyze, text="Analyze")
        nb.add(self.tab_screen, text="Screener")
        nb.add(self.tab_lead, text="Leaderboard")
        nb.add(self.tab_model, text="Model")
        self._build_analyze(self.tab_analyze)
        self._build_screener(self.tab_screen)
        self._build_leaderboard(self.tab_lead)
        self._build_model(self.tab_model)
        self.gui_queue = queue.Queue()
        try:
            self._price_cache = MeridianCache()      # persistent daily-bar disk cache
        except Exception:
            self._price_cache = None                 # fail open: run without caching
        self._intraday_cache = IntradayCache()       # short-TTL in-memory intraday cache
        self._process_queue()
        self._tick_clock()

    def _post(self, fn, *args):
        """Thread-safe: queue a callback for the Tk main thread. Worker threads
        must never touch Tk directly — they hand work here instead."""
        self.gui_queue.put((fn, args))

    def _process_queue(self):
        try:
            while True:
                fn, args = self.gui_queue.get_nowait()
                fn(*args)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._process_queue)

    def _tick_clock(self):
        from datetime import datetime
        try:
            from zoneinfo import ZoneInfo
            now = datetime.now(ZoneInfo("America/New_York"))
        except Exception:
            now = datetime.now()
        self.h_clock.configure(text=now.strftime("%H:%M:%S ET"))
        mins = now.hour * 60 + now.minute
        label, col = "CLOSED", DIM
        if now.weekday() < 5:
            if 240 <= mins < 570: label, col = "PRE-MARKET", BLUE
            elif 570 <= mins < 960: label, col = "MARKET OPEN", BUY
            elif 960 <= mins < 1200: label, col = "AFTER HOURS", AMBER
        self.h_pill.configure(text=label, fg=col)
        if getattr(self, "h_live", None):
            live = label in ("MARKET OPEN", "PRE-MARKET", "AFTER HOURS")
            # subtle blink of the streaming dot while a session is active
            dot = "●" if (not live or now.second % 2 == 0) else "○"
            self.h_live.configure(text=f"{dot} {'STREAMING' if live else 'IDLE'}",
                                  fg=BUY if live else DIM)
        self.root.after(1000, self._tick_clock)

    def _update_feeds(self):
        """Update feed status labels (checks connectivity every 30 sec)."""
        try:
            for name, (lbl, check_fn) in self.feed_labels.items():
                on = check_fn()
                lbl.configure(fg=BUY if on else "#3A4657")
        except Exception:
            pass
        self.root.after(30000, self._update_feeds)

    def _build_model(self, tab):
        out = self._out(tab)
        out.pack(fill="both", expand=True, pady=(6, 0))
        out.configure(state="normal")
        for text, tag in build_formula_reference():
            out.insert("end", text, tag)
        out.configure(state="disabled")

    # ================= LEADERBOARD TAB =================
    def _build_leaderboard(self, tab):
        row = tk.Frame(tab, bg=BG); row.pack(fill="x", pady=(6, 4))
        tk.Label(row, text="Universe", bg=BG, fg=DIM, font=self.ui).pack(side="left", padx=(0, 6))
        self.l_source = ttk.Combobox(row, width=30, state="readonly", values=[
            "Liquid US (~120)", "Active movers", "My watchlist"])
        self.l_source.set("Liquid US (~120)")
        self.l_source.pack(side="left")
        self.l_demo = tk.BooleanVar(value=False)
        self._check(row, "Demo", self.l_demo).pack(side="left", padx=(14, 0))
        self.l_btn = tk.Button(row, text="Build Leaderboard", command=self.run_leaderboard,
                               font=self.monob, bg=GOLD, fg="#191102",
                               activebackground="#a8863e", relief="flat", padx=22, pady=4)
        self.l_btn.pack(side="right")
        self.l_status = tk.Label(tab, text="Ranks the whole universe cross-sectionally — "
                                 "each factor scored against peers, not just the stock's own history.",
                                 bg=BG, fg=DIM, font=self.ui, anchor="w")
        self.l_status.pack(fill="x")
        self.l_out = self._out(tab); self.l_out.pack(fill="both", expand=True, pady=(6, 0))

    def run_leaderboard(self):
        src = self.l_source.get()
        source = ("watchlist" if src.startswith("My")
                  else "movers" if src.startswith("Active") else "liquid")
        tickers = self._parse_watchlist() if source == "watchlist" else []
        if source == "watchlist" and not tickers:
            self.l_status.configure(text="Watchlist is empty — add tickers on the Screener tab.", fg=SELL)
            return
        self.l_btn.configure(state="disabled")
        self.l_status.configure(text="Building leaderboard…", fg=AMBER)
        ctx = {"source": source, "tickers": tickers, "demo": self.l_demo.get()}
        threading.Thread(target=self._work_leaderboard, args=(ctx,), daemon=True).start()

    def _work_leaderboard(self, ctx):
        try:
            if ctx["source"] == "watchlist":
                tickers = ctx["tickers"]
            elif ctx["source"] == "movers":
                got = None if ctx["demo"] else fetch_movers()
                tickers = got or UNIVERSE_LIQUID
            else:
                tickers = UNIVERSE_LIQUID
            if ctx["demo"]:
                data = {t: qe.demo_data(t) for t in tickers}
            else:
                self._post(lambda n=len(tickers): self.l_status.configure(
                    text=f"Downloading {n} tickers…", fg=AMBER))
                data = fetch_daily_cached(self._price_cache, tickers, "6mo", "1d")
            board = lb.rank_universe(data)
            self._post(self._render_leaderboard, board, ctx["demo"])
        except Exception as e:
            self._post(self._leaderboard_failed, str(e))

    def _leaderboard_failed(self, msg):
        self.l_status.configure(text=f"Leaderboard failed: {msg}", fg=SELL)
        self.l_btn.configure(state="normal")

    def _render_leaderboard(self, board, demo):
        o = self.l_out
        o.configure(state="normal"); o.delete("1.0", "end")
        o.insert("end", f"LEADERBOARD  ", "big")
        o.insert("end", f"{board['universe']} names ranked vs peers"
                 + (" · DEMO DATA" if demo else "") + "\n", "dim")
        if not board["cross_sectional"]:
            o.insert("end", "Universe too small for peer ranking — using absolute scores.\n", "warn")
        o.insert("end", "\nTOP BUY CANDIDATES", "gold")
        buys = board["top_buys"]
        if len(buys) < lb.TOP_N:
            o.insert("end", f"  — only {len(buys)} pass the absolute BUY gate today\n", "warn")
        else:
            o.insert("end", "\n", "txt")
        if not buys:
            o.insert("end", "  none — no BUY-grade charts in this universe right now.\n", "dim")
        for r in buys:
            v = r["verdict"]
            o.insert("end", f"  #{r['rank']:<3}", "gold")
            o.insert("end", f"{r['ticker']:<8}", "big")
            o.insert("end", f"vs peers {r['xsec_score']:+.1f}", "buy")
            o.insert("end", f"  ·  own chart {r['abs_score']:+.0f} ({v['label']})", "txt")
            o.insert("end", "  [RISKY]" if v["risky"] else "", "sell")
            o.insert("end", f"  ·  {r['last']:.2f} ({r['chg']:+.2f}%) · {fmt_vol(r['dollar_vol'])}\n", "dim")
        o.insert("end", "\n" + "─" * 62 + "\n", "dim")
        o.insert("end", f"{'#':<4}{'TICKER':<8}{'VS PEERS':>9}{'OWN':>6}  VERDICT\n", "head")
        tag_map = {"good": "buy", "neutral": "warn", "bad": "sell"}
        for r in board["ranked"][:30]:
            v = r["verdict"]; tag = tag_map[v["tone"]]
            o.insert("end", f"{r['rank']:<4}", "dim")
            o.insert("end", f"{r['ticker']:<8}", "txt")
            o.insert("end", f"{r['xsec_score']:>+9.1f}", tag)
            o.insert("end", f"{r['abs_score']:>+6.0f}", "dim")
            o.insert("end", f"  {v['label']}", tag)
            o.insert("end", "  [RISKY]" if v["risky"] else "", "sell")
            o.insert("end", "\n", "txt")
        if len(board["ranked"]) > 30:
            o.insert("end", f"… +{len(board['ranked'])-30} more\n", "dim")
        if board["errors"]:
            o.insert("end", f"\nskipped ({len(board['errors'])}): "
                     + " ".join(t for t, _ in board["errors"][:12]) + "\n", "dim")
        o.insert("end", "\n'Vs peers' = factor strength ranked across this universe today; "
                 "the BUY gate still requires the stock's own chart to signal BUY.\n", "dim")
        o.configure(state="disabled")
        self.l_status.configure(
            text=f"Done — {board['universe']} ranked · {len(buys)} BUY candidate(s).", fg=BUY)
        self.l_btn.configure(state="normal")

    # ---- small widget helpers ----
    def _entry(self, parent, val, width):
        e = tk.Entry(parent, font=self.mono, width=width, bg=PANEL, fg=TXT,
                     insertbackground=TXT, relief="flat")
        e.insert(0, val)
        return e

    def _check(self, parent, text, var):
        return tk.Checkbutton(parent, text=text, variable=var, bg=BG, fg=TXT,
                              selectcolor=PANEL, activebackground=BG, activeforeground=TXT,
                              font=self.ui, highlightthickness=0, bd=0)

    def _out(self, parent):
        o = ScrolledText(parent, bg=PANEL, fg=TXT, font=self.mono, relief="flat",
                         padx=12, pady=10, wrap="word", borderwidth=0)
        o.tag_config("txt", foreground=TXT); o.tag_config("dim", foreground=DIM)
        o.tag_config("buy", foreground=BUY); o.tag_config("sell", foreground=SELL)
        o.tag_config("warn", foreground=AMBER)
        o.tag_config("head", foreground=TXT, font=self.monob)
        o.tag_config("big", foreground=TXT, font=self.big)
        o.tag_config("gold", foreground=GOLD, font=self.monob)
        o.tag_config("formula", foreground="#E8D9A8", font=self.mono)
        o.tag_config("blue", foreground=BLUE)
        o.configure(state="disabled")
        return o

    @staticmethod
    def _num(s, default):
        try: return float(str(s).replace(",", "").strip())
        except (ValueError, TypeError): return default

    # ================= ANALYZE TAB =================
    def _build_analyze(self, tab):
        top = tk.Frame(tab, bg=BG); top.pack(fill="x", pady=(6, 6))
        tk.Label(top, text="Ticker", bg=BG, fg=DIM, font=self.ui).grid(row=0, column=0, sticky="w")
        self.ticker = self._entry(top, "NVDA", 10)
        self.ticker.configure(font=self.monob)
        self.ticker.grid(row=1, column=0, padx=(0, 10), ipady=5, sticky="w")
        self.ticker.bind("<Return>", lambda e: self.run_single())
        tk.Label(top, text="Period", bg=BG, fg=DIM, font=self.ui).grid(row=0, column=1, sticky="w")
        self.period = ttk.Combobox(top, values=["1mo", "3mo", "6mo", "1y", "2y"], width=6, state="readonly")
        self.period.set("6mo"); self.period.grid(row=1, column=1, padx=(0, 10))
        tk.Label(top, text="Interval", bg=BG, fg=DIM, font=self.ui).grid(row=0, column=2, sticky="w")
        self.interval = ttk.Combobox(top, values=["1d", "1h", "30m", "15m", "5m", "2m", "1m"], width=5, state="readonly")
        self.interval.set("1d"); self.interval.grid(row=1, column=2, padx=(0, 10))
        tk.Label(top, text="Account $", bg=BG, fg=DIM, font=self.ui).grid(row=0, column=3, sticky="w")
        self.account = self._entry(top, "10000", 8); self.account.grid(row=1, column=3, padx=(0, 10), ipady=5)
        tk.Label(top, text="Risk %", bg=BG, fg=DIM, font=self.ui).grid(row=0, column=4, sticky="w")
        self.risk = self._entry(top, "1", 5); self.risk.grid(row=1, column=4, ipady=5)

        row2 = tk.Frame(tab, bg=BG); row2.pack(fill="x", pady=(2, 8))
        self.a_demo = tk.BooleanVar(value=False); self.a_opt = tk.BooleanVar(value=False)
        self.a_math = tk.BooleanVar(value=False)
        self._check(row2, "Demo (offline)", self.a_demo).pack(side="left", padx=(0, 14))
        self._check(row2, "Optimize weights", self.a_opt).pack(side="left", padx=(0, 14))
        self._check(row2, "Show the math", self.a_math).pack(side="left")
        self.a_btn = tk.Button(row2, text="Analyze", command=self.run_single, font=self.monob,
                               bg=BUY, fg="#04140c", activebackground="#25a877", relief="flat", padx=22, pady=4)
        self.a_btn.pack(side="right")
        self.a_status = tk.Label(tab, text="Ready — type a ticker and press Analyze (or tick Demo).",
                                 bg=BG, fg=DIM, font=self.ui, anchor="w")
        self.a_status.pack(fill="x")
        self.a_out = self._out(tab); self.a_out.pack(fill="both", expand=True, pady=(6, 0))
        self._welcome_single()

    def _welcome_single(self):
        self.a_out.configure(state="normal")
        self.a_out.insert("end", "Analyze one stock\n", "big")
        self.a_out.insert("end", "Type a ticker and press Analyze. First time? Tick “Demo (offline)”.\n\n", "txt")
        self.a_out.insert("end", "Not financial advice. Treat the verdict as one input, not a green light.\n", "dim")
        self.a_out.configure(state="disabled")

    def run_single(self):
        self.a_btn.configure(state="disabled")
        sym = self.ticker.get().strip().upper() or "NVDA"
        demo = self.a_demo.get()
        self.a_status.configure(text=f"Analyzing {'DEMO' if demo else sym}…", fg=AMBER)
        a = {"sym": sym, "demo": demo, "period": self.period.get(), "interval": self.interval.get(),
             "optimize": self.a_opt.get(), "account": self._num(self.account.get(), 0.0),
             "risk": self._num(self.risk.get(), 1.0) or 1.0, "math": self.a_math.get()}
        threading.Thread(target=self._work_single, args=(a,), daemon=True).start()

    def _work_single(self, a):
        try:
            try:
                res = screen_one(a["sym"], a["demo"], a["period"], a["interval"], a["optimize"],
                                 cache=self._price_cache, realtime_key=qe.FINNHUB_DEFAULT_KEY)
            except ValueError as e:
                # Young listing: too few daily bars → auto-retry on 1h (≈7× more bars).
                if "recent listing" in str(e) and a["interval"] == "1d" and not a["demo"]:
                    self._post(lambda: self.a_status.configure(
                        text=f"{a['sym']} is a recent listing — retrying on 1h bars…", fg=AMBER))
                    a = dict(a); a["interval"] = "1h"; a["period"] = "1mo"; a["_fellback"] = True
                    res = screen_one(a["sym"], a["demo"], a["period"], a["interval"], a["optimize"],
                                     cache=self._price_cache, realtime_key=qe.FINNHUB_DEFAULT_KEY)
                    res["interval_fallback"] = "1h"
                else:
                    raise
            recs = congress = market = insiders = whale = sentiment = None
            if a["demo"]:
                res["fund"] = fe.demo_fundamentals(a["sym"])
                sentiment = se.demo_sentiment(a["sym"])
            elif res.get("ineligible"):
                # BACKTEST GATE: the rules have historically lost on this name, so no
                # alt-data can rescue it — skip every API call (saves cost + rate limit).
                recs = congress = insiders = whale = market = sentiment = None
                res["fund"] = None; res["alt_skipped"] = True
            else:
                # Fan the independent context fetches across services in parallel —
                # they hit Finnhub / Quiver / Yahoo, which have separate rate limits,
                # so total time ≈ the slowest call (options) instead of the sum.
                sym = a["sym"]; fkey = qe.FINNHUB_DEFAULT_KEY
                qtok = os.environ.get("QUIVER_API_TOKEN")
                def _market():
                    if sym == "SPY":
                        return None
                    spy = _fetch_one_cached(self._price_cache, "SPY", a["period"], a["interval"])
                    return qe.market_context(res["d"], spy)
                jobs = {
                    "recs": lambda: qe.finnhub_recs(sym, fkey),
                    "congress": lambda: qe.quiver_congress(sym, qtok),
                    "insiders": lambda: qe.finnhub_insiders(sym, fkey),
                    "whale": lambda: qe.whale_signal(qe.options_whale_flow(sym),
                                                     qe.quiver_darkpool(sym, qtok)),
                    "market": _market,
                    "fund": lambda: fe.fetch_fundamentals(sym, fkey,
                                                          os.environ.get("ALPHA_VANTAGE_KEY")),
                    "sentiment": lambda: se.news_sentiment(sym, fkey,
                                                           os.environ.get("ALPHA_VANTAGE_KEY")),
                    "filings": lambda: edgar.recent_filings(sym, days=3),
                }
                akey, asec = alpaca_keys()
                if akey and asec:                       # real dark-pool block flow (SIP)
                    w0, w1 = of.after_hours_window()
                    jobs["orderflow"] = lambda: of.darkpool_blocks(sym, akey, asec, w0, w1,
                                                                   min_notional=200000)
                out = {}
                with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
                    futs = {ex.submit(fn): name for name, fn in jobs.items()}
                    for fu in as_completed(futs):
                        try: out[futs[fu]] = fu.result()
                        except Exception: out[futs[fu]] = None
                recs, congress = out.get("recs"), out.get("congress")
                insiders, whale, market = out.get("insiders"), out.get("whale"), out.get("market")
                res["fund"] = out.get("fund"); sentiment = out.get("sentiment")
                res["orderflow"] = out.get("orderflow")
                res["filings"] = out.get("filings")
            res["sentiment"] = sentiment
            try:
                macro = se.macro_signal(sentiment)
                tilt = qe.alt_data_tilt(congress, recs, insiders, whale, macro)
                if tilt or market: qe.apply_alt_tilt(res, tilt, market)
            except Exception: pass
            seg = build_report_segments(res, res.get("opt"), a["account"], a["risk"])
            if a.get("math"):
                seg = seg + build_live_math_segments(res)
            self._post(self._render_single, seg, recs, congress, a["sym"], a["demo"])
        except Exception as e:
            self._post(self._error_single, str(e))

    def _render_single(self, seg, recs, congress, sym, demo):
        self.a_out.configure(state="normal"); self.a_out.delete("1.0", "end")
        for text, tag in seg: self.a_out.insert("end", text, tag)
        if recs:
            self.a_out.insert("end", "\nANALYSTS (Finnhub)  ", "head")
            self.a_out.insert("end", f"strong buy {recs.get('strongBuy',0)} · buy {recs.get('buy',0)} · "
                              f"hold {recs.get('hold',0)} · sell {recs.get('sell',0)} · strong sell {recs.get('strongSell',0)}\n", "txt")
        if congress:
            self.a_out.insert("end", "\nCONGRESS (Quiver)  ", "head")
            if congress["total"]:
                self.a_out.insert("end", f"{congress['total']} disclosed trades · ", "txt")
                self.a_out.insert("end", f"{congress['buys']} buys", "buy")
                self.a_out.insert("end", " · ", "txt")
                self.a_out.insert("end", f"{congress['sells']} sells\n", "sell")
                for t in congress["latest"]:
                    is_buy = "purchase" in t["tx"].lower() or "buy" in t["tx"].lower()
                    self.a_out.insert("end", f"  {t['date']}  ", "dim")
                    self.a_out.insert("end", f"{t['rep']}", "txt")
                    party = f" ({t['party']})" if t["party"] else ""
                    self.a_out.insert("end", f"{party} {t['house']}".rstrip(), "dim")
                    self.a_out.insert("end", f"  {t['tx']}", "buy" if is_buy else "sell")
                    self.a_out.insert("end", f"  {t['range']}\n" if t["range"] else "\n", "dim")
            else:
                self.a_out.insert("end", "no disclosed congressional trades on record\n", "dim")
        self.a_out.configure(state="disabled")
        self.a_status.configure(text=f"Done — {'DEMO' if demo else sym}.", fg=BUY)
        self.a_btn.configure(state="normal")

    def _error_single(self, msg):
        self.a_out.configure(state="normal"); self.a_out.delete("1.0", "end")
        self.a_out.insert("end", "Couldn't analyze that.\n\n", "sell")
        self.a_out.insert("end", msg + "\n\n", "txt")
        self.a_out.insert("end", "Check the ticker spelling, or tick Demo to test offline.\n", "dim")
        self.a_out.configure(state="disabled")
        self.a_status.configure(text="Error — see above.", fg=SELL)
        self.a_btn.configure(state="normal")

    # ================= SCREENER TAB =================
    def _build_screener(self, tab):
        srcrow = tk.Frame(tab, bg=BG); srcrow.pack(fill="x", pady=(6, 2))
        tk.Label(srcrow, text="Scan", bg=BG, fg=DIM, font=self.ui).pack(side="left", padx=(0, 6))
        self.s_source = ttk.Combobox(srcrow, width=34, state="readonly", values=[
            "Discover — active movers", "Discover — liquid US (~120)", "My watchlist"])
        self.s_source.set("Discover — active movers")
        self.s_source.pack(side="left")
        tk.Label(srcrow, text="(Discover finds NEW stocks; watchlist ranks your own list)",
                 bg=BG, fg=DIM, font=self.ui).pack(side="left", padx=(8, 0))

        tk.Label(tab, text="Watchlist  (used only in “My watchlist” mode — comma / space / new line)",
                 bg=BG, fg=DIM, font=self.ui).pack(anchor="w", pady=(6, 2))
        self.wl = tk.Text(tab, height=3, font=self.mono, bg=PANEL, fg=TXT,
                          insertbackground=TXT, relief="flat", padx=10, pady=8, wrap="word")
        self.wl.insert("1.0", load_saved_watchlist())
        self.wl.pack(fill="x")

        opts = tk.Frame(tab, bg=BG); opts.pack(fill="x", pady=(8, 4))
        self.s_daytrade = tk.BooleanVar(value=False)
        dtc = self._check(opts, "⚡ Day-trade (5-min bars)", self.s_daytrade)
        dtc.grid(row=1, column=0, padx=(0, 14))
        self.s_daytrade.trace_add("write", lambda *a: self._apply_daytrade())
        tk.Label(opts, text="Interval", bg=BG, fg=DIM, font=self.ui).grid(row=0, column=1, sticky="w")
        self.s_interval = ttk.Combobox(opts, values=["1d", "1h", "30m", "15m", "5m", "2m", "1m"], width=5, state="readonly")
        self.s_interval.set("1d"); self.s_interval.grid(row=1, column=1, padx=(0, 12))
        tk.Label(opts, text="Min $ vol (millions)", bg=BG, fg=DIM, font=self.ui).grid(row=0, column=2, sticky="w")
        self.s_minvol = self._entry(opts, "0", 6); self.s_minvol.grid(row=1, column=2, padx=(0, 12), ipady=4)
        self.s_risky = tk.BooleanVar(value=False); self.s_buy = tk.BooleanVar(value=False)
        self.s_demo = tk.BooleanVar(value=False)
        self._check(opts, "Exclude RISKY", self.s_risky).grid(row=1, column=3, padx=(0, 10))
        self._check(opts, "Buy only", self.s_buy).grid(row=1, column=4, padx=(0, 10))
        self._check(opts, "Demo", self.s_demo).grid(row=1, column=5, padx=(0, 10))
        self.s_alert = tk.BooleanVar(value=True)
        self._check(opts, "Alert on new BUY", self.s_alert).grid(row=1, column=6, padx=(0, 10))
        self.s_congress = tk.BooleanVar(value=False)
        self._check(opts, "Congress col", self.s_congress).grid(row=1, column=7, padx=(0, 10))
        self.s_whale = tk.BooleanVar(value=False)
        self._check(opts, "🐋 Whale accum", self.s_whale).grid(row=1, column=8, padx=(0, 10))

        # ---- fundamental (Alpha Vantage) filter row ----
        frow = tk.Frame(tab, bg=BG); frow.pack(fill="x", pady=(0, 4))
        self.s_fund = tk.BooleanVar(value=False)
        self._check(frow, "✓ Fundamental Filter", self.s_fund).pack(side="left", padx=(0, 14))
        def _flabel(t): tk.Label(frow, text=t, bg=BG, fg=DIM, font=self.ui).pack(side="left", padx=(0, 3))
        _flabel("Max P/E"); self.s_maxpe = self._entry(frow, "50", 5); self.s_maxpe.pack(side="left", padx=(0, 10), ipady=3)
        _flabel("Max D/E"); self.s_maxde = self._entry(frow, "2.0", 5); self.s_maxde.pack(side="left", padx=(0, 10), ipady=3)
        _flabel("Min Growth %"); self.s_mingrow = self._entry(frow, "0", 5); self.s_mingrow.pack(side="left", padx=(0, 10), ipady=3)
        _flabel("Min Curr. Ratio"); self.s_mincur = self._entry(frow, "1.0", 5); self.s_mincur.pack(side="left", padx=(0, 10), ipady=3)

        btns = tk.Frame(tab, bg=BG); btns.pack(fill="x", pady=(4, 4))
        self.s_btn = tk.Button(btns, text="Screen now", command=self.run_screen, font=self.monob,
                               bg=BUY, fg="#04140c", activebackground="#25a877", relief="flat", padx=22, pady=4)
        self.s_btn.pack(side="left")
        self.s_chart_btn = tk.Button(btns, text="Open charts in browser", command=self.open_charts,
                                     font=self.ui, bg=PANEL, fg=TXT, activebackground=LINE,
                                     relief="flat", padx=16, pady=4, state="disabled")
        self.s_chart_btn.pack(side="left", padx=(8, 0))
        self.s_save_btn = tk.Button(btns, text="Save list", command=self.save_watchlist,
                                    font=self.ui, bg=PANEL, fg=TXT, activebackground=LINE,
                                    relief="flat", padx=14, pady=4)
        self.s_save_btn.pack(side="left", padx=(8, 0))
        self.s_ah_btn = tk.Button(btns, text="🌙 After-Hours", command=self.run_afterhours,
                                  font=self.ui, bg=PANEL, fg=AMBER, activebackground=LINE,
                                  relief="flat", padx=14, pady=4)
        self.s_ah_btn.pack(side="left", padx=(8, 0))
        self.s_mb_btn = tk.Button(btns, text="☀️ Morning Brief", command=self.run_morning,
                                  font=self.ui, bg=PANEL, fg=GOLD, activebackground=LINE,
                                  relief="flat", padx=14, pady=4)
        self.s_mb_btn.pack(side="left", padx=(8, 0))
        self.s_tr_btn = tk.Button(btns, text="📊 Track Record", command=self.run_trackrecord,
                                  font=self.ui, bg=PANEL, fg=TXT, activebackground=LINE,
                                  relief="flat", padx=14, pady=4)
        self.s_tr_btn.pack(side="left", padx=(8, 0))
        # live tracking controls (right side)
        self.s_auto = tk.BooleanVar(value=False)
        self._check(btns, "Track live", self.s_auto).pack(side="right")
        self.s_auto.trace_add("write", lambda *a: self.toggle_auto())
        self.s_int_sec = self._entry(btns, "900", 5)
        self.s_int_sec.pack(side="right", ipady=3, padx=(0, 6))
        tk.Label(btns, text="every (sec)", bg=BG, fg=DIM, font=self.ui).pack(side="right", padx=(0, 4))

        self.s_status = tk.Label(tab, text="Enter tickers → “Screen now”. Set “Min $ vol” to 5 to hide "
                                 "thin names. Tick “Track live” to auto-refresh.",
                                 bg=BG, fg=DIM, font=self.ui, anchor="w")
        self.s_status.pack(fill="x")
        self.s_updated = tk.Label(tab, text="", bg=BG, fg=DIM, font=self.ui, anchor="w")
        self.s_updated.pack(fill="x")
        self.s_out = self._out(tab); self.s_out.pack(fill="both", expand=True, pady=(6, 0))
        self._welcome_screen()
        self._last_results = None
        self._last_ctx = None
        self._auto_job = None
        self._prev_good = None
        self._result_cache = ResultCache(max_age_sec=300)  # Cache results for 5 min

    def _welcome_screen(self):
        self.s_out.configure(state="normal")
        self.s_out.insert("end", "Swing Screener\n", "big")
        self.s_out.insert("end", "Two ways to scan, chosen in the “Scan” dropdown:\n", "txt")
        self.s_out.insert("end", "  • Discover — searches a whole universe of stocks and surfaces the "
                          "ones setting up (finds NEW names you're not watching).\n", "txt")
        self.s_out.insert("end", "  • My watchlist — ranks only the tickers you typed.\n\n", "txt")
        self.s_out.insert("end", "For discovery, the filters do the work: tick “Buy only”, set “Min $ vol” "
                          "to 5, and it hands you just the liquid names that newly qualify — ranked, with "
                          "a Mac notification when a fresh BUY appears.\n\n", "dim")
        self.s_out.insert("end", "Daily bars + 6-month lookback is the swing setup; the ~15-min data delay "
                          "doesn't matter at that horizon. Watch the Sharpe column — negative means the "
                          "rules have historically lost on that name.\n\n", "dim")
        self.s_out.insert("end", "First time? Tick “Demo” and press “Scan now” to see discovery run offline.\n", "dim")
        self.s_out.configure(state="disabled")

    def _parse_watchlist(self):
        raw = self.wl.get("1.0", "end")
        toks = raw.replace(",", " ").replace("\n", " ").split()
        seen, out = set(), []
        for t in toks:
            u = t.strip().upper()
            if u and u not in seen:
                seen.add(u); out.append(u)
        return out

    def run_screen(self, auto=False):
        if self._auto_job:
            self.root.after_cancel(self._auto_job)
            self._auto_job = None
        src_label = self.s_source.get()
        source = ("movers" if src_label.startswith("Discover — active")
                  else "watchlist" if src_label.startswith("My")
                  else "curated")
        tickers = self._parse_watchlist()
        if source == "watchlist" and not tickers:
            self.s_status.configure(text="Add tickers, or pick a Discover option to find new stocks.", fg=SELL)
            return
        self.s_btn.configure(state="disabled"); self.s_chart_btn.configure(state="disabled")
        day = self.s_daytrade.get()
        interval = self.s_interval.get()      # day-trade keeps the selector live (5m/2m/1m)
        ctx = {"source": source, "tickers": tickers, "demo": self.s_demo.get(),
               "interval": interval, "period": period_for(interval, day),
               "exclude_risky": self.s_risky.get(), "buy_only": self.s_buy.get(),
               "min_vol": self._num(self.s_minvol.get(), 0.0) * 1e6,
               "congress": self.s_congress.get(),
               "whale_only": self.s_whale.get(),
               "fund": self.s_fund.get(),
               "max_pe": self._num(self.s_maxpe.get(), 0.0) or None,
               "max_de": self._num(self.s_maxde.get(), None),
               "min_growth": self._num(self.s_mingrow.get(), None),
               "min_current": self._num(self.s_mincur.get(), None),
               "note": ""}
        self.s_status.configure(text="Scanning…", fg=AMBER)
        threading.Thread(target=self._work_screen, args=(ctx,), daemon=True).start()

    def _work_screen(self, ctx):
        results, filtered, errors = [], [], []
        intraday = ctx["interval"] != "1d"
        # 1) resolve the universe of tickers to scan
        if ctx["source"] == "watchlist":
            tickers = ctx["tickers"]; note = f"Ranking your watchlist ({len(tickers)})"
        elif intraday and ctx["tickers"]:
            # Intraday is uncached + a flaky feed, so a full-universe discover is
            # slow. Prefer the watchlist for speed (fall back to a capped universe
            # only if the watchlist is empty).
            tickers = ctx["tickers"]
            note = f"Intraday {ctx['interval']} — ranking your watchlist ({len(tickers)}) for speed"
        elif ctx["source"] == "movers":
            got = None if ctx["demo"] else fetch_movers()
            if got:
                tickers, note = got, f"Discovering — {len(got)} active movers"
            else:
                tickers = UNIVERSE_LIQUID
                note = ("Discovering — demo universe" if ctx["demo"]
                        else "Movers feed unavailable — scanning curated liquid list")
        else:
            tickers = UNIVERSE_LIQUID; note = f"Discovering — curated liquid US ({len(tickers)})"
        if intraday and ctx["source"] != "watchlist" and len(tickers) > 40:
            tickers = tickers[:40]                       # cap intraday discover for speed
            note += " · capped to 40 (intraday)"
        ctx["note"] = note
        # 2) fetch data: use concurrent download for speed (or synthetic in demo)
        try:
            if ctx["demo"]:
                data = {t: qe.demo_data(t) for t in tickers}
            else:
                self._post(lambda n=len(tickers): self.s_status.configure(
                    text=f"Downloading {n} tickers (4 parallel)…", fg=AMBER))
                data = fetch_prices(self._price_cache, self._intraday_cache, tickers,
                                    ctx["period"], ctx["interval"])
        except Exception as e:
            self._post(self._screen_failed, str(e))
            return
        # 2a) intraday: patch each bar's last price with a live Finnhub quote
        #     (keyless-free real-time; only where the ~15-min delay actually matters)
        if intraday and not ctx["demo"]:
            self._post(lambda: self.s_status.configure(text="Fetching live prices…", fg=AMBER))
            patch_realtime_batch(data, tickers, qe.FINNHUB_DEFAULT_KEY)
        # 2b) one bulk congress fetch for the whole universe (opt-in, non-demo)
        congress_map = {}
        if ctx["congress"] and not ctx["demo"]:
            self._post(lambda: self.s_status.configure(text="Fetching congress feed…", fg=AMBER))
            try:
                congress_map = qe.quiver_congress_bulk(os.environ.get("QUIVER_API_TOKEN"))
            except Exception:
                congress_map = {}
        # 3) analyze each locally (with caching for live refresh)
        for t in tickers:
            df = data.get(t)
            if df is None or len(df) < 60:
                errors.append((t, "no data" if df is None else f"{len(df)} bars"))
                continue
            try:
                # Check cache first (saves ~100ms per cached result on refresh)
                cached_res = self._result_cache.get(t, ctx["interval"], ctx["period"], ctx["demo"])
                if cached_res:
                    res = cached_res
                else:
                    res = analyze_prefetched(t, df, ctx["interval"])
                    self._result_cache.set(t, ctx["interval"], ctx["period"], res, ctx["demo"])
                res["congress"] = congress_map.get(t)
                if passes_filters(res, ctx["exclude_risky"], ctx["buy_only"], ctx["min_vol"],
                                  ctx.get("whale_only")):
                    results.append(res)
                else:
                    filtered.append(t)
            except Exception as e:
                errors.append((t, str(e)))
        # 3b) fundamental filter (Alpha Vantage) — only on technically-passing names
        if ctx["fund"]:
            results, filtered = self._apply_fundamentals(results, filtered, ctx)
        self._post(self._render_screen, results, filtered, errors, ctx)

    def _apply_fundamentals(self, results, filtered, ctx):
        """Fetch fundamentals for technically-passing names and drop those that
        fail the value/quality thresholds. Demo uses synthetic data (no API);
        live uses Finnhub (60/min, reuses the app key) with Alpha Vantage fallback."""
        finnhub_key = qe.FINNHUB_DEFAULT_KEY
        av_key = os.environ.get("ALPHA_VANTAGE_KEY")
        # Cap so a huge universe doesn't blow the 60/min budget in one scan.
        cap = 10**9 if ctx["demo"] else 60
        targets = results if ctx["demo"] else results[:cap]
        over_cap = [] if ctx["demo"] else results[cap:]
        # Fetch fundamentals for the target names — CONCURRENTLY (Finnhub is 60/MIN,
        # not 1/sec), turning a ~minute of serial waits into a couple of seconds.
        if ctx["demo"]:
            funds = {r["ticker"]: fe.demo_fundamentals(r["ticker"]) for r in targets}
        else:
            self._post(lambda n=len(targets): self.s_status.configure(
                text=f"Fundamentals — {n} names (concurrent)…", fg=AMBER))
            funds = fe.fetch_fundamentals_batch([r["ticker"] for r in targets],
                                                finnhub_key, av_key)
        kept = []
        for r in targets:
            r["fund"] = funds.get(r["ticker"])
            if fe.passes_fundamental_filter(r["fund"], ctx["max_pe"], ctx["max_de"],
                                            ctx["min_growth"], ctx["min_current"]):
                kept.append(r)
            else:
                filtered.append(r["ticker"])
        for r in over_cap:                          # unchecked (over cap) — keep, don't penalize
            r["fund"] = None
            kept.append(r)
        if over_cap:
            ctx["note"] = (ctx.get("note", "") +
                           f"  ·  fundamentals checked for first {cap}")
        return kept, filtered

    # ================= TRACK RECORD =================
    def run_trackrecord(self):
        self.s_tr_btn.configure(state="disabled")
        self.s_status.configure(text="Grading the app's past verdicts…", fg=AMBER)
        threading.Thread(target=self._work_trackrecord, daemon=True).start()

    def _work_trackrecord(self, horizon=5):
        try:
            import trackrecord as tr
            tickers = sorted({e["ticker"] for e in tr._load()})
            data = fetch_daily_cached(self._price_cache, tickers, "6mo", "1d") if tickers else {}
            scored = tr.score(lambda t: data.get(t), horizon)
            summ = tr.summary(scored, horizon)
            self._post(self._render_trackrecord, summ, scored)
        except Exception as ex:
            err_msg = str(ex)
            self._post(lambda: (self.s_status.configure(text=f"Track Record failed: {err_msg}", fg=SELL),
                                self.s_tr_btn.configure(state="normal")))

    def _render_trackrecord(self, summ, scored):
        o = self.s_out
        o.configure(state="normal"); o.delete("1.0", "end")
        o.insert("end", "📊 VERDICT TRACK RECORD  ", "big")
        o.insert("end", f"(the app graded against reality, {summ['horizon']}-bar forward)\n", "dim")
        o.insert("end", f"{summ['total']} verdicts logged · {summ['graded']} graded · "
                 f"{summ['pending']} still pending\n", "txt")
        if not summ["graded"]:
            o.insert("end", "\nNothing graded yet — verdicts need to age "
                     f"{summ['horizon']} trading days before they can be scored.\n"
                     "Run live (non-demo) scans over a few days and check back.\n", "warn")
            o.configure(state="disabled")
            self.s_status.configure(text="Track Record — no graded verdicts yet.", fg=DIM)
            self.s_tr_btn.configure(state="normal"); return
        o.insert("end", "\nBY VERDICT\n", "gold")
        for name, s in summ["by_tone"].items():
            tag = "buy" if s["hit_rate"] >= 0.5 else "sell"
            o.insert("end", f"  {name:<7} {s['n']:>3} calls · ", "txt")
            o.insert("end", f"{s['hit_rate']*100:.0f}% correct", tag)
            o.insert("end", f" · avg fwd {s['avg_fwd']*100:+.1f}%\n", "dim")
        if summ["by_tag"]:
            o.insert("end", "\nWHICH SIGNALS ADD EDGE  ", "gold")
            o.insert("end", "(hit rate when this signal was present)\n", "dim")
            for t, s in sorted(summ["by_tag"].items(), key=lambda kv: -kv[1]["hit_rate"]):
                tag = "buy" if s["hit_rate"] >= 0.5 else "sell"
                o.insert("end", f"  {t:<14} {s['n']:>3} · ", "txt")
                o.insert("end", f"{s['hit_rate']*100:.0f}% correct", tag)
                o.insert("end", f" · avg fwd {s['avg_fwd']*100:+.1f}%\n", "dim")
        o.insert("end", "\nReal out-of-sample record (not a backtest) — but a small, slowly-growing "
                 "sample. Past hit rate ≠ future results; this exists to keep the app honest.\n", "dim")
        o.configure(state="disabled")
        self.s_status.configure(text=f"Track Record — {summ['graded']} graded.", fg=BUY)
        self.s_tr_btn.configure(state="normal")

    # ================= MORNING BRIEF =================
    def run_morning(self):
        tickers = self._parse_watchlist()
        if not tickers:
            self.s_status.configure(text="Add tickers to the watchlist first.", fg=SELL)
            return
        self.s_mb_btn.configure(state="disabled")
        self.s_status.configure(text="Building Morning Brief (overnight catalysts)…", fg=AMBER)
        threading.Thread(target=self._work_morning, args=(tickers,), daemon=True).start()

    def _work_morning(self, tickers):
        from concurrent.futures import ThreadPoolExecutor
        try:
            akey, asec = alpaca_keys()
            fkey = qe.FINNHUB_DEFAULT_KEY
            avkey = os.environ.get("ALPHA_VANTAGE_KEY")
            qtok = os.environ.get("QUIVER_API_TOKEN")
            data = fetch_daily_cached(self._price_cache, tickers, "6mo", "1d")
            briefs = {}

            def one(t):
                df = data.get(t)
                if df is None or len(df) < 60:
                    return
                try:
                    res = analyze_prefetched(t, df, "1d")
                except Exception:
                    return
                reg_close = float(df["Close"].iloc[-1])
                ah_px = qe.alpaca_latest_trade(t, akey, asec) if (akey and asec) else None
                ah_chg = (ah_px / reg_close - 1) * 100 if ah_px else 0.0
                ins = qe.insider_signal(qe.finnhub_insiders(t, fkey))
                fil = edgar.recent_filings(t, days=2)
                sen = se.news_sentiment(t, fkey, avkey)
                brief = mb.catalyst_score(res["score"], ah_chg, ins, fil,
                                          sen, res.get("whale_activity"))
                brief.update({"ticker": t, "ah_chg": ah_chg, "ah_px": ah_px,
                              "reg_close": reg_close, "tech": res["verdict"]["label"],
                              "tech_score": res["score"]})
                briefs[t] = brief
            with ThreadPoolExecutor(max_workers=8) as ex:
                list(ex.map(one, tickers))
            self._post(self._render_morning, briefs)
        except Exception as exc:
            err_msg = str(exc)
            self._post(lambda: (self.s_status.configure(text=f"Morning Brief failed: {err_msg}", fg=SELL),
                                self.s_mb_btn.configure(state="normal")))

    def _render_morning(self, briefs):
        o = self.s_out
        o.configure(state="normal"); o.delete("1.0", "end")
        o.insert("end", "☀️ MORNING BRIEF  ", "big")
        o.insert("end", f"(overnight catalysts · {len(briefs)} names)\n", "dim")
        ranked = sorted(briefs.values(), key=lambda b: -b["score"])
        buys = [b for b in ranked if b["verdict"] == "BUY candidate"]
        risks = [b for b in ranked if b["verdict"] == "RISK / avoid"]
        for b in buys[:5]:
            top = b["reasons"][0][0] if b["reasons"] else "overnight setup"
            mac_notify(f"☀️ {b['ticker']} — buy candidate", f"catalyst {b['score']:+d} · {top}", "Glass")

        def block(title, items, tag):
            o.insert("end", f"\n{title}\n", "gold")
            if not items:
                o.insert("end", "  (none)\n", "dim"); return
            for b in items:
                o.insert("end", f"  {b['ticker']:<7}", "big")
                o.insert("end", f"catalyst {b['score']:+d}", tag)
                o.insert("end", f"  · chart {b['tech_score']:+.0f} ({b['tech']})", "dim")
                if b["ah_px"]:
                    atag = "buy" if b["ah_chg"] >= 0 else "sell"
                    o.insert("end", "  · ", "dim")
                    o.insert("end", f"{b['ah_chg']:+.1f}% AH", atag)
                o.insert("end", "\n", "dim")
                for txt, d in b["reasons"][:4]:
                    rtag = "buy" if d > 0 else "sell" if d < 0 else "dim"
                    o.insert("end", f"       {'▲' if d>0 else '▼' if d<0 else '•'} {txt}\n", rtag)

        block("🟢 BUY CANDIDATES  (bullish overnight catalyst)", buys, "buy")
        block("🔴 RISK / AVOID  (dilution, selling, bad news)", risks, "sell")
        watch = [b for b in ranked if b["verdict"] == "watch"]
        if watch:
            o.insert("end", "\nWATCH (no strong catalyst): "
                     + " ".join(b["ticker"] for b in watch) + "\n", "dim")
        o.insert("end", "\nRanks overnight developments by how ACTIONABLE the sourced catalyst is "
                 "(insider buys & earnings drift weighted highest). Prep radar, NOT a prediction — "
                 "gaps can fade, and it reads disclosed catalysts, never order-flow intent.\n", "dim")
        o.configure(state="disabled")
        self.s_status.configure(text=f"Morning Brief — {len(buys)} buy candidate(s), {len(risks)} risk(s).", fg=BUY)
        self.s_mb_btn.configure(state="normal")

    # ================= AFTER-HOURS WATCH =================
    def run_afterhours(self):
        tickers = self._parse_watchlist()
        if not tickers:
            self.s_status.configure(text="Add tickers to the watchlist first.", fg=SELL)
            return
        akey, asec = alpaca_keys()
        if not (akey and asec):
            self.s_status.configure(text="After-Hours needs Alpaca keys (ALPACA_API_KEY/SECRET) "
                                    "for the live extended-hours price.", fg=SELL)
            return
        self.s_ah_btn.configure(state="disabled")
        self.s_status.configure(text="Checking after-hours prices…", fg=AMBER)
        threading.Thread(target=self._work_afterhours,
                         args=(tickers, akey, asec), daemon=True).start()

    def _work_afterhours(self, tickers, akey, asec):
        from concurrent.futures import ThreadPoolExecutor
        try:
            data = fetch_daily_cached(self._price_cache, tickers, "6mo", "1d")
            reads = {}

            def one(t):
                df = data.get(t)
                if df is None or len(df) < 2:
                    return
                reg_close, prev_close = float(df["Close"].iloc[-1]), float(df["Close"].iloc[-2])
                ah_px = qe.alpaca_latest_trade(t, akey, asec)
                r = (ah.read_one(t, reg_close, prev_close, ah_px) if ah_px else
                     {"ticker": t, "reg_close": reg_close, "ah_price": None, "ah_chg": 0.0,
                      "day_chg": (reg_close/prev_close-1)*100 if prev_close else None,
                      "diverges": False, "flag": False, "dpi": None, "dpi_avg": None})
                # The real 'institutional intent': what insiders/the company DISCLOSED to the SEC
                try:
                    r["filings"] = edgar.recent_filings(t, days=2)
                except Exception:
                    r["filings"] = []
                r["whale"] = qe.whale_score(df)      # observable large-money footprint
                if r["filings"]:                     # a fresh material filing is itself a flag
                    r["flag"] = True
                reads[t] = r
            with ThreadPoolExecutor(max_workers=8) as ex:
                list(ex.map(one, tickers))
            # REAL dark-pool block flow (Alpaca SIP / FINRA TRF) for flagged names
            if akey and asec:
                w0, w1 = of.after_hours_window()

                def flow(t):
                    try:
                        reads[t]["orderflow"] = of.darkpool_blocks(t, akey, asec, w0, w1)
                    except Exception:
                        pass
                with ThreadPoolExecutor(max_workers=5) as ex:
                    list(ex.map(flow, [t for t, r in reads.items() if r["flag"]]))
            self._post(self._render_afterhours, reads)
        except Exception as exc:
            err_msg = str(exc)
            self._post(lambda: (self.s_status.configure(text=f"After-Hours failed: {err_msg}", fg=SELL),
                                self.s_ah_btn.configure(state="normal")))

    def _render_afterhours(self, reads):
        o = self.s_out
        o.configure(state="normal"); o.delete("1.0", "end")
        o.insert("end", "🌙 AFTER-HOURS WATCH  ", "big")
        o.insert("end", f"({market_session()} · {len(reads)} names)\n", "dim")
        flagged = sorted([r for r in reads.values() if r["flag"]],
                         key=lambda r: -abs(r.get("ah_chg") or 0))
        # push-notify: pair the price move with the SEC filing that explains it
        for r in flagged[:5]:
            f = (r.get("filings") or [None])[0]
            cause = f"  — {f['form']}: {f['note']}" if f else ""
            move = ah.describe(r) if r.get("ah_price") else f"{r['ticker']} — material SEC filing"
            mac_notify("🌙 After-hours", move + cause, "Glass")
        if flagged:
            o.insert("end", "\n", "txt")
            for r in flagged:                          # one alert CARD per flagged name
                self._afterhours_card(o, r)
        else:
            o.insert("end", "\nNothing moving after hours and no fresh material filings.\n", "txt")
        o.insert("end", "\n" + "─" * 60 + "\n", "dim")
        o.insert("end", f"{'TICKER':<8}{'CLOSE':>9}{'AFTER-HRS':>11}{'MOVE':>8}  FILINGS\n", "head")
        for r in sorted(reads.values(), key=lambda r: -abs(r.get("ah_chg") or 0)):
            axp = f"{r['ah_price']:>11.2f}" if r.get("ah_price") else f"{'—':>11}"
            tag = "sell" if r["ah_chg"] < 0 else ("buy" if r["ah_chg"] > 0 else "dim")
            o.insert("end", f"{r['ticker']:<8}{r['reg_close']:>9.2f}{axp}", "txt")
            o.insert("end", f"{r['ah_chg']:>+7.1f}%", tag)
            nf = len(r.get("filings") or [])
            o.insert("end", f"   {nf} filing(s)\n" if nf else "\n", "dim")
        o.insert("end", "\nHOW THIS WORKS: it reports the after-hours PRICE MOVE and the actual "
                 "SEC filings (Form 4 insider trades, 8-K events, offerings) that CAUSE moves — "
                 "disclosed, sourced, timestamped. It does NOT guess 'institutional intent' from "
                 "anonymous order flow, because that can't be done reliably. The filing IS the intent.\n", "dim")
        o.configure(state="disabled")
        self.s_status.configure(text=f"After-Hours — {len(flagged)} flagged.", fg=BUY)
        self.s_ah_btn.configure(state="normal")

    def _afterhours_card(self, o, r):
        """One alert card: price move · SEC filing (the cause) · whale footprint.
        Every line is real/sourced — no fabricated 'institutional block prints'."""
        bar = "┌" + "─" * 56 + "┐\n"
        filings = r.get("filings") or []
        offering = next((f for f in filings if f["bias"] < 0), None)
        headline = (f"⚠ DILUTION/OFFERING" if offering else
                    "🔔 MATERIAL FILING" if filings else "🌙 AFTER-HOURS MOVE")
        o.insert("end", bar, "dim")
        o.insert("end", f"  🔔 {r['ticker']}  ", "big")
        o.insert("end", f"{headline}\n", "sell" if offering else "warn")
        # price line
        if r.get("ah_price"):
            tag = "sell" if r["ah_chg"] < 0 else "buy"
            o.insert("end", f"  Price: {r['ah_price']:.2f}  ", "txt")
            o.insert("end", f"({r['ah_chg']:+.1f}%)", tag)
            o.insert("end", f"  vs {r['reg_close']:.2f} close · {market_session()}\n", "dim")
        # EDGAR filing line(s) — the disclosed cause
        for f in filings[:2]:
            ahtag = " · ⏰ after-hours" if f["after_hours"] else ""
            o.insert("end", f"  📂 EDGAR {f['form']} — {f['note']}{ahtag}\n",
                     "sell" if f["bias"] < 0 else "buy" if f["bias"] > 0 else "txt")
            o.insert("end", f"     {f['url']}\n", "dim")
        # REAL dark-pool block flow from the FINRA TRF tape (Alpaca SIP)
        fl = r.get("orderflow")
        if fl and fl.get("n_blocks"):
            o.insert("end", "  🐋 Dark-pool blocks (FINRA TRF): ", "txt")
            o.insert("end", f"${fl['block_usd']/1e6:.1f}M across {fl['n_blocks']} block(s)", "warn")
            o.insert("end", f" · {fl['dp_share']*100:.0f}% of prints off-exchange\n", "dim")
            if fl.get("largest"):
                L = fl["largest"]
                o.insert("end", f"     largest ${L['usd']/1e6:.1f}M ({L['shares']:,} sh @ {L['price']:.2f})", "dim")
            if fl.get("sweeps"):
                o.insert("end", f" · {fl['sweeps']} intermarket sweep(s)", "dim")
            o.insert("end", "\n", "dim")
            # real aggressor split via NBBO (buy = printed above mid, sell = below)
            if "buy_usd" in fl and (fl["buy_usd"] or fl["sell_usd"] or fl["mid_usd"]):
                o.insert("end", "     aggressor: ", "dim")
                o.insert("end", f"${fl['buy_usd']/1e6:.1f}M buy", "buy")
                o.insert("end", " · ", "dim"); o.insert("end", f"${fl['sell_usd']/1e6:.1f}M sell", "sell")
                o.insert("end", f" · ${fl['mid_usd']/1e6:.1f}M ambiguous at midpoint\n", "dim")
                alert = of.block_alert(fl)
                if alert:
                    o.insert("end", f"     🟢 BUY-INITIATED BLOCKS: {alert['detail']}\n", "buy")
        else:
            w = r.get("whale")            # fall back to the volume/flow footprint
            if w:
                wtag = "buy" if w["direction"] == "accumulation" else "sell" if w["direction"] == "distribution" else "dim"
                o.insert("end", "  🐋 Whale footprint: ", "txt")
                o.insert("end", f"volume {w['rvol']:.1f}× avg, {w['direction']} pressure\n", wtag)
        o.insert("end", "     ↳ blocks are anonymous — 'institutional' is a reasonable inference, "
                 "not a nametag; buy/sell for midpoint dark prints is ambiguous\n", "dim")
        o.insert("end", "└" + "─" * 56 + "┘\n\n", "dim")

    def _screen_failed(self, msg):
        self.s_out.configure(state="normal"); self.s_out.delete("1.0", "end")
        self.s_out.insert("end", "Scan failed.\n\n", "sell")
        self.s_out.insert("end", msg + "\n\n", "txt")
        self.s_out.insert("end", "Check your internet, or tick Demo to test offline.\n", "dim")
        self.s_out.configure(state="disabled")
        self.s_status.configure(text="Scan failed — see above.", fg=SELL)
        self.s_btn.configure(state="normal")

    def _render_screen(self, results, filtered, errors, ctx):
        self._last_results = results
        self._last_filtered = filtered
        self._last_ctx = ctx
        log_verdicts(results, ctx.get("demo"))        # keep score on our own calls
        self.s_out.configure(state="normal"); self.s_out.delete("1.0", "end")
        ranked = sorted(results, key=lambda r: -r["score"])

        # detect stocks that just turned BUY / STRONG BUY, and alert
        first_run = self._prev_good is None
        new_signals, _ = detect_new_buys(self._prev_good, ranked, self.s_alert.get())
        self._prev_good = {r["ticker"] for r in ranked if r["verdict"]["tone"] == "good"}
        # Build dict for O(1) lookup instead of O(n) search per alert
        ranked_dict = {r["ticker"]: r for r in ranked}
        for t in sorted(new_signals):
            r = ranked_dict[t]
            mac_notify(f"🔔 {t} — {r['verdict']['label']}",
                       f"score {r['score']:+.0f}  ·  {r['last']:.2f} ({r['chg']:+.2f}%)  ·  vol {fmt_vol(r['dollar_vol'])}",
                       "Glass")

        # Batch all rendering as (tag, text) pairs — 60 inserts vs 200+ (3x faster)
        lines = []
        note = ctx.get("note", "")
        if note:
            lines.append(("dim", f"{note}\n"))
        
        total = len(results) + len(filtered) + len(errors)
        lines.append(("big", f"RANKED  ({len(results)} passed / {total} scanned)\n"))
        if not results and ctx.get("whale_only"):
            lines.append(("warn", "🐋 No whale-ACCUMULATION footprints right now — this is normal, "
                          "not a bug.\n"))
            lines.append(("dim", "   Whale bars need ≥1.5× average volume + buying pressure, which is "
                          "rare on a quiet day or a small watchlist.\n"
                          "   Try: source = “Discover — active movers” (high-volume by definition), "
                          "and un-tick “Buy only”.\n"))
        if new_signals:
            lines.append(("buy", f"🔔 NEW BUY SIGNAL — {', '.join(sorted(new_signals))} (you were notified)\n"))
        elif first_run and self.s_alert.get():
            lines.append(("dim", "Baseline set — you'll be alerted when a stock newly turns BUY.\n"))
        
        CAP = 30
        if len(ranked) > CAP:
            lines.append(("dim", f"Showing top {CAP} by score (of {len(ranked)} that passed).\n"))
        
        lines.append(("head", f"{'#':<3}{'':2}{'TICKER':<8}{'LAST':>9}{'CHG%':>8}{'SCORE':>7}  VERDICT\n"))
        lines.append(("dim", "─" * 58 + "\n"))
        
        new_signals_set = set(new_signals)
        tag_map = {"good": "buy", "neutral": "warn", "bad": "sell"}
        for i, r in enumerate(ranked[:CAP], 1):
            v = r["verdict"]; tag = tag_map[v["tone"]]
            is_new = r["ticker"] in new_signals_set
            lines.append(("dim", f"{i:<3}"))
            lines.append(("buy" if is_new else "txt", "★ " if is_new else "  "))
            lines.append(("buy" if is_new else "txt", f"{r['ticker']:<8}"))
            lines.append(("txt", f"{r['last']:>9.2f}"))
            lines.append(("buy" if r["chg"] >= 0 else "sell", f"{r['chg']:>+8.2f}"))
            lines.append((tag, f"{r['score']:>+7.0f}"))
            lines.append(("txt", "  "))
            lines.append((tag, v["label"]))
            if v["risky"]:
                lines.append(("sell", "  [RISKY]"))
            lines.append(("dim", f"   {fmt_vol(r['dollar_vol'])}"))
            w = r.get("whale_activity")
            if w and w["whale"]:
                wtag = "buy" if w["direction"] == "accumulation" else "sell"
                lines.append((wtag, f"  🐋{w['rvol']:.0f}×{'↑' if w['direction']=='accumulation' else '↓'}"))
            cnet = congress_net(r.get("congress"))
            if cnet:
                lines.append((cnet[1], f"  {cnet[0]}"))
            fsum = fe.fmt_fund(r.get("fund"))
            if fsum:
                lines.append(("dim", f"   {fsum}"))
            lines.append(("dim", "\n"))
        for tag, text in lines:
            self.s_out.insert("end", text, tag)
        if filtered:
            show = filtered[:20]
            more = f" +{len(filtered)-20} more" if len(filtered) > 20 else ""
            self.s_out.insert("end", f"\nFiltered out ({len(filtered)}): ", "dim")
            self.s_out.insert("end", " ".join(show) + more + "\n", "sell")
        if errors:
            self.s_out.insert("end", f"\nNo data / skipped ({len(errors)})", "dim")
            self.s_out.insert("end", ": " + " ".join(t for t, _ in errors[:15])
                              + (f" +{len(errors)-15} more" if len(errors) > 15 else "") + "\n", "dim")
        if ranked:
            self.s_out.insert("end", "\n▶ Click “Open charts in browser” for the visual report.\n", "warn")
        self.s_out.configure(state="disabled")
        self.s_status.configure(text=f"Done — {len(results)} passed.", fg=BUY)
        self.s_btn.configure(state="normal")
        self.s_chart_btn.configure(state="normal" if ranked else "disabled")
        self.s_updated.configure(
            text=f"Last updated {dt.datetime.now().strftime('%H:%M:%S')}"
            + (f" · tracking live, next in {max(MIN_REFRESH_SEC, int(self._num(self.s_int_sec.get(), 300)))}s"
               if self.s_auto.get() else ""))
        if self.s_auto.get():
            self._schedule_auto()

    def _schedule_auto(self):
        if self._auto_job:
            self.root.after_cancel(self._auto_job)
            self._auto_job = None
        if not self.s_auto.get():
            return
        sec = max(MIN_REFRESH_SEC, int(self._num(self.s_int_sec.get(), 300)))
        self._auto_job = self.root.after(sec * 1000, self._auto_tick)

    def _auto_tick(self):
        self._auto_job = None
        if self.s_auto.get():
            self.run_screen(auto=True)

    def toggle_auto(self):
        if self.s_auto.get():
            self.s_status.configure(text="Live tracking on — refreshing now…", fg=AMBER)
            self.run_screen(auto=True)
        else:
            if self._auto_job:
                self.root.after_cancel(self._auto_job)
                self._auto_job = None
            self.s_status.configure(text="Live tracking off.", fg=DIM)
            self.s_updated.configure(text=self.s_updated.cget("text").split(" · ")[0])

    def _apply_daytrade(self):
        """Optional intraday override. Swing (default, unchecked) uses daily bars."""
        if self.s_daytrade.get():
            # Default to 5m but leave the selector ENABLED so you can drop to
            # 2m/1m for finer intraday work. Intraday backtests now flatten
            # overnight, so stats reflect actual day-trading.
            if self.s_interval.get() not in ("1m", "2m", "5m", "15m"):
                self.s_interval.set("5m")
            self.s_interval.configure(state="readonly")
            self.s_int_sec.delete(0, "end"); self.s_int_sec.insert(0, "60")
            # Intraday can't be disk-cached and re-downloads every refresh, so
            # scanning 120 names is slow. Default to the (small) watchlist.
            self.s_source.set("My watchlist")
            self.s_status.configure(text="Day-trade mode on your watchlist — pick 5m/2m/1m bars. "
                                    "Backtest is flat overnight (accurate for day trading). "
                                    "Free data is ~15 min delayed.", fg=AMBER)
        else:
            self.s_interval.configure(state="readonly")
            self.s_interval.set("1d")
            self.s_int_sec.delete(0, "end"); self.s_int_sec.insert(0, "900")
            self.s_status.configure(text="Swing mode (daily bars) — the recommended setup. "
                                    "Set Min $ vol to 5, tick Track live.", fg=DIM)

    def save_watchlist(self):
        tickers = self._parse_watchlist()
        if not tickers:
            self.s_status.configure(text="Nothing to save — add tickers first.", fg=SELL)
            return
        try:
            save_watchlist_file(tickers)
            self.s_status.configure(text=f"Saved {len(tickers)} tickers — will auto-load next launch.", fg=BUY)
        except Exception as e:
            self.s_status.configure(text=f"Could not save: {e}", fg=SELL)

    def open_charts(self):
        if not self._last_results:
            return
        ctx = self._last_ctx
        filt = []
        if ctx["buy_only"]: filt.append("BUY only")
        if ctx["exclude_risky"]: filt.append("no RISKY")
        if ctx["min_vol"]: filt.append(f"$vol ≥ {fmt_vol(ctx['min_vol'])}")
        filt_txt = " · ".join(filt) if filt else "none"
        html_str = build_screener_html(self._last_results[:40] if len(self._last_results) > 40
                                        else self._last_results, self._last_filtered,
                                        ctx["interval"], ctx["demo"], filt_txt)
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screener_report.html")
        try:
            with open(out, "w") as f:
                f.write(html_str)
            webbrowser.open("file://" + os.path.abspath(out))
            self.s_status.configure(text="Charts opened in your browser.", fg=BUY)
        except Exception as e:
            self.s_status.configure(text=f"Could not open charts: {e}", fg=SELL)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Order-flow / dark-pool detection from the Alpaca SIP consolidated tape.

The SIP feed exposes what the free IEX feed can't: every exchange's prints,
including FINRA TRF (dark-pool) reports (exchange code 'D'), plus sale-condition
codes — F (Intermarket Sweep / ISO), T (Form T / extended hours). This isolates
the LARGE off-exchange BLOCK prints that are, in practice, institutional-sized.

HONEST SCOPE — what is FACT vs INFERENCE here:
  FACT (observable in the tape):
    • a $2.1M block printed off-exchange (TRF venue 'D') at 15:59:12
    • N intermarket sweeps fired; M Form-T extended-hours prints
    • total dark-pool block dollar-volume over a window
  INFERENCE (labelled, not claimed as certain):
    • "institutional" — a $2M dark-pool cross is almost certainly not retail, but
      it is still anonymous; could be a fund, a bank desk, or an index rebalance.
    • BUY vs SELL — dark prints usually cross at the NBBO MIDPOINT by design, so
      the aggressor side is genuinely ambiguous. We report a rough price-drift
      heuristic (are the blocks printing higher or lower through the window) and
      explicitly do NOT assert 'sell-side' when it's a coin flip.

Needs Alpaca SIP access (feed=sip). Falls back cleanly to {} without it.
"""
import bisect
import datetime as dt
import json
import urllib.parse
import urllib.request


def _pull(ticker, kind, key, secret, start_iso, end_iso, max_pages=8, timeout=25):
    """Generic paginated SIP puller for 'trades' or 'quotes'."""
    hdr = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    rows, page = [], None
    for _ in range(max_pages):
        q = {"start": start_iso, "end": end_iso, "limit": 10000, "feed": "sip"}
        if page:
            q["page_token"] = page
        url = f"https://data.alpaca.markets/v2/stocks/{ticker}/{kind}?" + urllib.parse.urlencode(q)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=timeout) as r:
                d = json.loads(r.read().decode())
        except Exception:
            break
        rows += d.get(kind, [])
        page = d.get("next_page_token")
        if not page:
            break
    return rows


def darkpool_blocks(ticker, key, secret, start_iso, end_iso, min_notional=100000, classify=True):
    """Isolate large FINRA-TRF (dark-pool) BLOCK prints in a window. Returns:
      n_trades       total prints scanned
      dp_share       fraction of prints that were off-exchange (TRF venue 'D')
      n_blocks       dark-pool prints >= min_notional
      block_usd      total $ of those blocks
      largest        {usd, shares, price, time} of the biggest block
      sweeps         count of Intermarket-Sweep (ISO, condition 'F') prints
      form_t         count of Form-T (extended-hours) prints
      drift          block VWAP vs early-vs-late split (rough +accum / -distrib heuristic)
    or {} if no data / no SIP access."""
    if not (key and secret):
        return {}
    trades = _pull(ticker, "trades", key, secret, start_iso, end_iso)
    if not trades:
        return {}
    n = len(trades)
    dp = [t for t in trades if t.get("x") == "D"]
    blocks = [t for t in dp if t["p"] * t["s"] >= min_notional]
    blocks.sort(key=lambda t: t["t"])
    block_usd = sum(t["p"] * t["s"] for t in blocks)
    sweeps = sum(1 for t in trades if "F" in t.get("c", []))
    form_t = sum(1 for t in trades if "T" in t.get("c", []))
    largest = max(blocks, key=lambda t: t["p"] * t["s"]) if blocks else None
    # rough drift heuristic: dollar-weighted avg price of the FIRST vs SECOND half
    # of blocks. Rising = leaning accumulation, falling = distribution. NOT a
    # buy/sell claim — just where the size printed over time.
    drift = None
    if len(blocks) >= 4:
        half = len(blocks) // 2
        def vwap(rows):
            num = sum(t["p"] * t["s"] for t in rows); den = sum(t["s"] for t in rows)
            return num / den if den else 0.0
        early, late = vwap(blocks[:half]), vwap(blocks[half:])
        if early:
            drift = (late / early - 1.0) * 100.0
    # Aggressor side via NBBO: a block above the midpoint = buy-initiated (a buyer
    # paid up), below = sell-initiated, AT the midpoint = genuinely ambiguous (how
    # most dark prints cross). This is the block-premium/discount method — real,
    # but honest that midpoint crosses can't be assigned a side.
    buy_usd = sell_usd = mid_usd = 0.0
    if classify and blocks:
        quotes = _pull(ticker, "quotes", key, secret, start_iso, end_iso)
        qt = [q["t"] for q in quotes]
        for t in blocks:
            usd = t["p"] * t["s"]
            i = bisect.bisect_left(qt, t["t"]) - 1
            if i < 0 or not quotes:
                mid_usd += usd; continue
            q = quotes[i]
            m = (q.get("bp", 0) + q.get("ap", 0)) / 2.0
            if m <= 0:
                mid_usd += usd
            elif t["p"] > m * 1.0001:
                buy_usd += usd
            elif t["p"] < m * 0.9999:
                sell_usd += usd
            else:
                mid_usd += usd
    return {"n_trades": n, "dp_share": len(dp) / n if n else 0.0,
            "n_blocks": len(blocks), "block_usd": block_usd,
            "sweeps": sweeps, "form_t": form_t, "drift": drift,
            "buy_usd": buy_usd, "sell_usd": sell_usd, "mid_usd": mid_usd,
            "net_usd": buy_usd - sell_usd,
            "largest": ({"usd": largest["p"] * largest["s"], "shares": largest["s"],
                         "price": largest["p"], "time": largest["t"]} if largest else None)}


def block_alert(fl, min_buy_usd=250000):
    """Honest institutional-block BUY read from classified flow. Fires ONLY on
    observable buy-initiated block volume (blocks printed above the NBBO mid) that
    outweighs sell-initiated — never a fabricated 'defending the floor' narrative.
    Returns {detail, buy_usd, net_usd} or None."""
    if not fl or not fl.get("buy_usd"):
        return None
    if fl["buy_usd"] >= min_buy_usd and fl["net_usd"] > 0:
        return {"buy_usd": fl["buy_usd"], "net_usd": fl["net_usd"],
                "detail": (f"${fl['buy_usd']/1e6:.1f}M buy-initiated blocks (above NBBO mid) "
                           f"vs ${fl['sell_usd']/1e6:.1f}M sell · "
                           f"${fl['mid_usd']/1e6:.1f}M ambiguous at midpoint")}
    return None


def after_hours_window():
    """(start_iso, end_iso) covering the most recent extended/overnight session."""
    now = dt.datetime.now(dt.timezone.utc)
    start = now - dt.timedelta(hours=18)          # covers post-close + overnight
    return start.strftime("%Y-%m-%dT%H:%M:%SZ"), now.strftime("%Y-%m-%dT%H:%M:%SZ")

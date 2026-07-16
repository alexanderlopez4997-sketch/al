#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
News / text sentiment for Meridian — lightweight, dependency-free.

Pulls recent company-news headlines+summaries from Finnhub (free tier, 60/min,
reuses the app's key) and scores them with a finance-tuned sentiment lexicon
(Loughran-McDonald style word lists, with simple negation handling). Produces a
"Macro/Text Sentiment" signal for the alt-data tilt, and a tone-shift flag that
tightens the ATR stop when coverage turns defensive.

HONEST SCOPE: this is lexicon sentiment over headlines, NOT a transformer model
analyzing full earnings-call transcripts. Free transcripts aren't available and
a real LLM/FinBERT model would be a heavy dependency the app deliberately avoids.
Lexicon sentiment on news flow is a coarse but genuine signal — weighted low and
capped like every other alt input, never overriding the technical read.
"""
import hashlib
import re

# Curated finance sentiment lexicon (high-signal subset; extend freely).
_POS = {
    "beat", "beats", "beating", "exceeded", "exceed", "surge", "surged", "soar",
    "soared", "rally", "rallied", "upgrade", "upgraded", "outperform", "strong",
    "strength", "growth", "grow", "record", "gains", "gain", "bullish", "optimistic",
    "robust", "momentum", "accelerate", "accelerating", "expansion", "profit",
    "profitable", "raised", "raise", "boost", "boosted", "breakthrough", "approval",
    "approved", "wins", "win", "won", "tailwind", "upside", "buyback", "dividend",
    "outperformed", "rebound", "recovery", "improve", "improved", "improving",
    "positive", "confident", "demand", "expanding", "leading", "innovative",
}
_NEG = {
    "miss", "missed", "missing", "plunge", "plunged", "plummet", "plummeted",
    "decline", "declined", "declining", "downgrade", "downgraded", "underperform",
    "weak", "weakness", "loss", "losses", "bearish", "cautious", "defensive",
    "warning", "warn", "warns", "cut", "cuts", "layoffs", "layoff", "lawsuit",
    "investigation", "probe", "recall", "delay", "delayed", "headwind", "headwinds",
    "pressure", "pressures", "slowdown", "slowing", "concern", "concerns", "risk",
    "risks", "bankruptcy", "default", "shortfall", "disappointing", "disappoint",
    "disappointed", "fell", "falling", "drop", "dropped", "sink", "slump", "fraud",
    "halt", "halted", "struggle", "struggling", "uncertainty", "uncertain", "fear",
}
_NEGATORS = {"not", "no", "never", "without", "less", "lower", "fails", "fail",
             "cannot", "hardly", "barely", "avoid", "isn't", "wasn't", "didn't"}

_WORD = re.compile(r"[a-z']+")


def score_text(text):
    """Return (sentiment in [-1,+1], hit_count). Negation within 2 tokens flips."""
    toks = _WORD.findall((text or "").lower())
    pos = neg = 0
    for i, w in enumerate(toks):
        s = 1 if w in _POS else (-1 if w in _NEG else 0)
        if not s:
            continue
        if any(t in _NEGATORS for t in toks[max(0, i - 2):i]):
            s = -s
        if s > 0:
            pos += 1
        else:
            neg += 1
    hits = pos + neg
    return ((pos - neg) / hits if hits else 0.0), hits


# ------------------------------------------------------------------ fetch ---
def fetch_news(ticker, key, days=7, timeout=10):
    """Recent company news from Finnhub (free tier). Returns list of dicts with
    headline, summary, datetime (epoch). Empty list on failure."""
    if not key:
        return []
    import datetime as _dt
    import json
    import urllib.request
    to = _dt.date.today()
    frm = to - _dt.timedelta(days=days)
    url = (f"https://finnhub.io/api/v1/company-news?symbol={ticker}"
           f"&from={frm}&to={to}&token={key}")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _wmean(rows):
    """Weighted mean of (score, weight) rows, or None if empty."""
    num = den = 0.0
    for s, w in rows:
        num += w * s; den += w
    return num / den if den else None


def _assemble(scored, days, recent_days, source):
    """Common aggregation for both sentiment sources. `scored` is a list of
    (score, weight, age_days) with score already in [-1,+1]."""
    if not scored:
        return None
    signal = _wmean([(s, w) for s, w, _ in scored])
    recent = _wmean([(s, w) for s, w, a in scored if a <= recent_days])
    prior = _wmean([(s, w) for s, w, a in scored if a > recent_days])
    defensive_shift = bool(recent is not None and prior is not None
                           and recent < -0.15 and (recent - prior) <= -0.30)
    return {"signal": float(max(-1.0, min(1.0, signal))),
            "confidence": min(1.0, len(scored) / 8.0),
            "n": len(scored), "source": source,
            "recent": recent, "prior": prior,
            "defensive_shift": defensive_shift,
            "detail": (f"{len(scored)} articles · tone {signal:+.2f} [{source}]"
                       + (f" · recent {recent:+.2f} vs prior {prior:+.2f}"
                          if recent is not None and prior is not None else "")
                       + (" · ⚠ DEFENSIVE SHIFT" if defensive_shift else ""))}


def _av_news_sentiment(ticker, av_key, days=7, recent_days=2):
    """PRIMARY: Alpha Vantage NEWS_SENTIMENT — professionally scored, per-ticker,
    relevance-weighted. Scores are on AV's scale where ±0.35 = Bearish/Bullish,
    so we map /0.35 into [-1,+1]. Needs a working AV key (premium for volume)."""
    import calendar
    import datetime as _dt
    import json
    import time as _t
    import urllib.request
    url = (f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
           f"&tickers={ticker}&limit=200&apikey={av_key}")
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            data = json.loads(r.read().decode())
    except Exception:
        return None
    feed = data.get("feed") if isinstance(data, dict) else None
    if not feed:
        return None
    now = _t.time()
    scored = []
    for art in feed:
        ts = next((t for t in art.get("ticker_sentiment", [])
                   if t.get("ticker") == ticker), None)
        if not ts:
            continue
        try:
            score = float(ts["ticker_sentiment_score"])
            rel = float(ts["relevance_score"])
        except (KeyError, ValueError, TypeError):
            continue
        if rel < 0.1:
            continue
        try:
            dtp = _dt.datetime.strptime(art.get("time_published", ""), "%Y%m%dT%H%M%S")
            age = max(0.0, (now - calendar.timegm(dtp.timetuple())) / 86400.0)
        except Exception:
            age = 0.0
        if age > days:
            continue
        recency = max(0.1, 1.0 - age / max(days, 1))
        scored.append((max(-1.0, min(1.0, score / 0.35)), rel * recency, age))
    return _assemble(scored, days, recent_days, "AlphaVantage")


def _finnhub_lexicon_sentiment(ticker, key, days=7, recent_days=2):
    """FALLBACK: Finnhub headlines scored by the built-in finance lexicon."""
    articles = fetch_news(ticker, key, days)
    if not articles:
        return None
    import time as _t
    now = _t.time()
    scored = []
    for a in articles:
        s, hits = score_text(f"{a.get('headline','')} . {a.get('summary','')}")
        if hits > 0:
            age = max(0.0, (now - (a.get("datetime") or now)) / 86400.0)
            scored.append((s, max(0.1, 1.0 - age / max(days, 1)) * min(hits, 4), age))
    return _assemble(scored, days, recent_days, "lexicon")


def news_sentiment(ticker, finnhub_key=None, av_key=None, days=7, recent_days=2):
    """Aggregate news sentiment. Tries Alpha Vantage's professionally-scored
    NEWS_SENTIMENT first (relevance-weighted, per-ticker), falls back to Finnhub
    headlines + the built-in lexicon. Returns dict (see _assemble) or None."""
    if av_key:
        av = _av_news_sentiment(ticker, av_key, days, recent_days)
        if av:
            return av
    return _finnhub_lexicon_sentiment(ticker, finnhub_key, days, recent_days)


def demo_sentiment(ticker):
    """Deterministic synthetic sentiment for offline/demo mode."""
    h = int(hashlib.md5(("news" + ticker.upper()).encode()).hexdigest(), 16)
    sig = ((h % 200) / 100.0) - 1.0                      # -1 .. +1
    recent = max(-1.0, min(1.0, sig - ((h >> 8) % 100) / 200.0))
    return {"signal": round(sig, 2), "confidence": 0.6, "n": (h >> 4) % 12 + 3,
            "source": "demo", "recent": round(recent, 2), "prior": round(sig, 2),
            "defensive_shift": bool(recent < -0.15 and (recent - sig) <= -0.30),
            "detail": f"{(h>>4)%12+3} articles · tone {sig:+.2f} (demo)"}


def macro_signal(sent):
    """News-sentiment summary -> {signal, confidence, detail} for the alt tilt."""
    if not sent:
        return None
    return {"signal": sent["signal"], "confidence": sent["confidence"],
            "detail": sent["detail"]}

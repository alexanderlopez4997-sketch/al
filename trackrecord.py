#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verdict track-record — the app keeps score on ITSELF.

Every live BUY/HOLD/AVOID the app issues is logged with a timestamp and price.
Later, each verdict is graded against what the stock ACTUALLY did over the next
`horizon` trading days:
  • a BUY  is a hit if the stock rose
  • an AVOID is a hit if it fell (correctly dodged)
  • HOLD is directional-neutral (tracked, not graded)

It also attributes outcomes to the signals that were present (whale accumulation,
congressional buying, insider buying, …), so you can see which of the app's many
signals ACTUALLY add edge — and which are noise worth cutting.

HONEST SCOPE: this is a real, forward-looking out-of-sample record (unlike an
in-sample backtest) — but it's still a small, self-selected sample that grows
slowly, and past hit rate does not guarantee future results. It's here to keep
the app honest, not to promise profit.
"""
import json
import os
from datetime import date

_PATH = os.path.expanduser("~/.meridian_cache/verdict_journal.json")


def _load():
    try:
        with open(_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def _save(entries):
    try:
        os.makedirs(os.path.dirname(_PATH), exist_ok=True)
        with open(_PATH, "w") as f:
            json.dump(entries, f)
    except Exception:
        pass


def log_verdicts(rows):
    """rows: list of {ticker, tone, label, score, price, tags}. Deduped by
    (ticker, date) so re-scanning the same day doesn't double-count. Returns
    how many NEW entries were added."""
    entries = _load()
    today = date.today().isoformat()
    seen = {(e["ticker"], e["date"]) for e in entries}
    added = 0
    for r in rows:
        if not r.get("price") or (r["ticker"], today) in seen:
            continue
        entries.append({"ticker": r["ticker"], "date": today, "tone": r["tone"],
                        "label": r["label"], "score": round(float(r["score"]), 1),
                        "price": round(float(r["price"]), 4), "tags": r.get("tags", [])})
        seen.add((r["ticker"], today))
        added += 1
    if added:
        _save(entries)
    return added


def _pos_on_or_after(index, date_str):
    y, m, d = (int(x) for x in date_str.split("-"))
    for i, ts in enumerate(index):
        t = ts.date() if hasattr(ts, "date") else ts
        if (t.year, t.month, t.day) >= (y, m, d):
            return i
    return None


def score(price_lookup, horizon=5):
    """Grade each logged verdict. price_lookup(ticker) -> daily DataFrame (or None).
    Returns the entries annotated with status/fwd_ret/win."""
    out = []
    for e in _load():
        df = price_lookup(e["ticker"])
        rec = dict(e)
        if df is None or len(df) < 2:
            rec["status"] = "no data"; out.append(rec); continue
        p = _pos_on_or_after(df.index, e["date"])
        if p is None or p + horizon >= len(df):
            rec["status"] = "pending"; out.append(rec); continue
        fwd = float(df["Close"].iloc[p + horizon])
        ret = fwd / e["price"] - 1.0
        win = (ret > 0) if e["tone"] == "good" else (ret < 0) if e["tone"] == "bad" else None
        rec.update({"status": "scored", "fwd_ret": ret, "win": win})
        out.append(rec)
    return out


def summary(scored, horizon=5):
    """Aggregate hit rate + avg forward return by verdict type, plus per-signal
    attribution (hit rate of graded calls that carried each tag)."""
    graded = [e for e in scored if e.get("status") == "scored" and e["win"] is not None]
    pending = sum(1 for e in scored if e.get("status") == "pending")
    res = {"total": len(scored), "graded": len(graded), "pending": pending, "horizon": horizon,
           "by_tone": {}, "by_tag": {}}
    for tone, name in (("good", "BUY"), ("bad", "AVOID")):
        g = [e for e in graded if e["tone"] == tone]
        if g:
            res["by_tone"][name] = {
                "n": len(g), "hit_rate": sum(e["win"] for e in g) / len(g),
                "avg_fwd": sum(e["fwd_ret"] for e in g) / len(g)}
    tags = {}
    for e in graded:
        for t in e.get("tags", []):
            tags.setdefault(t, []).append(e)
    for t, g in tags.items():
        if len(g) >= 5:                       # need a minimum sample to mean anything
            res["by_tag"][t] = {"n": len(g), "hit_rate": sum(e["win"] for e in g) / len(g),
                                "avg_fwd": sum(e["fwd_ret"] for e in g) / len(g)}
    return res

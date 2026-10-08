#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contract <-> EDGAR cross-check for Meridian.

A government award (contracts.py / dod_scraper.py) and the company's own SEC
filings (edgar.py) are scored independently, so they can quietly contradict
each other. This module lines them up on the award date:

  1. DILUTION -- small caps routinely raise money right after a win. An S-3 /
     424B* filed within DILUTION_WINDOW_DAYS of the award means the "bullish"
     contract may just be the pitch for an offering. Confidence x 0.5.
  2. INSIDER SELLING -- unplanned (non-10b5-1) insider sales within
     INSIDER_POST_DAYS after the award: the people who know the contract best
     are cashing out into it. Confidence x 0.6.
  3. INSIDER BUYING -- open-market insider purchases in the INSIDER_PRE_DAYS
     before the award: two independent sources pointing the same way.
     Confidence x 1.25 (capped at 1.0). Only applied when 1 and 2 are clear --
     a negative flag always beats a positive one.

Only a positive contract signal is cross-checked (an award is never a sell
signal, so there is nothing to "confirm" on a non-positive one), and only for a
RECENT award: both the Form 4 lookback and the decay in contracts.py make
anything older than MAX_AWARD_AGE_DAYS moot.

Everything is fail-open: missing dates, empty EDGAR results or any exception
mean "no cross-check" (the original signal passes through untouched), never a
crash and never a made-up flag. A failed EDGAR fetch is indistinguishable from
"nothing filed", so a cross-check can only ever be absent, not wrong in the
direction of a false alarm.

The pure core (evaluate / adjusted_signal) takes the EDGAR data as arguments so
it is testable without a network; crosscheck_for_ticker() is the I/O glue.
"""
import logging
from datetime import date, datetime, timedelta

logger = logging.getLogger("contract_crosscheck")

DILUTION_WINDOW_DAYS = 10     # offering filed this soon after an award is "the pitch"
INSIDER_PRE_DAYS = 30         # insider buying this far BEFORE the award counts
INSIDER_POST_DAYS = 10        # insider selling this far AFTER the award counts
MAX_AWARD_AGE_DAYS = 45       # older awards: skip (Form 4 coverage + signal decay)
MIN_INSIDER_USD = 100_000.0   # below this, insider flow is noise either way

DILUTION_MULT = 0.5
INSIDER_SELL_MULT = 0.6
INSIDER_BUY_MULT = 1.25


def _to_date(value):
    """'2026-10-07' / '2026-10-07T21:40:00+00:00' / 'Z'-suffixed -> date, else None."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except Exception:
        return None


def award_age_days(sig, today=None):
    """Days since sig["award_date"], or None if absent/unparseable/in the future."""
    d = _to_date((sig or {}).get("award_date"))
    if d is None:
        return None
    age = ((today or date.today()) - d).days
    return age if age >= 0 else None


def _dilution_after(award, filings):
    """Offering filings dated from the award date through DILUTION_WINDOW_DAYS later."""
    end = award + timedelta(days=DILUTION_WINDOW_DAYS)
    hits = []
    for f in filings or []:
        d = _to_date(f.get("date")) if isinstance(f, dict) else None
        if d is not None and award <= d <= end:
            hits.append({"form": f.get("form"), "date": d.isoformat(), "url": f.get("url")})
    return hits


def _insider_flow(award, txs):
    """-> (pre_buy_usd, pre_buyers, post_sell_usd, post_sellers) from
    form4_insider_bias()["transactions"]-shaped dicts. Only open-market P/S, and
    never 10b5-1 trades (pre-scheduled, so they say nothing about this award)."""
    pre_start = award - timedelta(days=INSIDER_PRE_DAYS)
    post_end = award + timedelta(days=INSIDER_POST_DAYS)
    pre_buy = post_sell = 0.0
    buyers, sellers = set(), set()
    for tx in txs or []:
        try:
            if not isinstance(tx, dict) or tx.get("is_10b5_1"):
                continue
            d = _to_date(tx.get("accepted"))
            usd = float(tx.get("usd") or 0.0)
            if d is None or usd <= 0:
                continue
            who = tx.get("owner") or "unknown"
            if tx.get("code") == "P" and pre_start <= d <= award:
                pre_buy += usd; buyers.add(who)
            elif tx.get("code") == "S" and award <= d <= post_end:
                post_sell += usd; sellers.add(who)
        except Exception:
            continue
    return pre_buy, len(buyers), post_sell, len(sellers)


def evaluate(sig, dilutive, insider_txs, today=None):
    """Cross-check one contract signal against EDGAR data already in hand.

    Args:
      sig: {signal, confidence, detail, award_date} from contracts.contract_signal
           or dod_scraper.dod_signal_for_ticker.
      dilutive: edgar.dilutive_filings() output ([{form, date, url}]).
      insider_txs: edgar.form4_insider_bias()["transactions"] ([] if none).

    Returns None when there is nothing to cross-check (no/non-positive signal,
    unknown or stale award date). Otherwise:
      {award_date, dilution:[...], insider_buy_usd, insider_sell_usd, flags:[str],
       negative: bool, confidence_mult: float, detail: str}
    """
    try:
        if not sig or sig.get("signal", 0) <= 0:
            return None
        age = award_age_days(sig, today)
        if age is None or age > MAX_AWARD_AGE_DAYS:
            return None
        award = _to_date(sig["award_date"])

        dil = _dilution_after(award, dilutive)
        buy, n_buy, sell, n_sell = _insider_flow(award, insider_txs)

        flags, mult = [], 1.0
        if dil:
            flags.append(f"{dil[0]['form']} offering filed {dil[0]['date']} after award")
            mult *= DILUTION_MULT
        if sell >= MIN_INSIDER_USD:
            flags.append(f"insider selling ${sell/1e6:.2f}M ({n_sell}) within {INSIDER_POST_DAYS}d after award")
            mult *= INSIDER_SELL_MULT
        negative = bool(flags)
        if not negative and buy >= MIN_INSIDER_USD:
            flags.append(f"insider buying ${buy/1e6:.2f}M ({n_buy}) in the {INSIDER_PRE_DAYS}d before award")
            mult = INSIDER_BUY_MULT
        return {"award_date": award.isoformat(), "dilution": dil,
                "insider_buy_usd": buy, "insider_sell_usd": sell,
                "flags": flags, "negative": negative, "confidence_mult": mult,
                "detail": "; ".join(flags)}
    except Exception:
        logger.warning("evaluate() failed", exc_info=True)
        return None


def adjusted_signal(sig, result):
    """Copy of `sig` with confidence scaled by the cross-check (clamped to [0, 1]) and
    the flags appended to its detail. `sig` itself is never mutated. Returns `sig`
    unchanged when there is no result or nothing to say."""
    if not sig or not result or not result.get("flags"):
        return sig
    out = dict(sig)
    out["confidence"] = float(max(0.0, min(1.0, sig.get("confidence", 0.0) * result["confidence_mult"])))
    out["detail"] = f"{sig.get('detail', '')} · EDGAR: {result['detail']}"
    out["crosscheck"] = result
    return out


def crosscheck_for_ticker(ticker, sig, today=None):
    """Fetch the EDGAR data that bears on `sig` and evaluate() it. Reaches EDGAR only
    for a positive, recent award (so it costs nothing for the vast majority of
    tickers). Never raises; None when there is nothing to cross-check."""
    try:
        age = award_age_days(sig, today) if sig and sig.get("signal", 0) > 0 else None
        if age is None or age > MAX_AWARD_AGE_DAYS:
            return None
        import edgar
        dilutive = edgar.dilutive_filings(ticker, days=age + 1)
        ib = edgar.form4_insider_bias(ticker, lookback_hours=(age + INSIDER_PRE_DAYS + 1) * 24,
                                      max_filings=25)
        return evaluate(sig, dilutive, (ib or {}).get("transactions") or [], today)
    except Exception:
        logger.warning("crosscheck_for_ticker(%s) failed", ticker, exc_info=True)
        return None


def apply_crosschecks(ticker, sigs, today=None):
    """Cross-check several contract signals for one ticker at once.

    `sigs` is {source_name: signal_or_None}. Returns (adjusted, results): `adjusted`
    has the same keys with each signal adjusted (or passed through), `results` holds
    the evaluate() output for the sources that had one -- what confirmation.py reads."""
    adjusted, results = {}, {}
    for name, sig in sigs.items():
        r = crosscheck_for_ticker(ticker, sig, today)
        if r:
            results[name] = r
        adjusted[name] = adjusted_signal(sig, r)
    return adjusted, results

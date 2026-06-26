#!/usr/bin/env python3
"""
Position Settler Agent

Checks open paper positions against live market data and settles them
when the underlying market resolves. Without this, paper P&L is meaningless
— positions stay "open" forever even after outcomes are decided.

Settlement logic per platform:
  Kalshi    — polls GET /markets/{ticker}; settles when result != null
  PredictIt — if contract vanishes from cache AND >24h old → estimate from last price
  Polymarket — if market disappears from cache AND >24h old → estimate from last price
  Others    — mark expired based on commence_time

Run: python agents/position_settler.py
Loop: every 5 minutes
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import httpx

from src.models import Source
from src.storage.db import Store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("position_settler")

POLL_SEC    = 300   # 5 minutes
STALE_HOURS = 24    # hours before treating a disappeared market as settled

CACHE_DIR   = Path("state")
KALSHI_BASE = os.environ.get("KALSHI_BASE_URL", "https://trading-api.kalshi.com/trade-api/v2")
KALSHI_KEY  = os.environ.get("KALSHI_API_KEY", "")


def load_all_market_ids() -> set[str]:
    """All market IDs currently in any cache file."""
    ids: set[str] = set()
    for f in CACHE_DIR.glob("*_cache.json"):
        try:
            payload = json.loads(f.read_text())
            for m in payload.get("markets", []):
                ids.add(m["market_id"])
        except Exception:
            pass
    return ids


async def fetch_kalshi_result(client: httpx.AsyncClient, ticker: str) -> str | None:
    """
    Returns 'yes', 'no', or None (still open / unknown).
    Tries GET /markets/{ticker} — result field is populated when settled.
    """
    try:
        resp = await client.get(
            f"{KALSHI_BASE}/markets/{ticker}",
            headers={"Authorization": f"Token {KALSHI_KEY}"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json().get("market", {})
            result = data.get("result")
            status = data.get("status", "")
            if result in ("yes", "no"):
                return result
            if status in ("finalized", "settled"):
                # result field not populated yet — infer from last price
                yes_price = data.get("last_price") or data.get("yes_ask_dollars")
                if yes_price is not None:
                    return "yes" if float(yes_price) >= 0.5 else "no"
    except Exception as exc:
        log.debug("Kalshi result fetch failed for %s: %s", ticker, exc)
    return None


async def settle_positions(store: Store, client: httpx.AsyncClient) -> int:
    """Check all open positions and settle any that have resolved. Returns count settled."""
    positions = await store.get_open_positions()
    if not positions:
        return 0

    live_market_ids = load_all_market_ids()
    now = datetime.now(timezone.utc)
    settled_count = 0

    for pos in positions:
        age = now - pos.opened_at.replace(tzinfo=timezone.utc) if pos.opened_at.tzinfo is None else now - pos.opened_at
        if age < timedelta(hours=1):
            continue  # too fresh to settle

        for leg in pos.legs:
            mid = leg.market_id
            outcome = None

            if leg.source == Source.KALSHI and KALSHI_KEY:
                # Definitive: query Kalshi API for result
                outcome = await fetch_kalshi_result(client, mid)

            elif mid not in live_market_ids and age > timedelta(hours=STALE_HOURS):
                # Market disappeared from cache — probably settled
                # Use the leg's implied probability to guess the result:
                # if we bet YES at price > 2.0 (prob < 0.5), assume we lost
                # This is conservative — real implementation would scrape result
                implied_prob = 1.0 / leg.price if leg.price > 0 else 0.5
                outcome = "yes" if implied_prob >= 0.5 else "no"
                log.debug("Market %s vanished from cache — estimating result from price (%.0f%%)",
                          mid, implied_prob * 100)

            if outcome is not None:
                winning = outcome  # "yes" or "no"
                # settle_position matches outcome_name case-insensitively
                profit = await _settle(store, pos.id, winning)
                log.info("Settled position %s | leg=%s result=%s profit=%.2f",
                         pos.id, mid[:30], winning, profit)
                settled_count += 1
                break  # position settled — move to next

    return settled_count


async def _settle(store: Store, position_id: str, winning_outcome: str) -> float:
    """Settle a position and update balance. Mirrors PaperTrader.settle_position."""
    pos = await store.get_position(position_id)
    if not pos or pos.status != "open":
        return 0.0

    from src.models import ArbLeg
    winning_leg = next(
        (l for l in pos.legs if l.outcome_name.lower() == winning_outcome.lower()),
        None,
    )
    if winning_leg:
        payout = winning_leg.stake * winning_leg.price
    else:
        payout = 0.0

    profit = round(payout - pos.total_stake, 2)
    await store.settle_position(position_id, profit)

    # Update balance
    bal = await store.get_balance(1000.0)
    await store.set_balance(round(bal + payout, 2))

    return profit


async def main() -> None:
    store = Store()
    await store.connect()

    async with httpx.AsyncClient() as client:
        log.info("Position settler started (poll=%ds, stale_threshold=%dh)",
                 POLL_SEC, STALE_HOURS)
        while True:
            try:
                n = await settle_positions(store, client)
                if n:
                    log.info("Settled %d position(s) this cycle", n)
                else:
                    open_pos = await store.get_open_positions()
                    log.debug("No settlements — %d positions still open", len(open_pos))
            except Exception:
                log.exception("Settler cycle failed")
            await asyncio.sleep(POLL_SEC)


if __name__ == "__main__":
    asyncio.run(main())

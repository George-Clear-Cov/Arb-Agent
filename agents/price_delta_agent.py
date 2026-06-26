#!/usr/bin/env python3
"""
Price Delta Agent

Watches confirmed market pairs for prices MOVING TOWARD an arb — catching
opportunities before they cross the detection threshold. Useful for pre-game
markets where odds shift minutes before kickoff.

Tracks a rolling 60-second price window per pair. When the margin delta
(cross-platform implied spread) moves >3% in under 60s, logs a "forming arb"
alert and forces an immediate detection pass on that pair.

Run: python agents/price_delta_agent.py
Loop: 10 seconds (reads from cache files; no API calls)
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.engine.detector import detect_arbs
from src.feeds.feed_cache import CACHE_DIR, _from_dict
from src.models import Source
from src.storage.db import Store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("price_delta")

BRAIN_FILE     = Path("brain/learned_pairs.json")
POLL_SEC       = 10     # check every 10 seconds
WINDOW_SEC     = 60     # rolling window for delta calc
DELTA_ALERT    = 0.03   # 3% move in window = forming-arb alert
MARGIN_WARN    = 0.03   # log NEAR-ARB when margin > 3%
MAX_PLAUSIBLE  = 0.30   # skip pairs with margin > 30% (bad brain matches)

# source name → Source enum (for reconstructing Market objects)
_SOURCE_MAP = {
    "polymarket":    Source.POLYMARKET,
    "kalshi":        Source.KALSHI,
    "kalshi_sports": Source.KALSHI_SPORTS,
    "predictit":     Source.PREDICTIT,
    "gemini":        Source.GEMINI,
    "hyperliquid":   Source.HYPERLIQUID,
}


def load_confirmed_pairs() -> set[tuple[str, str]]:
    """Load confirmed TRUE pairs from brain."""
    if not BRAIN_FILE.exists():
        return set()
    try:
        data = json.loads(BRAIN_FILE.read_text())
        now = time.time()
        confirmed: set[tuple[str, str]] = set()
        for key_str, entry in data.get("entries", {}).items():
            if not entry.get("result"):
                continue
            exp = entry.get("expires_at")
            if exp and now > exp:
                continue
            parts = key_str.split("__", 1)
            if len(parts) == 2:
                confirmed.add((parts[0], parts[1]))
        return confirmed
    except Exception:
        return set()


def load_all_markets() -> dict[str, dict]:
    """Load all markets from all cache files. Returns {market_id: market_dict}."""
    index: dict[str, dict] = {}
    for cache_file in sorted(CACHE_DIR.glob("*_cache.json")):
        source_name = cache_file.stem.replace("_cache", "")
        try:
            payload = json.loads(cache_file.read_text())
            for m in payload.get("markets", []):
                m["_source_name"] = source_name
                index[m["market_id"]] = m
        except Exception:
            pass
    return index


def _yes_prob(outcomes: list[dict]) -> float | None:
    """Return the YES implied probability for a binary market, or None if non-binary."""
    for o in outcomes:
        if o.get("name", "").lower() in ("yes", "true"):
            return o.get("implied_prob")
    return None


def implied_margin(ma: dict, mb: dict) -> tuple[float, float, float] | None:
    """
    Compute arb margin between two confirmed binary markets.
    Returns (margin, price_a, price_b) or None if markets are non-binary.

    For matched YES/NO markets:
      - If ya < yb: bet YES on A (cheaper), NO on B → margin = yb - ya
      - If ya > yb: bet NO on A, YES on B → margin = ya - yb
      - Result = abs(ya - yb)
    """
    try:
        ya = _yes_prob(ma.get("outcomes", []))
        yb = _yes_prob(mb.get("outcomes", []))
        if ya is None or yb is None:
            return None
        return abs(ya - yb), ya, yb
    except Exception:
        return None


async def main() -> None:
    store = Store()
    await store.connect()

    # {pair: deque of (timestamp, margin)} — in-memory rolling window for delta
    history: dict[tuple[str, str], deque] = {}
    alerted: set[tuple[str, str]] = set()

    log.info("Price delta agent started (poll=%ds, alert_threshold=%.0f%%)",
             POLL_SEC, DELTA_ALERT * 100)

    cycle = 0
    while True:
        try:
            confirmed = load_confirmed_pairs()
            markets   = load_all_markets()
            now       = time.time()
            cycle    += 1

            forming: list[tuple[float, str, str]] = []

            for pair in confirmed:
                id_a, id_b = pair
                ma = markets.get(id_a)
                mb = markets.get(id_b)
                if not ma or not mb:
                    continue

                result = implied_margin(ma, mb)
                if result is None:
                    continue
                margin, price_a, price_b = result
                # Skip pairs where margin is implausibly high — likely a bad LLM match
                if margin > MAX_PLAUSIBLE:
                    continue

                # ── Persist to DB (live tracking) ──────────────────────────
                src_name_a = ma.get("_source_name", "unknown")
                src_name_b = mb.get("_source_name", "unknown")
                pair_key   = f"{id_a}__{id_b}"
                event_name = ma.get("event_name", "")[:120]
                try:
                    await store.upsert_tracked_pair(
                        pair_key=pair_key,
                        market_name=event_name,
                        platform_a=src_name_a,
                        platform_b=src_name_b,
                        market_id_a=id_a,
                        market_id_b=id_b,
                        price_a=price_a,
                        price_b=price_b,
                        margin=margin,
                    )
                except Exception:
                    pass  # don't let DB errors break the detection loop

                # ── Rolling delta for "forming arb" alerts ─────────────────
                if pair not in history:
                    history[pair] = deque()
                history[pair].append((now, margin))
                while history[pair] and history[pair][0][0] < now - WINDOW_SEC:
                    history[pair].popleft()

                hist = history[pair]
                if len(hist) < 2:
                    continue

                delta = margin - hist[0][1]

                if margin >= MARGIN_WARN and pair not in alerted:
                    log.info("NEAR-ARB: margin=%.1f%% A=%.2f B=%.2f | %s <-> %s",
                             margin * 100, price_a, price_b,
                             event_name[:40], mb.get("event_name", "")[:40])

                if delta >= DELTA_ALERT:
                    forming.append((delta, event_name, mb.get("event_name", "")))
                    if pair not in alerted:
                        alerted.add(pair)
                        log.info(
                            "FORMING ARB: +%.1f%% in %ds | margin=%.1f%% | %s <-> %s",
                            delta * 100, int(now - hist[0][0]),
                            margin * 100, event_name[:40], mb.get("event_name", "")[:40],
                        )
                        src_a = _SOURCE_MAP.get(src_name_a, Source.POLYMARKET)
                        src_b = _SOURCE_MAP.get(src_name_b, Source.KALSHI)
                        mkt_a = _from_dict(ma, src_a)
                        mkt_b = _from_dict(mb, src_b)
                        if mkt_a and mkt_b:
                            arbs = detect_arbs([mkt_a, mkt_b], min_margin=0.01)
                            if arbs:
                                log.info("  → ARB CONFIRMED: %.1f%%", arbs[0].margin * 100)
                                for arb in arbs:
                                    await store.save_opportunity(arb)
                else:
                    alerted.discard(pair)

            if forming and cycle % 6 == 0:
                log.info("%d forming arbs in last 60s", len(forming))

        except Exception:
            log.exception("Price delta cycle failed")

        await asyncio.sleep(POLL_SEC)


if __name__ == "__main__":
    asyncio.run(main())

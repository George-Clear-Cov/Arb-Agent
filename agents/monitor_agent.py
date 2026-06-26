#!/usr/bin/env python3
"""
Per-Pair Arb Monitor Agent

One asyncio task per platform combination. Each task reads its two platform
cache files independently, runs arb detection, and writes to the shared DB.
Separation means a slow PredictIt poll doesn't delay Poly×Kalshi detection.

Platform pairs and their poll intervals:
  poly × kalshi_sports  → 15s  (live game markets, highest urgency)
  poly × kalshi         → 45s  (prediction + near-term events)
  poly × predictit      → 60s  (prediction markets)
  poly × gemini         → 60s
  poly × hyperliquid    → 60s
  kalshi × predictit    → 90s
  kalshi × gemini       → 90s

Run: python agents/monitor_agent.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.engine.detector import detect_arbs
from src.feeds.feed_cache import CACHE_DIR, load_cache
from src.models import Market, Source
from src.storage.db import Store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("monitor_agent")

MIN_ARB    = 0.02   # 2% minimum margin
STAKE      = 100.0

# (label, source_a, cache_a, source_b, cache_b, interval_seconds)
PAIRS: list[tuple[str, Source, str, Source, str, int]] = [
    ("poly×ksports",  Source.POLYMARKET,    "polymarket_cache",    Source.KALSHI_SPORTS, "kalshi_sports_cache", 15),
    ("poly×kalshi",   Source.POLYMARKET,    "polymarket_cache",    Source.KALSHI,        "kalshi_cache",        45),
    ("poly×predictit",Source.POLYMARKET,    "polymarket_cache",    Source.PREDICTIT,     "predictit_cache",     60),
    ("poly×gemini",   Source.POLYMARKET,    "polymarket_cache",    Source.GEMINI,        "gemini_cache",        60),
    ("poly×hl",       Source.POLYMARKET,    "polymarket_cache",    Source.HYPERLIQUID,   "hyperliquid_cache",   60),
    ("kalshi×pi",     Source.KALSHI,        "kalshi_cache",        Source.PREDICTIT,     "predictit_cache",     90),
    ("kalshi×gemini", Source.KALSHI,        "kalshi_cache",        Source.GEMINI,        "gemini_cache",        90),
]


def _read_cache(name: str, source: Source) -> list[Market]:
    return load_cache(CACHE_DIR / f"{name}.json", source)


async def pair_loop(
    label: str,
    src_a: Source, cache_a: str,
    src_b: Source, cache_b: str,
    interval: int,
    store: Store,
) -> None:
    """Detection loop for one platform pair."""
    log.info("[%s] loop started (interval=%ds)", label, interval)
    consecutive_errors = 0

    while True:
        try:
            start = time.time()
            markets_a = _read_cache(cache_a, src_a)
            markets_b = _read_cache(cache_b, src_b)

            if not markets_a or not markets_b:
                await asyncio.sleep(interval)
                continue

            arbs = detect_arbs(markets_a + markets_b, min_margin=MIN_ARB, total_stake=STAKE)

            elapsed = time.time() - start
            if arbs:
                log.info("[%s] %d arbs found in %.1fs: %s",
                         label, len(arbs), elapsed,
                         ", ".join(f"{a.margin:.1%} {a.event_name[:30]}" for a in arbs[:3]))
                for arb in arbs:
                    await store.save_opportunity(arb)
            else:
                log.debug("[%s] no arbs (%.1fs, %d+%d markets)",
                          label, elapsed, len(markets_a), len(markets_b))

            consecutive_errors = 0

        except Exception:
            consecutive_errors += 1
            log.exception("[%s] detection error (#%d)", label, consecutive_errors)
            if consecutive_errors > 10:
                log.error("[%s] too many consecutive errors — sleeping 5min", label)
                await asyncio.sleep(300)
                consecutive_errors = 0

        await asyncio.sleep(interval)


async def log_summary(store: Store) -> None:
    """Print arb summary every 10 minutes."""
    while True:
        await asyncio.sleep(600)
        try:
            cutoff = datetime.utcnow() - timedelta(hours=1)
            recent = await store.get_recent_opportunities(limit=100)
            last_hour = [a for a in recent if a.detected_at >= cutoff]
            if last_hour:
                by_sport: dict[str, int] = {}
                for a in last_hour:
                    by_sport[a.sport] = by_sport.get(a.sport, 0) + 1
                log.info("Last hour: %d arbs — %s",
                         len(last_hour),
                         ", ".join(f"{s}={c}" for s, c in sorted(by_sport.items())))
            else:
                log.info("Last hour: 0 arbs detected")
        except Exception:
            pass


async def main() -> None:
    store = Store()
    await store.connect()
    log.info("Monitor agent started — %d platform pairs", len(PAIRS))

    tasks = [
        pair_loop(label, src_a, cache_a, src_b, cache_b, interval, store)
        for label, src_a, cache_a, src_b, cache_b, interval in PAIRS
    ]
    tasks.append(log_summary(store))

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())

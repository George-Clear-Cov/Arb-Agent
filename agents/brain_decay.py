#!/usr/bin/env python3
"""
Brain Decay Agent

Nightly cleanup of brain/learned_pairs.json:
  1. Remove expired entries (TTL exceeded)
  2. Remove pairs where BOTH markets no longer exist in any cache file
     (markets have closed / delisted — the pair is dead)
  3. Flag confirmed pairs (TRUE) that have existed >30 days but never
     produced a detected arb — possible false positives from the LLM matcher
  4. Log a clean summary of brain health

Run: python agents/brain_decay.py [--once]
Loop: every 24 hours (runs once at startup, then waits)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("brain_decay")

BRAIN_FILE = Path("brain/learned_pairs.json")
CACHE_DIR  = Path("state")
DB_FILE    = Path("arbitrage.db")
POLL_SEC   = 86400   # 24 hours

# Pairs confirmed >30 days without producing an arb are flagged (not deleted)
STALE_CONFIRMED_DAYS = 30


def load_all_market_ids() -> set[str]:
    """All market IDs currently in any feed cache file."""
    ids: set[str] = set()
    for f in CACHE_DIR.glob("*_cache.json"):
        try:
            payload = json.loads(f.read_text())
            for m in payload.get("markets", []):
                ids.add(m["market_id"])
        except Exception:
            pass
    return ids


def load_arb_market_ids() -> set[str]:
    """Market IDs that have appeared in detected arbs (from SQLite)."""
    arb_ids: set[str] = set()
    try:
        import sqlite3
        db_path = DB_FILE if DB_FILE.exists() else Path("data/arbitrage.db")
        if not db_path.exists():
            return arb_ids
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT legs_json FROM arb_opportunities").fetchall()
        conn.close()
        for (legs_json,) in rows:
            for leg in json.loads(legs_json):
                arb_ids.add(leg.get("market_id", ""))
    except Exception as exc:
        log.warning("Could not load arb market IDs from DB: %s", exc)
    return arb_ids


def run_decay() -> dict:
    if not BRAIN_FILE.exists():
        log.info("Brain file not found — nothing to clean")
        return {}

    try:
        data = json.loads(BRAIN_FILE.read_text())
    except Exception as exc:
        log.error("Failed to read brain file: %s", exc)
        return {}

    entries: dict = data.get("entries", {})
    now = time.time()
    live_ids = load_all_market_ids()
    arb_ids  = load_arb_market_ids()

    removed_expired      = 0
    removed_dead_markets = 0
    flagged_stale        = 0
    kept                 = 0

    new_entries: dict = {}

    for key_str, entry in entries.items():
        parts = key_str.split("__", 1)
        if len(parts) != 2:
            continue
        id_a, id_b = parts

        # 1. Remove expired TTL
        exp = entry.get("expires_at")
        if exp and now > exp:
            removed_expired += 1
            continue

        # 2. Remove dead confirmed pairs (both markets gone >7 days)
        if entry.get("result") is True:
            a_alive = id_a in live_ids
            b_alive = id_b in live_ids
            saved_at = entry.get("saved_at", now)
            age_days = (now - saved_at) / 86400

            if not a_alive and not b_alive and age_days > 7:
                removed_dead_markets += 1
                log.debug("Removing dead pair: %s <-> %s (both markets gone)", id_a[:20], id_b[:20])
                continue

            # 3. Flag stale confirmed pairs (never produced an arb)
            if age_days > STALE_CONFIRMED_DAYS:
                produced_arb = id_a in arb_ids or id_b in arb_ids
                if not produced_arb and not entry.get("stale_flagged"):
                    entry["stale_flagged"] = True
                    entry["stale_flagged_at"] = now
                    flagged_stale += 1
                    log.info("Flagging stale pair (%.0f days, no arb): %s <-> %s",
                             age_days, id_a[:20], id_b[:20])

        new_entries[key_str] = entry
        kept += 1

    data["entries"] = new_entries
    data["last_decay_at"] = now

    tmp = str(BRAIN_FILE) + ".tmp"
    Path(tmp).write_text(json.dumps(data, separators=(",", ":")))
    os.replace(tmp, BRAIN_FILE)

    summary = {
        "total_before": len(entries),
        "total_after":  kept,
        "removed_expired":       removed_expired,
        "removed_dead_markets":  removed_dead_markets,
        "flagged_stale":         flagged_stale,
        "live_market_ids":       len(live_ids),
    }

    # Breakdown of TRUE vs FALSE pairs
    true_count  = sum(1 for e in new_entries.values() if e.get("result") is True)
    false_count = sum(1 for e in new_entries.values() if e.get("result") is False)
    summary["confirmed_pairs"] = true_count
    summary["rejected_pairs"]  = false_count

    log.info(
        "Brain decay complete: %d→%d entries | removed: %d expired, %d dead markets | "
        "flagged: %d stale | confirmed=%d rejected=%d",
        summary["total_before"], summary["total_after"],
        removed_expired, removed_dead_markets, flagged_stale,
        true_count, false_count,
    )

    return summary


async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    log.info("Brain decay agent started")

    while True:
        try:
            run_decay()
        except Exception:
            log.exception("Decay cycle failed")

        if args.once:
            break

        log.info("Next decay run in %.0fh", POLL_SEC / 3600)
        await asyncio.sleep(POLL_SEC)


if __name__ == "__main__":
    asyncio.run(main())

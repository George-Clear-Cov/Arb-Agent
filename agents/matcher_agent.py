#!/usr/bin/env python3
"""
Market Matching Agent

Continuously compares markets across all platforms using Claude Haiku.
Discovers pairs the fuzzy matcher misses (World Cup titles, cross-platform
naming divergence) and saves them to brain/learned_pairs.json.

Run: python agents/matcher_agent.py
Loop: every 5 minutes
Cost: ~$0.001-0.005/cycle (Haiku, batch 10 pairs/call)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import anthropic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("matcher_agent")

CACHE_DIR   = Path("state")
BRAIN_FILE  = Path("brain/learned_pairs.json")
LOOP_SEC    = 300   # 5 minutes
BATCH_SIZE  = 10    # pairs per Claude call
MAX_CALLS   = 20    # cap per cycle to control cost
_JSON_RE    = re.compile(r'\[[\s\w,]+\]')

# Sports to compare between each platform pair
SPORT_PAIRS: list[tuple[str, str, str]] = [
    # (sport, source_a_cache, source_b_cache)
    ("soccer",     "polymarket_cache",    "kalshi_cache"),
    ("soccer",     "polymarket_cache",    "kalshi_sports_cache"),
    ("baseball",   "polymarket_cache",    "kalshi_sports_cache"),
    ("baseball",   "polymarket_cache",    "kalshi_cache"),
    ("tennis",     "polymarket_cache",    "kalshi_sports_cache"),
    ("mma",        "polymarket_cache",    "kalshi_sports_cache"),
    ("basketball", "polymarket_cache",    "kalshi_sports_cache"),
    ("f1",         "polymarket_cache",    "kalshi_sports_cache"),
    ("prediction", "polymarket_cache",    "predictit_cache"),
    ("prediction", "polymarket_cache",    "gemini_cache"),
    ("prediction", "kalshi_cache",        "predictit_cache"),
    ("prediction", "kalshi_cache",        "gemini_cache"),
    ("prediction", "polymarket_cache",    "hyperliquid_cache"),
]


def load_brain() -> dict[tuple[str, str], bool]:
    """Load existing confirmed/rejected pairs from brain."""
    if not BRAIN_FILE.exists():
        return {}
    try:
        data = json.loads(BRAIN_FILE.read_text())
        now = time.time()
        pairs: dict[tuple[str, str], bool] = {}
        for key_str, entry in data.get("entries", {}).items():
            exp = entry.get("expires_at")
            if exp and now > exp:
                continue
            parts = key_str.split("__", 1)
            if len(parts) == 2:
                pairs[(parts[0], parts[1])] = entry["result"]
        return pairs
    except Exception as exc:
        log.warning("Brain load failed: %s", exc)
        return {}


def save_pairs(new_pairs: dict[tuple[str, str], bool]) -> None:
    """Merge new pairs into brain file."""
    try:
        data: dict = {"version": 1, "entries": {}}
        if BRAIN_FILE.exists():
            data = json.loads(BRAIN_FILE.read_text())
        now = time.time()
        for (k1, k2), result in new_pairs.items():
            ttl_days = 90 if result else 3
            data["entries"][f"{k1}__{k2}"] = {
                "result": result,
                "saved_at": now,
                "expires_at": now + ttl_days * 86400,
                "source": "matcher_agent",
            }
        BRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(BRAIN_FILE) + ".tmp"
        Path(tmp).write_text(json.dumps(data, separators=(",", ":")))
        os.replace(tmp, BRAIN_FILE)
    except Exception as exc:
        log.warning("Brain save failed: %s", exc)


def load_cache(name: str) -> list[dict]:
    """Load markets from a platform cache file."""
    path = CACHE_DIR / f"{name}.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
        return payload.get("markets", [])
    except Exception:
        return []


def ask_llm_sports(client: anthropic.Anthropic,
                   pairs: list[tuple[dict, dict]]) -> list[bool]:
    """Ask Claude whether each pair of sports markets is the same event."""
    lines = []
    for i, (a, b) in enumerate(pairs, 1):
        ao = " / ".join(o["name"] for o in a.get("outcomes", [])[:4])
        bo = " / ".join(o["name"] for o in b.get("outcomes", [])[:4])
        lines.append(f"Pair {i}:")
        lines.append(f"  A: sport={a['sport']} type={a['market_type']} "
                     f"event=\"{a['event_name']}\" outcomes=[{ao}]")
        lines.append(f"  B: sport={b['sport']} type={b['market_type']} "
                     f"event=\"{b['event_name']}\" outcomes=[{bo}]")

    prompt = (
        "You are a sports betting expert. For each pair determine if they refer to "
        "the SAME real-world event with the SAME market type.\n\n"
        "FIFA World Cup 2026 context: team abbreviations in Kalshi tickers "
        "(e.g. KXWCGAME-ARGBRA = Argentina vs Brazil). "
        "Polymarket uses full names. Consider them matching if same teams, same game, same bet type.\n\n"
        "Return TRUE only if: same teams/players, same date, same market type (both h2h OR both totals).\n"
        "Return FALSE if: different opponents, different market types, or uncertain.\n\n"
        + "\n".join(lines)
        + "\n\nRespond with ONLY a JSON array of booleans. Example: [true, false, true]"
    )
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        m = _JSON_RE.search(text)
        if m:
            text = m.group(0)
        parsed = json.loads(text)
        if isinstance(parsed, list) and len(parsed) == len(pairs):
            matched = sum(1 for v in parsed if v)
            log.info("LLM sports: %d/%d pairs matched", matched, len(pairs))
            return [bool(v) for v in parsed]
    except Exception as exc:
        log.warning("LLM sports call failed: %s", exc)
    return [False] * len(pairs)


def ask_llm_prediction(client: anthropic.Anthropic,
                       pairs: list[tuple[dict, dict]]) -> list[bool]:
    """Ask Claude whether each pair of prediction markets is the same question."""
    lines = []
    for i, (a, b) in enumerate(pairs, 1):
        lines.append(f"Pair {i}:")
        lines.append(f"  A: \"{a['event_name']}\"")
        lines.append(f"  B: \"{b['event_name']}\"")

    prompt = (
        "You are a prediction market expert. For each pair determine if both markets "
        "resolve on the EXACT same real-world event (same person, same contest, same year).\n\n"
        "Return TRUE only if a YES on one market definitively implies YES (or NO) on the other.\n"
        "Return FALSE if different people, different election cycles, or uncertain.\n\n"
        + "\n".join(lines)
        + "\n\nRespond with ONLY a JSON array of booleans. Example: [true, false, true]"
    )
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        m = _JSON_RE.search(text)
        if m:
            text = m.group(0)
        parsed = json.loads(text)
        if isinstance(parsed, list) and len(parsed) == len(pairs):
            matched = sum(1 for v in parsed if v)
            log.info("LLM prediction: %d/%d pairs matched", matched, len(pairs))
            return [bool(v) for v in parsed]
    except Exception as exc:
        log.warning("LLM prediction call failed: %s", exc)
    return [False] * len(pairs)


def run_cycle(client: anthropic.Anthropic) -> int:
    """One matching cycle. Returns number of new confirmed pairs found."""
    known = load_brain()
    new_pairs: dict[tuple[str, str], bool] = {}
    calls = 0
    total_new = 0

    for sport, cache_a, cache_b in SPORT_PAIRS:
        if calls >= MAX_CALLS:
            break

        markets_a = [m for m in load_cache(cache_a) if m.get("sport") == sport]
        markets_b = [m for m in load_cache(cache_b) if m.get("sport") == sport]

        if not markets_a or not markets_b:
            continue

        # Find candidate pairs not yet in brain
        candidates: list[tuple[dict, dict]] = []
        for ma in markets_a:
            for mb in markets_b:
                id_a, id_b = ma["market_id"], mb["market_id"]
                key = (id_a, id_b) if id_a <= id_b else (id_b, id_a)
                if key not in known and key not in new_pairs:
                    # Quick title filter: skip if no word overlap (fast pre-filter)
                    words_a = set(ma["event_name"].lower().split())
                    words_b = set(mb["event_name"].lower().split())
                    stopwords = {"will", "the", "a", "to", "in", "of", "vs", "win",
                                 "be", "for", "and", "or", "who", "what", "2026", "2028"}
                    overlap = (words_a - stopwords) & (words_b - stopwords)
                    if not overlap and sport not in ("soccer",):
                        continue
                    candidates.append((ma, mb))
            if len(candidates) >= 200:
                break

        if not candidates:
            continue

        log.info("%s×%s (%s): %d candidate pairs to check",
                 cache_a.replace("_cache", ""), cache_b.replace("_cache", ""), sport, len(candidates))

        for i in range(0, min(len(candidates), BATCH_SIZE * 5), BATCH_SIZE):
            if calls >= MAX_CALLS:
                break
            batch = candidates[i:i + BATCH_SIZE]
            if sport == "prediction":
                results = ask_llm_prediction(client, batch)
            else:
                results = ask_llm_sports(client, batch)
            calls += 1

            for (ma, mb), result in zip(batch, results):
                id_a, id_b = ma["market_id"], mb["market_id"]
                key = (id_a, id_b) if id_a <= id_b else (id_b, id_a)
                new_pairs[key] = result
                if result:
                    total_new += 1
                    log.info("NEW PAIR: \"%s\" <-> \"%s\"",
                             ma["event_name"][:50], mb["event_name"][:50])

    if new_pairs:
        save_pairs(new_pairs)
        log.info("Saved %d new pairs to brain (%d confirmed matches)", len(new_pairs), total_new)

    return total_new


async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    log.info("Matcher agent started — looping every %ds", LOOP_SEC)

    while True:
        try:
            start = time.time()
            found = run_cycle(client)
            elapsed = time.time() - start
            log.info("Cycle done in %.1fs — %d new confirmed pairs", elapsed, found)
        except Exception:
            log.exception("Cycle failed")
        await asyncio.sleep(LOOP_SEC)


if __name__ == "__main__":
    asyncio.run(main())

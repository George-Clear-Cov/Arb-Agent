#!/usr/bin/env python3
"""
Market Matching Agent

Compares markets across platforms using local sentence embeddings (free, no API cost).
Scores pairs by cosine similarity of (name + description) embeddings, with a
post-filter for economics/crypto that requires exact number/threshold matches.

Model: all-MiniLM-L6-v2 (~22MB, loads once, ~90MB RAM)
Run: python agents/matcher_agent.py
Loop: every 30 minutes
Cost: $0
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("matcher_agent")

CACHE_DIR     = Path("state")
BRAIN_FILE    = Path("brain/learned_pairs.json")
PATTERNS_FILE = Path("brain/match_patterns.json")
LOOP_SEC      = 1800   # 30 minutes
BATCH_SIZE    = 50     # pairs per embedding batch (fast locally)
MAX_PAIRS     = 500    # cap candidates per sport-pair to avoid memory spikes

# Similarity thresholds (cosine, 0-1)
# Sports: names tend to be short + specific; 0.72 catches cross-platform naming
# Prediction: descriptions help a lot; 0.78 avoids "same topic, different question"
# Economics/crypto: embedding alone insufficient — use 0.65 + exact number match
THRESHOLDS: dict[str, float] = {
    "soccer":     0.72,
    "baseball":   0.72,
    "basketball": 0.72,
    "football":   0.72,
    "tennis":     0.72,
    "golf":       0.72,
    "mma":        0.72,
    "f1":         0.72,
    "esports":    0.72,
    "boxing":     0.72,
    "prediction": 0.78,
    "economics":  0.65,  # lower — we rely on exact number post-filter
    "crypto":     0.65,  # lower — we rely on exact number post-filter
}

# Sports to compare between each platform pair
SPORT_PAIRS: list[tuple[str, str, str]] = [
    # (sport, source_a_cache, source_b_cache)

    # ── Economics (highest priority — CPI/Fed expire in days) ────────────────
    ("economics",  "kalshi_cache",        "polymarket_cache"),
    ("economics",  "kalshi_cache",        "predictit_cache"),
    ("economics",  "gemini_cache",        "kalshi_cache"),

    # ── Soccer — World Cup live right now ────────────────────────────────────
    ("soccer",     "gemini_cache",        "polymarket_cache"),
    ("soccer",     "gemini_cache",        "kalshi_cache"),
    ("soccer",     "polymarket_cache",    "kalshi_cache"),
    ("soccer",     "polymarket_cache",    "kalshi_sports_cache"),

    # ── Crypto ───────────────────────────────────────────────────────────────
    ("crypto",     "gemini_cache",        "polymarket_cache"),
    ("crypto",     "gemini_cache",        "kalshi_cache"),
    ("crypto",     "hyperliquid_cache",   "polymarket_cache"),
    ("crypto",     "hyperliquid_cache",   "kalshi_cache"),

    # ── Other sports ─────────────────────────────────────────────────────────
    ("golf",       "gemini_cache",        "polymarket_cache"),
    ("baseball",   "polymarket_cache",    "kalshi_sports_cache"),
    ("baseball",   "polymarket_cache",    "kalshi_cache"),
    ("tennis",     "polymarket_cache",    "kalshi_sports_cache"),
    ("mma",        "gemini_cache",        "polymarket_cache"),
    ("mma",        "polymarket_cache",    "kalshi_sports_cache"),
    ("basketball", "polymarket_cache",    "kalshi_sports_cache"),
    ("f1",         "polymarket_cache",    "kalshi_sports_cache"),

    # ── Prediction ───────────────────────────────────────────────────────────
    ("prediction", "polymarket_cache",    "predictit_cache"),
    ("prediction", "kalshi_cache",        "predictit_cache"),
    ("prediction", "kalshi_cache",        "gemini_cache"),
    ("prediction", "polymarket_cache",    "hyperliquid_cache"),
]

# ── Embedding model (loaded once at startup) ─────────────────────────────────

_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading sentence embedding model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("Model loaded.")
    return _model


def market_text(m: dict) -> str:
    """Combine event name + description into a single embedding input."""
    name = m.get("event_name", "").strip()
    desc = m.get("description", "").strip()
    if desc and desc.lower() != name.lower():
        return f"{name}. {desc}"[:512]
    return name[:512]


def score_pairs(pairs: list[tuple[dict, dict]]) -> list[float]:
    """Return cosine similarity scores for each pair using local embeddings."""
    if not pairs:
        return []
    model = get_model()
    import numpy as np

    texts_a = [market_text(a) for a, _ in pairs]
    texts_b = [market_text(b) for _, b in pairs]
    all_texts = texts_a + texts_b
    embs = model.encode(all_texts, batch_size=64, show_progress_bar=False,
                        normalize_embeddings=True)
    embs_a = embs[:len(pairs)]
    embs_b = embs[len(pairs):]
    # Normalized embeddings → dot product = cosine similarity
    scores = (embs_a * embs_b).sum(axis=1).tolist()
    return scores


# ── Number extraction for economics/crypto post-filter ───────────────────────

_NUM_RE = re.compile(r'\b(\d+\.?\d*)\s*(%|bps|basis points|k|m|b)?\b', re.IGNORECASE)

def extract_numbers(text: str) -> set[str]:
    """Extract all numbers+units from text for threshold comparison."""
    nums: set[str] = set()
    for m in _NUM_RE.finditer(text):
        val = m.group(1)
        unit = (m.group(2) or "").lower()
        nums.add(f"{val}{unit}")
    return nums


def numbers_compatible(ma: dict, mb: dict) -> bool:
    """For economics/crypto: require at least one shared number (threshold/date).

    If NEITHER market mentions any numbers, allow the embedding score to decide.
    If both have numbers but share none, they are measuring different thresholds.
    """
    text_a = market_text(ma)
    text_b = market_text(mb)
    nums_a = extract_numbers(text_a)
    nums_b = extract_numbers(text_b)
    if not nums_a or not nums_b:
        return True  # no numbers to compare — fall back to embedding
    return bool(nums_a & nums_b)


# ── Pattern memory ────────────────────────────────────────────────────────────

def load_patterns() -> dict:
    if not PATTERNS_FILE.exists():
        return {"version": 1, "entities": {}}
    try:
        return json.loads(PATTERNS_FILE.read_text())
    except Exception:
        return {"version": 1, "entities": {}}


def save_patterns(data: dict) -> None:
    PATTERNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(PATTERNS_FILE) + ".tmp"
    Path(tmp).write_text(json.dumps(data, separators=(",", ":")))
    os.replace(tmp, PATTERNS_FILE)


def extract_entities(name: str) -> set[str]:
    entities: set[str] = set()
    for word in re.findall(r'\b[A-Z][a-z]{2,}\b|\b\d{4}\b', name):
        entities.add(word.lower())
    return entities


def check_pattern(patterns: dict, sport: str, ma: dict, mb: dict) -> bool:
    ea = extract_entities(ma["event_name"])
    eb = extract_entities(mb["event_name"])
    shared = ea & eb
    if len(shared) < 2:
        return False
    for entry in patterns.get("entities", {}).values():
        if entry.get("sport") != sport:
            continue
        pattern_ents = set(entry.get("keywords", []))
        if len(shared & pattern_ents) >= min(2, len(pattern_ents)):
            return True
    return False


def record_pattern(patterns: dict, sport: str, ma: dict, mb: dict) -> None:
    ea = extract_entities(ma["event_name"])
    eb = extract_entities(mb["event_name"])
    shared = ea & eb
    if len(shared) < 2:
        return
    key = "::".join(sorted(shared))
    entry = patterns.setdefault("entities", {}).setdefault(key, {
        "sport": sport, "keywords": sorted(shared), "match_count": 0, "examples": [],
    })
    entry["match_count"] = entry.get("match_count", 0) + 1
    ex = f"{ma['market_id'][:25]}__{mb['market_id'][:25]}"
    if ex not in entry["examples"]:
        entry["examples"] = (entry["examples"] + [ex])[-10:]
    entry["last_updated"] = time.time()


# ── Brain I/O ─────────────────────────────────────────────────────────────────

def load_brain() -> dict[tuple[str, str], bool]:
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
    try:
        data: dict = {"version": 1, "entries": {}}
        if BRAIN_FILE.exists():
            data = json.loads(BRAIN_FILE.read_text())
        now = time.time()
        for (k1, k2), result in new_pairs.items():
            ttl_days = 90 if result else 3
            data["entries"][f"{k1}__{k2}"] = {
                "result": result,
                "confirmed_at": now,
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
    path = CACHE_DIR / f"{name}.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text()).get("markets", [])
    except Exception:
        return []


# ── Main cycle ────────────────────────────────────────────────────────────────

def run_cycle() -> int:
    known    = load_brain()
    patterns = load_patterns()
    new_pairs: dict[tuple[str, str], bool] = {}
    total_confirmed = 0
    pattern_hits    = 0

    for sport, cache_a, cache_b in SPORT_PAIRS:
        markets_a = [m for m in load_cache(cache_a) if m.get("sport") == sport]
        markets_b = [m for m in load_cache(cache_b) if m.get("sport") == sport]
        if not markets_a or not markets_b:
            continue

        stopwords = {"will", "the", "a", "to", "in", "of", "vs", "win", "be",
                     "for", "and", "or", "who", "what", "2026", "2028", "2027"}

        candidates: list[tuple[dict, dict]] = []
        for ma in markets_a:
            for mb in markets_b:
                id_a, id_b = ma["market_id"], mb["market_id"]
                key = (id_a, id_b) if id_a <= id_b else (id_b, id_a)
                if key in known or key in new_pairs:
                    continue
                # Word overlap pre-filter (skip for economics — CPI vs "consumer price index")
                if sport not in ("economics", "soccer"):
                    words_a = set(ma["event_name"].lower().split()) - stopwords
                    words_b = set(mb["event_name"].lower().split()) - stopwords
                    if not (words_a & words_b):
                        continue
                candidates.append((ma, mb))
            if len(candidates) >= MAX_PAIRS:
                break

        if not candidates:
            continue

        log.info("%s×%s (%s): %d candidates",
                 cache_a.replace("_cache", ""), cache_b.replace("_cache", ""),
                 sport, len(candidates))

        threshold = THRESHOLDS.get(sport, 0.75)
        auto_confirm: list[tuple[dict, dict]] = []
        needs_embed:  list[tuple[dict, dict]] = []

        for pair in candidates:
            if check_pattern(patterns, sport, pair[0], pair[1]):
                auto_confirm.append(pair)
            else:
                needs_embed.append(pair)

        # Pattern-memory auto-confirms (no embedding needed)
        for ma, mb in auto_confirm:
            id_a, id_b = ma["market_id"], mb["market_id"]
            key = (id_a, id_b) if id_a <= id_b else (id_b, id_a)
            new_pairs[key] = True
            total_confirmed += 1
            pattern_hits += 1
            log.info("PATTERN MATCH: \"%s\" <-> \"%s\"",
                     ma["event_name"][:50], mb["event_name"][:50])

        # Embedding similarity scoring
        if needs_embed:
            try:
                scores = score_pairs(needs_embed)
            except Exception as exc:
                log.warning("Embedding failed for %s: %s", sport, exc)
                scores = [0.0] * len(needs_embed)

            matched = 0
            for (ma, mb), score in zip(needs_embed, scores):
                id_a, id_b = ma["market_id"], mb["market_id"]
                key = (id_a, id_b) if id_a <= id_b else (id_b, id_a)

                result = score >= threshold
                # Economics/crypto post-filter: same score isn't enough if thresholds differ
                if result and sport in ("economics", "crypto"):
                    result = numbers_compatible(ma, mb)

                new_pairs[key] = result
                if result:
                    total_confirmed += 1
                    matched += 1
                    record_pattern(patterns, sport, ma, mb)
                    log.info("MATCH (%.2f): \"%s\" <-> \"%s\"",
                             score, ma["event_name"][:50], mb["event_name"][:50])

            log.info("Embedding %s: %d/%d matched (threshold=%.2f)",
                     sport, matched, len(needs_embed), threshold)

    if new_pairs:
        save_pairs(new_pairs)
        save_patterns(patterns)
        log.info("Saved %d pairs (%d confirmed, %d pattern hits)",
                 len(new_pairs), total_confirmed, pattern_hits)

    return total_confirmed


async def main() -> None:
    log.info("Matcher agent started (embedding mode, no API cost) — loop=%ds", LOOP_SEC)
    # Pre-load model at startup so first cycle isn't slow
    get_model()

    while True:
        try:
            start = time.time()
            found = run_cycle()
            elapsed = time.time() - start
            log.info("Cycle done in %.1fs — %d new confirmed pairs", elapsed, found)
        except Exception:
            log.exception("Cycle failed")
        await asyncio.sleep(LOOP_SEC)


if __name__ == "__main__":
    asyncio.run(main())

from __future__ import annotations

"""
LLM-powered market matching via Claude Haiku.

Fuzzy string matching is fast but brittle — it misses arbs when platform
name formats diverge and creates false positives when team names overlap
across unrelated markets.  This module adds a second verification pass for
pairs where the fuzzy score is uncertain (52–85).

Integration: matcher.py pre-computes fuzzy scores for all pairs in a game
bucket, collects the borderline ones, and sends them here as a batch.
Results are cached and persisted to brain/learned_pairs.json across restarts.
TRUE pairs (confirmed matches) persist 90 days; FALSE pairs persist 3 days.

Cost: ~$0.0001–$0.001 per detection cycle (Haiku, cached).
Latency: 1–3 s per batch; only on cache misses.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

_JSON_ARRAY_RE = re.compile(r'\[[\s\w,]+\]')

if TYPE_CHECKING:
    from src.models import Market

log = logging.getLogger(__name__)

_BATCH_SIZE = 10
_MAX_LLM_CALLS = 15       # higher cap — disk cache means cache is warm after first cycle

_BRAIN_FILE = Path(__file__).parent.parent.parent / "brain" / "learned_pairs.json"
_TRUE_TTL_DAYS  = 90   # confirmed matches — stable for the season
_FALSE_TTL_DAYS = 3    # rejected pairs — re-evaluate after a few days


class LLMMarketMatcher:
    def __init__(self, api_key: str) -> None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._cache: dict[tuple[str, str], bool] = {}
        self._calls_this_cycle: int = 0
        self._load_brain()

    # ── Brain persistence ─────────────────────────────────────────────────────

    def _load_brain(self) -> None:
        try:
            with open(_BRAIN_FILE) as f:
                data = json.load(f)
            now = time.time()
            loaded = 0
            for key_str, entry in data.get("entries", {}).items():
                exp = entry.get("expires_at")
                if exp and now > exp:
                    continue
                parts = key_str.split("__", 1)
                if len(parts) == 2:
                    self._cache[(parts[0], parts[1])] = entry["result"]
                    loaded += 1
            log.info("LLM brain: loaded %d cached pairs from disk", loaded)
        except FileNotFoundError:
            log.info("LLM brain: no brain file yet — starting fresh")
        except Exception as exc:
            log.warning("LLM brain load failed: %s", exc)

    def _save_brain(self) -> None:
        try:
            try:
                with open(_BRAIN_FILE) as f:
                    data = json.load(f)
            except Exception:
                data = {"version": 1, "entries": {}}

            now = time.time()
            for (k1, k2), result in self._cache.items():
                key_str = f"{k1}__{k2}"
                ttl_days = _TRUE_TTL_DAYS if result else _FALSE_TTL_DAYS
                data["entries"][key_str] = {
                    "result": result,
                    "saved_at": now,
                    "expires_at": now + ttl_days * 86400,
                }

            _BRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(_BRAIN_FILE) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, separators=(",", ":"))
            os.replace(tmp, _BRAIN_FILE)
        except Exception as exc:
            log.warning("LLM brain save failed: %s", exc)

    def dismiss(self, market_id_a: str, market_id_b: str) -> None:
        """Permanently mark a pair as non-matching (user feedback via /dismiss)."""
        x, y = market_id_a, market_id_b
        key = (x, y) if x <= y else (y, x)
        self._cache[key] = False
        try:
            try:
                with open(_BRAIN_FILE) as f:
                    data = json.load(f)
            except Exception:
                data = {"version": 1, "entries": {}}
            key_str = f"{key[0]}__{key[1]}"
            data["entries"][key_str] = {
                "result": False,
                "user_dismissed": True,
                "saved_at": time.time(),
                "expires_at": None,
            }
            _BRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(_BRAIN_FILE) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, separators=(",", ":"))
            os.replace(tmp, _BRAIN_FILE)
            log.info("Dismissed pair %s × %s (permanent)", market_id_a, market_id_b)
        except Exception as exc:
            log.warning("Brain dismiss write failed: %s", exc)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def reset_cycle(self) -> None:
        """Call once at the start of each detection cycle."""
        self._calls_this_cycle = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def verify_pairs(
        self,
        pairs: list[tuple["Market", "Market"]],
        category: str = "game",
    ) -> list[bool]:
        """
        Return True for each pair that represents the same real-world event.

        category="game"       → sports game (same teams, same date, same market type)
        category="prediction" → prediction market (same question / same resolution event)

        Cached pairs are answered instantly.  Uncached ones go to Claude in
        batches of up to _BATCH_SIZE.  If the per-cycle call budget is
        exhausted, uncached pairs default to False (conservative).
        """
        if not pairs:
            return []

        results: dict[int, bool] = {}
        uncached: list[tuple[int, "Market", "Market"]] = []

        for i, (a, b) in enumerate(pairs):
            ck = _cache_key(a, b)
            if ck in self._cache:
                results[i] = self._cache[ck]
            else:
                uncached.append((i, a, b))

        for start in range(0, len(uncached), _BATCH_SIZE):
            if self._calls_this_cycle >= _MAX_LLM_CALLS:
                log.debug("LLM match: per-cycle call cap reached — skipping %d pairs",
                          len(uncached) - start)
                break
            batch = uncached[start:start + _BATCH_SIZE]
            self._calls_this_cycle += 1
            call_fn = self._call_llm_prediction if category == "prediction" else self._call_llm
            batch_results = call_fn([(a, b) for _, a, b in batch])
            for (i, a, b), is_match in zip(batch, batch_results):
                ck = _cache_key(a, b)
                self._cache[ck] = is_match
                results[i] = is_match
            self._save_brain()

        return [results.get(i, False) for i in range(len(pairs))]

    # ── Internal ───────────────────────────────────────────────────────────────

    def _fmt(self, m: "Market") -> str:
        from src.engine.matcher import _sub_market_tag
        outcomes = " / ".join(o.name for o in m.outcomes[:5])
        home = f" home={m.home_team}" if m.home_team else ""
        away = f" away={m.away_team}" if m.away_team else ""
        sub = _sub_market_tag(m.event_name)
        sub_str = f" sub_market={sub}" if sub != "full" else ""
        return (
            f"[{m.source.value}] event=\"{m.event_name}\""
            f" sport={m.sport} type={m.market_type}{sub_str}{home}{away}"
            f" outcomes=[{outcomes}]"
        )

    def _call_llm(
        self,
        pairs: list[tuple["Market", "Market"]],
    ) -> list[bool]:
        lines: list[str] = []
        for i, (a, b) in enumerate(pairs, 1):
            lines.append(f"Pair {i}:")
            lines.append(f"  A: {self._fmt(a)}")
            lines.append(f"  B: {self._fmt(b)}")

        prompt = (
            "You are a sports betting expert verifying whether two market listings "
            "from different platforms represent the SAME game and market type.\n\n"
            "Kalshi ticker format: KXMLBGAME-26JUN07NYYBAL means MLB game on Jun 7 2026 "
            "between NYY (New York Yankees) and BAL (Baltimore Orioles). "
            "KXNFLGAME = NFL, KXNBAGAME = NBA, KXNHLGAME = NHL, KXMLBTOTAL = MLB totals.\n\n"
            "Team abbreviations: NYY=Yankees, BOS=Red Sox, LAD=Dodgers, SF=Giants, "
            "CHC=Cubs, ATL=Braves, PHI=Phillies, MIL=Brewers, STL=Cardinals, "
            "KC=Royals, MIN=Twins, CLE=Guardians, DET=Tigers, CWS=White Sox, "
            "NYM=Mets, WSH=Nationals, MIA=Marlins, TB=Rays, BAL=Orioles, TOR=Blue Jays, "
            "OAK=Athletics, SEA=Mariners, TEX=Rangers, HOU=Astros, LAA=Angels, "
            "COL=Rockies, ARI/AZ=Diamondbacks, SD=Padres.\n\n"
            "NFL: 'at' and 'vs' mean home/away but both sides are the same game. "
            "'A at B' = same game as 'B vs A'. Ignore home/away ordering differences.\n\n"
            "Return TRUE for each pair if and only if:\n"
            "- Same two teams/players (order doesn't matter)\n"
            "- Same sport and market type (both h2h/moneyline OR both totals OR both spread)\n"
            "- Same game date (if determinable)\n"
            "- SAME sub-market scope: both must be OUTRIGHT MATCH WINNER, or both a "
            "specific set/half/period — NEVER mix them\n\n"
            "Return FALSE if:\n"
            "- Teams/players differ\n"
            "- Sports differ\n"
            "- One is full-match winner and the other is a period/set/half market "
            "(e.g. 'Alcaraz vs Djokovic' outright vs 'Alcaraz wins 2nd set' = FALSE)\n"
            "- A sub_market= field is shown on one but not the other\n"
            "- You're uncertain\n\n"
            "Examples:\n"
            "  TRUE:  'Kansas City Chiefs at Dallas Cowboys [h2h]' vs "
            "'Dallas Cowboys vs Kansas City Chiefs [h2h]'\n"
            "  TRUE:  'KXNFLGAME-26SEP14KCEDAL [h2h]' vs 'Kansas City Chiefs at Dallas Cowboys [h2h]'\n"
            "  FALSE: 'Yankees at Red Sox' vs 'Yankees at Mets' (different opponent)\n"
            "  FALSE: 'NFL: Chiefs at Cowboys [h2h]' vs 'Chiefs at Cowboys [totals]' (different type)\n"
            "  FALSE: 'Alcaraz vs Djokovic [match winner]' vs 'Alcaraz wins second set vs Djokovic' "
            "(full match ≠ set market)\n"
            "  FALSE: 'Swiatek vs Sabalenka' outright vs 'Swiatek wins set 2 vs Sabalenka' "
            "(sub_market=set vs full match)\n\n"
            + "\n".join(lines)
            + "\n\nRespond with ONLY a JSON array of booleans, one per pair.\n"
            "Example for 3 pairs: [true, false, true]"
        )

        try:
            resp = self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            # Extract JSON array even if Claude prefixes with explanation text
            m = _JSON_ARRAY_RE.search(text)
            if m:
                text = m.group(0)
            parsed = json.loads(text)
            if isinstance(parsed, list) and len(parsed) == len(pairs):
                matched = sum(1 for v in parsed if v)
                log.info(
                    "LLM matcher: %d/%d pairs confirmed (%d verified this cycle)",
                    matched, len(pairs), self._calls_this_cycle,
                )
                return [bool(v) for v in parsed]
            log.warning("LLM matcher: unexpected response length %d (expected %d)",
                        len(parsed) if isinstance(parsed, list) else -1, len(pairs))
        except Exception as exc:
            log.warning("LLM matcher call failed: %s", exc)

        return [False] * len(pairs)

    def _call_llm_prediction(
        self,
        pairs: list[tuple["Market", "Market"]],
    ) -> list[bool]:
        """Verify prediction market pairs — do both questions resolve on the same event?"""
        lines: list[str] = []
        for i, (a, b) in enumerate(pairs, 1):
            lines.append(f"Pair {i}:")
            lines.append(f"  A: [{a.source.value}] \"{a.event_name}\"")
            lines.append(f"  B: [{b.source.value}] \"{b.event_name}\"")

        prompt = (
            "You are an expert at prediction market arbitrage. Determine whether each "
            "pair of prediction market titles refers to the EXACT same event so they "
            "can be arbed against each other.\n\n"
            "Platform naming conventions:\n"
            "- Kalshi often uses short tickers like 'PRES-2028-DEM-HARRIS' or "
            "'Will [X] win the [year] [party] nomination?'\n"
            "- Polymarket uses natural English: 'Will [X] become president in [year]?'\n"
            "- PredictIt uses short phrases: '[X] wins [party] [year] nomination'\n"
            "- Different phrasing of the SAME event = TRUE (alias, same resolution)\n\n"
            "Return TRUE if and only if:\n"
            "- Same person/entity as the subject\n"
            "- Same contest (nomination vs nomination, general vs general — NOT mixed)\n"
            "- Same year/election cycle\n"
            "- A 'yes' resolution on one market equals a 'yes' (or definitively 'no') on the other\n\n"
            "Return FALSE if:\n"
            "- Different people or entities\n"
            "- One is a primary/nomination and the other is a general election\n"
            "- Different years or geographic scope\n"
            "- Genuinely ambiguous — err on the side of caution\n\n"
            "Examples:\n"
            "  TRUE:  'Will Harris win 2028 Dem nomination?' vs 'Harris: 2028 Democratic nominee'\n"
            "  TRUE:  'Trump wins 2028 GOP primary' vs 'Will Donald Trump win the 2028 Republican primary?'\n"
            "  FALSE: 'Will Newsom win 2028 Dem nomination?' vs 'Will Harris win 2028 Dem nomination?'\n"
            "  FALSE: 'Trump wins 2028 nomination' vs 'Trump wins 2028 general election'\n"
            "  FALSE: 'Will [X] win CA governor 2026?' vs 'Will [X] win CA governor 2030?'\n\n"
            + "\n".join(lines)
            + "\n\nRespond with ONLY a JSON array of booleans, one per pair.\n"
            "Example for 3 pairs: [true, false, true]"
        )

        try:
            resp = self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            m = _JSON_ARRAY_RE.search(text)
            if m:
                text = m.group(0)
            parsed = json.loads(text)
            if isinstance(parsed, list) and len(parsed) == len(pairs):
                matched = sum(1 for v in parsed if v)
                log.info(
                    "LLM prediction matcher: %d/%d pairs confirmed (%d verified this cycle)",
                    matched, len(pairs), self._calls_this_cycle,
                )
                return [bool(v) for v in parsed]
            log.warning("LLM prediction matcher: unexpected response length %d (expected %d)",
                        len(parsed) if isinstance(parsed, list) else -1, len(pairs))
        except Exception as exc:
            log.warning("LLM prediction matcher call failed: %s", exc)

        return [False] * len(pairs)


# ── Module-level cache key helper (used by matcher.py) ────────────────────────

def _cache_key(a: "Market", b: "Market") -> tuple[str, str]:
    x, y = a.market_id, b.market_id
    return (x, y) if x <= y else (y, x)

from __future__ import annotations

"""
Polymarket CLOB-based feed.

Uses the Central Limit Order Book API instead of Gamma for more accurate
mid-prices and real liquidity data. Polls every 15s (vs 60s for Gamma).

CLOB base: https://clob.polymarket.com
Gamma (metadata): https://gamma-api.polymarket.com
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx

from src.feeds.base import BaseFeed
from src.feeds.feed_cache import CACHE_DIR, load_cache, save_cache
from src.models import BetSide, Market, Outcome, Source

log = logging.getLogger(__name__)

CLOB_BASE = "https://clob.polymarket.com"
GAMMA_BASE = "https://gamma-api.polymarket.com"

# Only include markets with at least this much liquidity ($)
MIN_LIQUIDITY = 50_000.0        # game markets — high bar keeps noise out
MIN_LIQUIDITY_PRED = 5_000.0    # prediction markets — lower bar catches niche topics

_CACHE_FILE = CACHE_DIR / "polymarket_cache.json"
_CACHE_MAX_AGE = 12 * 3600  # 12 hours — refresh overnight at most


class PolymarketCLOBFeed(BaseFeed):
    """
    Fetches Polymarket markets using the CLOB order book for accurate prices.

    Strategy:
    1. Gamma API → rich metadata (question text, close time, liquidity)
    2. CLOB book endpoint → best ask per token for accurate taker prices
    3. Filter by MIN_LIQUIDITY to avoid illiquid markets with slippage
    """

    def __init__(self, clob_url: str = CLOB_BASE, gamma_url: str = GAMMA_BASE):
        self.clob_url = clob_url
        self.gamma_url = gamma_url
        self._client = httpx.AsyncClient(timeout=20.0)
        self._market_cache: dict[str, dict] = {}  # condition_id → clob market metadata
        self._disk_markets = load_cache(_CACHE_FILE, Source.POLYMARKET, _CACHE_MAX_AGE)

    async def fetch(self) -> list[Market]:
        try:
            result = await self._fetch()
            if result:
                self._disk_markets = result
                save_cache(_CACHE_FILE, result)
            return result or self._disk_markets
        except Exception:
            log.exception("PolymarketCLOB fetch failed")
            return self._disk_markets

    async def _fetch(self) -> list[Market]:
        # Two parallel fetch strategies for full coverage:
        # A) Sorted by liquidityNum — top 1,400 markets by total liquidity
        # B) Sorted by volume24hr — top 600 markets by recent activity
        #    Catches short-term markets with high recent volume but depleted
        #    total liquidity (near expiry), which rank below offset 1300 in (A)
        _now = datetime.now(tz=timezone.utc)
        _30d = (_now + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        gamma_batches, volume_batches, game_batches = await asyncio.gather(
            asyncio.gather(
                self._fetch_gamma_markets(tag=None, offset=0),
                self._fetch_gamma_markets(tag=None, offset=100),
                self._fetch_gamma_markets(tag=None, offset=200),
                self._fetch_gamma_markets(tag=None, offset=300, min_liquidity=MIN_LIQUIDITY_PRED),
                self._fetch_gamma_markets(tag=None, offset=400, min_liquidity=MIN_LIQUIDITY_PRED),
                self._fetch_gamma_markets(tag=None, offset=500, min_liquidity=MIN_LIQUIDITY_PRED),
                self._fetch_gamma_markets(tag=None, offset=600, min_liquidity=MIN_LIQUIDITY_PRED),
                self._fetch_gamma_markets(tag=None, offset=700, min_liquidity=MIN_LIQUIDITY_PRED),
                self._fetch_gamma_markets(tag=None, offset=800, min_liquidity=MIN_LIQUIDITY_PRED),
                self._fetch_gamma_markets(tag=None, offset=900, min_liquidity=MIN_LIQUIDITY_PRED),
                self._fetch_gamma_markets(tag=None, offset=1000, min_liquidity=MIN_LIQUIDITY_PRED),
                self._fetch_gamma_markets(tag=None, offset=1100, min_liquidity=MIN_LIQUIDITY_PRED),
                self._fetch_gamma_markets(tag=None, offset=1200, min_liquidity=MIN_LIQUIDITY_PRED),
                self._fetch_gamma_markets(tag=None, offset=1300, min_liquidity=MIN_LIQUIDITY_PRED),
                return_exceptions=True,
            ),
            asyncio.gather(
                # Short-term volume-sorted: any market closing in 30 days, sorted by 24hr volume
                self._fetch_gamma_markets(tag=None, offset=0, min_liquidity=MIN_LIQUIDITY_PRED, order="volume24hr", end_date_max=_30d),
                self._fetch_gamma_markets(tag=None, offset=100, min_liquidity=MIN_LIQUIDITY_PRED, order="volume24hr", end_date_max=_30d),
                self._fetch_gamma_markets(tag=None, offset=200, min_liquidity=MIN_LIQUIDITY_PRED, order="volume24hr", end_date_max=_30d),
                self._fetch_gamma_markets(tag=None, offset=300, min_liquidity=MIN_LIQUIDITY_PRED, order="volume24hr", end_date_max=_30d),
                self._fetch_gamma_markets(tag=None, offset=400, min_liquidity=MIN_LIQUIDITY_PRED, order="volume24hr", end_date_max=_30d),
                self._fetch_gamma_markets(tag=None, offset=500, min_liquidity=MIN_LIQUIDITY_PRED, order="volume24hr", end_date_max=_30d),
                return_exceptions=True,
            ),
            asyncio.gather(
                self._fetch_live_game_events(),
                # Low-liquidity game event fetches for sports with smaller per-game markets
                self._fetch_game_events_by_tag("mlb"),
                self._fetch_game_events_by_tag("nhl"),
                self._fetch_game_events_by_tag("nba"),
                self._fetch_game_events_by_tag("mma"),
                return_exceptions=True,
            ),
        )
        gamma_markets: list[dict] = []
        seen_ids: set[str] = set()
        # game events first so event_title is preserved when same ID appears
        ordered = [*game_batches, *gamma_batches, *volume_batches]
        for batch in ordered:
            if isinstance(batch, list):
                for m in batch:
                    mid = str(m.get("id", ""))
                    if mid and mid not in seen_ids:
                        seen_ids.add(mid)
                        gamma_markets.append(m)

        if not gamma_markets:
            return []

        log.info("Polymarket: %d unique gamma markets before CLOB pricing", len(gamma_markets))

        # Step 2: only CLOB-price markets not already in disk cache.
        # Existing markets get live prices from the WS feed — no need to re-fetch.
        cached_ids = {m.market_id for m in self._disk_markets}
        need_clob = [m for m in gamma_markets
                     if str(m.get("slug") or m.get("id", "")) not in cached_ids]
        skip_clob = [m for m in gamma_markets
                     if str(m.get("slug") or m.get("id", "")) in cached_ids]

        if need_clob:
            log.info("Polymarket: CLOB-pricing %d new markets (%d served from cache)",
                     len(need_clob), len(skip_clob))
        else:
            log.info("Polymarket: all %d markets served from cache (0 new)", len(skip_clob))

        sem = asyncio.Semaphore(20)

        async def _guarded_price(m):
            async with sem:
                return await self._price_market(m)

        new_priced: list[Market] = []
        if need_clob:
            results = await asyncio.gather(
                *[_guarded_price(m) for m in need_clob], return_exceptions=True
            )
            new_priced = [r for r in results if isinstance(r, Market)]

        # Merge: cached markets still in gamma + freshly priced new markets
        cached_by_id = {m.market_id: m for m in self._disk_markets}
        gamma_ids = {str(m.get("slug") or m.get("id", "")) for m in gamma_markets}
        markets: list[Market] = (
            [cached_by_id[mid] for mid in gamma_ids if mid in cached_by_id]
            + new_priced
        )
        sport_counts: dict[str, int] = {}
        for m in markets:
            sport_counts[m.sport] = sport_counts.get(m.sport, 0) + 1
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(sport_counts.items()))
        log.info("Polymarket: %d markets (%s)", len(markets), breakdown)
        return markets

    async def _fetch_gamma_markets(
        self, tag: str | None = None, offset: int = 0,
        min_liquidity: float = MIN_LIQUIDITY,
        order: str = "liquidityNum",
        end_date_max: str | None = None,
    ) -> list[dict]:
        params: dict = {
            "active": "true",
            "closed": "false",
            "limit": 100,
            "liquidity_num_min": min_liquidity,
            "order": order,
            "ascending": "false",
        }
        if tag:
            params["tag_slug"] = tag
        if offset:
            params["offset"] = offset
        if end_date_max:
            params["end_date_max"] = end_date_max
        try:
            resp = await self._client.get(f"{self.gamma_url}/markets", params=params)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.warning("Polymarket gamma markets fetch error (tag=%s, offset=%d): %s",
                        tag, offset, exc)
            return []

    async def _fetch_live_game_events(self, max_days_ahead: int = 8) -> list[dict]:
        """Fetch live and upcoming game events (moneyline / spread / totals).

        Strategy: two parallel fetches —
          1. Sort by volume24hr — live games happening right now dominate.
          2. Sort by liquidityNum — pre-game markets for upcoming games have
             good liquidity but low 24hr volume (game hasn't started yet).
        Combine and deduplicate so both live and pre-game arbs are detected.

        Then filter to:
          1. Game-style titles only ("Team A vs. Team B" format)
          2. endDate within the next `max_days_ahead` days
        """
        base_params = {
            "active": "true",
            "closed": "false",
            "limit": 150,
        }
        try:
            r1, r2 = await asyncio.gather(
                self._client.get(
                    f"{self.gamma_url}/events",
                    params={**base_params, "order": "volume24hr", "ascending": "false"},
                    timeout=15.0,
                ),
                self._client.get(
                    f"{self.gamma_url}/events",
                    params={**base_params, "order": "volume", "ascending": "false"},
                    timeout=15.0,
                ),
                return_exceptions=True,
            )
        except Exception as exc:
            log.warning("Polymarket live game events fetch error: %s", exc)
            return []

        events: list[dict] = []
        seen_event_ids: set[str] = set()
        for resp in (r1, r2):
            if isinstance(resp, Exception):
                log.warning("Polymarket game events fetch error: %s", resp)
                continue
            try:
                resp.raise_for_status()
                for ev in (resp.json() if isinstance(resp.json(), list) else []):
                    eid = str(ev.get("id", ""))
                    if eid and eid not in seen_event_ids:
                        seen_event_ids.add(eid)
                        events.append(ev)
            except Exception as exc:
                log.warning("Polymarket game events parse error: %s", exc)

        now = datetime.now(tz=timezone.utc)
        cutoff = now + timedelta(days=max_days_ahead)
        _VS_PATTERNS = (" vs. ", " vs ", " v. ", " v ", " at ")

        markets: list[dict] = []
        seen: set[str] = set()
        for event in (events if isinstance(events, list) else []):
            title = event.get("title", "")
            # Only game-style events
            if not any(p in title for p in _VS_PATTERNS):
                continue

            # Filter expired and too-far-future markets
            end_date_raw = event.get("endDate") or ""
            if end_date_raw:
                try:
                    event_end = datetime.fromisoformat(
                        end_date_raw.replace("Z", "+00:00")
                    )
                    if event_end <= now or event_end > cutoff:
                        continue
                except ValueError:
                    pass

            event_tags = event.get("tags") or []
            for m in (event.get("markets") or []):
                mid = str(m.get("id", ""))
                if not mid or mid in seen:
                    continue
                # Propagate event-level metadata onto market dicts so that
                # _polymarket_sport can read the tags (nba, nhl, etc.) and
                # _price_market can use the event title as the event_name.
                if not m.get("event_title") or not m.get("tags"):
                    m = dict(m)
                    if not m.get("event_title"):
                        m["event_title"] = title
                    if not m.get("tags") and event_tags:
                        m["tags"] = event_tags
                # Skip resolved / untraded markets
                prices = m.get("outcomePrices")
                if isinstance(prices, str):
                    import json as _json
                    try:
                        prices = _json.loads(prices)
                    except Exception:
                        prices = None
                if prices and isinstance(prices, list) and len(prices) >= 1:
                    try:
                        p0 = float(prices[0])
                        if p0 <= 0.005 or p0 >= 0.995:
                            continue
                    except (ValueError, TypeError):
                        continue
                seen.add(mid)
                markets.append(m)

        log.debug(
            "Polymarket live game events: %d priced markets (cutoff=%s)",
            len(markets), cutoff.strftime("%Y-%m-%d"),
        )
        return markets

    async def _fetch_game_events_by_tag(self, tag: str) -> list[dict]:
        """Fetch game events for a specific sport tag, no liquidity floor.

        MLB/NHL game markets have lower per-game liquidity than prediction
        markets and won't rank in the top-150 by volume. Fetching by tag
        with no min-liquidity ensures today's games are always included.
        """
        try:
            resp = await self._client.get(
                f"{self.gamma_url}/events",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": 50,
                    "order": "volume24hr",
                    "ascending": "false",
                    "tag_slug": tag,
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            events = resp.json() if isinstance(resp.json(), list) else []
        except Exception as exc:
            log.warning("Polymarket game events by tag=%s error: %s", tag, exc)
            return []

        _VS_PATTERNS = (" vs. ", " vs ", " v. ", " v ", " at ")
        _now = datetime.now(tz=timezone.utc)
        markets: list[dict] = []
        seen: set[str] = set()
        for event in events:
            title = event.get("title", "")
            if not any(p in title for p in _VS_PATTERNS):
                continue
            end_date_raw = event.get("endDate") or ""
            if end_date_raw:
                try:
                    event_end = datetime.fromisoformat(end_date_raw.replace("Z", "+00:00"))
                    if event_end <= _now:
                        continue
                except ValueError:
                    pass
            event_tags = event.get("tags") or []
            for m in (event.get("markets") or []):
                mid = str(m.get("id", ""))
                if not mid or mid in seen:
                    continue
                prices = m.get("outcomePrices")
                if isinstance(prices, str):
                    try:
                        import json as _json
                        prices = _json.loads(prices)
                    except Exception:
                        prices = None
                if prices and isinstance(prices, list) and len(prices) >= 1:
                    try:
                        p0 = float(prices[0])
                        if p0 <= 0.005 or p0 >= 0.995:
                            continue
                    except (ValueError, TypeError):
                        continue
                m = dict(m)
                if not m.get("event_title"):
                    m["event_title"] = title
                if not m.get("tags") and event_tags:
                    m["tags"] = event_tags
                seen.add(mid)
                markets.append(m)
        return markets

    async def _price_market(self, m: dict) -> Market | None:
        mid = str(m.get("id", "?"))
        question = (m.get("question") or m.get("title") or "")[:60]

        # Gamma returns outcomes/clobTokenIds/outcomePrices as JSON-encoded strings
        outcomes_data = m.get("outcomes", [])
        if isinstance(outcomes_data, str):
            try:
                outcomes_data = json.loads(outcomes_data)
            except Exception:
                log.debug("Polymarket market %s: failed to parse outcomes JSON", mid)
                return None

        if not outcomes_data:
            log.debug("Polymarket market %s (%s): no outcomes", mid, question)
            return None

        clobTokenIds = m.get("clobTokenIds")
        if isinstance(clobTokenIds, str):
            try:
                clobTokenIds = json.loads(clobTokenIds)
            except Exception:
                clobTokenIds = None

        # Try to get CLOB order book prices for accurate taker costs
        outcome_prices: list[float] = []
        volumes: list[float] = []

        if clobTokenIds and isinstance(clobTokenIds, list):
            try:
                for token_id in clobTokenIds[:len(outcomes_data)]:
                    book = await self._fetch_book(str(token_id))
                    if book and "asks" in book:
                        asks = book.get("asks", [])
                        # Best ask = lowest ask price (taker cost to buy YES)
                        best_ask = min((float(a["price"]) for a in asks), default=None) if asks else None
                        volume = book.get("volume", 0) or 0
                        if best_ask and 0 < best_ask < 1:
                            outcome_prices.append(best_ask)
                            volumes.append(float(volume))
                        else:
                            outcome_prices.append(None)
                            volumes.append(0)
                    else:
                        outcome_prices.append(None)
                        volumes.append(0)
            except Exception as exc:
                log.debug("Polymarket market %s: CLOB fetch error: %s", mid, exc)
                outcome_prices = []

        # Fallback to Gamma outcomePrices if CLOB failed or incomplete
        if not outcome_prices or None in outcome_prices:
            gamma_prices = m.get("outcomePrices", [])
            if isinstance(gamma_prices, str):
                try:
                    gamma_prices = json.loads(gamma_prices)
                except Exception:
                    log.debug("Polymarket market %s: failed to parse outcomePrices JSON", mid)
                    return None
            try:
                outcome_prices = [float(p) for p in gamma_prices]
                volumes = [float(m.get("volume24hr", 0) or 0)] * len(outcome_prices)
            except (ValueError, TypeError) as exc:
                log.debug("Polymarket market %s (%s): price parse error: %s", mid, question, exc)
                return None

        if not outcome_prices or len(outcome_prices) != len(outcomes_data):
            log.debug(
                "Polymarket market %s (%s): price/outcome count mismatch (%d prices, %d outcomes)",
                mid, question, len(outcome_prices), len(outcomes_data),
            )
            return None

        total_vol = sum(volumes) if volumes else float(m.get("volume", 0) or 0)

        # For totals markets, include the line value in outcome names so the
        # arb detector can match "Over 7.5" (Polymarket) with "Over 7.5 runs scored"
        # (Kalshi) and reject mismatched lines ("Over 7.5" vs "Over 8.5").
        _totals_line: str | None = None
        question_for_line = m.get("question") or m.get("title") or ""
        if ": O/U " in question_for_line or ": o/u " in question_for_line.lower():
            import re as _re
            _lm = _re.search(r'[Oo][/][Uu]\s+(\d+\.?\d*)', question_for_line)
            if _lm:
                _totals_line = _lm.group(1)

        token_id_list = clobTokenIds if (clobTokenIds and isinstance(clobTokenIds, list)) else []
        outcomes = []
        outcome_idx = 0
        for name, prob, vol in zip(outcomes_data, outcome_prices, volumes or [0] * len(outcomes_data)):
            if prob <= 0 or prob >= 1:
                log.debug(
                    "Polymarket market %s (%s): skipping outcome '%s' — prob %.4f out of range",
                    mid, question, name, prob,
                )
                outcome_idx += 1
                continue
            dec_price = round(1 / prob, 4)
            clob_token = str(token_id_list[outcome_idx]) if outcome_idx < len(token_id_list) else None
            # Rename "Over"/"Under" to "Over X.5"/"Under X.5" for totals markets
            # so the arb detector can match same-line outcomes across platforms.
            outcome_name = name
            if _totals_line:
                nl = name.lower().strip()
                if nl == "over":
                    outcome_name = f"Over {_totals_line}"
                elif nl == "under":
                    outcome_name = f"Under {_totals_line}"
            outcomes.append(Outcome(
                name=outcome_name,
                price=dec_price,
                implied_prob=prob,
                source=Source.POLYMARKET,
                market_id=str(m.get("slug") or m.get("id", "")),
                bookmaker="Polymarket",
                side=BetSide.BACK,
                available_volume=float(vol) if vol else None,
                is_maker=False,  # assume taker; use limit orders in execution for 0% fee
                token_id=clob_token,
            ))
            outcome_idx += 1

        if len(outcomes) < 2:
            log.debug(
                "Polymarket market %s (%s): dropped — only %d valid outcome(s)",
                mid, question, len(outcomes),
            )
            return None

        question = m.get("question") or m.get("title") or ""
        market_type = _detect_market_type(question)

        # For game-level events (moneyline/spread/total), use the event title
        # as event_name so the matcher can pair e.g. "Phillies vs Padres" moneyline
        # with the same game's "Phillies vs Padres" total on another platform.
        event_title = m.get("event_title", "")
        if market_type in ("h2h", "spreads", "totals") and event_title:
            event_name = event_title
        else:
            event_name = question

        # Extract home/away for game events
        home, away = _extract_teams(event_name) if market_type in ("h2h", "spreads", "totals") else (None, None)

        return Market(
            source=Source.POLYMARKET,
            market_id=str(m.get("slug") or m.get("id", "")),
            sport=_polymarket_sport(m),
            event_name=event_name,
            commence_time=_parse_dt(m.get("endDate")),
            home_team=home,
            away_team=away,
            market_type=market_type,
            outcomes=outcomes,
            total_volume=total_vol,
            raw={"id": m.get("id"), "slug": m.get("slug"), "question": question},
        )

    async def _fetch_book(self, token_id: str) -> dict | None:
        try:
            resp = await self._client.get(
                f"{self.clob_url}/book",
                params={"token_id": token_id},
                timeout=5.0,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

    async def close(self) -> None:
        await self._client.aclose()


_SPORT_KEYWORDS: list[tuple[str, list[str]]] = [
    ("soccer",     ["fifa", "world cup", "soccer", "mls", "epl", "la liga",
                    "bundesliga", "serie a", "ligue 1", "champions league", "euros",
                    "copa america", "euro 2024", "euro 2026", "concacaf"]),
    ("basketball", ["nba", "basketball", "nba finals", "nba western conference",
                    "nba eastern conference"]),
    ("hockey",     ["nhl", "stanley cup", "hockey"]),
    ("football",   ["nfl", "super bowl", "american football", "ncaa football",
                    "college football"]),
    ("cricket",    ["ipl", "indian premier league", "cricket", "test match",
                    "ashes", "bbl", "t20", "one day international", "odi"]),
    ("baseball",   ["mlb", "world series", "baseball", "american league",
                    "national league", "al champion", "nl champion", "cy young",
                    "al mvp", "nl mvp", "hank aaron"]),
    ("tennis",     ["wimbledon", "us open", "french open", "roland garros",
                    "australian open", "atp", "wta", "tennis", "grand slam",
                    "djokovic", "sinner", "alcaraz", "swiatek", "sabalenka"]),
    ("golf",       ["masters", "pga", "golf", "us open golf", "the open championship",
                    "ryder cup"]),
    ("f1",         ["formula 1", "formula one", "f1", "grand prix"]),
    ("mma",        ["ufc", "mma", "bellator"]),
    ("boxing",     ["boxing", "wbc", "wba", "ibf", "wbo"]),
    ("esports",    ["esports", "league of legends", "cs:go", "counter-strike",
                    "valorant", "dota", "overwatch"]),
    ("cricket",    ["cricket", "ipl", "test match", "ashes"]),
    ("economics",  ["cpi", "inflation", "consumer price index", "federal reserve",
                    "fed rate", "interest rate", "rate cut", "rate hike", "fomc",
                    "unemployment", "jobs report", "nonfarm payroll", "gdp",
                    "gross domestic product", "recession", "pce", "tariff",
                    "trade war", "trade deficit", "treasury yield", "10-year",
                    "2-year yield", "basis points", "bps"]),
]


def _detect_market_type(question: str) -> str:
    """Infer market type from Polymarket question text.

    Polymarket game events use consistent question formats:
      "{Team A} vs. {Team B}"            → h2h (moneyline / game winner)
      "Spread: {Team} ({±X.5})"          → spreads
      "{Team A} vs. {Team B}: O/U {X.5}" → totals
      Everything else                     → binary (prediction market)
    """
    q = question.strip()
    ql = q.lower()
    if ql.startswith("spread:") or "run line" in ql or "puck line" in ql:
        return "spreads"
    if ": o/u " in ql or ": over/under " in ql:
        return "totals"
    # Simple "{A} vs. {B}" or "{A} vs {B}" with no extra clauses = moneyline
    vs_patterns = [" vs. ", " vs ", " v. ", " v "]
    if any(p in q for p in vs_patterns):
        # Exclude questions that are futures ("Will X win the World Series?")
        if not q.startswith("Will ") and "?" not in q[:q.find(" vs")] + q[q.find(" vs") + 4:]:
            return "h2h"
    return "binary"


def _extract_teams(event_name: str) -> tuple[str | None, str | None]:
    """Extract (home, away) from 'Team A vs. Team B' or 'Team A at Team B'."""
    for sep in (" vs. ", " vs ", " v. ", " v "):
        if sep in event_name:
            parts = event_name.split(sep, 1)
            # Strip trailing score/stat suffixes like ": O/U 8.5" or " (Game 3)"
            away = parts[0].strip().split(":")[0].strip()
            home = parts[1].strip().split(":")[0].strip()
            return home, away
    return None, None


def _polymarket_sport(m: dict) -> str:
    """Detect sport from market category, tags, or question text.

    Priority order:
    1. Gamma API `category` field (most reliable — set by Polymarket editors)
    2. Tag slugs attached to the market or event
    3. Keyword matching against the question text
    """
    # 1. category field — Gamma API sets this for sports markets
    category = (m.get("category") or "").lower().strip()
    _CATEGORY_MAP = {
        "sports": None,  # too broad — fall through to tags/keywords
        "baseball": "baseball",
        "basketball": "basketball",
        "hockey": "hockey",
        "football": "football",
        "soccer": "soccer",
        "tennis": "tennis",
        "golf": "golf",
        "mma": "mma",
        "boxing": "boxing",
        "cricket": "cricket",
        "esports": "esports",
        "formula 1": "f1",
        "f1": "f1",
        "economics": "economics",
        "economy": "economics",
        "finance": "economics",
        "crypto": "crypto",
    }
    if category and category in _CATEGORY_MAP and _CATEGORY_MAP[category]:
        return _CATEGORY_MAP[category]

    question = (m.get("question") or m.get("title") or "").lower()
    tags = [str(t.get("slug", "")).lower() for t in (m.get("tags") or []) if isinstance(t, dict)]
    text = question + " " + " ".join(tags)
    for sport, keywords in _SPORT_KEYWORDS:
        if any(kw in text for kw in keywords):
            return sport
    return "prediction"


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None

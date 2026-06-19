from __future__ import annotations

"""
Kalshi feed — updated for the v2 API format change.

The bulk /markets endpoint now returns only multi-leg sports parlay markets
(KXMVESPORTSMULTIGAMEEXTENDED / KXMVECROSSCATEGORY) that lack standard
yes/no pricing.  Real binary prediction markets (elections, politics,
economics, etc.) must be fetched via:

  GET /events?status=open&limit=200          → event list
  GET /markets?event_ticker=X&status=open   → markets per event

Prices in the new API are `yes_ask_dollars` / `no_ask_dollars` — string
floats in the 0–1 range (e.g. "0.6500" = 65 cents probability).

Rate-limit strategy: fetch up to MAX_EVENTS events per cycle, then fetch
their markets in parallel batches of BATCH_SIZE to keep total time < ~15s.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx

from src.feeds.kalshi_rate import kalshi_request
from src.models import BetSide, Market, Outcome, Source

log = logging.getLogger(__name__)

# Kalshi Sports events are long-horizon narrative prediction markets
# (player retirements, debut dates, team expansions) — not game outcomes.
# They are matchable with similar questions on Polymarket/PredictIt, so
# we no longer skip them.  The multi-leg parlay markets are skipped by
# the yes_ask_dollars/no_ask_dollars price check in _parse_market instead.
_SKIP_CATEGORIES: set[str] = set()  # nothing skipped at category level

MAX_EVENTS  = 200   # events per page (API maximum)
MAX_PAGES   = 3     # paginate up to 600 events to cover all categories
MAX_DIRECT_PAGES = 8  # /markets direct fetch: 8 pages × 200 = 1,600 short-term markets
MIN_PROB    = 0.03
MAX_PROB    = 0.97


class KalshiFeed:
    def __init__(self, api_key: str, base_url: str) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Token {api_key}"},
            timeout=20.0,
        )

    async def fetch(self) -> list[Market]:
        try:
            events_markets, direct_markets = await asyncio.gather(
                self._fetch(),
                self._fetch_direct_markets(),
            )
            # Merge: direct markets not already seen via events endpoint
            seen = {m.market_id for m in events_markets}
            new_markets = [m for m in direct_markets if m.market_id not in seen]
            log.info("Kalshi direct /markets: %d new markets (deduped from %d)", len(new_markets), len(direct_markets))
            return events_markets + new_markets
        except Exception:
            log.exception("Kalshi fetch failed")
            return []

    async def _fetch(self) -> list[Market]:
        """Fetch all open prediction markets using with_nested_markets=true.

        This collapses N+1 calls (1 events page + N per-event market fetches)
        into MAX_PAGES calls total — eliminating the main source of 429s.
        """
        markets: list[Market] = []
        cursor: str | None = None
        from collections import Counter
        cat_counts: Counter = Counter()

        for page in range(MAX_PAGES):
            params: dict = {
                "status": "open",
                "limit": MAX_EVENTS,
                "with_nested_markets": "true",
            }
            if cursor:
                params["cursor"] = cursor
            try:
                resp = await kalshi_request(self._client.get("/events", params=params))
                if resp.status_code == 429:
                    log.warning("Kalshi /events page %d rate limited", page + 1)
                    break
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                log.error("Kalshi /events page %d error: %s", page + 1, exc)
                break

            events = data.get("events", [])
            for event in events:
                if event.get("category") in _SKIP_CATEGORIES:
                    continue
                cat_counts[event.get("category", "")] += 1
                for raw in event.get("markets", []):
                    m = self._parse_market(raw)
                    if m:
                        markets.append(m)

            cursor = data.get("cursor")
            if not cursor or not events:
                break

        log.info(
            "Kalshi: %d markets across %d pages — %s",
            len(markets),
            page + 1,
            ", ".join(f"{k}={v}" for k, v in cat_counts.most_common(6)),
        )
        return markets

    async def _fetch_direct_markets(self) -> list[Market]:
        """Fetch short-term markets via GET /markets with a 90-day close window.

        The /events endpoint only exposes ~600 events; thousands of additional
        binary markets (economic indicators, earnings, movie scores, political
        events) are only accessible via the direct /markets endpoint filtered
        by close_time. This supplements the events-based fetch without replacing it.
        """
        markets: list[Market] = []
        cursor: str | None = None
        now = datetime.now(timezone.utc)
        end = now + timedelta(days=90)

        for page in range(MAX_DIRECT_PAGES):
            params: dict = {
                "status": "open",
                "limit": MAX_EVENTS,
                "min_close_ts": int(now.timestamp()),
                "max_close_ts": int(end.timestamp()),
            }
            if cursor:
                params["cursor"] = cursor
            try:
                resp = await kalshi_request(self._client.get("/markets", params=params))
                if resp.status_code == 429:
                    log.warning("Kalshi /markets direct page %d rate limited", page + 1)
                    break
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                log.error("Kalshi /markets direct page %d error: %s", page + 1, exc)
                break

            page_markets = data.get("markets", [])
            for raw in page_markets:
                m = self._parse_market(raw)
                if m:
                    markets.append(m)

            cursor = data.get("cursor")
            if not cursor or not page_markets:
                break

        return markets

    def _parse_market(self, m: dict) -> Optional[Market]:
        # New API: prices are string floats in 0-1 range ("0.6500")
        yes_ask_raw = m.get("yes_ask_dollars")
        no_ask_raw  = m.get("no_ask_dollars")

        if yes_ask_raw is None:
            return None
        try:
            yes_prob = float(yes_ask_raw)
        except (TypeError, ValueError):
            return None

        no_prob = (float(no_ask_raw) if no_ask_raw is not None else 1.0 - yes_prob)

        if not (MIN_PROB <= yes_prob <= MAX_PROB):
            return None
        if not (MIN_PROB <= no_prob <= MAX_PROB):
            return None

        ticker: str = m.get("ticker", "")
        title:  str = m.get("title",  ticker)

        # Many Kalshi markets share a generic title (e.g. "Who will win the next
        # presidential election?") with candidate differentiation only in
        # yes_sub_title.  Append it so the matcher can pair candidates correctly
        # across platforms (mirrors PredictIt's "Question: CandidateName" format).
        sub = m.get("yes_sub_title", "").strip()
        if sub and sub.lower() not in title.lower():
            title = f"{title}: {sub}"

        return Market(
            source=Source.KALSHI,
            market_id=ticker,
            sport=_kalshi_sport(ticker, title),
            event_name=title,
            commence_time=_parse_dt(m.get("close_time")),
            home_team=None,
            away_team=None,
            market_type="binary",
            outcomes=[
                Outcome(
                    name="Yes",
                    price=round(1 / yes_prob, 4),
                    implied_prob=round(yes_prob, 6),
                    source=Source.KALSHI,
                    market_id=ticker,
                    bookmaker="Kalshi",
                    side=BetSide.BACK,
                ),
                Outcome(
                    name="No",
                    price=round(1 / no_prob, 4),
                    implied_prob=round(no_prob, 6),
                    source=Source.KALSHI,
                    market_id=ticker,
                    bookmaker="Kalshi",
                    side=BetSide.BACK,
                ),
            ],
            raw=m,
        )

    async def close(self) -> None:
        await self._client.aclose()


_KALSHI_SPORT_PREFIXES: list[tuple[str, list[str]]] = [
    ("baseball",   ["KXMLB", "KXWORLDSERIES"]),
    ("basketball", ["KXNBA", "KXSONICS", "KXNBASEATTLE", "KXNBATEAM", "KXSPORTSOWNERLBJ"]),
    ("hockey",     ["KXNHL", "KXCANADACUP", "KXSTANLEY"]),
    ("football",   ["KXNFL", "KXSUPERBOWL", "KXNCAA"]),
    ("soccer",     ["KXSOCCER", "KXWORLDCUP", "KXEURO", "KXCL", "KXEPL", "KXMLS",
                    "KXLALIGA", "KXSERIEА", "KXBUNDESLIGA", "KXCOPAAMERICA"]),
    ("tennis",     ["KXATP", "KXWTA", "KXWIMBLEDON", "KXFRENCHOPEN", "KXUSOPEN"]),
    ("golf",       ["KXPGA", "KXMASTERS", "KXGOLF"]),
    ("f1",         ["KXF1", "KXFORMULA"]),
    ("mma",        ["KXUFC", "KXMMA"]),
    ("boxing",     ["KXBOXING"]),
    ("esports",    ["KXESPORTS", "KXLOL", "KXCSGO", "KXVALORANT", "KXDOTA"]),
]

_KALSHI_SPORT_KEYWORDS: list[tuple[str, list[str]]] = [
    ("baseball",   ["mlb", "world series", "baseball", "cy young", "al mvp", "nl mvp",
                    "american league champion", "national league champion", "debut date"]),
    ("basketball", ["nba", "basketball", "lebron", "steph curry",
                    "kevin durant", "kawhi", "kyrie irving", "draymond"]),
    ("hockey",     ["nhl", "stanley cup", "hockey"]),
    ("football",   ["nfl", "super bowl", "football"]),
    ("soccer",     ["world cup", "fifa", "soccer", "epl", "premier league", "la liga",
                    "bundesliga", "serie a", "ligue 1", "champions league", "europa league",
                    "copa america", "euros", "euro 2026", "concacaf", "mls",
                    "nations league"]),
    ("tennis",     ["wimbledon", "french open", "australian open", "us open tennis",
                    "grand slam", "djokovic", "sinner", "alcaraz", "swiatek"]),
    ("golf",       ["pga", "masters", "golf", "us open golf", "the open championship",
                    "ryder cup"]),
    ("f1",         ["formula 1", "formula one", "grand prix", "verstappen"]),
    ("mma",        ["ufc", "mma", "bellator"]),
    ("boxing",     ["boxing", "wbc", "wba", "ibf", "wbo"]),
    ("esports",    ["esports", "league of legends", "cs:go", "counter-strike",
                    "valorant", "dota", "overwatch"]),
]


def _kalshi_sport(ticker: str, title: str) -> str:
    """Detect sport category from Kalshi ticker prefix and market title."""
    tu = ticker.upper()
    for sport, prefixes in _KALSHI_SPORT_PREFIXES:
        if any(tu.startswith(p) for p in prefixes):
            return sport
    tl = title.lower()
    for sport, keywords in _KALSHI_SPORT_KEYWORDS:
        if any(kw and kw in tl for kw in keywords):
            return sport
    return "prediction"


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None

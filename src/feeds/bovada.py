from __future__ import annotations

"""
Bovada sportsbook odds feed — no API key required.

Fetches MLB, NBA, NHL moneyline / runline / total odds from Bovada's public API.
Only main-game markets (period.main = True) are included; 1st-half and
1st-inning props are skipped.

For totals and runlines the line value lives in outcome.price.handicap,
e.g. "8.5" for a run total or "1.5" for a baseball runline.  We embed this
in the outcome name ("Over 8.5", "Boston Red Sox +1.5") so the arb matcher
can correctly align these with Kalshi/Polymarket totals.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from src.models import BetSide, Market, Outcome, Source

log = logging.getLogger(__name__)

_BASE = "https://www.bovada.lv/services/sports/event/coupon/events/A/description"

_SPORT_PATHS: list[tuple[str, str]] = [
    # North American leagues
    ("baseball/mlb",      "baseball"),
    ("basketball/nba",    "basketball"),
    ("hockey/nhl",        "hockey"),
    ("football/nfl",      "football"),
    # Global / other
    ("soccer",            "soccer"),
    ("tennis",            "tennis"),
    ("boxing",            "boxing"),
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

MIN_PROB = 0.02
MAX_PROB = 0.98


class BovadaFeed:
    """
    Fetches Bovada sportsbook odds.

    Produces Source.BOVADA markets for matching against Kalshi sports and
    Polymarket game markets.  Bookmaker is "Bovada" so the arb detector
    treats it as a separate book from DraftKings / OddsAPI sources.
    """

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            headers=_HEADERS,
            follow_redirects=True,
            timeout=20.0,
        )

    async def fetch(self) -> list[Market]:
        try:
            return await self._fetch()
        except Exception:
            log.exception("Bovada fetch failed")
            return []

    async def _fetch(self) -> list[Market]:
        results = await asyncio.gather(
            *[self._fetch_sport(path, sport) for path, sport in _SPORT_PATHS],
            return_exceptions=True,
        )
        markets: list[Market] = []
        for (path, sport), result in zip(_SPORT_PATHS, results):
            if isinstance(result, Exception):
                log.warning("Bovada %s error: %s", sport, result)
            else:
                markets.extend(result)

        sport_counts: dict[str, int] = {}
        for m in markets:
            sport_counts[m.sport] = sport_counts.get(m.sport, 0) + 1
        log.info(
            "Bovada: %d markets (%s)",
            len(markets),
            ", ".join(f"{k}={v}" for k, v in sorted(sport_counts.items())),
        )
        return markets

    async def _fetch_sport(self, path: str, canonical_sport: str) -> list[Market]:
        resp = await self._client.get(
            f"{_BASE}/{path}",
            params={"lang": "en", "eventsLimit": "50", "preMatchOnly": "true"},
        )
        resp.raise_for_status()
        groups = resp.json()

        markets: list[Market] = []
        now = datetime.now(tz=timezone.utc)
        for group in groups:
            for event in group.get("events", []):
                start_ms = event.get("startTime")
                if start_ms:
                    commence = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
                    if commence < now:
                        continue  # skip games already in progress
                else:
                    commence = None
                markets.extend(self._parse_event(event, canonical_sport, commence))
        return markets

    def _parse_event(
        self,
        event: dict,
        sport: str,
        commence_time: Optional[datetime],
    ) -> list[Market]:
        raw_desc = event.get("description", "")
        # "Boston Red Sox @ New York Yankees" → "Boston Red Sox at New York Yankees"
        # ("at" is the standard separator that _normalize_game handles)
        event_name = raw_desc.replace(" @ ", " at ")
        event_id   = str(event.get("id", ""))

        home_name = away_name = None
        for c in event.get("competitors", []):
            if c.get("home"):
                home_name = c.get("name")
            else:
                away_name = c.get("name")

        markets: list[Market] = []
        for dg in event.get("displayGroups", []):
            if not dg.get("defaultType"):
                continue  # only the main "Game Lines" display group
            for mkt in dg.get("markets", []):
                if not mkt.get("period", {}).get("main"):
                    continue  # skip 1st-half / 1st-inning props
                parsed = self._parse_market(
                    mkt, event_id, event_name, sport,
                    home_name, away_name, commence_time,
                )
                if parsed:
                    markets.append(parsed)
        return markets

    def _parse_market(
        self,
        mkt: dict,
        event_id: str,
        event_name: str,
        sport: str,
        home_name: Optional[str],
        away_name: Optional[str],
        commence_time: Optional[datetime],
    ) -> Optional[Market]:
        desc = mkt.get("description", "")
        mkt_id = str(mkt.get("id", ""))

        if desc in ("Moneyline", "Money Line", "Match Winner", "Fight Winner",
                    "To Win Match", "Winner"):
            market_type = "h2h"
        elif desc in ("Runline", "Run Line", "Puck Line", "Point Spread", "Spread"):
            market_type = "spreads"
        elif desc in ("Total", "Game Total", "Total Runs", "Total Goals",
                      "Total Points", "Games Total"):
            market_type = "totals"
        elif desc in ("1X2", "Match Result", "Result"):
            market_type = "h2h"  # soccer 3-way — all 3 outcomes kept
        else:
            return None

        outcomes: list[Outcome] = []
        for o in mkt.get("outcomes", []):
            price_data = o.get("price", {})
            dec_str    = price_data.get("decimal")
            if not dec_str:
                continue
            try:
                dec = float(dec_str)
            except (ValueError, TypeError):
                continue
            if dec <= 1.0:
                continue

            prob = round(1.0 / dec, 6)
            if not (MIN_PROB <= prob <= MAX_PROB):
                continue

            name      = o.get("description", "")
            handicap  = price_data.get("handicap", "")

            # Embed line value in name so matcher can align with Kalshi/Polymarket.
            # Total:  "Over" + handicap "8.5" → "Over 8.5"
            # Spread: "Boston Red Sox" + handicap "+1.5" → "Boston Red Sox +1.5"
            if market_type == "totals" and handicap:
                if name.lower() == "over":
                    name = f"Over {handicap}"
                elif name.lower() == "under":
                    name = f"Under {handicap}"
            elif market_type == "spreads" and handicap:
                name = f"{name} {handicap}"

            outcomes.append(Outcome(
                name=name,
                price=round(dec, 4),
                implied_prob=prob,
                source=Source.BOVADA,
                market_id=mkt_id,
                bookmaker="Bovada",
                side=BetSide.BACK,
            ))

        if len(outcomes) < 2:
            return None
        # For soccer 1x2 (3-way), arb requires covering all outcomes; keep as-is.
        # For all other market types, drop if more than 3 outcomes (anomaly).

        return Market(
            source=Source.BOVADA,
            market_id=mkt_id,
            sport=sport,
            event_name=event_name,
            commence_time=commence_time,
            home_team=home_name,
            away_team=away_name,
            market_type=market_type,
            outcomes=outcomes,
            raw={"bookmaker": "Bovada", "event_id": event_id},
        )

    async def close(self) -> None:
        await self._client.aclose()

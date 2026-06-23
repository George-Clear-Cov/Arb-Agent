from __future__ import annotations

"""
ProphetX sports exchange feed via PolyRouter.

ProphetX has no self-serve public API — market data is accessed via PolyRouter,
which returns ProphetX prices normalized to implied probability alongside other
platforms in a single call.

Auth: set POLYROUTER_API_KEY env var (free tier at polyrouter.io)
Base: https://api-v2.polyrouter.io
Rate: 100 req/min on free tier
Coverage: NFL, NBA, MLB, NHL, UFC, college sports
"""
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import httpx

from src.models import BetSide, Market, Outcome, Source

log = logging.getLogger(__name__)

BASE = "https://api-v2.polyrouter.io"
LEAGUES = ["nfl", "nba", "mlb", "nhl", "ufc"]

# Mapping PolyRouter league → sport string used in bucket routing
_LEAGUE_SPORT = {
    "nfl": "football",
    "nba": "basketball",
    "mlb": "baseball",
    "nhl": "hockey",
    "ufc": "mma",
    "pga": "golf",
}

MIN_PROB = 0.02
MAX_PROB = 0.98
MIN_LIQUIDITY = 100.0   # USD minimum


class ProphetXFeed:
    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key or os.environ.get("POLYROUTER_API_KEY", "")
        self._client = httpx.AsyncClient(
            base_url=BASE,
            headers={"X-API-Key": self._api_key},
            timeout=20.0,
        )

    async def fetch(self) -> list[Market]:
        if not self._api_key:
            log.warning("ProphetX: no POLYROUTER_API_KEY — skipping")
            return []
        try:
            return await self._fetch()
        except Exception:
            log.exception("ProphetX/PolyRouter fetch failed")
            return []

    async def _fetch(self) -> list[Market]:
        markets: list[Market] = []
        for league in LEAGUES:
            try:
                resp = await self._client.get(
                    "/sports/games",
                    params={"league": league, "status": "not_started", "limit": 100},
                )
                resp.raise_for_status()
                body = resp.json()
                for game in body.get("data", {}).get("games", []):
                    parsed = self._parse_game(game, league)
                    if parsed:
                        markets.append(parsed)
            except Exception as exc:
                log.debug("ProphetX/PolyRouter error for %s: %s", league, exc)

        log.info("ProphetX: %d markets", len(markets))
        return markets

    def _parse_game(self, game: dict, league: str) -> Market | None:
        px_market = None
        for mkt in game.get("markets", []):
            if mkt.get("platform") == "prophetx" and mkt.get("market_type") in ("binary", "moneyline"):
                px_market = mkt
                break

        if not px_market:
            return None

        outcomes_raw = px_market.get("outcomes", [])
        if len(outcomes_raw) < 2:
            return None

        # current_prices is keyed by outcome_id, or fall back to outcomes list price field
        current_prices = px_market.get("current_prices", {})
        liquidity = px_market.get("liquidity", 0) or 0

        if liquidity < MIN_LIQUIDITY:
            return None

        outcomes: list[Outcome] = []
        market_id = str(game.get("id", ""))
        vol24h = px_market.get("volume_24h")

        for o in outcomes_raw:
            oid = o.get("id", "")
            price_info = current_prices.get(str(oid), {})
            # PolyRouter normalizes to implied probability
            ask_prob = price_info.get("ask") or price_info.get("price") or o.get("price")
            if ask_prob is None:
                continue
            ask_prob = float(ask_prob)
            if not (MIN_PROB < ask_prob < MAX_PROB):
                continue

            decimal_price = round(1 / ask_prob, 4)
            outcomes.append(Outcome(
                name=o.get("name", str(oid)),
                price=decimal_price,
                implied_prob=ask_prob,
                source=Source.PROPHETX,
                market_id=market_id,
                bookmaker="ProphetX",
                side=BetSide.BACK,
                available_volume=float(vol24h) if vol24h else None,
            ))

        if len(outcomes) < 2:
            return None

        sport = _LEAGUE_SPORT.get(league, "sports")
        teams = game.get("teams", [])
        home = teams[0] if teams else None
        away = teams[1] if len(teams) > 1 else None
        title = game.get("title") or (f"{away} at {home}" if away and home else "")

        return Market(
            source=Source.PROPHETX,
            market_id=market_id,
            sport=sport,
            event_name=title,
            commence_time=_parse_dt(game.get("scheduled_at")),
            home_team=home,
            away_team=away,
            market_type="binary",
            outcomes=outcomes,
            total_volume=float(vol24h) if vol24h else None,
            raw={"league": league, "game_id": market_id, "platform_id": px_market.get("platform_id")},
        )

    async def close(self) -> None:
        await self._client.aclose()


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None

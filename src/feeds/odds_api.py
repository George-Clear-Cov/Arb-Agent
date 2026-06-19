from __future__ import annotations

"""
Odds API feed — fetches sportsbook odds and structures them for arb detection.

Key design decisions:
- One Market per (event, market_type, bookmaker) so the arb detector can compare
  the same outcome across different bookmakers.
- Sport slugs are normalized to canonical names (baseball_mlb → baseball) so the
  matcher can pair OddsAPI markets with Betfair markets on the same sport.
- Event names use "Home vs Away" format matching Betfair/Polymarket conventions.
"""
import logging
from datetime import datetime
from typing import Optional

import httpx

from src.feeds.base import BaseFeed
from src.models import BetSide, Market, Outcome, Source

log = logging.getLogger(__name__)

# Map Odds API sport slugs → canonical sport names used throughout the system
_SPORT_SLUG_MAP: dict[str, str] = {
    "baseball_mlb":              "baseball",
    "basketball_nba":            "basketball",
    "basketball_ncaab":          "basketball",
    "icehockey_nhl":             "hockey",
    "americanfootball_nfl":      "football",
    "americanfootball_ncaaf":    "football",
    "soccer_epl":                "soccer",
    "soccer_usa_mls":            "soccer",
    "soccer_uefa_champs_league": "soccer",
    "soccer_spain_la_liga":      "soccer",
    "soccer_germany_bundesliga": "soccer",
    "soccer_italy_serie_a":      "soccer",
    "soccer_france_ligue_one":   "soccer",
    "tennis_atp_french_open":    "tennis",
    "tennis_wta_french_open":    "tennis",
    "tennis_atp_wimbledon":      "tennis",
    "tennis_wta_wimbledon":      "tennis",
    "tennis_atp_us_open":        "tennis",
    "tennis_wta_us_open":        "tennis",
    "tennis_atp_australian_open":"tennis",
    "tennis_wta_australian_open":"tennis",
    "mma_mixed_martial_arts":    "mma",
    "boxing_boxing":             "boxing",
    "golf_masters_tournament_winner": "golf",
    "golf_pga_championship":     "golf",
    "golf_us_open":              "golf",
    "golf_the_open_championship":"golf",
    "cricket_ipl":               "cricket",
    "cricket_test_match":        "cricket",
}


def _normalize_sport(slug: str) -> str:
    return _SPORT_SLUG_MAP.get(slug, slug.split("_")[0] if "_" in slug else slug)


class OddsApiFeed(BaseFeed):
    def __init__(self, api_key: str, base_url: str, sports: list[str],
                 markets: list[str], regions: list[str]):
        self.api_key = api_key
        self.base_url = base_url
        self.sports = sports
        self.markets = markets
        self.regions = regions
        self._client = httpx.AsyncClient(timeout=20.0)

    async def fetch(self) -> list[Market]:
        results: list[Market] = []
        for sport_slug in self.sports:
            try:
                markets = await self._fetch_sport(sport_slug)
                results.extend(markets)
                log.debug("OddsAPI %s: %d markets", sport_slug, len(markets))
            except Exception as exc:
                log.warning("OddsAPI fetch failed for sport=%s: %s", sport_slug, exc)
        log.info("OddsAPI: %d total markets across %d sports", len(results), len(self.sports))
        return results

    async def _fetch_sport(self, sport_slug: str) -> list[Market]:
        url = f"{self.base_url}/sports/{sport_slug}/odds"
        resp = await self._client.get(url, params={
            "apiKey": self.api_key,
            "regions": ",".join(self.regions),
            "markets": ",".join(self.markets),
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        })
        if resp.status_code == 422:
            log.debug("OddsAPI: sport %s not available (422)", sport_slug)
            return []
        resp.raise_for_status()
        events = resp.json()

        sport = _normalize_sport(sport_slug)
        markets: list[Market] = []

        for event in events:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            # Odds API uses "Home vs Away" ordering in their event name
            event_name = f"{home} vs {away}"
            commence_time = _parse_dt(event.get("commence_time"))
            event_id = event["id"]

            # Build one Market per (bookmaker, market_type) so every bookmaker's
            # prices are an independent market the arb detector can compare.
            for bookmaker in event.get("bookmakers", []):
                book_key   = bookmaker["key"]
                book_title = bookmaker["title"]

                for mkt in bookmaker.get("markets", []):
                    mkt_key = mkt["key"]

                    outcomes = []
                    for o in mkt.get("outcomes", []):
                        price = float(o.get("price", 0))
                        if price <= 1.0:
                            continue
                        # For spreads/totals include the point value in the name
                        # so "Mariners -1.5" and "Mariners +1.5" are distinct outcomes
                        # and won't be falsely matched as the same bet.
                        name = o["name"]
                        point = o.get("point")
                        if point is not None and mkt_key in ("spreads", "totals",
                                                              "alternate_spreads",
                                                              "alternate_totals"):
                            sign = "+" if float(point) > 0 else ""
                            name = f"{name} ({sign}{point})"

                        outcomes.append(Outcome(
                            name=name,
                            price=round(price, 4),
                            implied_prob=round(1 / price, 6),
                            source=Source.ODDS_API,
                            market_id=f"{event_id}_{mkt_key}_{book_key}",
                            bookmaker=book_title,
                            side=BetSide.BACK,
                        ))

                    if len(outcomes) < 2:
                        continue

                    markets.append(Market(
                        source=Source.ODDS_API,
                        market_id=f"{event_id}_{mkt_key}_{book_key}",
                        sport=sport,
                        event_name=event_name,
                        commence_time=commence_time,
                        home_team=home,
                        away_team=away,
                        market_type=_normalize_market_type(mkt_key),
                        outcomes=outcomes,
                        raw={
                            "event_id": event_id,
                            "bookmaker": book_key,
                            "sport_slug": sport_slug,
                        },
                    ))

        return markets

    async def fetch_props(
        self,
        prop_markets: list[str],
        max_events_per_sport: int = 5,
    ) -> list[Market]:
        """Fetch player prop markets for today's upcoming events.

        Uses the per-event endpoint /sports/{sport}/events/{id}/odds so we can
        request prop market types that aren't available in the bulk odds call.
        Fetches up to max_events_per_sport events per sport to stay within quota.
        """
        results: list[Market] = []
        for sport_slug in self.sports:
            try:
                events = await self._fetch_events(sport_slug, max_events_per_sport)
                for event in events:
                    try:
                        markets = await self._fetch_event_props(
                            sport_slug, event, prop_markets
                        )
                        results.extend(markets)
                    except Exception as exc:
                        log.debug("Props fetch failed for event %s: %s",
                                  event.get("id"), exc)
            except Exception as exc:
                log.warning("Props event list failed for %s: %s", sport_slug, exc)
        log.info(
            "OddsAPI props: %d markets across %d sports (%s)",
            len(results), len(self.sports), ", ".join(prop_markets),
        )
        return results

    async def _fetch_events(self, sport_slug: str, limit: int) -> list[dict]:
        """Get upcoming events for a sport (sorted by commence_time ascending)."""
        url = f"{self.base_url}/sports/{sport_slug}/events"
        resp = await self._client.get(url, params={
            "apiKey": self.api_key,
            "dateFormat": "iso",
        })
        if resp.status_code in (422, 404):
            return []
        resp.raise_for_status()
        events = resp.json()
        # Sort by start time, keep only the next N games
        events.sort(key=lambda e: e.get("commence_time", ""))
        return events[:limit]

    async def _fetch_event_props(
        self,
        sport_slug: str,
        event: dict,
        prop_markets: list[str],
    ) -> list[Market]:
        """Fetch prop odds for a single event."""
        event_id = event["id"]
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        event_name = f"{home} vs {away}"
        commence_time = _parse_dt(event.get("commence_time"))
        sport = _normalize_sport(sport_slug)

        url = f"{self.base_url}/sports/{sport_slug}/events/{event_id}/odds"
        resp = await self._client.get(url, params={
            "apiKey": self.api_key,
            "regions": ",".join(self.regions),
            "markets": ",".join(prop_markets),
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        })
        if resp.status_code in (422, 404):
            return []
        resp.raise_for_status()
        data = resp.json()

        markets: list[Market] = []
        for bookmaker in data.get("bookmakers", []):
            book_key   = bookmaker["key"]
            book_title = bookmaker["title"]

            for mkt in bookmaker.get("markets", []):
                mkt_key = mkt["key"]
                outcomes = []
                for o in mkt.get("outcomes", []):
                    price = float(o.get("price", 0))
                    if price <= 1.0:
                        continue
                    player = o.get("description", "").strip()
                    side   = o.get("name", "").strip()   # "Over" / "Under"
                    point  = o.get("point")
                    if not player or not side:
                        continue
                    # Composite name so "Over 4.5" for different players don't merge
                    if point is not None:
                        outcome_name = f"{player} {side} {point}"
                    else:
                        outcome_name = f"{player} {side}"

                    outcomes.append(Outcome(
                        name=outcome_name,
                        price=round(price, 4),
                        implied_prob=round(1 / price, 6),
                        source=Source.ODDS_API,
                        market_id=f"{event_id}_{mkt_key}_{book_key}",
                        bookmaker=book_title,
                        side=BetSide.BACK,
                    ))

                if len(outcomes) < 2:
                    continue

                markets.append(Market(
                    source=Source.ODDS_API,
                    market_id=f"{event_id}_{mkt_key}_{book_key}",
                    sport=sport,
                    event_name=event_name,
                    commence_time=commence_time,
                    home_team=home,
                    away_team=away,
                    market_type=_normalize_prop_type(mkt_key),
                    outcomes=outcomes,
                    raw={
                        "event_id": event_id,
                        "bookmaker": book_key,
                        "sport_slug": sport_slug,
                        "is_prop": True,
                    },
                ))

        return markets

    async def close(self) -> None:
        await self._client.aclose()


def _normalize_market_type(key: str) -> str:
    """Map OddsAPI market keys to canonical types."""
    _MAP = {
        "h2h":              "h2h",
        # h2h_lay is Betfair exchange data — keep separate so it's not
        # incorrectly paired with back markets in the arb detector.
        "h2h_lay":          "h2h_lay",
        "spreads":          "spreads",
        "totals":           "totals",
        "outrights":        "outright",
        "alternate_spreads":"spreads",
        "alternate_totals": "totals",
    }
    return _MAP.get(key, key)


# Player prop market type → readable label
_PROP_TYPE_MAP: dict[str, str] = {
    # MLB
    "pitcher_strikeouts":          "pitcher_strikeouts",
    "pitcher_record_a_win":        "pitcher_win",
    "batter_home_runs":            "batter_home_runs",
    "batter_hits":                 "batter_hits",
    "batter_total_bases":          "batter_total_bases",
    "batter_rbis":                 "batter_rbis",
    "batter_runs_scored":          "batter_runs_scored",
    "batter_stolen_bases":         "batter_stolen_bases",
    "batter_walks":                "batter_walks",
    # NBA
    "player_points":               "player_points",
    "player_rebounds":             "player_rebounds",
    "player_assists":              "player_assists",
    "player_threes":               "player_threes",
    "player_blocks":               "player_blocks",
    "player_steals":               "player_steals",
    "player_points_rebounds_assists": "player_pra",
    "player_points_rebounds":      "player_pr",
    "player_points_assists":       "player_pa",
    # Tennis
    "player_aces":                 "player_aces",
    "player_double_faults":        "player_double_faults",
}


def _normalize_prop_type(key: str) -> str:
    return _PROP_TYPE_MAP.get(key, key)


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None

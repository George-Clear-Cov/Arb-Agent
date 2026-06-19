from __future__ import annotations

"""
ESPN/DraftKings free odds feed — no API key required.

ESPN's public scoreboard API returns DraftKings moneyline, spread, and totals
odds for all major US sports.  No auth, no quota, refreshes on every poll.

Endpoint: https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard

Markets produced:
  h2h     — moneyline (team A wins / team B wins)
  spreads — point spread / runline
  totals  — over/under

All prices are sourced from DraftKings via ESPN's betting widget partnership.
Bookmaker is set to "DraftKings" so the matcher treats it as a sportsbook
and compares against Kalshi/Polymarket game markets.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from src.models import BetSide, Market, Outcome, Source

log = logging.getLogger(__name__)

# ESPN scoreboard endpoints for each sport/league
# format: (sport_slug, league_slug, canonical_sport_name)
_SPORT_ENDPOINTS: list[tuple[str, str, str]] = [
    ("baseball",    "mlb",              "baseball"),
    ("basketball",  "nba",              "basketball"),
    ("hockey",      "nhl",              "hockey"),
    ("football",    "nfl",              "football"),
    ("football",    "college-football", "football"),  # NCAAF — confirmed 51 events
    ("soccer",      "usa.1",            "soccer"),    # MLS
]

# Game status values that indicate a game is over or already started.
# We skip these — stale odds create phantom arbs.
_SKIP_STATUSES = frozenset({
    "STATUS_FINAL",
    "STATUS_FINAL_OT",
    "STATUS_FINAL_PEN",
    "STATUS_IN_PROGRESS",
    "STATUS_HALFTIME",
    "STATUS_END_PERIOD",
    "STATUS_DELAYED",
    "STATUS_POSTPONED",
    "STATUS_CANCELED",
    "STATUS_SUSPENDED",
})

_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

MIN_PROB = 0.02
MAX_PROB = 0.98


def _american_to_decimal(american_str: str) -> Optional[float]:
    """Convert American odds string ('+136', '-162') to decimal odds."""
    try:
        val = int(str(american_str).replace("+", "").strip())
    except (TypeError, ValueError):
        return None
    if val > 0:
        return round(1 + val / 100, 4)
    elif val < 0:
        return round(1 + 100 / abs(val), 4)
    return None


def _decimal_to_prob(decimal: float) -> float:
    return round(1 / decimal, 6) if decimal and decimal > 1 else 0.0


class ESPNOddsFeed:
    """
    Fetches DraftKings odds from ESPN's public scoreboard API.

    Produces sportsbook-category markets (Source.ESPN_DK) for matching
    against Kalshi/Polymarket game markets and OddsAPI/Betfair.

    Poll interval: 60s (updates with each game's line movement).
    No rate limiting needed — ESPN's API has no documented quota.
    """

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            headers={"User-Agent": "Mozilla/5.0 (compatible; ArbitrageBot/1.0)"},
            timeout=15.0,
            follow_redirects=True,
        )

    async def fetch(self) -> list[Market]:
        try:
            return await self._fetch()
        except Exception:
            log.exception("ESPNOdds fetch failed")
            return []

    async def _fetch(self) -> list[Market]:
        tasks = [
            self._fetch_sport(sport, league, canonical)
            for sport, league, canonical in _SPORT_ENDPOINTS
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        markets: list[Market] = []
        for (sport, league, _), result in zip(_SPORT_ENDPOINTS, results):
            if isinstance(result, Exception):
                log.warning("ESPNOdds %s/%s error: %s", sport, league, result)
            else:
                markets.extend(result)

        sport_counts: dict[str, int] = {}
        for m in markets:
            sport_counts[m.sport] = sport_counts.get(m.sport, 0) + 1
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(sport_counts.items()))
        log.info("ESPNOdds (DraftKings): %d markets (%s)", len(markets), breakdown)
        return markets

    async def _fetch_sport(self, sport: str, league: str,
                           canonical_sport: str) -> list[Market]:
        """Fetch scoreboard for one sport and parse all game odds."""
        url = f"{_ESPN_BASE}/{sport}/{league}/scoreboard"
        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.debug("ESPNOdds %s/%s fetch error: %s", sport, league, exc)
            return []

        markets: list[Market] = []
        for event in data.get("events", []):
            parsed = self._parse_event(event, canonical_sport)
            markets.extend(parsed)
        return markets

    def _parse_event(self, event: dict, sport: str) -> list[Market]:
        """Parse one ESPN event into h2h, spreads, and totals markets."""
        competitions = event.get("competitions", [])
        if not competitions:
            return []
        comp = competitions[0]

        # Skip games that are in progress or already finished
        status_name = comp.get("status", {}).get("type", {}).get("name", "")
        if status_name in _SKIP_STATUSES:
            return []

        # Extract team info
        competitors = comp.get("competitors", [])
        home_team: Optional[str] = None
        away_team: Optional[str] = None
        for c in competitors:
            team_name = c.get("team", {}).get("displayName", "")
            if c.get("homeAway") == "home":
                home_team = team_name
            else:
                away_team = team_name

        if not home_team or not away_team:
            return []

        event_name = event.get("name", f"{away_team} at {home_team}")
        event_id = str(event.get("id", ""))

        # Parse start time
        date_str = event.get("date", "")
        commence_time: Optional[datetime] = None
        if date_str:
            try:
                commence_time = datetime.fromisoformat(
                    date_str.replace("Z", "+00:00")
                )
            except ValueError:
                pass

        # Find DraftKings odds (provider id "100" or name "DraftKings")
        odds_list = [o for o in comp.get("odds", []) if o]  # filter out None entries
        dk_odds: Optional[dict] = None
        for o in odds_list:
            provider = o.get("provider", {})
            if (provider.get("id") == "100"
                    or "DraftKings" in provider.get("name", "")):
                dk_odds = o
                break
        if not dk_odds and odds_list:
            dk_odds = odds_list[0]  # fallback to first available
        if not dk_odds:
            return []

        markets: list[Market] = []

        # ── Moneyline (h2h) ──────────────────────────────────────────────
        ml = dk_odds.get("moneyline", {})
        home_ml_str = ml.get("home", {}).get("close", {}).get("odds")
        away_ml_str = ml.get("away", {}).get("close", {}).get("odds")

        home_dec = _american_to_decimal(home_ml_str) if home_ml_str else None
        away_dec = _american_to_decimal(away_ml_str) if away_ml_str else None

        if home_dec and away_dec:
            home_prob = _decimal_to_prob(home_dec)
            away_prob = _decimal_to_prob(away_dec)
            if (MIN_PROB <= home_prob <= MAX_PROB
                    and MIN_PROB <= away_prob <= MAX_PROB):
                markets.append(Market(
                    source=Source.ESPN_DK,
                    market_id=f"{event_id}_h2h",
                    sport=sport,
                    event_name=event_name,
                    commence_time=commence_time,
                    home_team=home_team,
                    away_team=away_team,
                    market_type="h2h",
                    outcomes=[
                        Outcome(
                            name=home_team,
                            price=home_dec,
                            implied_prob=home_prob,
                            source=Source.ESPN_DK,
                            market_id=f"{event_id}_h2h",
                            bookmaker="DraftKings",
                            side=BetSide.BACK,
                        ),
                        Outcome(
                            name=away_team,
                            price=away_dec,
                            implied_prob=away_prob,
                            source=Source.ESPN_DK,
                            market_id=f"{event_id}_h2h",
                            bookmaker="DraftKings",
                            side=BetSide.BACK,
                        ),
                    ],
                    raw={"bookmaker": "DraftKings", "event_id": event_id},
                ))

        # ── Point Spread ─────────────────────────────────────────────────
        ps = dk_odds.get("pointSpread", {})
        home_ps_close = ps.get("home", {}).get("close", {})
        away_ps_close = ps.get("away", {}).get("close", {})
        home_line = home_ps_close.get("line", "")
        away_line = away_ps_close.get("line", "")
        home_sp_str = home_ps_close.get("odds")
        away_sp_str = away_ps_close.get("odds")

        home_sp_dec = _american_to_decimal(home_sp_str) if home_sp_str else None
        away_sp_dec = _american_to_decimal(away_sp_str) if away_sp_str else None

        if home_sp_dec and away_sp_dec and home_line and away_line:
            home_sp_prob = _decimal_to_prob(home_sp_dec)
            away_sp_prob = _decimal_to_prob(away_sp_dec)
            if (MIN_PROB <= home_sp_prob <= MAX_PROB
                    and MIN_PROB <= away_sp_prob <= MAX_PROB):
                markets.append(Market(
                    source=Source.ESPN_DK,
                    market_id=f"{event_id}_spreads",
                    sport=sport,
                    event_name=event_name,
                    commence_time=commence_time,
                    home_team=home_team,
                    away_team=away_team,
                    market_type="spreads",
                    outcomes=[
                        Outcome(
                            name=f"{home_team} ({home_line})",
                            price=home_sp_dec,
                            implied_prob=home_sp_prob,
                            source=Source.ESPN_DK,
                            market_id=f"{event_id}_spreads",
                            bookmaker="DraftKings",
                            side=BetSide.BACK,
                        ),
                        Outcome(
                            name=f"{away_team} ({away_line})",
                            price=away_sp_dec,
                            implied_prob=away_sp_prob,
                            source=Source.ESPN_DK,
                            market_id=f"{event_id}_spreads",
                            bookmaker="DraftKings",
                            side=BetSide.BACK,
                        ),
                    ],
                    raw={"bookmaker": "DraftKings", "event_id": event_id},
                ))

        # ── Totals (over/under) ──────────────────────────────────────────
        tot = dk_odds.get("total", {})
        over_close = tot.get("over", {}).get("close", {})
        under_close = tot.get("under", {}).get("close", {})
        over_line = over_close.get("line", "")   # e.g. "o9.5"
        under_line = under_close.get("line", "") # e.g. "u9.5"
        over_odds_str = over_close.get("odds")
        under_odds_str = under_close.get("odds")

        over_dec = _american_to_decimal(over_odds_str) if over_odds_str else None
        under_dec = _american_to_decimal(under_odds_str) if under_odds_str else None

        # Extract clean line number from "o9.5" → "9.5"
        line_num = over_line.lstrip("o").lstrip("u") if over_line else ""

        if over_dec and under_dec and line_num:
            over_prob = _decimal_to_prob(over_dec)
            under_prob = _decimal_to_prob(under_dec)
            if (MIN_PROB <= over_prob <= MAX_PROB
                    and MIN_PROB <= under_prob <= MAX_PROB):
                markets.append(Market(
                    source=Source.ESPN_DK,
                    market_id=f"{event_id}_totals",
                    sport=sport,
                    event_name=event_name,
                    commence_time=commence_time,
                    home_team=home_team,
                    away_team=away_team,
                    market_type="totals",
                    outcomes=[
                        Outcome(
                            name=f"Over {line_num}",
                            price=over_dec,
                            implied_prob=over_prob,
                            source=Source.ESPN_DK,
                            market_id=f"{event_id}_totals",
                            bookmaker="DraftKings",
                            side=BetSide.BACK,
                        ),
                        Outcome(
                            name=f"Under {line_num}",
                            price=under_dec,
                            implied_prob=under_prob,
                            source=Source.ESPN_DK,
                            market_id=f"{event_id}_totals",
                            bookmaker="DraftKings",
                            side=BetSide.BACK,
                        ),
                    ],
                    raw={"bookmaker": "DraftKings", "event_id": event_id,
                         "line": line_num},
                ))

        return markets

    async def close(self) -> None:
        await self._client.aclose()

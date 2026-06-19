from __future__ import annotations

"""
Action Network multi-book odds feed — no API key required.

Action Network's public scoreboard API returns odds from up to 10 major US
sportsbooks in a single request: FanDuel, Caesars, BetMGM, DraftKings,
Pinnacle, bet365, BetRivers, Underdog, and Fanatics.

Endpoint:
  GET https://api.actionnetwork.com/web/v1/scoreboard/{league}
      ?period=game&bookIds=3,13,68,69,71,75,79,3348,2396

Odds are in American format (e.g. +130, -154); we convert to decimal.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Callable

import httpx

from src.models import BetSide, Market, Outcome, Source

log = logging.getLogger(__name__)

_BASE = "https://api.actionnetwork.com/web/v1/scoreboard"

# Book IDs → canonical name. All are major US legal sportsbooks.
_BOOKS: dict[int, str] = {
    3:    "Pinnacle",
    13:   "Caesars",
    68:   "DraftKings",
    69:   "FanDuel",
    71:   "BetRivers",
    75:   "BetMGM",
    79:   "bet365",
    2396: "Fanatics",
    3348: "Underdog",
}
_BOOK_IDS = ",".join(str(k) for k in _BOOKS)

# Leagues to poll: (action_network_slug, canonical_sport)
_LEAGUES: list[tuple[str, str]] = [
    ("mlb",   "baseball"),
    ("nba",   "basketball"),
    ("nhl",   "hockey"),
    ("nfl",   "football"),
    ("ncaaf", "football"),
    ("mls",   "soccer"),
]

# Game statuses where the game is over or in progress — skip these.
_SKIP_STATUSES = frozenset({
    "final", "in progress", "halftime", "delayed",
    "postponed", "canceled", "suspended", "forfeit",
})

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Origin": "https://www.actionnetwork.com",
    "Referer": "https://www.actionnetwork.com/",
}

MIN_PROB = 0.02
MAX_PROB = 0.98

# Fire detection event when any moneyline moves by this many American odds points.
# 5 points ≈ $0.02 implied probability move at -110 (~0.5%).
LINE_MOVE_THRESHOLD_AMERICAN = 5


def _american_to_decimal(american: int) -> Optional[float]:
    if american > 0:
        return round(1 + american / 100, 4)
    elif american < 0:
        return round(1 + 100 / abs(american), 4)
    return None


def _decimal_to_prob(d: float) -> float:
    return round(1 / d, 6) if d and d > 1 else 0.0


class ActionNetworkFeed:
    """
    Fetches odds from FanDuel, Caesars, BetMGM, DraftKings, Pinnacle,
    bet365, BetRivers, Underdog, and Fanatics via Action Network's
    public scoreboard API.

    Each game produces one Market per bookmaker per market type (h2h /
    spreads / totals), tagged with Source.ACTION_NETWORK and the
    bookmaker name.  The arb detector compares across bookmakers.

    Tracks previous moneylines and fires on_line_move() when any book
    moves a line by ≥ LINE_MOVE_THRESHOLD_AMERICAN points — allowing the
    detection loop to wake immediately on a meaningful odds shift.
    """

    def __init__(self, on_line_move: Callable[[], None] | None = None) -> None:
        self._client = httpx.AsyncClient(headers=_HEADERS, timeout=20.0)
        self._on_line_move = on_line_move
        # (game_id, book_id) → (ml_away, ml_home) from previous fetch
        self._prev_lines: dict[tuple[str, int], tuple[int, int]] = {}

    async def fetch(self) -> list[Market]:
        try:
            return await self._fetch()
        except Exception:
            log.exception("ActionNetwork fetch failed")
            return []

    async def _fetch(self) -> list[Market]:
        tasks = [self._fetch_league(slug, sport) for slug, sport in _LEAGUES]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        markets: list[Market] = []
        for (slug, _), result in zip(_LEAGUES, results):
            if isinstance(result, Exception):
                log.warning("ActionNetwork %s error: %s", slug, result)
            else:
                markets.extend(result)

        sport_counts: dict[str, int] = {}
        for m in markets:
            sport_counts[m.sport] = sport_counts.get(m.sport, 0) + 1
        log.info(
            "ActionNetwork: %d markets (%s)",
            len(markets),
            ", ".join(f"{k}={v}" for k, v in sorted(sport_counts.items())),
        )
        return markets

    async def _fetch_league(self, league: str, canonical_sport: str) -> list[Market]:
        resp = await self._client.get(
            f"{_BASE}/{league}",
            params={"period": "game", "bookIds": _BOOK_IDS},
        )
        resp.raise_for_status()
        data = resp.json()

        now = datetime.now(tz=timezone.utc)
        markets: list[Market] = []
        for game in data.get("games", []):
            # Skip games that are in progress or finished
            status = (game.get("real_status") or game.get("status") or "").lower()
            if any(s in status for s in _SKIP_STATUSES):
                continue

            # Parse start time
            start_str = game.get("start_time", "")
            commence_time: Optional[datetime] = None
            if start_str:
                try:
                    # Format: "2026-06-07T17:35:00.000Z" or "2026-06-07T17:35"
                    commence_time = datetime.fromisoformat(
                        start_str.replace("Z", "+00:00").split(".")[0]
                    )
                    if commence_time.tzinfo is None:
                        commence_time = commence_time.replace(tzinfo=timezone.utc)
                    if commence_time <= now:
                        continue  # already started
                except ValueError:
                    pass

            # Extract team names using the game-level away/home ID fields
            away_team_id = game.get("away_team_id")
            home_team_id = game.get("home_team_id")
            teams = game.get("teams", [])
            team_by_id: dict[int, str] = {
                t["id"]: t.get("full_name") or t.get("name", "")
                for t in teams if t.get("id")
            }
            away_team = team_by_id.get(away_team_id)
            home_team = team_by_id.get(home_team_id)
            # Fallback if IDs not present (older data)
            if not away_team and len(teams) >= 1:
                away_team = teams[0].get("full_name") or teams[0].get("name")
            if not home_team and len(teams) >= 2:
                home_team = teams[1].get("full_name") or teams[1].get("name")

            if not away_team or not home_team:
                continue

            event_name = f"{away_team} at {home_team}"
            game_id = str(game.get("id", ""))

            game_markets = self._parse_odds(
                game.get("odds", []),
                game_id, event_name, canonical_sport,
                home_team, away_team, commence_time,
            )
            markets.extend(game_markets)

            # Track moneyline movements per book — fire callback on significant shifts
            if self._on_line_move:
                for o in game.get("odds", []):
                    book_id = o.get("book_id")
                    ml_away = o.get("ml_away")
                    ml_home = o.get("ml_home")
                    if not (book_id and isinstance(ml_away, int) and isinstance(ml_home, int)):
                        continue
                    key = (game_id, book_id)
                    prev = self._prev_lines.get(key)
                    self._prev_lines[key] = (ml_away, ml_home)
                    if prev is None:
                        continue
                    moved = (
                        abs(ml_away - prev[0]) >= LINE_MOVE_THRESHOLD_AMERICAN
                        or abs(ml_home - prev[1]) >= LINE_MOVE_THRESHOLD_AMERICAN
                    )
                    if moved:
                        book_name = _BOOKS.get(book_id, str(book_id))
                        log.info(
                            "ActionNetwork line move: %s | %s | away %+d→%+d home %+d→%+d",
                            event_name[:40], book_name,
                            prev[0], ml_away, prev[1], ml_home,
                        )
                        self._on_line_move()

        return markets

    def _parse_odds(
        self,
        odds_list: list[dict],
        game_id: str,
        event_name: str,
        sport: str,
        home_team: str,
        away_team: str,
        commence_time: Optional[datetime],
    ) -> list[Market]:
        markets: list[Market] = []

        for o in odds_list:
            book_id = o.get("book_id")
            book_name = _BOOKS.get(book_id)
            if not book_name:
                continue  # skip Consensus/Open pseudo-books

            ml_away = o.get("ml_away")
            ml_home = o.get("ml_home")
            spread_away = o.get("spread_away")
            spread_away_line = o.get("spread_away_line")
            spread_home_line = o.get("spread_home_line")
            total = o.get("total")
            over_line = o.get("over")
            under_line = o.get("under")

            # ── Moneyline (h2h) ──────────────────────────────────────────
            if ml_away and ml_home:
                away_dec = _american_to_decimal(ml_away)
                home_dec = _american_to_decimal(ml_home)
                if away_dec and home_dec:
                    away_prob = _decimal_to_prob(away_dec)
                    home_prob = _decimal_to_prob(home_dec)
                    if (MIN_PROB <= away_prob <= MAX_PROB
                            and MIN_PROB <= home_prob <= MAX_PROB):
                        markets.append(Market(
                            source=Source.ACTION_NETWORK,
                            market_id=f"{game_id}_{book_id}_h2h",
                            sport=sport,
                            event_name=event_name,
                            commence_time=commence_time,
                            home_team=home_team,
                            away_team=away_team,
                            market_type="h2h",
                            outcomes=[
                                Outcome(
                                    name=away_team,
                                    price=away_dec,
                                    implied_prob=away_prob,
                                    source=Source.ACTION_NETWORK,
                                    market_id=f"{game_id}_{book_id}_h2h",
                                    bookmaker=book_name,
                                    side=BetSide.BACK,
                                ),
                                Outcome(
                                    name=home_team,
                                    price=home_dec,
                                    implied_prob=home_prob,
                                    source=Source.ACTION_NETWORK,
                                    market_id=f"{game_id}_{book_id}_h2h",
                                    bookmaker=book_name,
                                    side=BetSide.BACK,
                                ),
                            ],
                            raw={"bookmaker": book_name, "game_id": game_id},
                        ))

            # ── Point Spread ─────────────────────────────────────────────
            if spread_away is not None and spread_away_line and spread_home_line:
                spread_home = -spread_away if spread_away else None
                away_sp_dec = _american_to_decimal(spread_away_line)
                home_sp_dec = _american_to_decimal(spread_home_line)
                if away_sp_dec and home_sp_dec and spread_home is not None:
                    away_sp_prob = _decimal_to_prob(away_sp_dec)
                    home_sp_prob = _decimal_to_prob(home_sp_dec)
                    if (MIN_PROB <= away_sp_prob <= MAX_PROB
                            and MIN_PROB <= home_sp_prob <= MAX_PROB):
                        line = f"+{spread_away}" if spread_away > 0 else str(spread_away)
                        home_line = f"+{spread_home}" if spread_home > 0 else str(spread_home)
                        markets.append(Market(
                            source=Source.ACTION_NETWORK,
                            market_id=f"{game_id}_{book_id}_spreads",
                            sport=sport,
                            event_name=event_name,
                            commence_time=commence_time,
                            home_team=home_team,
                            away_team=away_team,
                            market_type="spreads",
                            outcomes=[
                                Outcome(
                                    name=f"{away_team} ({line})",
                                    price=away_sp_dec,
                                    implied_prob=away_sp_prob,
                                    source=Source.ACTION_NETWORK,
                                    market_id=f"{game_id}_{book_id}_spreads",
                                    bookmaker=book_name,
                                    side=BetSide.BACK,
                                ),
                                Outcome(
                                    name=f"{home_team} ({home_line})",
                                    price=home_sp_dec,
                                    implied_prob=home_sp_prob,
                                    source=Source.ACTION_NETWORK,
                                    market_id=f"{game_id}_{book_id}_spreads",
                                    bookmaker=book_name,
                                    side=BetSide.BACK,
                                ),
                            ],
                            raw={"bookmaker": book_name, "game_id": game_id},
                        ))

            # ── Totals (over/under) ──────────────────────────────────────
            if total and over_line and under_line:
                over_dec = _american_to_decimal(over_line)
                under_dec = _american_to_decimal(under_line)
                if over_dec and under_dec:
                    over_prob = _decimal_to_prob(over_dec)
                    under_prob = _decimal_to_prob(under_dec)
                    if (MIN_PROB <= over_prob <= MAX_PROB
                            and MIN_PROB <= under_prob <= MAX_PROB):
                        markets.append(Market(
                            source=Source.ACTION_NETWORK,
                            market_id=f"{game_id}_{book_id}_totals",
                            sport=sport,
                            event_name=event_name,
                            commence_time=commence_time,
                            home_team=home_team,
                            away_team=away_team,
                            market_type="totals",
                            outcomes=[
                                Outcome(
                                    name=f"Over {total}",
                                    price=over_dec,
                                    implied_prob=over_prob,
                                    source=Source.ACTION_NETWORK,
                                    market_id=f"{game_id}_{book_id}_totals",
                                    bookmaker=book_name,
                                    side=BetSide.BACK,
                                ),
                                Outcome(
                                    name=f"Under {total}",
                                    price=under_dec,
                                    implied_prob=under_prob,
                                    source=Source.ACTION_NETWORK,
                                    market_id=f"{game_id}_{book_id}_totals",
                                    bookmaker=book_name,
                                    side=BetSide.BACK,
                                ),
                            ],
                            raw={"bookmaker": book_name, "game_id": game_id,
                                 "line": total},
                        ))

        return markets

    async def close(self) -> None:
        await self._client.aclose()

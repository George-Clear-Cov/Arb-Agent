from __future__ import annotations

"""
Kalshi live sports feed — game-by-game spread, total, and winner markets.

Kalshi's sports product covers in-season game events across MLB, NBA, NHL,
NFL, soccer, tennis, MMA, and boxing.  These are fetched from dedicated
series endpoints (e.g. KXMLBGAME, KXNBAGAME) rather than the general
/events list, because game events don't appear in the standard 200-event
prediction-market page.

Market types produced:
  h2h      — game winner (both teams as separate outcomes)
  totals   — over/under run/point totals (standard line closest to 50/50)

Poll interval: 30s (live games update quickly).

Rate-limit note: we fetch series events in parallel but gate individual
event-market fetches with a small semaphore to avoid hammering the API.
"""

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Optional

import httpx

from src.feeds.feed_cache import CACHE_DIR, SPORTS_MAX_AGE, load_cache, save_cache
from src.feeds.kalshi_rate import kalshi_sports_request as kalshi_request
from src.models import BetSide, Market, Outcome, Source

_CACHE_FILE = CACHE_DIR / "kalshi_sports_cache.json"

log = logging.getLogger(__name__)

MIN_PROB = 0.03
MAX_PROB = 0.97

# Series tickers grouped by sport and market type.
# Each entry: (sport, market_type, series_ticker)
# ORDER MATTERS — fetched top-to-bottom within the rate-limit budget.
# Put currently-active playoff/Finals series FIRST so they aren't squeezed out.
SPORT_SERIES: list[tuple[str, str, str]] = [
    # ── TOP PRIORITY: reorder to match what's actually live right now ──────────
    # Soccer — World Cup 2026 is live; highest-value opportunity
    ("soccer",     "h2h",    "KXWORLDCUP"),
    ("soccer",     "h2h",    "KXCHAMPIONSLEAGUE"),
    ("soccer",     "h2h",    "KXSOCCERGAME"),
    ("soccer",     "totals", "KXSOCCERTOTAL"),
    # Baseball — daily games, high arb density
    ("baseball",   "h2h",    "KXMLBGAME"),
    ("baseball",   "totals", "KXMLBTOTAL"),
    # Tennis — live set/match markets, short-expiry arbs
    ("tennis",     "h2h",    "KXATPGWINNER"),
    ("tennis",     "h2h",    "KXWTASETWINNER"),
    # MMA / Boxing — fight cards
    ("mma",        "h2h",    "KXUFCFIGHT"),
    ("boxing",     "h2h",    "KXBOXINGFIGHT"),
    # Basketball
    ("basketball", "h2h",    "KXNBAGAME"),
    ("basketball", "totals", "KXNBATOTAL"),
    ("basketball", "h2h",    "KXWNBAGAME"),
    # Hockey
    ("hockey",     "h2h",    "KXNHLGAME"),
    ("hockey",     "totals", "KXNHLTOTAL"),
    # F1
    ("f1",         "h2h",    "KXF1RACE"),
    # American football (off-season — lowest priority)
    ("football",   "h2h",    "KXNFLGAME"),
    ("football",   "totals", "KXNFLTOTAL"),
    ("football",   "h2h",    "KXCFBGAME"),
    # College / Cricket / Golf / Esports
    ("basketball", "h2h",    "KXNCAABGAME"),
    ("cricket",    "h2h",    "KXCRICKET"),
    ("golf",       "h2h",    "KXGOLFTOURNAMENT"),
    ("esports",    "h2h",    "KXCSGOGAME"),
    ("esports",    "h2h",    "KXLOLGAME"),
]


class KalshiSportsFeed:
    """
    Fetches live sports game markets from Kalshi's series-based endpoints.

    Unlike the prediction market KalshiFeed (which polls /events every 10 min),
    this feed polls every 30 s because game probabilities change quickly during
    live matches.

    Cold-series skip: series that return 0 events for COLD_THRESHOLD consecutive
    cycles are skipped for COLD_SKIP_CYCLES cycles before being retried.
    Off-season series (NFL in June, WNBA in winter) stop consuming quota until
    they have active games, leaving budget for active sports like soccer.
    """

    COLD_THRESHOLD  = 5    # empty cycles before backing off
    COLD_SKIP_CYCLES = 30  # skip N cycles (~5 min at 10s poll) before retry

    def __init__(self, api_key: str, base_url: str) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Token {api_key}"},
            timeout=20.0,
        )
        # Per-series state: consecutive empty cycles
        self._empty: dict[str, int] = {}
        self._disk_markets = load_cache(_CACHE_FILE, Source.KALSHI_SPORTS,
                                        max_age=SPORTS_MAX_AGE)

    async def fetch(self) -> list[Market]:
        try:
            result = await self._fetch()
            if result:
                self._disk_markets = result
                save_cache(_CACHE_FILE, result)
            return result or self._disk_markets
        except Exception:
            log.exception("KalshiSports fetch failed")
            return self._disk_markets

    async def _fetch(self) -> list[Market]:
        markets: list[Market] = []
        skipped: list[str] = []
        for sport, mtype, series in SPORT_SERIES:
            empty = self._empty.get(series, 0)
            # Skip cold series: only retry every COLD_SKIP_CYCLES cycles
            if empty >= self.COLD_THRESHOLD and empty % self.COLD_SKIP_CYCLES != 0:
                self._empty[series] = empty + 1
                skipped.append(series)
                continue
            result = await self._fetch_series(sport, mtype, series)
            if result:
                self._empty[series] = 0
            else:
                self._empty[series] = empty + 1
            markets.extend(result)
        if skipped:
            log.debug("KalshiSports: skipped %d cold series: %s", len(skipped), skipped)
        log.info(
            "KalshiSports: %d markets (%s)",
            len(markets),
            ", ".join(
                f"{s}={sum(1 for m in markets if m.sport == s)}"
                for s in sorted({m.sport for m in markets})
            ),
        )
        return markets

    async def _fetch_series(self, sport: str, market_type: str,
                             series: str) -> list[Market]:
        """Get events + nested markets for a series in a single API call."""
        try:
            resp = await kalshi_request(self._client.get("/events", params={
                "status": "open",
                "limit": 50,
                "series_ticker": series,
                "with_nested_markets": "true",
            }))
            if resp.status_code == 429:
                log.warning("KalshiSports: rate limited on series %s", series)
                return []
            resp.raise_for_status()
            events = resp.json().get("events", [])
        except Exception as exc:
            log.warning("KalshiSports series %s fetch error: %s", series, exc)
            return []

        if not events:
            log.debug("KalshiSports series %s: 0 events returned", series)
            return []

        results: list[Market] = []
        for event in events:
            raw_markets = event.get("markets", [])
            if not raw_markets:
                continue
            if market_type == "h2h":
                m = self._parse_h2h(event, raw_markets, sport)
            elif market_type == "totals":
                m = self._parse_totals(event, raw_markets, sport)
            else:
                m = None
            if m:
                results.append(m)
        return results

    def _parse_h2h(self, event: dict, raw_markets: list[dict],
                   sport: str) -> Market | None:
        """
        Game winner: each raw market is "Does [team] win?" (yes/no).
        Combine all team markets into a single 2+ outcome Market.
        """
        event_title = event.get("title", event.get("event_ticker", ""))
        outcomes: list[Outcome] = []
        seen_teams: set[str] = set()

        for m in raw_markets:
            team = (m.get("yes_sub_title") or "").strip()
            if not team or team in seen_teams:
                continue
            seen_teams.add(team)

            try:
                yes_prob = float(m.get("yes_ask_dollars", 0) or 0)
            except (TypeError, ValueError):
                continue

            if not (MIN_PROB <= yes_prob <= MAX_PROB):
                continue

            outcomes.append(Outcome(
                name=team,
                price=round(1 / yes_prob, 4),
                implied_prob=yes_prob,
                source=Source.KALSHI_SPORTS,
                market_id=m.get("ticker", event.get("event_ticker", "")),
                bookmaker="Kalshi",
                side=BetSide.BACK,
            ))

        if len(outcomes) < 2:
            return None

        # Sanity check: the sum of all yes_ask probabilities for a winner market
        # should be close to 1.0 (small overround).  Games days away often have
        # placeholder prices (e.g. 0.72 + 0.72 = 1.44 overround) — skip those.
        # Soccer allows slightly wider initial spreads (heavy favorites vs minnows).
        total_prob = sum(o.implied_prob for o in outcomes)
        max_overround = 1.25 if sport == "soccer" else 1.10
        if total_prob > max_overround:
            log.debug("Skipping %s h2h: overround %.0f%% too wide (placeholder prices)",
                      event_title[:40], (total_prob - 1) * 100)
            return None

        # Extract home/away from title ("Team A vs Team B" or "Team A at Team B")
        home, away = _parse_teams(event_title)

        event_ticker = event.get("event_ticker", "")
        # Parse the actual game time from the ticker (e.g. KXMLBGAME-26JUN091907PHITOR
        # → 2026-06-09 19:07 UTC). Kalshi's close_time is the settlement DEADLINE
        # (3-7 days after the game) — using it caused the date-guard to reject
        # legitimate same-day matches and accept next-day false arbs.
        commence_time = _parse_ticker_datetime(event_ticker)
        close_time = event.get("close_time") or next(
            (m.get("close_time") for m in raw_markets if m.get("close_time")), None
        )

        return Market(
            source=Source.KALSHI_SPORTS,
            market_id=event_ticker,
            sport=sport,
            event_name=event_title,
            commence_time=commence_time,
            home_team=home,
            away_team=away,
            market_type="h2h",
            outcomes=outcomes,
            raw={"event_ticker": event.get("event_ticker"), "series": event.get("series_ticker"),
                 "kalshi_close_time": close_time},
        )

    def _parse_totals(self, event: dict, raw_markets: list[dict],
                      sport: str) -> Market | None:
        """
        Over/under totals: pick the most balanced line (closest to 50/50).
        Returns a single binary market: Over X.5 / Under X.5.
        """
        event_title = event.get("title", event.get("event_ticker", ""))

        # Find the market whose yes_ask is closest to 0.50 (most balanced)
        best: Optional[dict] = None
        best_dist = 1.0
        for m in raw_markets:
            try:
                yes_prob = float(m.get("yes_ask_dollars", 0) or 0)
            except (TypeError, ValueError):
                continue
            if not (MIN_PROB <= yes_prob <= MAX_PROB):
                continue
            dist = abs(yes_prob - 0.50)
            if dist < best_dist:
                best_dist = dist
                best = m

        if not best:
            return None

        try:
            yes_prob = float(best["yes_ask_dollars"])
            no_prob  = float(best["no_ask_dollars"])
        except (KeyError, TypeError, ValueError):
            return None

        if not (MIN_PROB <= yes_prob <= MAX_PROB) or not (MIN_PROB <= no_prob <= MAX_PROB):
            return None

        sub = (best.get("yes_sub_title") or "").strip()
        over_label = sub if sub else "Over"
        under_label = over_label.replace("Over", "Under") if "Over" in over_label else "Under"

        home, away = _parse_teams(event_title)

        event_ticker = event.get("event_ticker", "")
        commence_time = _parse_ticker_datetime(event_ticker)
        close_time = event.get("close_time") or best.get("close_time")

        return Market(
            source=Source.KALSHI_SPORTS,
            market_id=best.get("ticker", event_ticker),
            sport=sport,
            event_name=event_title,
            commence_time=commence_time,
            home_team=home,
            away_team=away,
            market_type="totals",
            outcomes=[
                Outcome(
                    name=over_label,
                    price=round(1 / yes_prob, 4),
                    implied_prob=yes_prob,
                    source=Source.KALSHI_SPORTS,
                    market_id=best.get("ticker", ""),
                    bookmaker="Kalshi",
                    side=BetSide.BACK,
                ),
                Outcome(
                    name=under_label,
                    price=round(1 / no_prob, 4),
                    implied_prob=no_prob,
                    source=Source.KALSHI_SPORTS,
                    market_id=best.get("ticker", ""),
                    bookmaker="Kalshi",
                    side=BetSide.BACK,
                ),
            ],
            raw={"event_ticker": event.get("event_ticker"), "line": sub},
        )

    async def close(self) -> None:
        await self._client.aclose()


_TICKER_DT_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})(\d{2})(\d{2})")
_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _parse_ticker_datetime(ticker: str) -> Optional[datetime]:
    """
    Extract UTC game time from a Kalshi event ticker.
    Format: {SERIES}-{YY}{MON}{DD}{HH}{MM}{TEAMS}
    e.g. KXMLBGAME-26JUN091907PHITOR → 2026-06-09 19:07 UTC
    Kalshi's close_time is the settlement DEADLINE (days after the game),
    so we use the ticker-encoded date instead for the date-guard in matcher.
    """
    m = _TICKER_DT_RE.search(ticker)
    if not m:
        return None
    yy, mon, dd, hh, mm = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
    month = _MONTH_MAP.get(mon.upper())
    if not month:
        return None
    try:
        return datetime(2000 + int(yy), month, int(dd), int(hh), int(mm),
                        tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_teams(title: str) -> tuple[str | None, str | None]:
    """Extract (home, away) from 'Away at Home', 'Team A vs Team B', or similar."""
    t = title
    # Strip prefixes like "Game 6: "
    if ": " in t:
        t = t.split(": ", 1)[-1]
    # "Team A at Home B" — away "at" home
    if " at " in t:
        parts = t.split(" at ", 1)
        return parts[1].strip(), parts[0].strip()
    # "Team A vs Team B"
    for sep in (" vs ", " v ", " - "):
        if sep in t:
            parts = t.split(sep, 1)
            return parts[0].strip(), parts[1].strip()
    return None, None


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None

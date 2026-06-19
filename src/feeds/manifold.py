from __future__ import annotations

"""
Manifold Markets feed — real-money sweepstakes (mana) prediction markets.

Manifold is primarily a play-money platform, but its community prices are
excellent signal for detecting mispricings between Kalshi and Polymarket.
We include all markets but tag them with is_play_money=True in raw so the
UI can distinguish them from real-money positions.

API docs: https://docs.manifold.markets/api
Rate limit: generous (~100 req/min), no auth required.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from src.feeds.base import BaseFeed
from src.models import BetSide, Market, Outcome, Source

log = logging.getLogger(__name__)

API_BASE = "https://api.manifold.markets/v0"

# Minimum liquidity in mana (≈ USD on sweepstakes markets)
MIN_LIQUIDITY = 500

# Map Manifold category slugs → our canonical sport names
_CATEGORY_MAP: dict[str, str] = {
    "sports":         "prediction",
    "politics":       "prediction",
    "economics":      "prediction",
    "crypto":         "prediction",
    "science":        "prediction",
    "technology":     "prediction",
    "entertainment":  "prediction",
    "culture":        "prediction",
    "gaming":         "esports",
    "esports":        "esports",
    "baseball":       "baseball",
    "basketball":     "basketball",
    "football":       "football",
    "soccer":         "soccer",
    "hockey":         "hockey",
    "tennis":         "tennis",
    "mma":            "mma",
    "golf":           "golf",
    "f1":             "f1",
    "cricket":        "cricket",
}


class ManifoldFeed(BaseFeed):
    """Fetches active binary markets from Manifold sorted by liquidity."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=20.0)

    async def fetch(self) -> list[Market]:
        try:
            return await self._fetch()
        except Exception:
            log.exception("Manifold fetch failed")
            return []

    async def _fetch(self) -> list[Market]:
        # Fetch top markets sorted by liquidity using search-markets endpoint
        # /v0/markets doesn't support sort=liquidity; /v0/search-markets does
        all_raw: list[dict] = []
        seen_ids: set[str] = set()

        # Pull 3 pages × 500 = up to 1500 markets, offset-based pagination
        for page in range(3):
            params: dict = {
                "limit": 500,
                "offset": page * 500,
                "sort": "liquidity",
                "filter": "open",
                "contractType": "BINARY",
            }

            resp = await self._client.get(f"{API_BASE}/search-markets", params=params)
            resp.raise_for_status()
            batch: list[dict] = resp.json()
            if not batch:
                break

            for m in batch:
                mid = m.get("id", "")
                if mid and mid not in seen_ids:
                    seen_ids.add(mid)
                    all_raw.append(m)

            if len(batch) < 500:
                break  # last page

        markets: list[Market] = []
        for raw in all_raw:
            m = self._parse(raw)
            if m:
                markets.append(m)

        log.info("Manifold: %d markets loaded", len(markets))
        return markets

    def _parse(self, raw: dict) -> Market | None:
        # Only binary markets (YES/NO)
        if raw.get("outcomeType") != "BINARY":
            return None
        if raw.get("isResolved"):
            return None

        liquidity = raw.get("totalLiquidity", 0) or 0
        if liquidity < MIN_LIQUIDITY:
            return None

        # Probability → decimal odds
        prob = raw.get("probability")
        if prob is None or not (0.02 <= prob <= 0.98):
            return None

        yes_price = round(1 / prob, 4)
        no_price  = round(1 / (1 - prob), 4)

        question = raw.get("question", "").strip()
        if not question:
            return None

        # Classify sport from group slugs
        groups = [g.get("slug", "").lower() for g in raw.get("groupLinks", [])]
        sport = "prediction"
        for slug in groups:
            if slug in _CATEGORY_MAP:
                sport = _CATEGORY_MAP[slug]
                break
            # partial match
            for key, val in _CATEGORY_MAP.items():
                if key in slug:
                    sport = val
                    break

        close_time = raw.get("closeTime")
        commence_time: Optional[datetime] = None
        if close_time:
            try:
                commence_time = datetime.fromtimestamp(
                    close_time / 1000, tz=timezone.utc
                )
            except (TypeError, ValueError, OSError):
                pass

        market_id = raw.get("id", "")
        slug = raw.get("slug", market_id)

        return Market(
            source=Source.MANIFOLD,
            market_id=market_id,
            sport=sport,
            event_name=question,
            commence_time=commence_time,
            home_team=None,
            away_team=None,
            market_type="binary",
            total_volume=float(liquidity),
            outcomes=[
                Outcome(
                    name="Yes",
                    price=yes_price,
                    implied_prob=round(prob, 6),
                    source=Source.MANIFOLD,
                    market_id=market_id,
                    bookmaker="Manifold",
                    side=BetSide.BACK,
                    available_volume=float(liquidity) / 2,
                ),
                Outcome(
                    name="No",
                    price=no_price,
                    implied_prob=round(1 - prob, 6),
                    source=Source.MANIFOLD,
                    market_id=market_id,
                    bookmaker="Manifold",
                    side=BetSide.BACK,
                    available_volume=float(liquidity) / 2,
                ),
            ],
            raw={
                "slug": slug,
                "url": f"https://manifold.markets/{raw.get('creatorUsername','')}/{slug}",
                "is_play_money": True,
                "total_liquidity": liquidity,
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

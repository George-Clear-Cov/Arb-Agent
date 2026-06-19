import logging
from datetime import datetime
from typing import Optional

import httpx

from src.feeds.base import BaseFeed
from src.models import BetSide, Market, Outcome, Source

log = logging.getLogger(__name__)


class PolymarketFeed(BaseFeed):
    def __init__(self, clob_url: str, gamma_url: str):
        self.clob_url = clob_url
        self.gamma_url = gamma_url
        self._client = httpx.AsyncClient(timeout=15.0)

    async def fetch(self) -> list[Market]:
        try:
            return await self._fetch_markets()
        except Exception:
            log.exception("Polymarket fetch failed")
            return []

    async def _fetch_markets(self) -> list[Market]:
        # Gamma API gives richer metadata
        resp = await self._client.get(f"{self.gamma_url}/markets", params={
            "active": "true",
            "closed": "false",
            "limit": 100,
        })
        resp.raise_for_status()
        raw_markets = resp.json()

        markets: list[Market] = []
        for m in raw_markets:
            # Binary markets: outcomes are "Yes"/"No"
            outcomes_data = m.get("outcomes", [])
            prices = m.get("outcomePrices", [])

            if not outcomes_data or not prices:
                continue

            outcomes: list[Outcome] = []
            for name, price_str in zip(outcomes_data, prices):
                try:
                    prob = float(price_str)
                except (ValueError, TypeError):
                    continue
                if prob <= 0:
                    continue
                outcomes.append(Outcome(
                    name=name,
                    price=round(1 / prob, 4),
                    implied_prob=prob,
                    source=Source.POLYMARKET,
                    market_id=str(m.get("id", "")),
                    bookmaker="Polymarket",
                    side=BetSide.BACK,
                ))

            if not outcomes:
                continue

            markets.append(Market(
                source=Source.POLYMARKET,
                market_id=str(m.get("id", "")),
                sport="prediction",
                event_name=m.get("question", m.get("title", "")),
                commence_time=_parse_dt(m.get("endDate")),
                home_team=None,
                away_team=None,
                market_type="binary",
                outcomes=outcomes,
                raw={"id": m.get("id"), "slug": m.get("slug")},
            ))

        return markets

    async def close(self) -> None:
        await self._client.aclose()


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None

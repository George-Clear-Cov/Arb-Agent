from __future__ import annotations

"""
Gemini Prediction Markets feed.

CFTC-regulated exchange. Binary and categorical events across crypto, sports,
politics, and entertainment. No API key required for read-only market data.

Base: https://api.gemini.com/v1/prediction-markets
Docs: https://developer.gemini.com/rest-api/prediction-markets
Rate: undocumented — poll at 1 req/cycle per endpoint
"""
import logging
from datetime import datetime
from typing import Optional

import httpx

from src.models import BetSide, Market, Outcome, Source

log = logging.getLogger(__name__)

BASE = "https://api.gemini.com/v1/prediction-markets"
PAGE_SIZE = 500   # API max is 500

# Skip near-settled contracts (spread blows out, arb noise)
MIN_PROB = 0.02
MAX_PROB = 0.98


class GeminiFeed:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=20.0)

    async def fetch(self) -> list[Market]:
        try:
            return await self._fetch()
        except Exception:
            log.exception("Gemini fetch failed")
            return []

    async def _fetch(self) -> list[Market]:
        raw_events: list[dict] = []
        offset = 0
        while True:
            resp = await self._client.get(
                f"{BASE}/events",
                params={"status": "active", "limit": PAGE_SIZE, "offset": offset},
            )
            resp.raise_for_status()
            body = resp.json()
            batch = body.get("data", [])
            raw_events.extend(batch)
            total = body.get("pagination", {}).get("total", 0)
            offset += PAGE_SIZE
            if offset >= total:
                break

        markets: list[Market] = []
        for event in raw_events:
            if event.get("type") != "binary":
                continue
            parsed = self._parse_event(event)
            if parsed:
                markets.append(parsed)

        log.info("Gemini: %d markets", len(markets))
        return markets

    def _parse_event(self, event: dict) -> Market | None:
        contracts = event.get("contracts", [])
        if len(contracts) < 2:
            return None

        # Binary events have exactly 2 contracts: typically "Up"/"Down" or "Yes"/"No"
        # Find the YES-side and NO-side prices from bestBid/bestAsk
        outcomes: list[Outcome] = []
        for contract in contracts:
            if contract.get("marketState") != "open":
                continue
            prices = contract.get("prices", {})
            best_ask = prices.get("bestAsk")
            best_bid = prices.get("bestBid")
            if best_ask is None or best_bid is None:
                continue

            ask_prob = float(best_ask)   # cost to buy this side = ask price = implied prob
            if not (MIN_PROB < ask_prob < MAX_PROB):
                continue

            decimal_price = round(1 / ask_prob, 4)
            label = contract.get("label", contract.get("ticker", "?"))
            market_id = contract.get("id", "")
            vol = event.get("volume24h")

            outcomes.append(Outcome(
                name=label,
                price=decimal_price,
                implied_prob=ask_prob,
                source=Source.GEMINI,
                market_id=market_id,
                bookmaker="Gemini",
                side=BetSide.BACK,
                available_volume=float(vol) if vol else None,
            ))

        if len(outcomes) < 2:
            return None

        expiry = event.get("expiryDate")
        vol_str = event.get("volume24h") or event.get("volume")

        return Market(
            source=Source.GEMINI,
            market_id=event.get("ticker", event["id"]),
            sport=_category_to_sport(event.get("category", "")),
            event_name=event.get("title", ""),
            commence_time=_parse_dt(event.get("startTime") or expiry),
            home_team=None,
            away_team=None,
            market_type="binary",
            outcomes=outcomes,
            total_volume=float(vol_str) if vol_str else None,
            raw={
                "ticker": event.get("ticker"),
                "series": event.get("series"),
                "category": event.get("category"),
                "expiry": expiry,
                "source_index": (event.get("sourceDetails") or {}).get("index"),
            },
        )

    async def close(self) -> None:
        await self._client.aclose()


def _category_to_sport(category: str) -> str:
    return {
        "Crypto": "prediction",
        "Politics": "prediction",
        "Sports": "sports",
        "Entertainment": "prediction",
    }.get(category, "prediction")


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None

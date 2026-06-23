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
            parsed = self._parse_event(event)
            markets.extend(parsed)

        log.info("Gemini: %d markets", len(markets))
        return markets

    def _parse_event(self, event: dict) -> list[Market]:
        """
        Binary events: 1 contract with YES+NO prices in prices.buy.yes / prices.buy.no
        Categorical events: N contracts, each becomes a standalone binary market
        (e.g. "Men's World Cup Winner?" → 56 separate "Will France win?" markets)
        """
        evt_type = event.get("type", "")
        contracts = event.get("contracts", [])
        if not contracts:
            return []

        expiry = event.get("expiryDate")
        vol_str = event.get("volume24h") or event.get("volume")
        vol = float(vol_str) if vol_str else None
        category = event.get("category", "")
        sport = _category_to_sport(category)
        commence = _parse_dt(event.get("startTime") or expiry)

        markets: list[Market] = []

        if evt_type == "binary":
            # Single contract has YES and NO prices side-by-side
            contract = contracts[0]
            if contract.get("marketState") != "open":
                return []
            prices = contract.get("prices", {})
            yes_ask_str = prices.get("bestAsk") or prices.get("buy", {}).get("yes")
            no_ask_str  = prices.get("buy", {}).get("no")
            if not yes_ask_str or not no_ask_str:
                return []
            yes_ask = float(yes_ask_str)
            no_ask  = float(no_ask_str)
            if not (MIN_PROB < yes_ask < MAX_PROB and MIN_PROB < no_ask < MAX_PROB):
                return []
            # Filter severely illiquid markets (spread > 30%)
            best_bid_str = prices.get("bestBid")
            if best_bid_str and (yes_ask - float(best_bid_str)) > 0.30:
                return []
            market_id = event.get("ticker", str(event.get("id", "")))
            markets.append(Market(
                source=Source.GEMINI,
                market_id=market_id,
                sport=sport,
                event_name=event.get("title", ""),
                commence_time=commence,
                home_team=None,
                away_team=None,
                market_type="binary",
                outcomes=[
                    Outcome(name="Yes", price=round(1 / yes_ask, 4), implied_prob=yes_ask,
                            source=Source.GEMINI, market_id=market_id, bookmaker="Gemini",
                            side=BetSide.BACK, available_volume=vol),
                    Outcome(name="No", price=round(1 / no_ask, 4), implied_prob=no_ask,
                            source=Source.GEMINI, market_id=market_id, bookmaker="Gemini",
                            side=BetSide.BACK, available_volume=vol),
                ],
                total_volume=vol,
                raw={"ticker": event.get("ticker"), "series": event.get("series"),
                     "category": category, "expiry": expiry},
            ))

        elif evt_type == "categorical":
            # Each contract = one possible outcome; model each as "Will X happen?" binary market
            event_title = event.get("title", "")
            event_ticker = event.get("ticker", str(event.get("id", "")))
            for contract in contracts:
                if contract.get("marketState") != "open":
                    continue
                prices = contract.get("prices", {})
                yes_ask_str = prices.get("bestAsk")
                yes_bid_str = prices.get("bestBid")
                if not yes_ask_str or not yes_bid_str:
                    continue
                yes_ask = float(yes_ask_str)
                yes_bid = float(yes_bid_str)
                if not (MIN_PROB < yes_ask < MAX_PROB):
                    continue
                if (yes_ask - yes_bid) > 0.20:  # skip illiquid
                    continue
                # NO side = cost to bet against this outcome ≈ 1 - yes_bid
                no_ask = 1.0 - yes_bid
                if not (MIN_PROB < no_ask < MAX_PROB):
                    continue
                label = contract.get("label", "")
                market_name = f"{event_title}: {label}" if label else event_title
                contract_id = f"{event_ticker}-{contract.get('ticker', label)}"
                markets.append(Market(
                    source=Source.GEMINI,
                    market_id=contract_id,
                    sport=sport,
                    event_name=market_name,
                    commence_time=commence,
                    home_team=None,
                    away_team=None,
                    market_type="binary",
                    outcomes=[
                        Outcome(name="Yes", price=round(1 / yes_ask, 4), implied_prob=yes_ask,
                                source=Source.GEMINI, market_id=contract_id, bookmaker="Gemini",
                                side=BetSide.BACK, available_volume=vol),
                        Outcome(name="No", price=round(1 / no_ask, 4), implied_prob=no_ask,
                                source=Source.GEMINI, market_id=contract_id, bookmaker="Gemini",
                                side=BetSide.BACK, available_volume=vol),
                    ],
                    total_volume=vol,
                    raw={"event_ticker": event_ticker, "contract_ticker": contract.get("ticker"),
                         "category": category, "expiry": expiry},
                ))

        return markets

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

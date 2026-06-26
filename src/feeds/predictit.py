from __future__ import annotations

"""
PredictIt dedicated feed.

Uses PredictIt's public unauthenticated API:
  GET https://www.predictit.org/api/marketdata/all/

Returns all active markets in a single request — no auth, no quota.
Each PredictIt market has multiple contracts (candidates / outcomes).
We model each contract as a separate binary market (Yes/No).

Prices are in the 0–1 probability range (not 0–100 cents like PH API).
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

from src.feeds.feed_cache import CACHE_DIR, load_cache, save_cache
from src.models import BetSide, Market, Outcome, Source

log = logging.getLogger(__name__)

API_URL = "https://www.predictit.org/api/marketdata/all/"

# Filter out near-resolved contracts
MIN_PROB = 0.03
MAX_PROB = 0.97

_CACHE_FILE = CACHE_DIR / "predictit_cache.json"


class PredictItFeed:
    """Single-request PredictIt feed. No API key needed."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=20.0)
        self._disk_markets = load_cache(_CACHE_FILE, Source.PREDICTIT)

    async def fetch(self) -> list[Market]:
        try:
            resp = await self._client.get(API_URL)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.error("PredictIt fetch error: %s", exc)
            return self._disk_markets

        markets: list[Market] = []
        for raw_mkt in data.get("markets", []):
            if raw_mkt.get("status") != "Open":
                continue
            parsed = self._parse_market(raw_mkt)
            markets.extend(parsed)

        log.info("PredictIt: %d markets", len(markets))
        if markets:
            self._disk_markets = markets
            save_cache(_CACHE_FILE, markets)
        return markets or self._disk_markets

    def _parse_market(self, mkt: dict) -> list[Market]:
        """
        Each PredictIt market can have multiple contracts (e.g. candidates).
        We turn every contract into its own binary Yes/No Market so the
        matcher can pair individual outcomes against other platforms.
        """
        parent_name: str = mkt.get("name", "Unknown")
        parent_url: str = mkt.get("url", "")
        results: list[Market] = []

        for contract in mkt.get("contracts", []):
            if contract.get("status") != "Open":
                continue

            yes_prob = contract.get("bestBuyYesCost")  # already 0–1
            no_prob = contract.get("bestBuyNoCost")
            contract_name: str = contract.get("shortName") or contract.get("name", "")
            contract_id = str(contract.get("id", ""))

            # Fall back to last trade price if no bid/ask
            if yes_prob is None:
                yes_prob = contract.get("lastTradePrice")
            if yes_prob is None:
                continue

            # Derive no_prob if missing
            if no_prob is None:
                no_prob = 1.0 - yes_prob

            if not (MIN_PROB <= yes_prob <= MAX_PROB):
                continue
            if not (MIN_PROB <= no_prob <= MAX_PROB):
                continue

            yes_decimal = round(1 / yes_prob, 4)
            no_decimal = round(1 / no_prob, 4)

            # Build event name: "Parent Market Name: Contract Name"
            if contract_name and contract_name.lower() != parent_name.lower():
                event_name = f"{parent_name}: {contract_name}"
            else:
                event_name = parent_name

            # Direct URL for this contract
            market_url = f"{parent_url}#contract={contract_id}" if parent_url else ""

            results.append(Market(
                source=Source.PREDICTIT,
                market_id=contract_id,
                sport="prediction",
                event_name=event_name,
                commence_time=_parse_dt(contract.get("dateEnd")),
                home_team=None,
                away_team=None,
                market_type="binary",
                outcomes=[
                    Outcome(
                        name="Yes",
                        price=yes_decimal,
                        implied_prob=round(yes_prob, 6),
                        source=Source.PREDICTIT,
                        market_id=contract_id,
                        bookmaker="PredictIt",
                        side=BetSide.BACK,
                    ),
                    Outcome(
                        name="No",
                        price=no_decimal,
                        implied_prob=round(no_prob, 6),
                        source=Source.PREDICTIT,
                        market_id=contract_id,
                        bookmaker="PredictIt",
                        side=BetSide.BACK,
                    ),
                ],
                raw={
                    "source_url": market_url,
                    "parent_name": parent_name,
                    "contract_name": contract_name,
                },
            ))

        return results

    async def close(self) -> None:
        await self._client.aclose()


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None

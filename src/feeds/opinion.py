from __future__ import annotations

"""
Opinion prediction market feed.

Opinion is a BNB-Chain CLOB prediction market, 3rd largest globally by volume.
API requires an apikey header even for read-only data.

Auth: set OPINION_API_KEY env var (apply at https://docs.opinion.trade)
Base: https://proxy.opinion.trade:8443/openapi
Rate: 15 req/s per key
"""
import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import httpx

from src.models import BetSide, Market, Outcome, Source

log = logging.getLogger(__name__)

BASE = "https://proxy.opinion.trade:8443/openapi"
PAGE_SIZE = 20
MIN_VOLUME_24H = 500.0   # skip ultra-thin markets
MAX_PAGES = 10           # cap at 200 markets to stay within rate budget


class OpinionFeed:
    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key or os.environ.get("OPINION_API_KEY", "")
        self._client = httpx.AsyncClient(
            base_url=BASE,
            headers={"apikey": self._api_key},
            timeout=20.0,
        )

    async def fetch(self) -> list[Market]:
        if not self._api_key:
            log.warning("Opinion: no API key set — skipping")
            return []
        try:
            return await self._fetch()
        except Exception:
            log.exception("Opinion fetch failed")
            return []

    async def _fetch(self) -> list[Market]:
        # Paginate through active binary markets sorted by 24h volume
        raw_markets: list[dict] = []
        for page in range(1, MAX_PAGES + 1):
            resp = await self._client.get("/market", params={
                "status": "activated",
                "marketType": 0,   # binary only
                "sortBy": 6,       # sort by 24h volume desc
                "page": page,
                "limit": PAGE_SIZE,
            })
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("result", {}).get("list", [])
            if not batch:
                break
            raw_markets.extend(batch)
            total = data.get("result", {}).get("total", 0)
            if len(raw_markets) >= total:
                break
            await asyncio.sleep(0.1)   # stay under 15 req/s

        # Filter by volume before fetching orderbooks
        raw_markets = [
            m for m in raw_markets
            if float(m.get("volume24h") or 0) >= MIN_VOLUME_24H
        ]

        if not raw_markets:
            return []

        # Fetch YES orderbooks in parallel (batches of 10 to respect rate limit)
        yes_books = await self._fetch_orderbooks(
            [m["yesTokenId"] for m in raw_markets]
        )

        markets: list[Market] = []
        for mkt, yes_book in zip(raw_markets, yes_books):
            parsed = self._parse(mkt, yes_book)
            if parsed:
                markets.append(parsed)

        log.info("Opinion: %d markets", len(markets))
        return markets

    async def _fetch_orderbooks(self, token_ids: list[str]) -> list[dict | None]:
        results: list[dict | None] = []
        batch_size = 10
        for i in range(0, len(token_ids), batch_size):
            batch = token_ids[i:i + batch_size]
            tasks = [self._get_orderbook(tid) for tid in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in batch_results:
                results.append(r if not isinstance(r, Exception) else None)
            await asyncio.sleep(0.08)   # 10 req per 80ms = 125 req/s → stay at ~12/s
        return results

    async def _get_orderbook(self, token_id: str) -> dict | None:
        try:
            resp = await self._client.get("/token/orderbook", params={"token_id": token_id})
            resp.raise_for_status()
            return resp.json().get("result")
        except Exception as exc:
            log.debug("Opinion orderbook error for %s: %s", token_id[:10], exc)
            return None

    def _parse(self, mkt: dict, yes_book: dict | None) -> Market | None:
        if not yes_book:
            return None

        bids = yes_book.get("bids", [])
        asks = yes_book.get("asks", [])
        if not bids or not asks:
            return None

        yes_ask_prob = float(asks[0]["price"])   # best ask = price to buy YES
        no_ask_prob  = 1.0 - float(bids[0]["price"])  # best bid on YES = best ask on NO

        if not (0.01 < yes_ask_prob < 0.99 and 0.01 < no_ask_prob < 0.99):
            return None

        yes_price = round(1 / yes_ask_prob, 4)
        no_price  = round(1 / no_ask_prob, 4)
        market_id = str(mkt["marketId"])
        title     = mkt.get("marketTitle", "")
        cutoff    = mkt.get("cutoffAt")
        vol24h    = float(mkt.get("volume24h") or 0)

        return Market(
            source=Source.OPINION,
            market_id=market_id,
            sport="prediction",
            event_name=title,
            commence_time=datetime.fromtimestamp(cutoff, tz=timezone.utc) if cutoff else None,
            home_team=None,
            away_team=None,
            market_type="binary",
            outcomes=[
                Outcome(
                    name="Yes",
                    price=yes_price,
                    implied_prob=yes_ask_prob,
                    source=Source.OPINION,
                    market_id=market_id,
                    bookmaker="Opinion",
                    side=BetSide.BACK,
                    available_volume=vol24h,
                ),
                Outcome(
                    name="No",
                    price=no_price,
                    implied_prob=no_ask_prob,
                    source=Source.OPINION,
                    market_id=market_id,
                    bookmaker="Opinion",
                    side=BetSide.BACK,
                    available_volume=vol24h,
                ),
            ],
            total_volume=vol24h,
            raw={"title": title, "yesTokenId": mkt.get("yesTokenId"), "noTokenId": mkt.get("noTokenId")},
        )

    async def close(self) -> None:
        await self._client.aclose()

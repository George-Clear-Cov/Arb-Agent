import logging
from datetime import datetime, timezone
from typing import Optional

import betfairlightweight
from betfairlightweight import APIClient
from betfairlightweight.filters import market_filter, price_projection

from src.feeds.base import BaseFeed
from src.models import BetSide, Market, Outcome, Source

log = logging.getLogger(__name__)

# Betfair market types we care about
MARKET_TYPES = ["MATCH_ODDS", "NEXT_GOAL", "ASIAN_HANDICAP", "BOTH_TEAMS_TO_SCORE"]


class BetfairFeed(BaseFeed):
    def __init__(self, username: str, password: str, app_key: str,
                 event_type_ids: list[str], max_results: int = 50):
        self.event_type_ids = event_type_ids
        self.max_results = max_results
        self._client: Optional[APIClient] = None
        self._creds = (username, password, app_key)

    def _ensure_client(self) -> APIClient:
        if self._client is None:
            username, password, app_key = self._creds
            self._client = betfairlightweight.APIClient(
                username=username, password=password, app_key=app_key
            )
            self._client.login()
        return self._client

    async def fetch(self) -> list[Market]:
        try:
            return await _run_sync(self._fetch_sync)
        except Exception:
            log.exception("Betfair fetch failed")
            return []

    def _fetch_sync(self) -> list[Market]:
        client = self._ensure_client()

        mkt_filter = market_filter(
            event_type_ids=self.event_type_ids,
            market_countries=["GB", "US", "AU"],
            market_type_codes=MARKET_TYPES,
            in_play_only=False,
        )

        catalogues = client.betting.list_market_catalogue(
            filter=mkt_filter,
            market_projection=["EVENT", "RUNNERS", "MARKET_START_TIME"],
            sort="FIRST_TO_START",
            max_results=self.max_results,
        )

        if not catalogues:
            return []

        market_ids = [c.market_id for c in catalogues]
        books = client.betting.list_market_book(
            market_ids=market_ids,
            price_projection={"priceData": ["EX_BEST_OFFERS"]},
        )

        cat_map = {c.market_id: c for c in catalogues}
        markets: list[Market] = []

        for book in books:
            cat = cat_map.get(book.market_id)
            if not cat:
                continue

            runner_map = {r.selection_id: r.runner_name for r in (cat.runners or [])}
            outcomes: list[Outcome] = []

            for runner in book.runners:
                name = runner_map.get(runner.selection_id, str(runner.selection_id))
                # Best available back price
                back_prices = runner.ex.available_to_back if runner.ex else []
                lay_prices = runner.ex.available_to_lay if runner.ex else []

                if back_prices:
                    bp = back_prices[0]
                    if bp.price > 1:
                        outcomes.append(Outcome(
                            name=name,
                            price=bp.price,
                            implied_prob=round(1 / bp.price, 6),
                            source=Source.BETFAIR,
                            market_id=book.market_id,
                            bookmaker="Betfair",
                            side=BetSide.BACK,
                            available_volume=bp.size,
                        ))
                if lay_prices:
                    lp = lay_prices[0]
                    if lp.price > 1:
                        outcomes.append(Outcome(
                            name=name,
                            price=lp.price,
                            implied_prob=round(1 / lp.price, 6),
                            source=Source.BETFAIR,
                            market_id=f"{book.market_id}_lay",
                            bookmaker="Betfair",
                            side=BetSide.LAY,
                            available_volume=lp.size,
                        ))

            if not outcomes:
                continue

            event = cat.event
            event_name = event.name if event else cat.market_name

            markets.append(Market(
                source=Source.BETFAIR,
                market_id=book.market_id,
                sport=str(cat.event_type.id if cat.event_type else "unknown"),
                event_name=event_name or "",
                commence_time=cat.market_start_time,
                home_team=None,
                away_team=None,
                market_type=cat.market_name or "unknown",
                outcomes=outcomes,
                raw={"market_id": book.market_id, "status": book.status},
            ))

        return markets

    async def close(self) -> None:
        if self._client:
            try:
                self._client.logout()
            except Exception:
                pass


async def _run_sync(fn):
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn)

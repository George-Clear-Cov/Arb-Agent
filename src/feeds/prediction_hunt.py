from __future__ import annotations

"""
Prediction Hunt unified feed.

Uses the /markets endpoint to bulk-fetch active markets from PredictIt,
Opinion, and ProphetX (platforms not covered by dedicated feeds).

Prices come in a 0–100 cent range; we convert to decimal odds.
We use the mid of yes_ask/yes_bid as the effective price.
Cursor-based pagination handles large result sets.
"""
import asyncio
import logging
from datetime import datetime
from typing import Optional

import httpx

from src.models import BetSide, Market, Outcome, Source

log = logging.getLogger(__name__)

BASE_URL = "https://www.predictionhunt.com/api/v2"

_PLATFORM_SOURCE: dict[str, Source] = {
    "predictit":  Source.PREDICTIT,
    "opinion":    Source.OPINION,
    "prophetx":   Source.PROPHETX,
    "predictfun": Source.PREDICTFUN,
}


class RateLimitedError(Exception):
    """Raised when PredictionHunt API returns 429 on all platforms."""


class PredictionHuntFeed:
    def __init__(
        self,
        api_key: str,
        platforms: Optional[list[str]] = None,
        page_size: int = 200,            # safe for free-tier burst limit
        rate_limit_delay: float = 1.5,  # seconds between page requests
        platform_cooldown: float = 3.0, # extra pause between platforms
        max_markets_per_platform: int = 2000,
    ):
        self._headers = {"X-API-Key": api_key}
        # Default: all platforms with live prices (ProphetX excluded — always null via PH API)
        self._platforms = platforms or ["predictit", "opinion", "predictfun"]
        self._page_size = page_size
        self._delay = rate_limit_delay
        self._platform_cooldown = platform_cooldown
        self._max_per_platform = max_markets_per_platform
        self._client: Optional[httpx.AsyncClient] = None
        # Cache last successful fetch per platform so rate-limited cycles
        # still return data instead of going blank.
        self._cache: dict[str, list[Market]] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=BASE_URL,
                headers=self._headers,
                timeout=20.0,
            )
        return self._client

    async def fetch(self) -> list[Market]:
        """Fetch all platforms.  Raises RateLimitedError if every platform returned 429."""
        client = await self._get_client()
        markets: list[Market] = []
        rate_limited_count = 0

        for i, platform in enumerate(self._platforms):
            source = _PLATFORM_SOURCE.get(platform)
            if not source:
                continue
            if i > 0:
                await asyncio.sleep(self._platform_cooldown)
            fetched, was_429 = await self._fetch_platform(client, platform, source)
            if was_429:
                rate_limited_count += 1
            if fetched:
                self._cache[platform] = fetched
                markets.extend(fetched)
                log.info("PredictionHunt[%s]: %d markets", platform, len(fetched))
            elif platform in self._cache:
                cached = self._cache[platform]
                markets.extend(cached)
                log.warning("PredictionHunt[%s]: rate-limited — using %d cached markets",
                            platform, len(cached))
            else:
                log.warning("PredictionHunt[%s]: 0 markets (no cache available)", platform)

        log.info("PredictionHunt total: %d markets", len(markets))

        # If every platform was rate-limited, signal the caller so it doesn't
        # count this as a genuine "empty" fetch and compound the backoff.
        if rate_limited_count == len(self._platforms):
            raise RateLimitedError("all platforms returned 429")

        return markets

    async def _fetch_platform(
        self, client: httpx.AsyncClient, platform: str, source: Source
    ) -> tuple[list[Market], bool]:
        """Fetch one platform.  Returns (markets, was_rate_limited).

        Returns immediately on 429 without retrying — the agent-level backoff
        controls when we come back.
        """
        results: list[Market] = []
        cursor: Optional[str] = None

        while True:
            params: dict = {
                "limit": self._page_size,
                "status": "active",
                "platform": platform,
            }
            if cursor:
                params["cursor"] = cursor

            try:
                resp = await client.get("/markets", params=params)
                if resp.status_code == 429:
                    log.warning("PredictionHunt[%s] rate limited (429) — aborting, will retry later", platform)
                    return [], True   # signal rate-limit to caller
                resp.raise_for_status()
                body = resp.json()
            except Exception as exc:
                log.error("PredictionHunt[%s] fetch error: %s: %r", platform, type(exc).__name__, str(exc))
                break

            for raw in body.get("markets", []):
                m = self._parse_market(raw, source)
                if m:
                    results.append(m)

            cursor = body.get("next_cursor")
            if not cursor or len(results) >= self._max_per_platform:
                break

            await asyncio.sleep(self._delay)

        return results, False

    def _parse_market(self, raw: dict, source: Source) -> Optional[Market]:
        price = raw.get("price", {})

        yes_price_cents = self._mid(price.get("yes_ask"), price.get("yes_bid"))
        no_price_cents  = self._mid(price.get("no_ask"),  price.get("no_bid"))

        # Fall back to last_price for yes if no bid/ask
        if yes_price_cents is None and price.get("last_price") is not None:
            yes_price_cents = price["last_price"]

        # Filter out near-resolved markets (< 3% or > 97% probability)
        MIN_CENTS = 3.0
        MAX_CENTS = 97.0

        if yes_price_cents is None or yes_price_cents < MIN_CENTS or yes_price_cents > MAX_CENTS:
            return None

        # Derive no_price from yes if missing (binary: P(no) = 1 - P(yes))
        if no_price_cents is None:
            no_price_cents = 100 - yes_price_cents

        if no_price_cents < MIN_CENTS or no_price_cents > MAX_CENTS:
            return None

        yes_prob = yes_price_cents / 100.0
        no_prob  = no_price_cents  / 100.0

        yes_decimal = round(1 / yes_prob, 4)
        no_decimal  = round(1 / no_prob,  4)

        market_id = str(raw.get("market_id", raw.get("id", "")))
        title: str = raw.get("title", "Unknown")
        category: str = raw.get("category", "prediction")

        exp = raw.get("expiration_date")
        expire_dt: Optional[datetime] = None
        if exp:
            try:
                expire_dt = datetime.fromisoformat(exp.replace("Z", "+00:00"))
            except Exception:
                pass

        volume = price.get("volume")
        liquidity = price.get("liquidity")
        total_vol = liquidity if liquidity is not None else volume

        # For PredictIt sub-markets, reconstruct the full parent question from
        # the source_url slug (e.g. ".../Who-will-win-the-2028-Republican-...")
        # so that fuzzy matching uses the full question, not just a candidate name.
        event_name = self._build_event_name(raw, title, source)

        return Market(
            source=source,
            market_id=market_id,
            sport=category,
            event_name=event_name,
            commence_time=expire_dt,
            home_team=None,
            away_team=None,
            market_type="binary",
            outcomes=[
                Outcome(
                    name="Yes",
                    price=yes_decimal,
                    implied_prob=round(yes_prob, 6),
                    source=source,
                    market_id=market_id,
                    bookmaker=raw.get("platform", "").title(),
                    side=BetSide.BACK,
                    available_volume=total_vol,
                ),
                Outcome(
                    name="No",
                    price=no_decimal,
                    implied_prob=round(no_prob, 6),
                    source=source,
                    market_id=market_id,
                    bookmaker=raw.get("platform", "").title(),
                    side=BetSide.BACK,
                    available_volume=total_vol,
                ),
            ],
            total_volume=total_vol,
            raw=raw,
        )

    @staticmethod
    def _build_event_name(raw: dict, title: str, source: Source) -> str:
        """
        For PredictIt sub-markets, extract the parent question from the source_url
        slug so the matcher uses the full question rather than just a candidate name.
        e.g. "JD Vance" → "Who will win the 2028 Republican presidential nomination: JD Vance"
        """
        if source != Source.PREDICTIT:
            return title
        url: str = raw.get("source_url", "")
        # URL form: .../markets/detail/{id}/{parent-slug}
        try:
            slug = url.rstrip("/").split("/")[-1]
            if slug and not slug.isdigit():
                parent = slug.replace("-", " ").title()
                if parent.lower() != title.lower():
                    return f"{parent}: {title}"
        except Exception:
            pass
        return title

    @staticmethod
    def _mid(ask: Optional[float], bid: Optional[float]) -> Optional[float]:
        if ask is not None and bid is not None:
            return (ask + bid) / 2
        if ask is not None:
            return ask
        if bid is not None:
            return bid
        return None

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

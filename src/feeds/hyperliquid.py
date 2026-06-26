from __future__ import annotations

"""
Hyperliquid HIP-4 Outcome Markets feed.

Binary prediction markets on Hyperliquid CLOB. Launched May 2, 2026.
Outcomes settle 0 or 1 USDC. YES + NO prices always sum to 1.0.

All reads are unauthenticated. Three POST calls to /info:
  - outcomeMeta  → market names/descriptions
  - allMids      → mid prices for all #XXXX coins
  - spotMetaAndAssetCtxs → 24h volume

Coin notation: #{10 * outcome_id + side}  (side 0=YES, 1=NO)
"""
import logging
import re
from datetime import datetime, timezone
from typing import Optional

import httpx

from src.feeds.feed_cache import CACHE_DIR, load_cache, save_cache
from src.models import BetSide, Market, Outcome, Source

log = logging.getLogger(__name__)

BASE = "https://api.hyperliquid.xyz/info"
HEADERS = {"Content-Type": "application/json"}

MIN_PROB = 0.02
MAX_PROB = 0.98
MIN_VOLUME_24H = 100.0  # USDC — HIP-4 is newer, lower threshold

_CACHE_FILE = CACHE_DIR / "hyperliquid_cache.json"


class HyperliquidFeed:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=20.0)
        self._disk_markets = load_cache(_CACHE_FILE, Source.HYPERLIQUID)

    async def fetch(self) -> list[Market]:
        try:
            result = await self._fetch()
            if result:
                self._disk_markets = result
                save_cache(_CACHE_FILE, result)
            return result or self._disk_markets
        except Exception:
            log.exception("Hyperliquid HIP-4 fetch failed")
            return self._disk_markets

    async def _fetch(self) -> list[Market]:
        # Three parallel-ish calls
        meta_resp, mids_resp, ctxs_resp = await _gather(
            self._client,
            {"type": "outcomeMeta"},
            {"type": "allMids"},
            {"type": "spotMetaAndAssetCtxs"},
        )

        outcomes_meta: list[dict] = meta_resp.get("outcomes", [])
        all_mids: dict[str, str] = mids_resp  # {"#XXXX": "0.1234"}

        # Build volume index: outcome_id → 24h notional USD
        vol_index: dict[int, float] = {}
        for ctx in ctxs_resp[1]:
            coin = ctx.get("coin", "")
            if not coin.startswith("#"):
                continue
            encoding = int(coin[1:])
            oid = encoding // 10
            side = encoding % 10
            if side == 0:   # only index YES side (both sides report same volume)
                vol_index[oid] = float(ctx.get("dayNtlVlm", 0) or 0)

        markets: list[Market] = []
        for outcome in outcomes_meta:
            oid = outcome["outcome"]
            yes_coin = f"#{10 * oid}"
            yes_mid_str = all_mids.get(yes_coin)
            if yes_mid_str is None:
                continue

            yes_mid = float(yes_mid_str)
            no_mid = 1.0 - yes_mid

            if not (MIN_PROB < yes_mid < MAX_PROB):
                continue

            vol24h = vol_index.get(oid, 0.0)
            if vol24h < MIN_VOLUME_24H:
                continue

            yes_price = round(1 / yes_mid, 4)
            no_price  = round(1 / no_mid, 4)
            name = outcome.get("name", "")
            desc = outcome.get("description", "")
            sport = _classify_sport(name, desc)
            expire = _parse_expiry(desc)
            mkt_id = str(oid)

            markets.append(Market(
                source=Source.HYPERLIQUID,
                market_id=mkt_id,
                sport=sport,
                event_name=name,
                commence_time=expire,
                home_team=None,
                away_team=None,
                market_type="binary",
                outcomes=[
                    Outcome(
                        name="Yes",
                        price=yes_price,
                        implied_prob=yes_mid,
                        source=Source.HYPERLIQUID,
                        market_id=mkt_id,
                        bookmaker="Hyperliquid",
                        side=BetSide.BACK,
                        available_volume=vol24h,
                    ),
                    Outcome(
                        name="No",
                        price=no_price,
                        implied_prob=no_mid,
                        source=Source.HYPERLIQUID,
                        market_id=mkt_id,
                        bookmaker="Hyperliquid",
                        side=BetSide.BACK,
                        available_volume=vol24h,
                    ),
                ],
                total_volume=vol24h,
                raw={"yes_coin": yes_coin, "description": desc},
            ))

        log.info("Hyperliquid HIP-4: %d markets", len(markets))
        return markets

    async def close(self) -> None:
        await self._client.aclose()


async def _gather(client: httpx.AsyncClient, *payloads: dict) -> list:
    import asyncio
    tasks = [client.post(BASE, json=p, headers=HEADERS) for p in payloads]
    responses = await asyncio.gather(*tasks)
    results = []
    for r in responses:
        r.raise_for_status()
        results.append(r.json())
    return results


# Pipe-delimited recurring market description e.g.:
# "class:priceBinary|underlying:BTC|expiry:20260623-0600|targetPrice:64209|period:1d"
_DESC_RE = re.compile(r"(\w+):([^|]+)")


def _parse_expiry(desc: str) -> Optional[datetime]:
    if "|" in desc:
        fields = dict(_DESC_RE.findall(desc))
        expiry_str = fields.get("expiry")
        if expiry_str:
            try:
                # "20260623-0600" → YYYYMMDD-HHMM UTC
                dt = datetime.strptime(expiry_str, "%Y%m%d-%H%M")
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                pass
    return None


def _classify_sport(name: str, desc: str) -> str:
    text = (name + " " + desc).lower()
    if any(w in text for w in ("btc", "eth", "sol", "crypto", "price", "bitcoin", "ethereum")):
        return "prediction"
    if any(w in text for w in ("world cup", "fifa", "nfl", "nba", "mlb", "nhl", "soccer",
                                "football", "basketball", "baseball", "hockey")):
        return "sports"
    return "prediction"

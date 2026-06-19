from __future__ import annotations

"""
Polymarket CLOB real-time WebSocket feed.

Subscribes to best_ask price updates for active game/prediction markets
and updates in-memory Market objects immediately on price change — no
polling delay.  Fires an asyncio.Event on each update so the detection
loop wakes up without waiting for its 30s sleep.

WebSocket endpoint: wss://ws-subscriptions-clob.polymarket.com/ws/market
Authentication: none (public orderbook data)

Message types received:
  book         — initial snapshot: bids/asks arrays
  price_change — live update: best_bid, best_ask per token
"""

import asyncio
import json
import logging
from typing import TYPE_CHECKING

import websockets

from src.models import Market

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
RECONNECT_DELAY = 3.0     # seconds between reconnect attempts
BATCH_SIZE = 500          # max tokens per subscription (WS limit)
MIN_PRICE_MOVE = 0.005    # only wake detection if price moved ≥ 0.5%


class PolymarketWSFeed:
    """
    Maintains a live WebSocket connection to Polymarket CLOB.

    Usage:
        feed = PolymarketWSFeed(detection_event)
        asyncio.create_task(feed.run(get_markets_fn))

    get_markets_fn: callable returning the current list of Polymarket Market
    objects.  Called on each reconnect to refresh subscriptions.
    """

    def __init__(self, detection_event: asyncio.Event) -> None:
        self._event = detection_event
        # token_id → Market (updated in-place on price change)
        self._token_to_market: dict[str, Market] = {}
        self._token_to_outcome_idx: dict[str, int] = {}
        self._running = False

    def _build_index(self, markets: list[Market]) -> None:
        self._token_to_market.clear()
        self._token_to_outcome_idx.clear()
        for m in markets:
            for i, o in enumerate(m.outcomes):
                if o.market_id:
                    self._token_to_market[o.market_id] = m
                    self._token_to_outcome_idx[o.market_id] = i

    def _apply_price_change(self, token_id: str, best_ask: float) -> bool:
        """Update outcome implied_prob/price. Returns True if change was significant."""
        m = self._token_to_market.get(token_id)
        if not m:
            return False
        idx = self._token_to_outcome_idx.get(token_id, -1)
        if idx < 0 or idx >= len(m.outcomes):
            return False
        o = m.outcomes[idx]
        old_prob = o.implied_prob
        if best_ask <= 0 or best_ask >= 1:
            return False
        moved = abs(best_ask - old_prob) >= MIN_PRICE_MOVE
        if moved:
            o.implied_prob = best_ask
            o.price = round(1 / best_ask, 4)
        return moved

    async def run(self, get_markets_fn) -> None:
        """Connect, subscribe, and stream updates forever with auto-reconnect."""
        self._running = True
        while self._running:
            try:
                await self._connect_and_stream(get_markets_fn)
            except Exception as exc:
                log.warning("Polymarket WS disconnected: %s — reconnecting in %.0fs", exc, RECONNECT_DELAY)
            await asyncio.sleep(RECONNECT_DELAY)

    async def _connect_and_stream(self, get_markets_fn) -> None:
        markets = get_markets_fn()
        self._build_index(markets)

        all_token_ids = list(self._token_to_market.keys())
        if not all_token_ids:
            log.debug("Polymarket WS: no token IDs yet, waiting...")
            await asyncio.sleep(10)
            return

        log.info("Polymarket WS: subscribing to %d tokens", len(all_token_ids))

        async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=30) as ws:
            # Subscribe in batches (WS has a message size limit)
            for i in range(0, len(all_token_ids), BATCH_SIZE):
                batch = all_token_ids[i : i + BATCH_SIZE]
                await ws.send(json.dumps({"assets_ids": batch, "type": "market"}))

            updates = 0
            async for raw in ws:
                try:
                    msgs = json.loads(raw)
                    if not isinstance(msgs, list):
                        msgs = [msgs]
                    for msg in msgs:
                        fired = self._handle_message(msg)
                        if fired:
                            updates += 1
                            self._event.set()
                except Exception as exc:
                    log.debug("Polymarket WS parse error: %s", exc)

    def _handle_message(self, msg: dict) -> bool:
        """Process one WS message. Returns True if a significant price moved."""
        event_type = msg.get("event_type")

        if event_type == "price_change":
            # Single asset price_change
            token = msg.get("asset_id", "")
            best_ask_str = msg.get("best_ask")
            if best_ask_str and token:
                try:
                    return self._apply_price_change(token, float(best_ask_str))
                except (ValueError, TypeError):
                    pass
            return False

        # Nested price_changes array (batch update)
        changes = msg.get("price_changes")
        if changes:
            fired = False
            for change in changes:
                token = change.get("asset_id", "")
                best_ask_str = change.get("best_ask")
                if best_ask_str and token:
                    try:
                        if self._apply_price_change(token, float(best_ask_str)):
                            fired = True
                    except (ValueError, TypeError):
                        pass
            return fired

        # Initial book snapshot — use best ask from asks array
        asks = msg.get("asks")
        token = msg.get("asset_id", "")
        if asks and token:
            try:
                best = min(float(a["price"]) for a in asks if a.get("price"))
                return self._apply_price_change(token, best)
            except (ValueError, TypeError, KeyError):
                pass

        return False

    def stop(self) -> None:
        self._running = False

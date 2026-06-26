from __future__ import annotations

"""
Kalshi real-time WebSocket feed.

Subscribes to the 'ticker' channel for all active near-term Kalshi markets
and updates in-memory Market objects on price change — no polling delay.
Fires an asyncio.Event so the detection loop wakes up immediately.

WebSocket endpoint: wss://external-api-ws.kalshi.com/trade-api/ws/v2
Authentication: RSA-PSS headers (same as HTTP executor)
"""

import asyncio
import base64
import json
import logging
import time
from pathlib import Path
from typing import Callable

import websockets
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from src.models import Market

log = logging.getLogger(__name__)

WS_URL       = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
WS_PATH      = "/trade-api/ws/v2"
RECONNECT_DELAY = 5.0
BATCH_SIZE   = 200   # tickers per subscribe message (keep messages small)
MIN_PRICE_MOVE = 0.005


def _load_key(path: str):
    return load_pem_private_key(Path(path).expanduser().read_bytes(), password=None)


def _auth_headers(api_key: str, private_key) -> dict:
    ts_ms = int(time.time() * 1000)
    msg   = f"{ts_ms}GET{WS_PATH}".encode()
    sig   = private_key.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY":       api_key,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "KALSHI-ACCESS-TIMESTAMP": str(ts_ms),
    }


class KalshiWSFeed:
    def __init__(self, api_key: str, private_key_path: str, detection_event: asyncio.Event) -> None:
        self._api_key      = api_key
        self._private_key  = _load_key(private_key_path)
        self._event        = detection_event
        self._ticker_to_market: dict[str, Market] = {}
        self._running      = False

    def _build_index(self, markets: list[Market]) -> None:
        self._ticker_to_market.clear()
        for m in markets:
            self._ticker_to_market[m.market_id] = m

    def _apply_update(self, ticker: str, yes_ask_raw, no_ask_raw) -> bool:
        m = self._ticker_to_market.get(ticker)
        if not m or len(m.outcomes) < 2:
            return False
        fired = False
        for raw, idx in ((yes_ask_raw, 0), (no_ask_raw, 1)):
            if raw is None:
                continue
            try:
                prob = float(raw)
            except (TypeError, ValueError):
                continue
            if not (0 < prob < 1):
                continue
            o = m.outcomes[idx]
            if abs(prob - o.implied_prob) >= MIN_PRICE_MOVE:
                o.implied_prob = prob
                o.price        = round(1 / prob, 4)
                fired = True
        return fired

    async def run(self, get_markets_fn: Callable[[], list[Market]]) -> None:
        self._running = True
        while self._running:
            try:
                await self._connect_and_stream(get_markets_fn)
            except Exception as exc:
                log.warning("Kalshi WS disconnected: %s — reconnecting in %.0fs", exc, RECONNECT_DELAY)
            await asyncio.sleep(RECONNECT_DELAY)

    async def _connect_and_stream(self, get_markets_fn: Callable[[], list[Market]]) -> None:
        markets = get_markets_fn()
        self._build_index(markets)
        tickers = list(self._ticker_to_market.keys())
        if not tickers:
            log.debug("Kalshi WS: no tickers yet, waiting 10s")
            await asyncio.sleep(10)
            return

        log.info("Kalshi WS: connecting, subscribing to %d tickers", len(tickers))
        headers = _auth_headers(self._api_key, self._private_key)

        async with websockets.connect(
            WS_URL,
            extra_headers=headers,
            ping_interval=20,
            ping_timeout=30,
        ) as ws:
            msg_id = 1
            for i in range(0, len(tickers), BATCH_SIZE):
                batch = tickers[i : i + BATCH_SIZE]
                await ws.send(json.dumps({
                    "id": msg_id,
                    "cmd": "subscribe",
                    "params": {"channels": ["ticker"], "market_tickers": batch},
                }))
                msg_id += 1

            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    if self._handle_message(msg):
                        self._event.set()
                except Exception as exc:
                    log.debug("Kalshi WS parse error: %s", exc)

    def _handle_message(self, msg: dict) -> bool:
        if msg.get("type") != "ticker":
            return False
        data   = msg.get("msg", {})
        ticker = data.get("market_ticker", "")
        if not ticker:
            return False
        # Kalshi sends yes_ask / no_ask as strings (cents × 100, so 0–1 range)
        return self._apply_update(
            ticker,
            data.get("yes_ask") or data.get("yes_ask_dollars"),
            data.get("no_ask")  or data.get("no_ask_dollars"),
        )

    def stop(self) -> None:
        self._running = False

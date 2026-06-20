"""
Polymarket execution engine — places real limit orders via the Polymarket CLOB API.

Auth flow:
    L1: Sign a timestamp+nonce with your Ethereum private key → get API key/secret
    L2: Sign each request with the API key credentials (HMAC-SHA256)

Order flow:
    1. Build an Order struct (EIP-712 typed data on Polygon)
    2. Sign with wallet private key
    3. POST to /order with the signed payload

Contract addresses (Polygon mainnet):
    CTF Exchange: 0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982
    Neg Risk Exchange: 0xC5d563A36AE78145C45a50134d48A1215220f80a

Amounts:
    Polymarket uses 6-decimal USDC (like USDC on Polygon).
    makerAmount = USDC to spend  (e.g. $10.00 = 10_000_000)
    takerAmount = contracts to receive (also 6 decimals, e.g. 10 contracts = 10_000_000)
    price = makerAmount / takerAmount  (e.g. 0.65 = buying YES at 65 cents)
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import httpx
from eth_account import Account
from eth_account.messages import encode_defunct
from eth_account.structured_data.hashing import hash_message

from src.models import ArbLeg, ArbOpportunity, Source

log = logging.getLogger(__name__)

CLOB_HOST          = "https://clob.polymarket.com"
CHAIN_ID           = 137   # Polygon mainnet
CTF_EXCHANGE       = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982"
NEG_RISK_EXCHANGE  = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
USDC_DECIMALS      = 6     # USDC on Polygon has 6 decimals

FILL_TIMEOUT       = 60    # seconds to wait for fill
FILL_POLL_INTERVAL = 3.0

# EIP-712 domain and type for Polymarket orders
_DOMAIN = {
    "name": "Polymarket CTF Exchange",
    "version": "1",
    "chainId": CHAIN_ID,
    "verifyingContract": CTF_EXCHANGE,
}

_ORDER_TYPES = {
    "EIP712Domain": [
        {"name": "name",              "type": "string"},
        {"name": "version",           "type": "string"},
        {"name": "chainId",           "type": "uint256"},
        {"name": "verifyingContract", "type": "address"},
    ],
    "Order": [
        {"name": "salt",          "type": "uint256"},
        {"name": "maker",         "type": "address"},
        {"name": "signer",        "type": "address"},
        {"name": "taker",         "type": "address"},
        {"name": "tokenId",       "type": "uint256"},
        {"name": "makerAmount",   "type": "uint256"},
        {"name": "takerAmount",   "type": "uint256"},
        {"name": "expiration",    "type": "uint256"},
        {"name": "nonce",         "type": "uint256"},
        {"name": "feeRateBps",    "type": "uint256"},
        {"name": "side",          "type": "uint8"},
        {"name": "signatureType", "type": "uint8"},
    ],
}

_SIDE_BUY  = 0
_SIDE_SELL = 1
_SIG_EOA   = 0  # standard EOA signature


@dataclass
class PolyOrderResult:
    leg_idx: int
    token_id: str
    side: int            # 0=BUY 1=SELL
    size: float          # contracts
    price: float         # 0–1
    order_id: Optional[str] = None
    status: str = "pending"
    filled_size: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status in ("MATCHED", "partially filled")


@dataclass
class PolyExecutionResult:
    arb_id: str
    leg_results: list[PolyOrderResult]
    executed_at: datetime = field(default_factory=datetime.utcnow)
    fully_filled: bool = False
    hedged: bool = False
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.fully_filled


class PolymarketExecutor:
    def __init__(self, private_key: str) -> None:
        # Normalise key
        if not private_key.startswith("0x"):
            private_key = "0x" + private_key
        self._account  = Account.from_key(private_key)
        self._priv_key = private_key
        self._address  = self._account.address
        self._client   = httpx.AsyncClient(base_url=CLOB_HOST, timeout=20.0)
        self._api_key:    Optional[str] = None
        self._api_secret: Optional[str] = None
        self._api_passphrase: Optional[str] = None

    # ------------------------------------------------------------------
    # L1 auth — derive API credentials from wallet signature
    # ------------------------------------------------------------------

    async def init_credentials(self) -> None:
        """Derive Polymarket CLOB API credentials from wallet signature (L1 auth)."""
        ts    = str(int(time.time()))
        nonce = "0"
        msg   = encode_defunct(text=f"{ts}{nonce}")
        sig   = self._account.sign_message(msg)
        sig_hex = sig.signature.hex()
        if not sig_hex.startswith("0x"):
            sig_hex = "0x" + sig_hex

        resp = await self._client.post(
            "/auth/api-key",
            json={"address": self._address, "signature": sig_hex,
                  "timestamp": ts, "nonce": nonce},
        )
        resp.raise_for_status()
        data = resp.json()
        self._api_key        = data["apiKey"]
        self._api_secret     = data["secret"]
        self._api_passphrase = data["passphrase"]
        log.info("Polymarket: API credentials initialised for %s", self._address[:10])

    def _l2_headers(self, method: str, path: str, body: str = "") -> dict:
        """HMAC-SHA256 L2 auth headers for authenticated requests."""
        ts  = str(int(time.time() * 1000))
        msg = ts + method.upper() + path + body
        sig = hmac.new(
            self._api_secret.encode(),
            msg.encode(),
            hashlib.sha256,
        ).hexdigest()
        return {
            "POLY-ADDRESS":    self._address,
            "POLY-SIGNATURE":  sig,
            "POLY-TIMESTAMP":  ts,
            "POLY-API-KEY":    self._api_key,
            "POLY-PASSPHRASE": self._api_passphrase,
            "Content-Type":    "application/json",
        }

    # ------------------------------------------------------------------
    # EIP-712 order signing
    # ------------------------------------------------------------------

    def _sign_order(self, token_id: str, side: int,
                    maker_amount: int, taker_amount: int) -> dict:
        """Build and sign an EIP-712 order. Returns the full order payload."""
        salt = random.randint(1, 2**256 - 1)
        order_data = {
            "salt":          salt,
            "maker":         self._address,
            "signer":        self._address,
            "taker":         "0x0000000000000000000000000000000000000000",
            "tokenId":       int(token_id),
            "makerAmount":   maker_amount,
            "takerAmount":   taker_amount,
            "expiration":    0,
            "nonce":         0,
            "feeRateBps":    0,
            "side":          side,
            "signatureType": _SIG_EOA,
        }
        structured = {
            "types":       _ORDER_TYPES,
            "primaryType": "Order",
            "domain":      _DOMAIN,
            "message":     order_data,
        }
        signed  = self._account.sign_typed_data(full_message=structured)
        sig_hex = signed.signature.hex()
        if not sig_hex.startswith("0x"):
            sig_hex = "0x" + sig_hex

        return {
            "order":     {k: str(v) for k, v in order_data.items()},
            "signature": sig_hex,
            "owner":     self._address,
        }

    # ------------------------------------------------------------------
    # Order placement + polling
    # ------------------------------------------------------------------

    async def place_order(self, token_id: str, side: int,
                          price: float, size: float, leg_idx: int) -> PolyOrderResult:
        """Place a GTC limit order on the Polymarket CLOB."""
        if not self._api_key:
            await self.init_credentials()

        # Amounts in 6-decimal USDC units
        # BUY:  makerAmount = USDC spent,  takerAmount = contracts received
        # SELL: makerAmount = contracts given, takerAmount = USDC received
        size_units  = int(round(size * 10 ** USDC_DECIMALS))
        price_clamped = max(0.01, min(0.99, price))

        if side == _SIDE_BUY:
            maker_amount = int(round(price_clamped * size_units))
            taker_amount = size_units
        else:
            maker_amount = size_units
            taker_amount = int(round(price_clamped * size_units))

        result = PolyOrderResult(
            leg_idx=leg_idx, token_id=token_id,
            side=side, size=size, price=price_clamped,
        )

        payload      = self._sign_order(token_id, side, maker_amount, taker_amount)
        payload_json = json.dumps(payload)

        log.info(
            "Polymarket: placing %s %s contracts @ %.4f (token %s...) arb leg %d",
            "BUY" if side == _SIDE_BUY else "SELL", size, price_clamped,
            token_id[:10], leg_idx,
        )

        try:
            resp = await self._client.post(
                "/order",
                content=payload_json,
                headers=self._l2_headers("POST", "/order", payload_json),
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            result.status = "error"
            result.error  = f"HTTP {e.response.status_code}: {e.response.text}"
            log.error("Polymarket order failed: %s", result.error)
            return result
        except Exception as e:
            result.status = "error"
            result.error  = str(e)
            return result

        result.order_id = data.get("orderID")
        result.status   = data.get("status", "pending")

        if result.status == "MATCHED":
            log.info("Polymarket order %s filled immediately", result.order_id)
            return result

        if result.order_id:
            result = await self._poll_fill(result)

        return result

    async def _poll_fill(self, result: PolyOrderResult) -> PolyOrderResult:
        deadline = time.monotonic() + FILL_TIMEOUT
        while time.monotonic() < deadline:
            await asyncio.sleep(FILL_POLL_INTERVAL)
            try:
                resp = await self._client.get(
                    f"/order/{result.order_id}",
                    headers=self._l2_headers("GET", f"/order/{result.order_id}"),
                )
                resp.raise_for_status()
                data = resp.json()
                result.status      = data.get("status", result.status)
                result.filled_size = float(data.get("size_matched", 0) or 0)
            except Exception as e:
                log.warning("Polymarket poll_fill %s: %s", result.order_id, e)
                continue

            if result.status in ("MATCHED", "partially filled"):
                log.info("Polymarket order %s %s", result.order_id, result.status)
                return result
            if result.status == "CANCELED":
                log.warning("Polymarket order %s was cancelled", result.order_id)
                return result

        # Timeout — cancel
        log.warning("Polymarket order %s timed out — cancelling", result.order_id)
        try:
            await self._client.delete(
                f"/order/{result.order_id}",
                headers=self._l2_headers("DELETE", f"/order/{result.order_id}"),
            )
        except Exception as e:
            log.warning("Polymarket cancel failed: %s", e)
        result.status = "CANCELED"
        return result

    async def execute_arb(
        self, arb: ArbOpportunity,
        max_stake: float = 500.0,
        dry_run: bool = False,
    ) -> PolyExecutionResult:
        poly_legs = [
            (i, leg) for i, leg in enumerate(arb.legs)
            if leg.source == Source.POLYMARKET
        ]
        if not poly_legs:
            return PolyExecutionResult(arb_id=arb.id, leg_results=[],
                                       error="No Polymarket legs")

        # Validate token IDs are present
        missing = [(i, leg) for i, leg in poly_legs if not leg.token_id]
        if missing:
            return PolyExecutionResult(
                arb_id=arb.id, leg_results=[],
                error=f"Missing token_id on {len(missing)} leg(s) — feed data incomplete",
            )

        # Cap stake
        total = sum(leg.stake for _, leg in poly_legs)
        if total > max_stake:
            scale = max_stake / total
            poly_legs = [(i, _scale_leg(leg, scale)) for i, leg in poly_legs]

        if dry_run:
            for i, leg in poly_legs:
                log.info("[DRY RUN] Polymarket: %s token %s size %.2f @ %.4f",
                         "BUY", leg.token_id[:10], leg.stake, 1 / leg.price)
            return PolyExecutionResult(arb_id=arb.id, leg_results=[], error="dry_run")

        # Determine BUY/SELL per leg
        # For a YES leg at price p: BUY YES token at p
        # For a NO leg at price p: BUY NO token at p (NO token = 1 - YES price)
        tasks = []
        for i, leg in poly_legs:
            implied = 1.0 / leg.price
            tasks.append(self.place_order(
                token_id=leg.token_id,
                side=_SIDE_BUY,
                price=implied,
                size=leg.stake / implied,
                leg_idx=i,
            ))

        leg_results: list[PolyOrderResult] = await asyncio.gather(*tasks)
        exec_result = PolyExecutionResult(arb_id=arb.id, leg_results=list(leg_results))

        filled = [r for r in leg_results if r.ok]
        failed = [r for r in leg_results if not r.ok]

        if len(filled) == len(leg_results):
            exec_result.fully_filled = True
            log.info("Polymarket arb %s fully executed: %d legs", arb.id, len(filled))
            return exec_result

        if not filled:
            exec_result.error = f"All {len(failed)} legs failed"
            return exec_result

        # Partial: cancel filled orders
        exec_result.hedged = True
        cancel_tasks = [
            self._client.delete(
                f"/order/{r.order_id}",
                headers=self._l2_headers("DELETE", f"/order/{r.order_id}"),
            )
            for r in filled if r.order_id
        ]
        await asyncio.gather(*cancel_tasks, return_exceptions=True)
        log.error("Polymarket arb %s: partial fill — MANUAL HEDGE REQUIRED", arb.id)
        return exec_result

    async def get_balance(self) -> float:
        """Return available USDC balance in dollars."""
        if not self._api_key:
            await self.init_credentials()
        resp = await self._client.get(
            "/balance-allowance",
            params={"asset_type": "USDC", "signature_type": 0},
            headers=self._l2_headers("GET", "/balance-allowance?asset_type=USDC&signature_type=0"),
        )
        resp.raise_for_status()
        data = resp.json()
        return float(data.get("balance", 0)) / 10 ** USDC_DECIMALS

    async def close(self) -> None:
        await self._client.aclose()


def _scale_leg(leg: ArbLeg, scale: float) -> ArbLeg:
    from src.models import ArbLeg as AL
    return AL(
        source=leg.source, market_id=leg.market_id, bookmaker=leg.bookmaker,
        outcome_name=leg.outcome_name, price=leg.price,
        effective_price=leg.effective_price, stake=round(leg.stake * scale, 2),
        side=leg.side, token_id=leg.token_id,
    )

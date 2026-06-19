from __future__ import annotations

"""
Kalshi execution engine — places real limit orders via the Kalshi REST API v2.

Authentication:
    Portfolio endpoints require RSA-PKCS1v15 + SHA256 signing, not the simple
    Token header (which is read-only market-data access only).

    Headers sent on every request:
        KALSHI-ACCESS-KEY        : API key UUID from .env
        KALSHI-ACCESS-SIGNATURE  : base64(RSA-PKCS1v15-SHA256(ts + METHOD + /path))
        KALSHI-ACCESS-TIMESTAMP  : millisecond timestamp as string

Order math:
    Kalshi contracts pay $1 if the outcome wins.
    Each contract costs `price_cents / 100` dollars.
    To bet `stake` dollars at price `price_cents`:
        count = round(stake * 100 / price_cents)

    We use limit orders at the current ask so we're always a maker (0% fee on
    prediction markets, though Kalshi charges a small platform fee on profit).

Execution flow:
    execute_arb() fires both legs concurrently via asyncio.gather.
    If leg 1 fills but leg 2 fails, we attempt to cancel leg 1.  Since Kalshi
    limit orders rest in the book before matching, cancellation is usually
    possible in the brief window between placement and fill.  If cancellation
    fails (already filled), the position is flagged as a one-legged hedge
    that requires manual resolution.
"""

import asyncio
import base64
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from src.models import ArbLeg, ArbOpportunity, Source

log = logging.getLogger(__name__)

# How long to poll for a fill before giving up and cancelling
FILL_TIMEOUT_SECONDS = 45
FILL_POLL_INTERVAL   = 2.0   # seconds between status checks


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class OrderResult:
    leg_idx: int
    market_id: str
    side: str                    # "yes" or "no"
    count: int                   # contracts placed
    price_cents: int             # price paid per contract in cents
    client_order_id: str
    order_id: Optional[str] = None
    status: str = "pending"      # pending | resting | filled | partially_filled | canceled | error
    filled_count: int = 0
    error: Optional[str] = None
    raw_response: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in ("filled", "partially_filled")

    @property
    def stake_dollars(self) -> float:
        return round(self.count * self.price_cents / 100, 2)


@dataclass
class ExecutionResult:
    arb_id: str
    leg_results: list[OrderResult]
    executed_at: datetime = field(default_factory=datetime.utcnow)
    fully_filled: bool = False
    partial: bool = False       # one leg filled, other didn't
    hedged: bool = False        # one-legged — manual intervention needed
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.fully_filled


# ---------------------------------------------------------------------------
# RSA signing
# ---------------------------------------------------------------------------

def _load_private_key(pem_path: str):
    """Load RSA private key from PEM file."""
    path = Path(pem_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Kalshi private key not found at {path}.\n"
            "Run setup: generate a key pair and upload the public key to "
            "https://app.kalshi.com/account → API Keys."
        )
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def _make_auth_headers(private_key, api_key: str, method: str, path: str) -> dict:
    """Build Kalshi RSA-signed auth headers for a single request."""
    ts_ms = int(time.time() * 1000)
    msg   = f"{ts_ms}{method.upper()}{path}".encode()

    signature = private_key.sign(msg, padding.PKCS1v15(), hashes.SHA256())
    sig_b64   = base64.b64encode(signature).decode()

    return {
        "KALSHI-ACCESS-KEY":       api_key,
        "KALSHI-ACCESS-SIGNATURE": sig_b64,
        "KALSHI-ACCESS-TIMESTAMP": str(ts_ms),
        "Content-Type":            "application/json",
    }


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class KalshiExecutor:
    def __init__(self, api_key: str, private_key_path: str, base_url: str) -> None:
        self._api_key     = api_key
        self._private_key = _load_private_key(private_key_path)
        self._base_url    = base_url.rstrip("/")
        self._client      = httpx.AsyncClient(base_url=self._base_url, timeout=15.0)

    # ------------------------------------------------------------------
    # Internal request helper
    # ------------------------------------------------------------------

    def _auth(self, method: str, path: str) -> dict:
        return _make_auth_headers(self._private_key, self._api_key, method, path)

    async def _get(self, path: str) -> dict:
        resp = await self._client.get(path, headers=self._auth("GET", path))
        resp.raise_for_status()
        return resp.json()

    async def _post(self, path: str, body: dict) -> dict:
        resp = await self._client.post(path, json=body, headers=self._auth("POST", path))
        resp.raise_for_status()
        return resp.json()

    async def _delete(self, path: str) -> dict:
        resp = await self._client.delete(path, headers=self._auth("DELETE", path))
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_balance(self) -> float:
        """Return available balance in dollars."""
        data = await self._get("/portfolio/balance")
        cents = data.get("balance", 0)
        return cents / 100.0

    async def get_order(self, order_id: str) -> dict:
        return await self._get(f"/portfolio/orders/{order_id}")

    async def cancel_order(self, order_id: str) -> bool:
        """Attempt to cancel an order.  Returns True if successfully cancelled."""
        try:
            data = await self._delete(f"/portfolio/orders/{order_id}")
            status = data.get("order", {}).get("status", "")
            return status in ("canceled", "cancelled")
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (404, 409):
                # 404 = already settled / never existed, 409 = already filled
                log.warning("cancel_order %s: %s — order may already be filled", order_id, e)
                return False
            raise

    async def execute_leg(self, leg: ArbLeg, arb_id: str, leg_idx: int) -> OrderResult:
        """Place a single limit order for one leg of an arb.

        Polls for a fill up to FILL_TIMEOUT_SECONDS, then cancels if still resting.
        """
        side = leg.outcome_name.lower()  # "yes" or "no"

        # Convert decimal odds to cent price.  leg.price is raw odds (e.g. 1.4286
        # for 70-cent YES).  implied_prob = 1/price.
        implied_prob = 1.0 / leg.price
        price_cents  = round(implied_prob * 100)
        price_cents  = max(1, min(99, price_cents))

        # Number of contracts.  Each costs price_cents/100 dollars; pays $1 on win.
        count = max(1, round(leg.stake * 100 / price_cents))

        client_order_id = f"arb-{arb_id}-leg{leg_idx}-{uuid.uuid4().hex[:6]}"

        order_body: dict = {
            "ticker":          leg.market_id,
            "client_order_id": client_order_id,
            "type":            "limit",
            "action":          "buy",
            "side":            side,
            "count":           count,
        }
        if side == "yes":
            order_body["yes_price"] = price_cents
        else:
            order_body["no_price"] = price_cents

        result = OrderResult(
            leg_idx=leg_idx,
            market_id=leg.market_id,
            side=side,
            count=count,
            price_cents=price_cents,
            client_order_id=client_order_id,
        )

        log.info(
            "Placing Kalshi order: %s %s %d contracts @ %d¢  (stake $%.2f arb=%s)",
            leg.market_id, side.upper(), count, price_cents, leg.stake, arb_id,
        )

        try:
            data = await self._post("/portfolio/orders", order_body)
        except httpx.HTTPStatusError as e:
            result.status = "error"
            result.error  = f"HTTP {e.response.status_code}: {e.response.text}"
            log.error("Kalshi order failed [%s]: %s", leg.market_id, result.error)
            return result
        except Exception as e:
            result.status = "error"
            result.error  = str(e)
            log.error("Kalshi order exception [%s]: %s", leg.market_id, e)
            return result

        result.raw_response = data
        order = data.get("order", {})
        result.order_id = order.get("order_id") or order.get("id")
        result.status   = order.get("status", "resting")
        result.filled_count = order.get("filled_count") or order.get("num_fills", 0)

        if result.status == "filled":
            log.info(
                "Kalshi order %s filled immediately: %s %s %d contracts",
                result.order_id, leg.market_id, side.upper(), count,
            )
            return result

        # Poll for fill
        if result.order_id:
            result = await self._poll_fill(result)

        return result

    async def _poll_fill(self, result: OrderResult) -> OrderResult:
        """Poll /portfolio/orders/{id} until filled, timeout, or error."""
        deadline = time.monotonic() + FILL_TIMEOUT_SECONDS

        while time.monotonic() < deadline:
            await asyncio.sleep(FILL_POLL_INTERVAL)
            try:
                data  = await self.get_order(result.order_id)
                order = data.get("order", data)
                result.status       = order.get("status", result.status)
                result.filled_count = order.get("filled_count") or order.get("num_fills", result.filled_count)
            except Exception as e:
                log.warning("poll_fill %s: %s", result.order_id, e)
                continue

            if result.status in ("filled", "partially_filled"):
                log.info(
                    "Kalshi order %s %s: %d/%d contracts filled",
                    result.order_id, result.status, result.filled_count, result.count,
                )
                return result

            if result.status in ("canceled", "cancelled"):
                log.warning("Kalshi order %s was cancelled externally", result.order_id)
                return result

        # Timed out — cancel the resting order
        log.warning(
            "Kalshi order %s not filled within %ds — cancelling",
            result.order_id, FILL_TIMEOUT_SECONDS,
        )
        cancelled = await self.cancel_order(result.order_id)

        if cancelled:
            result.status = "canceled"
        else:
            # Couldn't cancel — may have just filled; re-check one more time
            try:
                data  = await self.get_order(result.order_id)
                order = data.get("order", data)
                result.status       = order.get("status", result.status)
                result.filled_count = order.get("filled_count") or order.get("num_fills", result.filled_count)
            except Exception:
                pass

        return result

    async def execute_arb(
        self,
        arb: ArbOpportunity,
        max_stake: float = 500.0,
        dry_run: bool = False,
    ) -> ExecutionResult:
        """Execute both legs of an arb concurrently.

        Args:
            arb:       The opportunity to trade.
            max_stake: Hard cap on total dollars at risk per arb.
            dry_run:   If True, log what would be placed but don't actually place.

        Leg failure handling:
            - Both legs fire simultaneously.
            - If both fill → ExecutionResult.fully_filled = True.
            - If leg 1 fills but leg 2 fails → attempt cancel of leg 1.
              If cancel succeeds → no net position.
              If cancel fails (already filled) → ExecutionResult.hedged = True,
              meaning we hold a one-legged position that needs manual hedging.
        """
        kalshi_legs = [
            (i, leg)
            for i, leg in enumerate(arb.legs)
            if leg.source in (Source.KALSHI, Source.KALSHI_SPORTS)
        ]

        if not kalshi_legs:
            return ExecutionResult(
                arb_id=arb.id,
                leg_results=[],
                error="No Kalshi legs in this arb",
            )

        # Safety: cap stake
        total_kalshi_stake = sum(leg.stake for _, leg in kalshi_legs)
        if total_kalshi_stake > max_stake:
            scale = max_stake / total_kalshi_stake
            kalshi_legs = [(i, _scale_leg(leg, scale)) for i, leg in kalshi_legs]

        if dry_run:
            for i, leg in kalshi_legs:
                side = leg.outcome_name.lower()
                prob = 1.0 / leg.price
                cents = round(prob * 100)
                count = max(1, round(leg.stake * 100 / cents))
                log.info(
                    "[DRY RUN] Would place: %s %s %d contracts @ %d¢  stake=$%.2f",
                    leg.market_id, side.upper(), count, cents, leg.stake,
                )
            return ExecutionResult(arb_id=arb.id, leg_results=[], error="dry_run")

        # Fire all Kalshi legs concurrently
        tasks = [
            self.execute_leg(leg, arb.id, i)
            for i, leg in kalshi_legs
        ]
        leg_results: list[OrderResult] = await asyncio.gather(*tasks, return_exceptions=False)

        exec_result = ExecutionResult(arb_id=arb.id, leg_results=list(leg_results))

        filled  = [r for r in leg_results if r.ok]
        failed  = [r for r in leg_results if not r.ok]

        if len(filled) == len(leg_results):
            exec_result.fully_filled = True
            log.info(
                "Arb %s fully executed: %d legs, stake $%.2f expected margin %.1f%%",
                arb.id, len(filled), arb.total_stake, arb.margin * 100,
            )
            return exec_result

        if not filled:
            exec_result.error = f"All {len(failed)} legs failed"
            log.error("Arb %s: all legs failed — %s", arb.id, [r.error for r in failed])
            return exec_result

        # Partial: some legs filled, some didn't → try to cancel filled ones
        exec_result.partial = True
        log.warning(
            "Arb %s: %d legs filled, %d failed — attempting cancellation of filled legs",
            arb.id, len(filled), len(failed),
        )

        cancel_tasks = [
            self.cancel_order(r.order_id) for r in filled if r.order_id
        ]
        cancel_results = await asyncio.gather(*cancel_tasks, return_exceptions=True)

        all_cancelled = all(r is True for r in cancel_results)

        if all_cancelled:
            log.info("Arb %s: partial fill rolled back — no net position", arb.id)
            exec_result.partial = False
        else:
            exec_result.hedged = True
            log.error(
                "Arb %s: one-legged position — MANUAL HEDGE REQUIRED. "
                "Filled legs: %s. Failed legs: %s",
                arb.id,
                [(r.market_id, r.side, r.filled_count) for r in filled],
                [(r.market_id, r.side, r.error) for r in failed],
            )

        return exec_result

    async def close(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scale_leg(leg: ArbLeg, scale: float) -> ArbLeg:
    from src.models import ArbLeg as AL
    return AL(
        source=leg.source,
        market_id=leg.market_id,
        bookmaker=leg.bookmaker,
        outcome_name=leg.outcome_name,
        price=leg.price,
        effective_price=leg.effective_price,
        stake=round(leg.stake * scale, 2),
        side=leg.side,
    )

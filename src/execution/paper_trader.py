from __future__ import annotations

"""
Paper trading engine — simulates arb execution with a virtual balance.
All positions are stored in-memory and persisted to SQLite via the store.
"""
import logging
import uuid
from datetime import datetime

from src.models import ArbOpportunity, PaperPosition
from src.storage.db import Store

log = logging.getLogger(__name__)


class PaperTrader:
    def __init__(self, store: Store, starting_balance: float, max_position_pct: float):
        self._store = store
        self._starting_balance = starting_balance
        self._max_stake = starting_balance * max_position_pct
        self._balance: float | None = None  # lazy-loaded

    async def balance(self) -> float:
        if self._balance is None:
            self._balance = await self._store.get_balance(self._starting_balance)
        return self._balance

    async def open_position(self, arb: ArbOpportunity) -> PaperPosition | None:
        bal = await self.balance()
        stake = min(arb.total_stake, self._max_stake, bal)

        if stake < 1:
            log.warning("Insufficient balance (%.2f) to open arb %s", bal, arb.id)
            return None

        # Scale legs proportionally if we're constrained
        scale = stake / arb.total_stake
        from src.models import ArbLeg
        scaled_legs = [
            ArbLeg(
                source=leg.source,
                market_id=leg.market_id,
                bookmaker=leg.bookmaker,
                outcome_name=leg.outcome_name,
                price=leg.price,
                effective_price=leg.effective_price,
                stake=round(leg.stake * scale, 2),
                side=leg.side,
            )
            for leg in arb.legs
        ]

        pos = PaperPosition(
            id=str(uuid.uuid4())[:8],
            arb_id=arb.id,
            opened_at=datetime.utcnow(),
            legs=scaled_legs,
            total_stake=round(stake, 2),
            expected_profit=round(stake * arb.margin, 2),
        )

        self._balance = bal - stake
        await self._store.save_position(pos)
        await self._store.set_balance(self._balance)

        log.info(
            "Opened paper position %s | stake=%.2f expected_profit=%.2f margin=%.2f%%",
            pos.id, pos.total_stake, pos.expected_profit, arb.margin * 100,
        )
        return pos

    async def settle_position(self, position_id: str, winning_outcome: str) -> float:
        pos = await self._store.get_position(position_id)
        if not pos or pos.status != "open":
            return 0.0

        winning_leg = next(
            (l for l in pos.legs if l.outcome_name.lower() == winning_outcome.lower()),
            None,
        )
        payout = winning_leg.payout if winning_leg else 0.0
        profit = round(payout - pos.total_stake, 2)

        self._balance = (await self.balance()) + payout
        await self._store.set_balance(self._balance)
        await self._store.settle_position(position_id, profit)

        log.info("Settled position %s | profit=%.2f", position_id, profit)
        return profit

    async def get_stats(self) -> dict:
        return await self._store.get_portfolio_stats(self._starting_balance)

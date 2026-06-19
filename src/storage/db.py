from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import aiosqlite

from src.models import ArbLeg, ArbOpportunity, BetSide, PaperPosition, Source

log = logging.getLogger(__name__)

# DATA_DIR env var lets Fly.io (and other cloud hosts) redirect persistent files
# to a mounted volume.  Falls back to the project root for local dev.
_DATA_DIR = Path(os.environ.get("DATA_DIR", "."))
DB_PATH = str(_DATA_DIR / "arbitrage.db")


class Store:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._create_tables()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def _create_tables(self) -> None:
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS notified_arbs (
                arb_id TEXT PRIMARY KEY,
                notified_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS balance (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                amount REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS arb_opportunities (
                id TEXT PRIMARY KEY,
                sport TEXT,
                event_name TEXT,
                market_type TEXT,
                margin REAL,
                total_stake REAL,
                legs_json TEXT,
                detected_at TEXT
            );

            CREATE TABLE IF NOT EXISTS paper_positions (
                id TEXT PRIMARY KEY,
                arb_id TEXT,
                opened_at TEXT,
                legs_json TEXT,
                total_stake REAL,
                expected_profit REAL,
                status TEXT DEFAULT 'open',
                actual_profit REAL,
                settled_at TEXT
            );
        """)
        await self._db.commit()

    # ── Notification dedup ─────────────────────────────────────────────────

    async def has_notified(self, arb_id: str) -> bool:
        async with self._db.execute(
            "SELECT 1 FROM notified_arbs WHERE arb_id = ?", (arb_id,)
        ) as cur:
            return await cur.fetchone() is not None

    async def mark_notified(self, arb_id: str) -> None:
        await self._db.execute(
            "INSERT OR IGNORE INTO notified_arbs (arb_id, notified_at) VALUES (?, ?)",
            (arb_id, datetime.utcnow().isoformat()),
        )
        await self._db.commit()

    # ── Balance ────────────────────────────────────────────────────────────

    async def get_balance(self, starting: float) -> float:
        async with self._db.execute("SELECT amount FROM balance WHERE id = 1") as cur:
            row = await cur.fetchone()
        if row:
            return row["amount"]
        await self._db.execute(
            "INSERT INTO balance (id, amount) VALUES (1, ?)", (starting,)
        )
        await self._db.commit()
        return starting

    async def set_balance(self, amount: float) -> None:
        await self._db.execute(
            "INSERT INTO balance (id, amount) VALUES (1, ?) "
            "ON CONFLICT(id) DO UPDATE SET amount = excluded.amount",
            (amount,),
        )
        await self._db.commit()

    # ── Opportunities ──────────────────────────────────────────────────────

    async def save_opportunity(self, arb: ArbOpportunity) -> None:
        legs_json = json.dumps([
            {
                "source": l.source.value, "market_id": l.market_id,
                "bookmaker": l.bookmaker, "outcome_name": l.outcome_name,
                "price": l.price, "effective_price": l.effective_price,
                "stake": l.stake, "side": l.side.value,
            }
            for l in arb.legs
        ])
        await self._db.execute(
            """INSERT OR REPLACE INTO arb_opportunities
               (id, sport, event_name, market_type, margin, total_stake, legs_json, detected_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (arb.id, arb.sport, arb.event_name, arb.market_type,
             arb.margin, arb.total_stake, legs_json,
             arb.detected_at.isoformat()),
        )
        await self._db.commit()

    async def get_recent_opportunities(self, limit: int = 50) -> list[ArbOpportunity]:
        async with self._db.execute(
            "SELECT * FROM arb_opportunities ORDER BY detected_at DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_arb(r) for r in rows]

    # ── Positions ──────────────────────────────────────────────────────────

    async def save_position(self, pos: PaperPosition) -> None:
        legs_json = json.dumps([
            {
                "source": l.source.value, "market_id": l.market_id,
                "bookmaker": l.bookmaker, "outcome_name": l.outcome_name,
                "price": l.price, "effective_price": l.effective_price,
                "stake": l.stake, "side": l.side.value,
            }
            for l in pos.legs
        ])
        await self._db.execute(
            """INSERT INTO paper_positions
               (id, arb_id, opened_at, legs_json, total_stake, expected_profit, status)
               VALUES (?, ?, ?, ?, ?, ?, 'open')""",
            (pos.id, pos.arb_id, pos.opened_at.isoformat(),
             legs_json, pos.total_stake, pos.expected_profit),
        )
        await self._db.commit()

    async def get_position(self, position_id: str) -> PaperPosition | None:
        async with self._db.execute(
            "SELECT * FROM paper_positions WHERE id = ?", (position_id,)
        ) as cur:
            row = await cur.fetchone()
        return _row_to_position(row) if row else None

    async def settle_position(self, position_id: str, profit: float) -> None:
        await self._db.execute(
            """UPDATE paper_positions
               SET status = 'settled', actual_profit = ?, settled_at = ?
               WHERE id = ?""",
            (profit, datetime.utcnow().isoformat(), position_id),
        )
        await self._db.commit()

    async def get_open_positions(self) -> list[PaperPosition]:
        async with self._db.execute(
            "SELECT * FROM paper_positions WHERE status = 'open' ORDER BY opened_at DESC"
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_position(r) for r in rows]

    async def get_all_positions(self, limit: int = 100) -> list[PaperPosition]:
        async with self._db.execute(
            "SELECT * FROM paper_positions ORDER BY opened_at DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_position(r) for r in rows]

    # ── Stats ──────────────────────────────────────────────────────────────

    async def get_pnl_history(self) -> list[dict]:
        """Time-series of cumulative P&L for the TradingView equity chart."""
        async with self._db.execute(
            """SELECT settled_at,
                      SUM(actual_profit) OVER (ORDER BY settled_at ROWS UNBOUNDED PRECEDING) AS cum_pnl
               FROM paper_positions
               WHERE status = 'settled' AND settled_at IS NOT NULL
               ORDER BY settled_at"""
        ) as cur:
            rows = await cur.fetchall()
        return [
            {"time": row["settled_at"][:10], "value": round(row["cum_pnl"], 2)}
            for row in rows
        ]

    async def get_portfolio_stats(self, starting_balance: float) -> dict:
        balance = await self.get_balance(starting_balance)
        async with self._db.execute(
            "SELECT COUNT(*), COALESCE(SUM(actual_profit), 0), "
            "COALESCE(SUM(CASE WHEN actual_profit > 0 THEN 1 ELSE 0 END), 0) "
            "FROM paper_positions WHERE status = 'settled'"
        ) as cur:
            row = await cur.fetchone()
        settled, total_profit, wins = row[0], row[1], row[2]
        async with self._db.execute(
            "SELECT COUNT(*) FROM paper_positions WHERE status = 'open'"
        ) as cur:
            open_count = (await cur.fetchone())[0]
        return {
            "balance": round(balance, 2),
            "starting_balance": starting_balance,
            "total_pnl": round(total_profit, 2),
            "pnl_pct": round((balance - starting_balance) / starting_balance * 100, 2),
            "total_positions": settled,
            "open_positions": open_count,
            "win_rate": round(wins / settled * 100, 1) if settled else 0,
        }


def _parse_legs(legs_json: str) -> list[ArbLeg]:
    legs = []
    for l in json.loads(legs_json):
        legs.append(ArbLeg(
            source=Source(l["source"]),
            market_id=l["market_id"],
            bookmaker=l.get("bookmaker"),
            outcome_name=l["outcome_name"],
            price=l["price"],
            effective_price=l.get("effective_price", l["price"]),
            stake=l["stake"],
            side=BetSide(l.get("side", "back")),
        ))
    return legs


def _row_to_arb(row) -> ArbOpportunity:
    return ArbOpportunity(
        id=row["id"],
        sport=row["sport"],
        event_name=row["event_name"],
        market_type=row["market_type"],
        gross_margin=row.get("gross_margin") or row["margin"],
        margin=row["margin"],
        total_stake=row["total_stake"],
        legs=_parse_legs(row["legs_json"]),
        detected_at=datetime.fromisoformat(row["detected_at"]),
    )


def _row_to_position(row) -> PaperPosition:
    return PaperPosition(
        id=row["id"],
        arb_id=row["arb_id"],
        opened_at=datetime.fromisoformat(row["opened_at"]),
        legs=_parse_legs(row["legs_json"]),
        total_stake=row["total_stake"],
        expected_profit=row["expected_profit"],
        status=row["status"],
        actual_profit=row["actual_profit"],
        settled_at=datetime.fromisoformat(row["settled_at"]) if row["settled_at"] else None,
    )

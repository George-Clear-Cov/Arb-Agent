from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class Source(str, Enum):
    ODDS_API = "odds_api"
    BETFAIR = "betfair"
    ESPN_DK = "espn_dk"
    BOVADA = "bovada"
    KALSHI = "kalshi"
    KALSHI_SPORTS = "kalshi_sports"  # Kalshi game markets (h2h/spreads/totals) — treated as sportsbook
    POLYMARKET = "polymarket"
    PREDICTIT = "predictit"
    PROPHETX = "prophetx"
    OPINION = "opinion"
    GEMINI = "gemini"
    HYPERLIQUID = "hyperliquid"
    PREDICTFUN = "predictfun"
    MANIFOLD = "manifold"
    ACTION_NETWORK = "action_network"  # FanDuel/Caesars/BetMGM/DK/Pinnacle/bet365/BetRivers/Underdog/Fanatics


class BetSide(str, Enum):
    BACK = "back"
    LAY = "lay"


@dataclass
class Outcome:
    name: str
    price: float           # raw decimal odds quoted by platform
    implied_prob: float    # 1 / price
    source: Source
    market_id: str
    bookmaker: Optional[str] = None
    side: BetSide = BetSide.BACK
    available_volume: Optional[float] = None  # liquidity in $, Betfair/CLOB
    is_maker: bool = False                    # Polymarket limit order = 0% fee
    token_id: Optional[str] = None           # Polymarket CLOB token ID for order placement


@dataclass
class Market:
    source: Source
    market_id: str
    sport: str
    event_name: str
    commence_time: Optional[datetime]
    home_team: Optional[str]
    away_team: Optional[str]
    market_type: str
    outcomes: list[Outcome]
    total_volume: Optional[float] = None     # total $ volume — used for liquidity filter
    description: str = ""                    # market rules / description text from source
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    raw: dict = field(default_factory=dict, repr=False)


@dataclass
class ArbOpportunity:
    id: str
    sport: str
    event_name: str
    market_type: str
    gross_margin: float            # pre-fee margin
    margin: float                  # net margin after all platform fees
    total_stake: float
    legs: list[ArbLeg]
    detected_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    @property
    def profit(self) -> float:
        return round(self.total_stake * self.margin, 2)

    @property
    def gross_profit(self) -> float:
        return round(self.total_stake * self.gross_margin, 2)

    @property
    def sources(self) -> list[str]:
        return list({leg.source.value for leg in self.legs})

    @property
    def annualized_margin(self) -> float:
        """Return annualised net margin.  0.0 if no expiry or already expired."""
        if self.expires_at is None:
            return 0.0
        now = datetime.now(tz=timezone.utc)
        exp = self.expires_at
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        days = (exp - now).total_seconds() / 86400
        if days <= 0:
            return 0.0
        return self.margin * 365 / days


@dataclass
class ArbLeg:
    source: Source
    market_id: str
    bookmaker: Optional[str]
    outcome_name: str
    price: float           # raw price
    effective_price: float  # fee-adjusted price used in arb math
    stake: float
    side: BetSide = BetSide.BACK
    token_id: Optional[str] = None  # Polymarket CLOB token ID

    @property
    def payout(self) -> float:
        return self.stake * self.effective_price


@dataclass
class PaperPosition:
    id: str
    arb_id: str
    opened_at: datetime
    legs: list[ArbLeg]
    total_stake: float
    expected_profit: float
    status: str = "open"
    actual_profit: Optional[float] = None
    settled_at: Optional[datetime] = None

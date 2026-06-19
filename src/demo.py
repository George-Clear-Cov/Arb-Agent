from __future__ import annotations

"""
Synthetic data generator for UI preview without real API keys.
Run with: python agent.py --demo
"""
import random
import uuid
from datetime import datetime, timedelta

from src.models import ArbLeg, ArbOpportunity, BetSide, Market, Outcome, Source

_EVENTS = [
    ("nfl",        "Chiefs vs Eagles",           "h2h",    [Source.ODDS_API, Source.BETFAIR]),
    ("nba",        "Lakers vs Celtics",           "h2h",    [Source.ODDS_API, Source.BETFAIR]),
    ("soccer_epl", "Arsenal vs Liverpool",        "h2h",    [Source.ODDS_API, Source.BETFAIR]),
    ("prediction", "Fed cuts rates in June",      "binary", [Source.POLYMARKET, Source.KALSHI]),
    ("prediction", "Bitcoin above 100k by Aug",   "binary", [Source.POLYMARKET, Source.KALSHI]),
    ("prediction", "Trump signs bill by Q3",      "binary", [Source.KALSHI, Source.POLYMARKET]),
    ("nfl",        "49ers vs Cowboys",            "spreads",[Source.ODDS_API, Source.BETFAIR]),
    ("nba",        "Bucks vs Heat totals",        "totals", [Source.ODDS_API, Source.BETFAIR]),
    ("prediction", "Elon Musk leaves DOGE",       "binary", [Source.POLYMARKET, Source.KALSHI]),
]

_BOOKS = {
    Source.ODDS_API: ["DraftKings", "FanDuel", "BetMGM", "Caesars"],
    Source.BETFAIR:  ["Betfair"],
    Source.KALSHI:   ["Kalshi"],
    Source.POLYMARKET: ["Polymarket"],
}


def _make_arb(event: tuple, net_margin: float) -> ArbOpportunity:
    sport, name, mtype, sources = event
    src_a, src_b = sources[0], sources[1]

    # Two-outcome arb: split prob with a gap
    prob_a = random.uniform(0.38, 0.60)
    prob_b = 1 - prob_a - net_margin * 0.9  # leave room for margin

    price_a = round(1 / prob_a, 3)
    price_b = round(1 / prob_b, 3)

    # Slight fee reduction for effective prices
    eff_a = round(price_a * random.uniform(0.97, 0.995), 3)
    eff_b = round(price_b * random.uniform(0.97, 0.995), 3)

    gross = round((1 - prob_a - prob_b) / (prob_a + prob_b), 4)
    stake = 100.0
    impl_sum = 1 / eff_a + 1 / eff_b
    s_a = round(stake / impl_sum / eff_a, 2)
    s_b = round(stake / impl_sum / eff_b, 2)

    if mtype == "binary":
        outcomes = ["Yes", "No"]
    elif mtype in ("h2h", "spreads"):
        parts = name.split(" vs ")
        outcomes = parts if len(parts) == 2 else ["Home", "Away"]
    else:
        outcomes = ["Over", "Under"]

    detected = datetime.utcnow() - timedelta(seconds=random.randint(0, 120))

    return ArbOpportunity(
        id=str(uuid.uuid4())[:8],
        sport=sport,
        event_name=name,
        market_type=mtype,
        gross_margin=gross,
        margin=net_margin,
        total_stake=round(s_a + s_b, 2),
        detected_at=detected,
        legs=[
            ArbLeg(
                source=src_a,
                market_id=f"demo_{uuid.uuid4().hex[:6]}",
                bookmaker=random.choice(_BOOKS[src_a]),
                outcome_name=outcomes[0],
                price=price_a,
                effective_price=eff_a,
                stake=s_a,
                side=BetSide.BACK,
            ),
            ArbLeg(
                source=src_b,
                market_id=f"demo_{uuid.uuid4().hex[:6]}",
                bookmaker=random.choice(_BOOKS[src_b]),
                outcome_name=outcomes[1],
                price=price_b,
                effective_price=eff_b,
                stake=s_b,
                side=BetSide.BACK,
            ),
        ],
    )


def _make_market(source: Source, event_name: str, sport: str, mtype: str) -> Market:
    book = random.choice(_BOOKS[source])
    outcomes = []
    for name, prob in [("Yes" if mtype == "binary" else "Home", random.uniform(0.4, 0.65))]:
        rest = 1 - prob
        for n, p in [(name, prob), ("No" if mtype == "binary" else "Away", rest)]:
            outcomes.append(Outcome(
                name=n,
                price=round(1 / p, 3),
                implied_prob=round(p, 4),
                source=source,
                market_id=f"demo_{uuid.uuid4().hex[:6]}",
                bookmaker=book,
                side=BetSide.BACK,
                available_volume=random.uniform(5_000, 200_000),
            ))
    return Market(
        source=source,
        market_id=f"demo_{uuid.uuid4().hex[:6]}",
        sport=sport,
        event_name=event_name,
        commence_time=datetime.utcnow() + timedelta(days=random.randint(1, 14)),
        home_team=None, away_team=None,
        market_type=mtype,
        outcomes=outcomes,
        total_volume=random.uniform(50_000, 500_000),
    )


def generate_demo_state() -> tuple[list[Market], list[ArbOpportunity], dict]:
    margins = [
        random.uniform(0.02, 0.08),
        random.uniform(0.015, 0.05),
        random.uniform(0.025, 0.06),
        random.uniform(0.018, 0.04),
    ]

    arbs = [_make_arb(random.choice(_EVENTS), m) for m in sorted(margins, reverse=True)]

    markets: list[Market] = []
    for sport, name, mtype, sources in _EVENTS:
        for src in sources:
            markets.append(_make_market(src, name, sport, mtype))

    stats = {
        "balance": round(10000 + random.uniform(-200, 800), 2),
        "starting_balance": 10000.0,
        "total_pnl": round(random.uniform(-200, 800), 2),
        "pnl_pct": round(random.uniform(-2, 8), 2),
        "total_positions": random.randint(5, 40),
        "open_positions": random.randint(1, 5),
        "win_rate": round(random.uniform(55, 85), 1),
    }

    return markets, arbs, stats

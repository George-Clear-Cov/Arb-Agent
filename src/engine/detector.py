from __future__ import annotations

"""
Fee-aware arbitrage detection.

Uses effective (post-fee) prices for arb math so detected margins are real
net profits, not inflated gross figures that disappear after fees.

Also detects back/lay arbs between sportsbooks and Betfair exchange.
"""
import hashlib
import re
from datetime import datetime, timedelta, timezone

from src.engine.fees import (
    effective_back_price,
    gross_margin as calc_gross_margin,
    lay_arb_margin,
)
from src.engine.matcher import group_matching_markets
from src.models import ArbLeg, ArbOpportunity, BetSide, Market, Outcome, Source

# Sources that refresh fast enough that stale data is a sign of a feed problem,
# not a design decision.  Age limit before we reject any arb leg from that source.
_MAX_STALE_SECONDS: dict[Source, int] = {
    Source.KALSHI_SPORTS: 900,  # polls every 30s — stale after 15 min
}

# Minimum 24hr volume to consider a sportsbook/exchange outcome tradeable.
# Prediction markets use a much lower floor — their volume is in the hundreds,
# but a $50-200 stake at the CLOB best ask is still fully executable.
MIN_OUTCOME_LIQUIDITY = 5_000.0       # Betfair, OddsAPI (exchange/book depth)
MIN_PREDICTION_LIQUIDITY = 500.0      # Polymarket 24hr vol floor

# Sources whose available_volume represents prediction market CLOB/pool volume,
# not exchange order-book depth — use the lower floor for these.
_PREDICTION_SOURCES = frozenset({
    Source.POLYMARKET,
    Source.KALSHI,
    Source.KALSHI_SPORTS,
    Source.PREDICTIT,
})


MAX_PLAUSIBLE_MARGIN = 0.20


def _arb_id(sport: str, event_name: str, legs: list) -> str:
    """Deterministic 8-char ID so the same arb always gets the same ID.

    Prevents notification spam: without this, uuid4() generates a new ID every
    detection cycle and the notifier fires for every cycle instead of once.
    """
    key = f"{sport}|{event_name}|" + "|".join(
        sorted(f"{l.bookmaker}:{l.outcome_name}" for l in legs)
    )
    return hashlib.md5(key.encode()).hexdigest()[:8]


def _earliest_expiry(markets: list[Market]) -> datetime | None:
    times = []
    for m in markets:
        ct = m.commence_time
        if ct is None:
            continue
        if ct.tzinfo is None:
            ct = ct.replace(tzinfo=timezone.utc)
        times.append(ct)
    return min(times) if times else None


def detect_arbs_with_groups(
    markets: list[Market],
    min_margin: float = 0.01,
    total_stake: float = 100.0,
    min_liquidity: float = MIN_OUTCOME_LIQUIDITY,
) -> tuple[list[ArbOpportunity], list[list[Market]]]:
    """Like detect_arbs but also returns the matched groups for pair registration."""
    groups = group_matching_markets(markets)
    arbs: list[ArbOpportunity] = []
    for group in groups:
        opp = _check_back_arb(group, min_margin, total_stake, min_liquidity)
        if opp:
            arbs.append(opp)
    now = datetime.now(tz=timezone.utc)
    live_arbs = []
    for a in arbs:
        exp = a.expires_at
        if exp is not None:
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            if exp <= now:
                continue
        live_arbs.append(a)
    _now = datetime.now(tz=timezone.utc)

    def _sort_key(a: ArbOpportunity) -> tuple:
        exp = a.expires_at
        if exp is None:
            bucket = 3
        else:
            fat = exp if exp.tzinfo else exp.replace(tzinfo=timezone.utc)
            secs = (fat - _now).total_seconds()
            if secs < 6 * 3600:
                bucket = 0
            elif secs < 24 * 3600:
                bucket = 1
            elif secs < 7 * 24 * 3600:
                bucket = 2
            else:
                bucket = 3
        return (bucket, -a.margin)

    return sorted(live_arbs, key=_sort_key), groups


def detect_arbs(
    markets: list[Market],
    min_margin: float = 0.01,
    total_stake: float = 100.0,
    min_liquidity: float = MIN_OUTCOME_LIQUIDITY,
) -> list[ArbOpportunity]:
    groups = group_matching_markets(markets)
    arbs: list[ArbOpportunity] = []

    for group in groups:
        opp = _check_back_arb(group, min_margin, total_stake, min_liquidity)
        if opp:
            arbs.append(opp)

    now = datetime.now(tz=timezone.utc)

    # Drop arbs whose market has already closed — stale feeds can keep returning
    # prices for resolved markets which show up as phantom high-margin arbs.
    live_arbs = []
    for a in arbs:
        exp = a.expires_at
        if exp is not None:
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            if exp <= now:
                continue  # already expired
        live_arbs.append(a)

    # Sort: expiry bucket (sooner = higher priority), then ROI descending.
    # Buckets: 0=<6h, 1=<24h, 2=<7d, 3=no expiry/>7d
    _now = datetime.now(tz=timezone.utc)

    def _sort_key(a: ArbOpportunity) -> tuple:
        exp = a.expires_at
        if exp is None:
            bucket = 3
        else:
            fat = exp if exp.tzinfo else exp.replace(tzinfo=timezone.utc)
            secs = (fat - _now).total_seconds()
            if secs < 6 * 3600:
                bucket = 0
            elif secs < 24 * 3600:
                bucket = 1
            elif secs < 7 * 24 * 3600:
                bucket = 2
            else:
                bucket = 3
        return (bucket, -a.margin)

    return sorted(live_arbs, key=_sort_key)


def _check_back_arb(
    markets: list[Market],
    min_margin: float,
    total_stake: float,
    min_liquidity: float,
) -> ArbOpportunity | None:
    # Best fee-adjusted back price per outcome name across all sources.
    # Tuple: (eff_price, raw_price, outcome, fetched_at)
    best: dict[str, tuple[float, float, Outcome, datetime]] = {}

    for mkt in markets:
        if mkt.raw.get("is_play_money"):
            continue  # Manifold / play-money — can't execute real-money arbs
        for outcome in mkt.outcomes:
            if outcome.side != BetSide.BACK:
                continue
            if outcome.available_volume is not None:
                floor = (MIN_PREDICTION_LIQUIDITY
                         if outcome.source in _PREDICTION_SOURCES
                         else min_liquidity)
                if outcome.available_volume < floor:
                    continue
            # Normalise outcome name so cross-platform outcomes can be compared:
            # "Over 211.5 points scored" (Kalshi) == "Over 211.5" (Polymarket)
            # "New York M" (Kalshi) == "New York Mets" (Polymarket)
            name = _normalize_outcome_name(outcome.name, mkt.market_type)
            eff = effective_back_price(outcome.price, outcome.source, outcome.is_maker)
            if name not in best or eff > best[name][0]:
                best[name] = (eff, outcome.price, outcome, mkt.fetched_at)

    # Merge partial city/region names into full team names.
    # Kalshi uses city-only names ("New York", "San Antonio") while Polymarket
    # uses nicknames ("Knicks", "Spurs") which expand to full names.  After
    # _normalize_outcome_name both expand "Spurs"→"san antonio spurs" and
    # "San Antonio"→"san antonio spurs" (same key), but "New York" stays
    # "new york" while "Knicks"→"new york knicks" (different keys).
    # Fix: if key A is a proper substring of key B, merge A into B keeping
    # the better effective price.
    if any(m.sport not in ("prediction",) for m in markets):
        keys = list(best.keys())
        merged: set[str] = set()
        for ka in keys:
            if ka in merged or len(ka) < 5:
                continue
            # Never merge over/under lines — "under 8" ⊂ "under 8.5" is a number
            # mismatch, not a partial name.  Merging them creates phantom arbs.
            if ka.startswith(("over ", "under ")):
                continue
            for kb in keys:
                if ka == kb or kb in merged:
                    continue
                if ka in kb:  # "new york" ⊂ "new york knicks"
                    eff_a, raw_a, out_a, fat_a = best[ka]
                    eff_b, _, _, _ = best[kb]
                    if eff_a > eff_b:
                        best[kb] = (eff_a, raw_a, out_a, fat_a)
                    merged.add(ka)
                    break
        for k in merged:
            del best[k]

    if len(best) < 2:
        return None

    # For totals markets: "over 8.5" (Polymarket) vs "under 8" (FanDuel) is NOT a real
    # arb — there's a gap [8, 8.5) where both bets lose.  Remove mis-paired over/under
    # outcomes (different line values) before proceeding.
    if any(m.market_type in _TOTALS_MARKET_TYPES for m in markets):
        over_lines = {k[5:] for k in best if k.startswith("over ")}   # {"8.5", "9"}
        under_lines = {k[6:] for k in best if k.startswith("under ")} # {"8", "9"}
        paired_lines = over_lines & under_lines                        # {"9"}
        to_delete = [
            k for k in list(best)
            if k.startswith(("over ", "under "))
            and (k[5:] if k.startswith("over ") else k[6:]) not in paired_lines
        ]
        for k in to_delete:
            del best[k]

    # Require legs from at least 2 different bookmakers.
    # For prediction markets this means 2 different platforms (Polymarket vs Kalshi).
    # For sportsbooks this means 2 different books (FanDuel vs BetMGM).
    # Using bookmaker rather than source handles intra-OddsAPI arbs correctly.
    books_in_best = {o.bookmaker for _, _, o, _ in best.values()}
    if len(books_in_best) < 2:
        return None

    # Reject arb if any fast-polling sportsbook leg has stale prices.
    now_utc = datetime.now(tz=timezone.utc)
    for _, _, outcome, fetched_at in best.values():
        max_stale = _MAX_STALE_SECONDS.get(outcome.source)
        if max_stale:
            fat = fetched_at if fetched_at.tzinfo else fetched_at.replace(tzinfo=timezone.utc)
            if (now_utc - fat).total_seconds() > max_stale:
                return None  # stale sportsbook price — skip

    eff_prices = [eff for eff, _, _, _ in best.values()]
    raw_prices = [raw for _, raw, _, _ in best.values()]

    eff_sum = sum(1 / p for p in eff_prices)
    if eff_sum >= 1.0:
        return None

    net_mgn = round((1 - eff_sum) / eff_sum, 6)
    if net_mgn < min_margin:
        return None
    if net_mgn > MAX_PLAUSIBLE_MARGIN:
        return None

    gross_mgn = calc_gross_margin(raw_prices)

    legs: list[ArbLeg] = []
    for name, (eff, raw, outcome, _fetched_at) in best.items():
        leg_stake = round((total_stake / eff_sum) / eff, 2)
        legs.append(ArbLeg(
            source=outcome.source,
            market_id=outcome.market_id,
            bookmaker=outcome.bookmaker,
            outcome_name=outcome.name,
            price=raw,
            effective_price=eff,
            stake=leg_stake,
            side=BetSide.BACK,
            token_id=outcome.token_id,
        ))

    ref = markets[0]
    opp = ArbOpportunity(
        id="",
        sport=ref.sport,
        event_name=ref.event_name,
        market_type=ref.market_type,
        gross_margin=gross_mgn,
        margin=net_mgn,
        total_stake=round(sum(l.stake for l in legs), 2),
        legs=legs,
        detected_at=datetime.utcnow(),
        expires_at=_earliest_expiry(markets),
    )
    opp.id = _arb_id(opp.sport, opp.event_name, legs)
    return opp


# Tokens that follow a line value in Kalshi totals outcome names.
# "Over 211.5 points scored" → "over 211.5"
# "Under 7.5 runs scored"   → "under 7.5"
_TOTALS_NOISE_RE = re.compile(
    r'\b(points|runs|goals|assists|scored|made|total)\b.*$', re.IGNORECASE
)
_LINE_RE = re.compile(r'\b(\d+\.?\d*)\b')

# Market type strings whose outcomes should be normalised to "over X.5" / "under X.5".
_TOTALS_MARKET_TYPES = frozenset({
    "totals", "over_under", "total_goals", "over/under",
})
# Market type strings whose outcomes are team names and should have abbreviations expanded.
_H2H_MARKET_TYPES = frozenset({
    "h2h", "match_odds", "match odds", "moneyline", "1x2", "winner", "to win", "binary",
})


def _normalize_outcome_name(name: str, market_type: str) -> str:
    """Normalise an outcome name for cross-platform outcome matching.

    For totals markets:  "Over 211.5 points scored" → "over 211.5"
                         "Over 211.5"               → "over 211.5"
                         "Over"  (no line)           → "over"
    For h2h markets:     "New York M"  → "new york mets"  (via team expansions)
                         "New York Mets" → "new york mets"
    """
    n = name.lower().strip()
    mt = market_type.lower()

    if mt in _TOTALS_MARKET_TYPES:
        if n.startswith("over"):
            rest = _TOTALS_NOISE_RE.sub("", n[4:]).strip()
            lm = _LINE_RE.search(rest)
            return f"over {lm.group()}" if lm else "over"
        if n.startswith("under"):
            rest = _TOTALS_NOISE_RE.sub("", n[5:]).strip()
            lm = _LINE_RE.search(rest)
            return f"under {lm.group()}" if lm else "under"
        return n

    if mt in _H2H_MARKET_TYPES:
        from src.engine.matcher import _expand_team_names
        return _expand_team_names(n)

    return n

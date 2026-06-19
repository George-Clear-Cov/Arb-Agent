from __future__ import annotations

"""
Platform fee models.

Every source has a different fee structure that reduces effective odds.
Ignoring fees causes the detector to flag losing trades as profitable.
"""
import math
from src.models import Source

# Polymarket: 0% maker (limit orders on book), 2% taker on net profit
POLYMARKET_TAKER_FEE = 0.02
# Betfair: commission on net winnings — varies 2-5%, use conservative 5%
BETFAIR_COMMISSION = 0.05
# Kalshi: ceil(0.07 * price * (1-price)) per contract
KALSHI_FEE_COEFF = 0.07
# PredictIt: 10% fee on profits + 5% withdrawal fee; model as 10% on winnings
PREDICTIT_FEE = 0.10
# ProphetX: peer-to-peer exchange, 2% commission on net winnings
PROPHETX_COMMISSION = 0.02
# Opinion: CLOB taker fee 2% on net profit (same model as Polymarket taker)
OPINION_TAKER_FEE = 0.02
# Predict.fun: Solana-based CLOB, 2% taker fee on net profit
PREDICTFUN_TAKER_FEE = 0.02


def effective_back_price(raw_price: float, source: Source, is_maker: bool = False) -> float:
    """
    Decimal odds after platform fees. Always <= raw_price.

    is_maker: if True (Polymarket limit order sitting on book), fee is 0%.
    """
    if raw_price <= 1:
        return raw_price

    net_win = raw_price - 1  # profit per $1 staked if we win

    if source == Source.POLYMARKET:
        if is_maker:
            return raw_price
        return 1 + net_win * (1 - POLYMARKET_TAKER_FEE)

    if source in {Source.KALSHI, Source.KALSHI_SPORTS}:
        # Per $1 staked at implied prob p:
        # cost per contract = p + 0.07*p*(1-p) = p*(1 + 0.07*(1-p))
        # payout if correct = 1
        # KALSHI_SPORTS uses the same fee structure as KALSHI prediction markets.
        p = 1 / raw_price
        effective_cost = p * (1 + KALSHI_FEE_COEFF * (1 - p))
        return 1 / effective_cost

    if source == Source.BETFAIR:
        return 1 + net_win * (1 - BETFAIR_COMMISSION)

    if source == Source.PREDICTIT:
        return 1 + net_win * (1 - PREDICTIT_FEE)

    if source == Source.PROPHETX:
        return 1 + net_win * (1 - PROPHETX_COMMISSION)

    if source == Source.OPINION:
        if is_maker:
            return raw_price
        return 1 + net_win * (1 - OPINION_TAKER_FEE)

    if source == Source.PREDICTFUN:
        if is_maker:
            return raw_price
        return 1 + net_win * (1 - PREDICTFUN_TAKER_FEE)

    # ODDS_API sportsbooks — vig already baked into quoted odds
    return raw_price


def gross_margin(legs_raw_prices: list[float]) -> float:
    """Pre-fee margin from a list of raw decimal prices (one per outcome)."""
    implied_sum = sum(1 / p for p in legs_raw_prices if p > 1)
    if implied_sum >= 1:
        return 0.0
    return round((1 - implied_sum) / implied_sum, 6)


def net_margin(legs_effective_prices: list[float]) -> float:
    """Post-fee margin from a list of fee-adjusted decimal prices."""
    implied_sum = sum(1 / p for p in legs_effective_prices if p > 1)
    if implied_sum >= 1:
        return 0.0
    return round((1 - implied_sum) / implied_sum, 6)


def lay_arb_margin(back_price: float, lay_price: float,
                   commission: float = BETFAIR_COMMISSION) -> float:
    """
    Margin on a back/lay arb: back at a sportsbook, lay on Betfair.

    Returns positive margin if arb exists, 0 otherwise.
    Condition: back_price > 1 + (lay_price - 1) / (1 - commission)
    """
    if back_price <= 1 or lay_price <= 1:
        return 0.0
    k = 1 / (1 - commission)
    numerator = back_price - 1 - k * (lay_price - 1)
    denominator = 1 + k * (lay_price - 1)
    if numerator <= 0 or denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)

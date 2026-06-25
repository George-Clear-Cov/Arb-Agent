from __future__ import annotations


def kelly_stake(
    margin: float,
    bankroll: float,
    kelly_fraction: float = 0.25,
    max_stake: float = 100.0,
    min_stake: float = 5.0,
) -> float:
    """Fractional Kelly stake for a guaranteed-edge arb.

    For a two-leg arb the Kelly fraction equals the margin (guaranteed return
    on capital at risk), scaled by a conservative fraction to account for
    model error and execution risk.
    """
    raw = bankroll * margin * kelly_fraction
    return round(max(min_stake, min(raw, max_stake)), 2)

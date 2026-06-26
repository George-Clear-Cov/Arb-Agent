from __future__ import annotations

"""
Confirmed market pair index for fast arbitrage detection.

Instead of running O(n²) fuzzy matching + LLM on every WS price event,
we maintain an index of confirmed matched pairs and run the arb check
only on those pairs. This reduces detection from 60-90s to <100ms.

Flow:
  1. Startup: load confirmed pairs from LLM brain into memory
  2. Every 5 min (SCAN_INTERVAL): run full group_matching_markets, update index
  3. Every WS event: check confirmed pairs only (pure arithmetic, <100ms)
"""

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models import ArbOpportunity, Market

log = logging.getLogger(__name__)

SCAN_INTERVAL = 300.0  # full matching scan every 5 minutes


class ConfirmedPairMonitor:
    """Fast arbitrage detection on pre-matched market pairs."""

    def __init__(self) -> None:
        # Sorted tuple (id_a, id_b) where id_a <= id_b
        self._confirmed_keys: set[tuple[str, str]] = set()
        # market_id → Market (updated each cycle)
        self._market_index: dict[str, "Market"] = {}
        self.last_full_scan_ts: float = 0.0

    def should_run_full_scan(self) -> bool:
        return (time.time() - self.last_full_scan_ts) > SCAN_INTERVAL

    def load_from_llm_cache(self, llm_matcher) -> None:
        """Seed confirmed pairs from the LLM matcher's in-memory cache."""
        if not llm_matcher:
            return
        cache: dict[tuple[str, str], bool] = getattr(llm_matcher, "_cache", {})
        for key, result in cache.items():
            if result:
                self._confirmed_keys.add(key)
        log.info("Pair monitor: seeded %d confirmed pairs from brain", len(self._confirmed_keys))

    def register_groups(self, groups: list[list["Market"]]) -> int:
        """Register newly matched groups from a full matching scan."""
        added = 0
        for group in groups:
            for i, m_a in enumerate(group):
                for m_b in group[i + 1:]:
                    if m_a.source == m_b.source:
                        continue
                    a, b = m_a.market_id, m_b.market_id
                    key = (a, b) if a <= b else (b, a)
                    if key not in self._confirmed_keys:
                        self._confirmed_keys.add(key)
                        added += 1
        if added:
            log.info("Pair monitor: registered %d new confirmed pairs (total=%d)",
                     added, len(self._confirmed_keys))
        return added

    def update_market_index(self, markets: list["Market"]) -> None:
        self._market_index = {m.market_id: m for m in markets}

    def check_confirmed_pairs(
        self,
        min_margin: float,
        total_stake: float,
    ) -> list["ArbOpportunity"]:
        """Check all confirmed pairs for arbs using in-memory prices.

        Runs in <100ms — no API calls, no fuzzy matching. Pure arithmetic.
        """
        from src.engine.detector import _check_back_arb, MIN_OUTCOME_LIQUIDITY
        from datetime import datetime, timezone

        arbs: list["ArbOpportunity"] = []
        now = datetime.now(tz=timezone.utc)

        for (id_a, id_b) in self._confirmed_keys:
            m_a = self._market_index.get(id_a)
            m_b = self._market_index.get(id_b)
            if not m_a or not m_b:
                continue
            opp = _check_back_arb([m_a, m_b], min_margin, total_stake, MIN_OUTCOME_LIQUIDITY)
            if opp:
                # Filter expired
                exp = opp.expires_at
                if exp is not None:
                    if exp.tzinfo is None:
                        exp = exp.replace(tzinfo=timezone.utc)
                    if exp <= now:
                        continue
                arbs.append(opp)

        return arbs

    @property
    def pair_count(self) -> int:
        return len(self._confirmed_keys)

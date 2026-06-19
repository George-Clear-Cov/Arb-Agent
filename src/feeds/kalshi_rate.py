from __future__ import annotations

"""
Kalshi API rate limiter — token bucket via asynciolimiter.

Observed behaviour: ~20 reads/sec advertised, but burst 429s appear at >6/s
in practice (probably a per-second burst window, not a sustained cap).
We use 6/sec to stay safely under the burst threshold.
Both KalshiFeed and KalshiSportsFeed share one limiter — same API quota.
"""
from typing import Optional
from asynciolimiter import Limiter

RATE_PER_SEC = 6.0   # req/s — conservative, burst-safe under observed 429 behaviour

_limiter: Optional[Limiter] = None


def _get_limiter() -> Limiter:
    global _limiter
    if _limiter is None:
        _limiter = Limiter(RATE_PER_SEC)
    return _limiter


async def kalshi_request(coro):
    """Rate-limit a Kalshi API call. Usage: resp = await kalshi_request(client.get(...))"""
    await _get_limiter().wait()
    return await coro


# Alias — sports feed uses the same shared limiter
kalshi_sports_request = kalshi_request


async def reset() -> None:
    global _limiter
    _limiter = Limiter(RATE_PER_SEC)

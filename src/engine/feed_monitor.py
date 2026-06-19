from __future__ import annotations

"""
Feed health monitor — tracks per-source fetch status, error rates, and market counts.

Each feed calls feed_monitor.record_success() / record_error() after every fetch.
The dashboard's /api/debug endpoint exposes the full health report.

Design goals:
- Zero blocking (pure in-memory, no I/O)
- Thread-safe via asyncio (single-threaded event loop)
- Easy to add to any feed: one line per fetch
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class FeedStatus:
    source: str
    last_success: Optional[datetime] = None
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    last_count: int = 0
    peak_count: int = 0
    consecutive_errors: int = 0
    total_fetches: int = 0
    total_errors: int = 0

    @property
    def healthy(self) -> bool:
        """Healthy = at least one success AND fewer than 3 consecutive errors."""
        return self.last_success is not None and self.consecutive_errors < 3

    @property
    def seconds_since_success(self) -> Optional[float]:
        if self.last_success is None:
            return None
        return (datetime.utcnow() - self.last_success).total_seconds()

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "healthy": self.healthy,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_error": self.last_error,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None,
            "last_count": self.last_count,
            "peak_count": self.peak_count,
            "consecutive_errors": self.consecutive_errors,
            "total_fetches": self.total_fetches,
            "total_errors": self.total_errors,
            "seconds_since_success": self.seconds_since_success,
        }


class FeedMonitor:
    """Singleton that tracks health of all data feeds."""

    def __init__(self) -> None:
        self._feeds: dict[str, FeedStatus] = {}

    def record_success(self, source: str, count: int) -> None:
        """Call after a successful fetch, even if count == 0."""
        s = self._get(source)
        s.last_success = datetime.utcnow()
        s.last_count = count
        s.peak_count = max(s.peak_count, count)
        s.consecutive_errors = 0
        s.total_fetches += 1
        if count == 0:
            log.warning("Feed %s: fetch succeeded but returned 0 markets", source)

    def record_error(self, source: str, error: str) -> None:
        """Call when a fetch fails completely."""
        s = self._get(source)
        s.last_error = str(error)[:300]
        s.last_error_time = datetime.utcnow()
        s.consecutive_errors += 1
        s.total_fetches += 1
        s.total_errors += 1
        log.error(
            "Feed %s error (consecutive=%d): %s",
            source, s.consecutive_errors, error,
        )

    def status(self, source: str) -> Optional[FeedStatus]:
        return self._feeds.get(source)

    def all_statuses(self) -> list[dict]:
        return [s.to_dict() for s in self._feeds.values()]

    def unhealthy_feeds(self) -> list[str]:
        return [s.source for s in self._feeds.values() if not s.healthy]

    def _get(self, source: str) -> FeedStatus:
        if source not in self._feeds:
            self._feeds[source] = FeedStatus(source=source)
        return self._feeds[source]


# Module-level singleton shared across agent.py and app.py
feed_monitor = FeedMonitor()

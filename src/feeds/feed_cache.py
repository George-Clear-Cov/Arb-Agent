from __future__ import annotations

"""Shared disk-cache helpers for all prediction market feeds.

Each feed calls load_cache() in __init__ and save_cache() after a
successful fetch. This gives instant startup across all platforms —
stale prices are overwritten by the first live poll.
"""
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from src.models import BetSide, Market, Outcome, Source

log = logging.getLogger(__name__)

CACHE_DIR = Path("state")
DEFAULT_MAX_AGE = 6 * 3600   # 6 hours — prediction markets change slowly
SPORTS_MAX_AGE  = 3600       # 1 hour — live game market structure


_SOURCE_BOOKMAKER: dict[Source, str] = {
    Source.POLYMARKET:    "Polymarket",
    Source.KALSHI:        "Kalshi",
    Source.KALSHI_SPORTS: "Kalshi",
    Source.PREDICTIT:     "PredictIt",
    Source.GEMINI:        "Gemini",
    Source.HYPERLIQUID:   "Hyperliquid",
}


def save_cache(path: Path, markets: list[Market]) -> None:
    try:
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({
            "saved_at": time.time(),
            "markets": [_to_dict(m) for m in markets],
        }))
    except Exception as exc:
        log.warning("Cache save failed (%s): %s", path.name, exc)


def load_cache(path: Path, source: Source,
               max_age: float = DEFAULT_MAX_AGE) -> list[Market]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
        age = time.time() - payload.get("saved_at", 0)
        if age > max_age:
            return []
        markets = [m for d in payload["markets"] if (m := _from_dict(d, source))]
        log.info("%s: loaded %d markets from disk cache (%.0f min old)",
                 source.value, len(markets), age / 60)
        return markets
    except Exception as exc:
        log.warning("Cache load failed (%s): %s", path.name, exc)
        return []


def _to_dict(m: Market) -> dict:
    return {
        "market_id": m.market_id,
        "sport": m.sport,
        "event_name": m.event_name,
        "description": m.description,
        "market_type": m.market_type,
        "home_team": m.home_team,
        "away_team": m.away_team,
        "commence_time": m.commence_time.isoformat() if m.commence_time else None,
        "outcomes": [
            {
                "name": o.name,
                "price": o.price,
                "implied_prob": o.implied_prob,
                "token_id": getattr(o, "token_id", None),
                "available_volume": o.available_volume,
            }
            for o in m.outcomes
        ],
    }


def _from_dict(d: dict, source: Source) -> Market | None:
    try:
        bookmaker = _SOURCE_BOOKMAKER.get(source, source.value.title())
        outcomes = [
            Outcome(
                name=o["name"],
                price=o["price"],
                implied_prob=o["implied_prob"],
                source=source,
                market_id=d["market_id"],
                bookmaker=bookmaker,
                side=BetSide.BACK,
                token_id=o.get("token_id"),
                available_volume=o.get("available_volume"),
            )
            for o in d.get("outcomes", [])
        ]
        if len(outcomes) < 2:
            return None
        ct = d.get("commence_time")
        return Market(
            source=source,
            market_id=d["market_id"],
            sport=d["sport"],
            event_name=d["event_name"],
            description=d.get("description", ""),
            market_type=d["market_type"],
            home_team=d.get("home_team"),
            away_team=d.get("away_team"),
            commence_time=datetime.fromisoformat(ct) if ct else None,
            outcomes=outcomes,
            fetched_at=datetime.utcnow(),
            raw={},
        )
    except Exception:
        return None

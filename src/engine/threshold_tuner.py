from __future__ import annotations

"""Adaptive alert threshold tuner.

Queries the arb_opportunities DB weekly to compute the 60th-percentile margin
by sport bucket, then saves to brain/thresholds.json. Notifier loads on startup
and re-applies daily. Falls back to hardcoded defaults when insufficient data.
"""
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

_DATA_DIR = Path(os.environ.get("DATA_DIR", "."))
_BRAIN_DIR = _DATA_DIR / "brain"
_THRESHOLDS_FILE = _BRAIN_DIR / "thresholds.json"
_DB_PATH = _DATA_DIR / "arbitrage.db"

_GAME_SPORTS = frozenset({
    "baseball", "football", "basketball", "hockey", "tennis",
    "soccer", "mma", "boxing", "sports",
})


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, int(len(s) * p) - 1)
    return s[idx]


def compute_and_save(lookback_days: int = 14, min_samples: int = 50) -> dict:
    """Compute adaptive thresholds from recent arb history and persist to disk."""
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()
        cur = conn.execute(
            "SELECT margin, sport FROM arb_opportunities WHERE detected_at > ?",
            (cutoff,),
        )
        rows = cur.fetchall()
        conn.close()
    except Exception as exc:
        log.warning("Threshold tuner DB query failed: %s", exc)
        return load()

    if len(rows) < min_samples:
        log.info(
            "Threshold tuner: %d samples in last %dd — not enough to tune (need %d)",
            len(rows), lookback_days, min_samples,
        )
        return load()

    game_margins: list[float] = []
    pred_margins: list[float] = []
    for margin, sport in rows:
        m = (margin or 0) * 100
        if m <= 0:
            continue
        if sport in _GAME_SPORTS:
            game_margins.append(m)
        else:
            pred_margins.append(m)

    thresholds: dict = {}
    if len(game_margins) >= 20:
        thresholds["game"] = round(_percentile(game_margins, 0.60), 3)
    if len(pred_margins) >= 20:
        thresholds["prediction"] = round(_percentile(pred_margins, 0.60), 3)

    if not thresholds:
        return load()

    _BRAIN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        existing = json.loads(_THRESHOLDS_FILE.read_text()) if _THRESHOLDS_FILE.exists() else {}
    except Exception:
        existing = {}
    existing["thresholds"] = thresholds
    existing["updated_at"] = datetime.utcnow().isoformat()
    existing["sample_sizes"] = {"game": len(game_margins), "prediction": len(pred_margins)}
    _THRESHOLDS_FILE.write_text(json.dumps(existing, indent=2))

    log.info(
        "Threshold tuner updated — game=%.2f%% (n=%d) prediction=%.2f%% (n=%d)",
        thresholds.get("game", 0), len(game_margins),
        thresholds.get("prediction", 0), len(pred_margins),
    )
    return thresholds


def load() -> dict:
    """Load persisted thresholds; returns {} if file missing or unreadable."""
    try:
        if _THRESHOLDS_FILE.exists():
            return json.loads(_THRESHOLDS_FILE.read_text()).get("thresholds", {})
    except Exception:
        pass
    return {}

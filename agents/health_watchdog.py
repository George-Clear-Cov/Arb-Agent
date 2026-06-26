#!/usr/bin/env python3
"""
Feed Health Watchdog

Monitors all feed cache files and the main agent process. Alerts when:
  - A cache file hasn't been updated in > STALE_THRESHOLD seconds
  - The main agent process has died
  - The FastAPI dashboard isn't responding

Auto-restarts the main agent if it dies. Logs health summary every cycle.

Run: python agents/health_watchdog.py
Loop: every 2 minutes
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("health_watchdog")

PROJECT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR   = PROJECT_DIR / "state"
LOG_FILE    = Path("/tmp/arb-agent.log")
POLL_SEC    = 120   # 2 minutes
DASHBOARD   = "http://localhost:8000/api/state"

# Max age (seconds) before a cache file is considered stale
STALE_THRESHOLDS: dict[str, int] = {
    "polymarket_cache":    900,   # 15 min (CLOB polling)
    "kalshi_cache":        900,
    "kalshi_sports_cache": 120,   # 2 min (live games)
    "predictit_cache":     600,
    "gemini_cache":        600,
    "hyperliquid_cache":   600,
}


def is_agent_running() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", "python agent.py"],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def restart_agent() -> None:
    log.warning("Restarting main agent...")
    subprocess.run(["pkill", "-f", "python agent.py"], capture_output=True)
    time.sleep(2)
    lock = PROJECT_DIR / "state" / "agent.lock"
    lock.unlink(missing_ok=True)
    log_fh = open(str(LOG_FILE), "a")
    subprocess.Popen(
        [sys.executable, "agent.py"],
        cwd=str(PROJECT_DIR),
        stdout=log_fh,
        stderr=log_fh,
    )
    log.info("Main agent restarted")


def check_cache_freshness() -> list[str]:
    """Returns list of stale cache names."""
    stale = []
    now = time.time()
    for name, max_age in STALE_THRESHOLDS.items():
        path = CACHE_DIR / f"{name}.json"
        if not path.exists():
            stale.append(f"{name} (missing)")
            continue
        age = now - path.stat().st_mtime
        if age > max_age:
            stale.append(f"{name} ({int(age)}s old > {max_age}s)")
    return stale


def cache_market_counts() -> dict[str, int]:
    counts = {}
    for path in sorted(CACHE_DIR.glob("*_cache.json")):
        name = path.stem.replace("_cache", "")
        try:
            payload = json.loads(path.read_text())
            counts[name] = len(payload.get("markets", []))
        except Exception:
            counts[name] = -1
    return counts


async def check_dashboard(client: httpx.AsyncClient) -> bool:
    try:
        resp = await client.get(DASHBOARD, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


async def main() -> None:
    log.info("Health watchdog started (poll=%ds)", POLL_SEC)
    consecutive_dead = 0
    consecutive_stale: dict[str, int] = {}

    async with httpx.AsyncClient() as client:
        while True:
            try:
                issues: list[str] = []

                # 1. Check main agent process
                if not is_agent_running():
                    consecutive_dead += 1
                    issues.append(f"Main agent DOWN (#{consecutive_dead})")
                    if consecutive_dead >= 2:
                        restart_agent()
                        consecutive_dead = 0
                else:
                    if consecutive_dead > 0:
                        log.info("Main agent recovered")
                    consecutive_dead = 0

                # 2. Check dashboard responsiveness
                dashboard_ok = await check_dashboard(client)
                if not dashboard_ok:
                    issues.append("Dashboard not responding")

                # 3. Check cache freshness
                stale = check_cache_freshness()
                for s in stale:
                    name = s.split(" ")[0]
                    consecutive_stale[name] = consecutive_stale.get(name, 0) + 1
                    if consecutive_stale[name] >= 3:
                        issues.append(f"STALE: {s}")
                for name in list(consecutive_stale):
                    if not any(name in s for s in stale):
                        consecutive_stale.pop(name, None)

                # 4. Log summary
                counts = cache_market_counts()
                count_str = " ".join(f"{k}={v}" for k, v in counts.items())
                if issues:
                    log.warning("HEALTH ISSUES: %s | markets: %s", " | ".join(issues), count_str)
                else:
                    log.info("All feeds healthy | %s", count_str)

            except Exception:
                log.exception("Health check cycle failed")

            await asyncio.sleep(POLL_SEC)


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Audit Agent — Bug Detection + Self-Improvement

Uses Claude Code CLI (Max Pro plan, no API cost) to autonomously:
1. Read recent logs and detect errors / anomalies
2. Read the relevant source files
3. Diagnose root causes
4. Write fixes directly to files
5. Create a git commit with the changes
6. Restart the main agent to pick up the fixes

Requires: `claude` CLI installed and authenticated on this machine.
Run once: `claude auth login`

Run: python agents/audit_agent.py [--once] [--no-restart]
Loop: every 6 hours (errors) or 24 hours (healthy)
Cost: $0 (uses Max Pro subscription)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("audit_agent")

LOG_FILE    = Path("/tmp/arb-agent.log")
PROJECT_DIR = Path(__file__).resolve().parent.parent
LOOP_ERRORS = 6 * 3600   # 6h when errors detected
LOOP_CLEAN  = 24 * 3600  # 24h when all clear

SYSTEM_PROMPT = """You are a senior Python engineer auditing an arbitrage detection bot for bugs.

Your ONLY job is to find and fix real bugs — not refactor, not add features.

The codebase:
- Python asyncio, FastAPI, httpx, aiosqlite
- Main loop: agent.py
- Feeds: src/feeds/ (polymarket_clob.py, kalshi.py, kalshi_sports.py, predictit.py, gemini.py, hyperliquid.py)
- Engine: src/engine/ (matcher.py, detector.py, pair_monitor.py)
- Agents: agents/ (matcher_agent.py uses local embeddings, monitor_agent.py, price_delta_agent.py)
- Storage: src/storage/db.py
- Dashboard: src/dashboard/app.py

Common bug classes to look for:
1. Exceptions in logs (TypeError, AttributeError, KeyError, ValueError)
2. Markets being dropped silently (wrong sport tag, missing prices, bad parsing)
3. Matching failures (markets that should pair but don't)
4. Cache/state inconsistencies
5. Detection gaps (arbs present in data but not detected)

When you find a bug:
1. Read the relevant source file to confirm
2. Write the minimal fix — change only what's broken
3. Do NOT add comments explaining what you did
4. When done, print a JSON summary as the last line of output:
   {"fixes": ["description1", "description2"], "skipped": ["reason1"]}

Be conservative: if you're not certain about a fix, skip it."""


def _run(cmd: str, cwd: str = str(PROJECT_DIR), timeout: int = 30) -> str:
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )
    return (result.stdout + result.stderr).strip()[:4000]


def gather_context() -> str:
    errors = _run(f"grep -E 'ERROR|Traceback|Exception' {LOG_FILE} | tail -30") if LOG_FILE.exists() else "(no log)"
    recent = _run(f"tail -50 {LOG_FILE}") if LOG_FILE.exists() else "(no log)"

    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:8000/api/state", timeout=3) as r:
            state = json.loads(r.read())
        arbs = state.get("arbs", [])
        arb_summary = f"{len(arbs)} arbs detected"
        mkt_counts = json.dumps(state.get("market_counts", {}))
    except Exception:
        arb_summary = "(agent not reachable)"
        mkt_counts = "{}"

    git_status = _run("git status --short | head -10")

    return f"""=== AUDIT CONTEXT ===
Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
Arb status: {arb_summary}
Market counts: {mkt_counts}

Recent errors in log:
{errors}

Last 50 log lines:
{recent}

Git status:
{git_status}

Project root: {PROJECT_DIR}
=== END CONTEXT ===

Investigate the errors above. Read source files as needed. Apply minimal fixes.
End your response with a JSON summary on the last line:
{{"fixes": ["what you fixed"], "skipped": ["what you left for human review"]}}"""


def has_errors() -> bool:
    if not LOG_FILE.exists():
        return False
    recent = _run(f"tail -200 {LOG_FILE}")
    return any(kw in recent for kw in ["ERROR", "Traceback", "Exception", "CRITICAL"])


def run_audit(auto_restart: bool) -> dict:
    context = gather_context()
    log.info("Running audit via Claude Code CLI...")

    prompt = f"{SYSTEM_PROMPT}\n\n{context}"

    try:
        result = subprocess.run(
            ["claude", "--print", "--allowedTools", "Bash,Read,Edit,Write",
             "--max-turns", "10", "-p", prompt],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute max
        )
        output = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0:
            log.warning("Claude CLI exited %d: %s", result.returncode, stderr[:500])

        log.info("Claude output:\n%s", output[-2000:] if len(output) > 2000 else output)

        # Parse JSON summary from last line
        fixes, skipped = [], []
        for line in reversed(output.splitlines()):
            line = line.strip()
            if line.startswith("{") and "fixes" in line:
                try:
                    summary = json.loads(line)
                    fixes   = summary.get("fixes", [])
                    skipped = summary.get("skipped", [])
                except json.JSONDecodeError:
                    pass
                break

        # If files were changed, commit and optionally restart
        git_diff = _run("git diff --name-only")
        files_changed = [f for f in git_diff.splitlines() if f.strip()]

        if files_changed:
            _run('git add -A && git commit -m "audit_agent: auto-fix bugs via Claude Code"')
            log.info("Committed changes: %s", files_changed)

            if auto_restart:
                agent_pid = _run("pgrep -f 'python agent.py'")
                if agent_pid:
                    _run(f"kill {agent_pid.strip()}")
                    log.info("Restarting agent.py...")
                    subprocess.Popen(
                        [".venv/bin/python", "agent.py"],
                        cwd=str(PROJECT_DIR),
                        stdout=open("/tmp/arb-agent.log", "a"),
                        stderr=subprocess.STDOUT,
                    )

        return {"fixes": fixes, "skipped": skipped, "files_changed": files_changed}

    except subprocess.TimeoutExpired:
        log.warning("Claude CLI timed out after 5 minutes")
        return {"fixes": [], "skipped": ["timed out"], "files_changed": []}
    except FileNotFoundError:
        log.error("'claude' CLI not found — run: npm install -g @anthropic-ai/claude-code && claude auth login")
        return {"fixes": [], "skipped": ["claude not installed"], "files_changed": []}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once",       action="store_true")
    parser.add_argument("--no-restart", action="store_true")
    args = parser.parse_args()

    auto_restart = not args.no_restart
    log.info("Audit agent started (Claude Code CLI, $0 cost) — auto_restart=%s", auto_restart)

    while True:
        try:
            if has_errors() or args.once:
                summary = run_audit(auto_restart)
                log.info("Audit complete — fixes: %s, changed: %s, skipped: %s",
                         summary["fixes"], summary["files_changed"], summary["skipped"])
            else:
                log.info("No errors in recent logs — skipping audit cycle")

        except Exception:
            log.exception("Audit cycle failed")

        if args.once:
            break

        loop_sec = LOOP_ERRORS if has_errors() else LOOP_CLEAN
        log.info("Next audit in %dh", loop_sec // 3600)
        await asyncio.sleep(loop_sec)


if __name__ == "__main__":
    asyncio.run(main())

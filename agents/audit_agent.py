#!/usr/bin/env python3
"""
Audit Agent — Bug Detection + Self-Improvement

Uses Claude Sonnet with tool_use to autonomously:
1. Read recent logs and detect errors / anomalies
2. Read the relevant source files
3. Diagnose root causes
4. Write fixes directly to files
5. Create a git commit with the changes
6. Restart the main agent to pick up the fixes

Run: python agents/audit_agent.py [--once] [--no-restart]
  --once       Run one audit cycle and exit (default: loop every hour)
  --no-restart Don't restart the main agent after applying fixes

The agent is scoped to bug fixes only — it will not refactor or add features.
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

import anthropic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("audit_agent")

LOG_FILE    = Path("/tmp/arb-agent.log")
PROJECT_DIR = Path(__file__).resolve().parent.parent
LOOP_SEC    = 3600  # 1 hour

SYSTEM_PROMPT = """You are a senior Python engineer auditing an arbitrage detection bot for bugs.

Your ONLY job is to find and fix real bugs — not refactor, not add features.

The codebase:
- Python asyncio, FastAPI, httpx, aiosqlite
- Main loop: agent.py
- Feeds: src/feeds/ (polymarket_clob.py, kalshi.py, kalshi_sports.py, predictit.py, gemini.py, hyperliquid.py)
- Engine: src/engine/ (matcher.py, detector.py, llm_matcher.py, pair_monitor.py)
- Storage: src/storage/db.py
- Dashboard: src/dashboard/app.py

Common bug classes to look for:
1. Exceptions in logs (TypeError, AttributeError, KeyError, ValueError)
2. Markets being dropped silently (wrong sport tag, missing prices, bad parsing)
3. Matching failures (markets that should pair but don't due to format issues)
4. Cache/state inconsistencies
5. Detection gaps (arbs present in data but not detected)

When you find a bug:
1. Read the relevant source file to confirm
2. Write the minimal fix — change only what's broken
3. Do NOT add comments explaining what you did
4. Call finish_audit() with a summary of all fixes applied

Be conservative: if you're not certain about a fix, skip it and note it in the summary."""


def _run(cmd: str, cwd: str = str(PROJECT_DIR)) -> str:
    """Run a shell command and return stdout."""
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=30
    )
    return (result.stdout + result.stderr).strip()[:4000]


# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "read_file",
        "description": "Read a source file from the project. Path is relative to project root.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path, e.g. src/feeds/kalshi.py"},
                "start_line": {"type": "integer", "description": "First line to read (1-indexed, optional)"},
                "end_line":   {"type": "integer", "description": "Last line to read (optional)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write (overwrite) a source file with a bug fix applied. Only use for minimal, targeted fixes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path":    {"type": "string", "description": "Relative file path"},
                "content": {"type": "string", "description": "Complete new file content"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "read_logs",
        "description": "Read recent agent logs, optionally filtered by a grep pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "lines":   {"type": "integer", "description": "Number of recent lines (default 100)"},
                "pattern": {"type": "string",  "description": "grep pattern to filter (optional)"},
            },
        },
    },
    {
        "name": "run_command",
        "description": "Run a safe read-only command (grep, find, git diff, python -c syntax check). No destructive commands.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run from project root"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "finish_audit",
        "description": "Signal that the audit is complete. Call this when all bugs are fixed or investigated.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fixes_applied": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of bug fixes applied (one sentence each)",
                },
                "bugs_found_not_fixed": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Bugs found but not fixed (too risky / need human review)",
                },
            },
            "required": ["fixes_applied"],
        },
    },
]


def handle_tool(name: str, inputs: dict, files_changed: list[str]) -> str:
    if name == "read_file":
        path = PROJECT_DIR / inputs["path"]
        if not path.exists():
            return f"File not found: {inputs['path']}"
        lines = path.read_text().splitlines()
        start = inputs.get("start_line", 1) - 1
        end   = inputs.get("end_line", len(lines))
        chunk = lines[max(0, start):end]
        numbered = "\n".join(f"{i+start+1:4d}  {l}" for i, l in enumerate(chunk))
        return numbered[:6000]

    elif name == "write_file":
        # Safety: only allow src/ and agents/ directories
        rel = inputs["path"]
        if not (rel.startswith("src/") or rel.startswith("agents/")):
            return f"BLOCKED: can only write to src/ or agents/ — got '{rel}'"
        path = PROJECT_DIR / rel
        # Syntax check before writing
        syntax = subprocess.run(
            [sys.executable, "-c", f"import ast; ast.parse(open('{path}').read())"],
            capture_output=True, text=True,
        )
        # Write and syntax-check new content
        tmp = path.with_suffix(".tmp_audit")
        tmp.write_text(inputs["content"])
        syntax_new = subprocess.run(
            [sys.executable, "-m", "py_compile", str(tmp)],
            capture_output=True, text=True,
        )
        if syntax_new.returncode != 0:
            tmp.unlink(missing_ok=True)
            return f"SYNTAX ERROR — file NOT written:\n{syntax_new.stderr}"
        tmp.rename(path)
        files_changed.append(rel)
        log.info("Wrote fix to %s", rel)
        return f"Written: {rel}"

    elif name == "read_logs":
        if not LOG_FILE.exists():
            return "Log file not found"
        lines = inputs.get("lines", 100)
        pattern = inputs.get("pattern", "")
        if pattern:
            out = _run(f"grep -E '{pattern}' {LOG_FILE} | tail -{lines}")
        else:
            out = _run(f"tail -{lines} {LOG_FILE}")
        return out or "(empty)"

    elif name == "run_command":
        cmd = inputs["command"]
        # Blocklist destructive commands
        blocked = ["rm ", "mv ", "kill", "pkill", "reboot", "shutdown", "sudo", "> /"]
        for b in blocked:
            if b in cmd:
                return f"BLOCKED: '{b}' not allowed"
        return _run(cmd) or "(no output)"

    elif name == "finish_audit":
        return "__DONE__"

    return f"Unknown tool: {name}"


def gather_context() -> str:
    """Build initial context for Claude."""
    # Recent errors from logs
    errors = _run(f"grep -E 'ERROR|Traceback|Exception' {LOG_FILE} | tail -30") if LOG_FILE.exists() else "(no log)"
    # Arb stats from API
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:8000/api/state", timeout=3) as r:
            state = json.loads(r.read())
        arbs = state.get("arbs", [])
        sports = {}
        for a in arbs:
            sports[a.get("sport", "?")] = sports.get(a.get("sport", "?"), 0) + 1
        arb_summary = f"{len(arbs)} arbs detected — " + ", ".join(f"{s}={c}" for s, c in sorted(sports.items()))
        mkt_counts = json.dumps(state.get("market_counts", {}))
    except Exception:
        arb_summary = "(agent not running or not reachable)"
        mkt_counts = "{}"
    # Git status
    git_status = _run("git status --short | head -10")

    return f"""=== AUDIT CONTEXT ===
Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
Arb status: {arb_summary}
Market counts: {mkt_counts}

Recent errors in log:
{errors}

Git status:
{git_status}
=== END CONTEXT ===

Start by reading recent logs for errors, then investigate the root cause in the source files.
Focus on bugs that explain missing arbs or runtime errors. Fix only what you're confident about."""


def run_audit(client: anthropic.Anthropic, auto_restart: bool) -> dict:
    """Run one audit cycle. Returns summary of changes."""
    files_changed: list[str] = []
    context = gather_context()
    log.info("Starting audit cycle")

    messages = [{"role": "user", "content": context}]
    fixes_applied: list[str] = []
    bugs_not_fixed: list[str] = []
    max_turns = 20

    for turn in range(max_turns):
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Collect assistant message
        messages.append({"role": "assistant", "content": resp.content})

        # Process tool calls
        tool_results = []
        done = False
        for block in resp.content:
            if block.type == "tool_use":
                result = handle_tool(block.name, block.input, files_changed)
                if result == "__DONE__":
                    done = True
                    fixes_applied = block.input.get("fixes_applied", [])
                    bugs_not_fixed = block.input.get("bugs_found_not_fixed", [])
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result if result != "__DONE__" else "Audit complete.",
                })

        if done:
            break

        if resp.stop_reason == "end_turn" and not tool_results:
            break

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    # Commit changes if any
    if files_changed:
        staged = " ".join(files_changed)
        _run(f"git add {staged}")
        commit_msg = "fix: audit agent bug fixes\n\n" + "\n".join(f"- {f}" for f in fixes_applied)
        _run(f'git commit -m "{commit_msg}"')
        log.info("Committed %d changed files", len(files_changed))

        if auto_restart:
            log.info("Restarting main agent to pick up fixes...")
            _run("pkill -f 'python agent.py' || true")
            time.sleep(2)
            _run("rm -f state/agent.lock")
            subprocess.Popen(
                [sys.executable, "agent.py"],
                cwd=str(PROJECT_DIR),
                stdout=open("/tmp/arb-agent.log", "a"),
                stderr=subprocess.STDOUT,
            )
            log.info("Main agent restarted")

    return {
        "files_changed": files_changed,
        "fixes_applied": fixes_applied,
        "bugs_not_fixed": bugs_not_fixed,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once",       action="store_true", help="Run once and exit")
    parser.add_argument("--no-restart", action="store_true", help="Don't restart agent after fixes")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    auto_restart = not args.no_restart

    log.info("Audit agent started (loop=%s, auto_restart=%s)",
             not args.once, auto_restart)

    while True:
        try:
            summary = run_audit(client, auto_restart)
            log.info("Audit complete — fixes: %d, changed: %s, skipped: %d",
                     len(summary["fixes_applied"]),
                     summary["files_changed"] or "none",
                     len(summary["bugs_not_fixed"]))
            for fix in summary["fixes_applied"]:
                log.info("  FIXED: %s", fix)
            for bug in summary["bugs_not_fixed"]:
                log.info("  SKIPPED: %s", bug)
        except Exception:
            log.exception("Audit cycle failed")

        if args.once:
            break

        log.info("Next audit in %ds", LOOP_SEC)
        await asyncio.sleep(LOOP_SEC)


if __name__ == "__main__":
    asyncio.run(main())

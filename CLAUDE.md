# Arbitrage Agent — Claude Behavior Rules

## Anti-Token-Maxxing Protocol

**No explanations before doing.** Do the thing, then say what you did in one line.

**No restarts unless broken.** The agent is running. Don't kill and restart it to pick up a code change unless the change is in a file that doesn't hot-reload. Check first.

**One fix at a time.** Make the change, verify it worked, move on. Don't batch 3 fixes and introduce 2 new bugs.

**No lengthy diagnosis monologues.** Read the log, find the line that matters, fix it.

**Verify before reporting.** Don't say "arbs should show now" — check the API and confirm they do.

**Log reading discipline.** When checking logs, grep for the specific signal. Don't tail 30 lines and describe them.

**No apology loops.** Don't acknowledge being wrong, then explain why, then apologize again. Fix it.

## Project Context

- **Stack:** Python 3.9, FastAPI/uvicorn, httpx async, single `agent.py` process
- **Venv:** `.venv/bin/python` — always use this, never bare `python`
- **Start:** `cd /Users/georgenagib/arbitrage-agent && nohup .venv/bin/python agent.py >> /tmp/arb-agent.log 2>&1 &`
- **Logs:** `tail -f /tmp/arb-agent.log` or `grep` for specific signals
- **API:** `curl http://localhost:8000/api/state` to check current arbs and market counts
- **Lock file:** `state/agent.lock` — delete it if agent crashes and won't restart

## Key Files

- `src/feeds/polymarket_clob.py` — Polymarket feed
- `src/feeds/kalshi_sports.py` — Kalshi live sports feed
- `src/feeds/kalshi.py` — Kalshi prediction markets feed
- `src/feeds/kalshi_rate.py` — shared rate limiter (both Kalshi feeds use `kalshi_request`)
- `src/engine/matcher.py` — cross-platform market matching
- `src/engine/detector.py` — arb detection with fee-adjusted prices
- `src/engine/fees.py` — per-platform fee models
- `agent.py` — main loop, feed orchestration, FastAPI server

## Known Issues / Current State

- Kalshi prediction feed runs as background task after first cycle; first cycle fetches synchronously when cache is empty
- KalshiSports and Kalshi prediction share one rate limiter (`kalshi_request`) to avoid competing for API quota
- Polymarket game events fetched with dual sort (volume24hr + volume) to catch both live and pre-game markets

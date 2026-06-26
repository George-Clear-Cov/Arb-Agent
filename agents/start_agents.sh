#!/usr/bin/env bash
# Start all background agents. Run from project root.
# Usage: bash agents/start_agents.sh

set -e
cd "$(dirname "$0")/.."

VENV=".venv/bin/python"
LOGS="/tmp"

echo "Starting all agents..."

# Kill any existing agent processes
pkill -f "agents/matcher_agent.py" 2>/dev/null || true
pkill -f "agents/monitor_agent.py" 2>/dev/null || true
pkill -f "agents/audit_agent.py"   2>/dev/null || true

sleep 1

# Matcher Agent — LLM market matching loop
nohup $VENV agents/matcher_agent.py >> $LOGS/matcher-agent.log 2>&1 &
echo "Matcher agent PID: $!"

# Monitor Agent — per-pair arb detection
nohup $VENV agents/monitor_agent.py >> $LOGS/monitor-agent.log 2>&1 &
echo "Monitor agent PID: $!"

# Audit Agent — runs once on startup then every hour
nohup $VENV agents/audit_agent.py >> $LOGS/audit-agent.log 2>&1 &
echo "Audit agent PID: $!"

echo ""
echo "Logs:"
echo "  Matcher:  tail -f $LOGS/matcher-agent.log"
echo "  Monitor:  tail -f $LOGS/monitor-agent.log"
echo "  Audit:    tail -f $LOGS/audit-agent.log"

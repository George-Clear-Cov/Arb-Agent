#!/usr/bin/env bash
# Start all background agents. Run from project root.
# Usage: bash agents/start_agents.sh

set -e
cd "$(dirname "$0")/.."

VENV=".venv/bin/python"
LOGS="/tmp"

echo "Stopping existing agents..."
pkill -f "agents/matcher_agent.py"    2>/dev/null || true
pkill -f "agents/monitor_agent.py"    2>/dev/null || true
pkill -f "agents/audit_agent.py"      2>/dev/null || true
pkill -f "agents/price_delta_agent.py" 2>/dev/null || true
pkill -f "agents/position_settler.py" 2>/dev/null || true
pkill -f "agents/health_watchdog.py"  2>/dev/null || true
pkill -f "agents/brain_decay.py"      2>/dev/null || true
sleep 1

echo "Starting agents..."

nohup $VENV agents/matcher_agent.py     >> $LOGS/matcher-agent.log     2>&1 & echo "matcher:       $!"
nohup $VENV agents/monitor_agent.py     >> $LOGS/monitor-agent.log     2>&1 & echo "monitor:       $!"
nohup $VENV agents/price_delta_agent.py >> $LOGS/price-delta-agent.log 2>&1 & echo "price_delta:   $!"
nohup $VENV agents/position_settler.py  >> $LOGS/position-settler.log  2>&1 & echo "position:      $!"
nohup $VENV agents/health_watchdog.py   >> $LOGS/health-watchdog.log   2>&1 & echo "watchdog:      $!"
nohup $VENV agents/brain_decay.py       >> $LOGS/brain-decay.log       2>&1 & echo "brain_decay:   $!"

echo ""
echo "All agents started. Logs:"
echo "  tail -f $LOGS/matcher-agent.log"
echo "  tail -f $LOGS/monitor-agent.log"
echo "  tail -f $LOGS/price-delta-agent.log"
echo "  tail -f $LOGS/position-settler.log"
echo "  tail -f $LOGS/health-watchdog.log"
echo "  tail -f $LOGS/brain-decay.log"

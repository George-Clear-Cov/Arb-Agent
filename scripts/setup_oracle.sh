#!/bin/bash
# Run this ONCE on the Oracle Cloud VM to install dependencies and create the systemd service.
# Usage: bash setup_oracle.sh
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_USER="$(whoami)"
DATA_DIR="$APP_DIR/data"

echo "==> App dir: $APP_DIR"
echo "==> Service user: $SERVICE_USER"

# ---- System packages --------------------------------------------------------
echo "==> Installing system packages..."
sudo apt-get update -q
sudo apt-get install -y -q python3.9 python3.9-venv python3.9-dev gcc

# ---- Python venv ------------------------------------------------------------
if [ ! -d "$APP_DIR/.venv" ]; then
    echo "==> Creating venv..."
    python3.9 -m venv "$APP_DIR/.venv"
fi

echo "==> Installing Python requirements..."
"$APP_DIR/.venv/bin/pip" install --quiet --upgrade pip
"$APP_DIR/.venv/bin/pip" install --quiet -r "$APP_DIR/requirements.txt"

# ---- Data directory ---------------------------------------------------------
mkdir -p "$DATA_DIR/state"
echo "==> Data dir: $DATA_DIR"

# ---- Systemd service --------------------------------------------------------
echo "==> Creating systemd service..."
sudo tee /etc/systemd/system/arb-agent.service > /dev/null <<EOF
[Unit]
Description=Arbitrage Agent
After=network-online.target
Wants=network-online.target

[Service]
User=$SERVICE_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
Environment=DATA_DIR=$DATA_DIR
ExecStart=$APP_DIR/.venv/bin/python agent.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=arb-agent

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable arb-agent
sudo systemctl restart arb-agent

echo ""
echo "==> Done! Service status:"
sleep 2
sudo systemctl status arb-agent --no-pager -l | head -20
echo ""
echo "==> Logs: sudo journalctl -u arb-agent -f"
echo "==> Dashboard: http://$(curl -s ifconfig.me):8000"

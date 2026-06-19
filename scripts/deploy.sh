#!/bin/bash
# Push code to Oracle Cloud VM and restart the service.
# Usage: ORACLE_HOST=<your-vm-ip> bash scripts/deploy.sh
#
# First deploy (fresh VM):
#   ORACLE_HOST=1.2.3.4 FIRST_DEPLOY=1 bash scripts/deploy.sh
set -euo pipefail

ORACLE_HOST="${ORACLE_HOST:?Error: set ORACLE_HOST to your Oracle VM public IP}"
ORACLE_USER="${ORACLE_USER:-ubuntu}"
SSH_KEY="${SSH_KEY:-}"   # optional: path to private key, e.g. ~/.ssh/oracle_arb
LOCAL_DIR="/Users/georgenagib/arbitrage-agent"
REMOTE_DIR="/home/$ORACLE_USER/arbitrage-agent"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
[ -n "$SSH_KEY" ] && SSH_OPTS="$SSH_OPTS -i $SSH_KEY"

echo "==> Syncing code to $ORACLE_USER@$ORACLE_HOST:$REMOTE_DIR ..."
rsync -az \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.db' \
  --exclude='state/' \
  --exclude='data/' \
  --exclude='logs/' \
  --exclude='.git/' \
  ${SSH_KEY:+-e "ssh -i $SSH_KEY"} \
  "$LOCAL_DIR/" "$ORACLE_USER@$ORACLE_HOST:$REMOTE_DIR/"

echo "==> Copying .env ..."
scp $SSH_OPTS "$LOCAL_DIR/.env" "$ORACLE_USER@$ORACLE_HOST:$REMOTE_DIR/.env"

if [ "${FIRST_DEPLOY:-0}" = "1" ]; then
    echo "==> First deploy — running setup_oracle.sh on VM ..."
    ssh $SSH_OPTS "$ORACLE_USER@$ORACLE_HOST" "bash $REMOTE_DIR/scripts/setup_oracle.sh"
else
    echo "==> Updating pip requirements and restarting service ..."
    ssh $SSH_OPTS "$ORACLE_USER@$ORACLE_HOST" "
        $REMOTE_DIR/.venv/bin/pip install --quiet -r $REMOTE_DIR/requirements.txt
        sudo systemctl restart arb-agent
        sleep 2
        sudo systemctl status arb-agent --no-pager | head -10
    "
fi

echo "==> Deployed! Dashboard: http://$ORACLE_HOST:8000"

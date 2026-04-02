#!/bin/bash
# Dryft — Deploy to VPS
#
# Run this locally after a Claude Code session.
# Pushes code to GitHub, then SSHs to VPS to pull and restart.
#
# Prerequisites:
#   - Git remote "origin" pointed at your private GitHub repo
#   - VPS_HOST set in your environment (or edit the default below)
#   - SSH key configured for VPS access
#   - VPS has deploy key configured for the GitHub repo (see deploy/README.md)
#
# Usage:
#   bash deploy/push.sh

set -e

VPS_HOST="${VPS_HOST:-dryft}"  # SSH config alias or user@host
VPS_DIR="/opt/dryft"
SERVICE="dryft-bot"

echo "=== Dryft Deploy ==="

# Step 1: Push to GitHub
echo ""
echo "[1/4] Pushing to GitHub..."
git push origin master
echo "  Done."

# Step 2: SSH to VPS, pull as dryft user (owns repo + deploy key), install deps, restart
echo ""
echo "[2/4] Pulling on VPS..."
ssh "$VPS_HOST" "su - dryft -s /bin/bash -c 'cd $VPS_DIR && git pull origin master'"

echo ""
echo "[3/4] Installing dependencies..."
ssh "$VPS_HOST" "pip install -r $VPS_DIR/requirements-prod.txt --quiet --break-system-packages"

echo ""
echo "[4/4] Restarting service..."
ssh "$VPS_HOST" "systemctl restart $SERVICE"

# Verify
echo ""
echo "=== Verifying ==="
ssh "$VPS_HOST" "systemctl is-active $SERVICE && echo 'Service is running.' || echo 'WARNING: Service failed to start. Check: journalctl -u $SERVICE -n 30'"

echo ""
echo "Deploy complete."

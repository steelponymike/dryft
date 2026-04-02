#!/bin/bash
# Dryft — VPS Initial Setup
#
# Run as root on a fresh Ubuntu VPS.
# This is a reference script. Read it first, then run step by step.
#
# After running this, you still need to:
#   1. Add the deploy key to GitHub (printed at the end)
#   2. Clone the repo (printed at the end)
#   3. Edit /opt/dryft/.env with real values
#   4. Copy state files from local machine via scp
#   5. Enable and start the service

set -e

echo "=== Dryft VPS Setup ==="

# Create dedicated user (no login shell, no home directory beyond /opt/dryft)
echo "[1/7] Creating dryft user..."
useradd -r -s /bin/false -d /opt/dryft dryft 2>/dev/null || echo "  User already exists."

# Install Python
echo "[2/7] Installing Python..."
apt update -qq && apt install -y -qq python3 python3-venv python3-pip git ffmpeg > /dev/null

# Create app directory structure
echo "[3/7] Creating directories..."
mkdir -p /opt/dryft/state
mkdir -p /opt/dryft/backups
chown -R dryft:dryft /opt/dryft
chmod 700 /opt/dryft/state

# Generate deploy key for GitHub (read-only access to private repo)
echo "[4/7] Generating deploy key..."
if [ ! -f /opt/dryft/.ssh/id_ed25519 ]; then
    mkdir -p /opt/dryft/.ssh
    ssh-keygen -t ed25519 -f /opt/dryft/.ssh/id_ed25519 -N "" -C "dryft-vps-deploy"
    chown -R dryft:dryft /opt/dryft/.ssh
    chmod 700 /opt/dryft/.ssh
    chmod 600 /opt/dryft/.ssh/id_ed25519

    # Configure SSH to use deploy key for GitHub
    cat > /opt/dryft/.ssh/config << 'SSHCONF'
Host github.com
    HostName github.com
    User git
    IdentityFile /opt/dryft/.ssh/id_ed25519
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
SSHCONF
    chmod 600 /opt/dryft/.ssh/config
    chown dryft:dryft /opt/dryft/.ssh/config
    echo "  Deploy key generated."
else
    echo "  Deploy key already exists."
fi

# Install systemd service
echo "[5/7] Installing systemd service..."
cp /opt/dryft/deploy/dryft-bot.service /etc/systemd/system/ 2>/dev/null || echo "  Service file not found yet (clone repo first, then re-run this step)."
systemctl daemon-reload

# Set up daily backup cron
echo "[6/7] Setting up daily backup..."
cp /opt/dryft/deploy/backup.sh /opt/dryft/backup.sh 2>/dev/null || true
chmod +x /opt/dryft/backup.sh 2>/dev/null || true
# Add cron job (daily at 3am)
(crontab -u dryft -l 2>/dev/null; echo "0 3 * * * /opt/dryft/backup.sh") | sort -u | crontab -u dryft - 2>/dev/null || echo "  Cron setup skipped (set up after clone)."

# Allow dryft user to restart its own service without full sudo
echo "[7/7] Configuring sudoers for service restart..."
cat > /etc/sudoers.d/dryft << 'SUDOERS'
dryft ALL=(ALL) NOPASSWD: /bin/systemctl restart dryft-bot
dryft ALL=(ALL) NOPASSWD: /bin/systemctl stop dryft-bot
dryft ALL=(ALL) NOPASSWD: /bin/systemctl start dryft-bot
dryft ALL=(ALL) NOPASSWD: /bin/systemctl status dryft-bot
SUDOERS
chmod 440 /etc/sudoers.d/dryft

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo ""
echo "1. Add this deploy key to your GitHub repo:"
echo "   (Settings > Deploy keys > Add deploy key, read-only access)"
echo ""
cat /opt/dryft/.ssh/id_ed25519.pub
echo ""
echo "2. Clone the repo as the dryft user:"
echo "   sudo -u dryft git clone git@github.com:YOUR_USER/dryft.git /opt/dryft/repo"
echo "   # Then move contents: cp -r /opt/dryft/repo/* /opt/dryft/"
echo "   # Or clone directly into /opt/dryft if it's empty"
echo ""
echo "3. Set up Python venv:"
echo "   cd /opt/dryft && python3 -m venv venv"
echo "   source venv/bin/activate && pip install -r requirements-prod.txt"
echo ""
echo "4. Create .env with real values:"
echo "   cp .env.example .env && nano .env"
echo ""
echo "5. Copy state files from local machine:"
echo "   scp -r state/ dryft@YOUR_VPS:/opt/dryft/state/"
echo ""
echo "6. Start the service:"
echo "   systemctl enable dryft-bot && systemctl start dryft-bot"
echo ""
echo "7. Check logs:"
echo "   journalctl -u dryft-bot -f"

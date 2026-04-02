# Dryft Cloud Deployment

Single-user Telegram bot on a cheap Linux VPS. No Docker, no CI/CD, no web server. The bot polls Telegram over outbound HTTPS. SSH is the only inbound port.


## VPS Requirements

- 1 vCPU, 512MB to 1GB RAM (all heavy compute is API calls)
- Ubuntu 22.04 or 24.04
- SSH access
- Any VPS with 2+ vCPU, 4GB RAM (e.g. Hetzner, DigitalOcean, ~$4-6/month)


## One-Time Setup

### 1. Provision the VPS

Spin up the VPS through your provider's dashboard. Note the IP address. Add an SSH key during creation or copy one after.

Add an SSH alias to your local `~/.ssh/config`:

```
Host dryft
    HostName YOUR_VPS_IP
    User root
    IdentityFile ~/.ssh/your_key
```

Verify: `ssh dryft`

### 2. Create the private GitHub repo

Create a private repo (e.g., `dryft`) on GitHub. Do not initialize with a README.

Locally, add the remote and push:

```bash
cd /c/Projects/Dryft
git remote add origin git@github.com:YOUR_USER/dryft.git
git push -u origin master
```

### 3. Run the VPS setup script

```bash
ssh dryft
# Download or paste setup.sh, then:
bash setup.sh
```

This creates the dryft user, installs Python, generates a deploy key, and prints the next steps.

### 4. Add the deploy key to GitHub

The setup script prints the public key. Copy it.

Go to your GitHub repo: Settings > Deploy keys > Add deploy key.
- Title: `dryft-vps`
- Key: paste the public key
- Allow write access: NO (read-only)

### 5. Clone the repo on the VPS

```bash
# As root, switch to dryft user for the clone
sudo -u dryft git clone git@github.com:YOUR_USER/dryft.git /opt/dryft/code

# Move files into place (or re-clone directly if /opt/dryft is clean)
cp -r /opt/dryft/code/* /opt/dryft/
cp -r /opt/dryft/code/.gitignore /opt/dryft/
cp -r /opt/dryft/code/.git /opt/dryft/
rm -rf /opt/dryft/code
```

Or if /opt/dryft is empty except for .ssh and state:
```bash
sudo -u dryft git clone git@github.com:YOUR_USER/dryft.git /tmp/dryft-clone
cp -r /tmp/dryft-clone/* /opt/dryft/
cp -r /tmp/dryft-clone/.git /opt/dryft/
cp -r /tmp/dryft-clone/.gitignore /opt/dryft/
rm -rf /tmp/dryft-clone
chown -R dryft:dryft /opt/dryft
```

### 6. Set up the Python environment

```bash
cd /opt/dryft
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-prod.txt
```

### 7. Create the .env file

```bash
cp .env.example .env
nano .env
# Fill in all four values with NEW keys (rotated from local dev)
```

### 8. Copy state files from local machine

From your local machine:
```bash
scp -r /c/Projects/Dryft/state/* dryft:/opt/dryft/state/
```

Then on the VPS:
```bash
chown -R dryft:dryft /opt/dryft/state
chmod 700 /opt/dryft/state
```

### 9. Install and start the service

```bash
cp /opt/dryft/deploy/dryft-bot.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable dryft-bot
systemctl start dryft-bot
```

### 10. Verify

```bash
systemctl status dryft-bot
journalctl -u dryft-bot -f
```

Send a message to the bot in Telegram. You should get a response.


## Deploying Updates

After a Claude Code session (or any local code change):

```bash
bash deploy/push.sh
```

This pushes to GitHub, SSHs to the VPS, pulls the new code, and restarts the service.

If you want to do it manually:

```bash
git push origin master
ssh dryft "cd /opt/dryft && git pull origin master && sudo systemctl restart dryft-bot"
```

### What gets deployed

Only tracked files in git. The .gitignore ensures:
- state/ never enters git (stays on VPS, untouched by pulls)
- .env never enters git (stays on VPS, untouched by pulls)
- __pycache__/ is ignored
- conversations.json is ignored

A `git pull` on the VPS updates code only. State and secrets are untouched.


## Daily Operations

### Check logs
```bash
ssh dryft "journalctl -u dryft-bot -n 50"
```

### Follow logs live
```bash
ssh dryft "journalctl -u dryft-bot -f"
```

### Restart the bot
```bash
ssh dryft "sudo systemctl restart dryft-bot"
```

### Check service status
```bash
ssh dryft "systemctl status dryft-bot"
```

### Force-save state
Send `/save` to the bot in Telegram.

### Check herd health
Send `/status` to the bot in Telegram.


## Backups

State files are backed up daily at 3am VPS time via cron. Backups are compressed tar archives in `/opt/dryft/backups/`. Last 14 days retained.

### Manual backup
```bash
ssh dryft "bash /opt/dryft/backup.sh"
```

### Download a backup locally
```bash
scp dryft:/opt/dryft/backups/state_LATEST.tar.gz ./
```

### Restore from backup
```bash
ssh dryft
sudo systemctl stop dryft-bot
cd /opt/dryft
tar xzf backups/state_YYYYMMDD_HHMMSS.tar.gz
chown -R dryft:dryft state/
sudo systemctl start dryft-bot
```


## Environment Variables

| Variable | Required | Description |
|---|---|---|
| ANTHROPIC_API_KEY | Yes | Anthropic API key for conversation model |
| TELEGRAM_BOT_TOKEN | Yes | Telegram bot token from @BotFather |
| TELEGRAM_USER_ID | Yes | Owner's Telegram user ID (bot rejects all others) |
| GROQ_API_KEY | No | Groq API key for voice transcription (Whisper). Without it, voice input is disabled. |
| TOMORROW_API_KEY | No | Tomorrow.io API key for hyperlocal weather. Without it, weather context is disabled. |
| GOOGLE_SHEETS_CREDENTIALS | No | Path to Google Sheets service account JSON. Without it, sheet reading is disabled. |


## Cron Jobs

### Morning Message (Phase 12A)

Sends a proactive morning briefing at 8:15 AM MDT (14:15 UTC) on weekdays.

```
15 14 * * 1-5 cd /opt/dryft && python morning_message.py >> /var/log/dryft-morning.log 2>&1
```

Add manually on VPS after deploy:
```bash
ssh dryft
crontab -e
# Add the line above, save
```

To test manually before waiting for cron:
```bash
ssh dryft "cd /opt/dryft && python morning_message.py"
```

Morning message preferences accumulate in the grass layer. The user can reply to any morning message to adjust future messages.

## Security Notes

- The dryft user has no login shell and no home directory outside /opt/dryft
- systemd sandboxing: ProtectSystem=strict, ProtectHome=true, PrivateTmp=true, NoNewPrivileges=true
- Only /opt/dryft/state is writable by the service
- The deploy key is read-only (cannot push to the repo)
- SSH is the only inbound port. The bot uses outbound HTTPS polling only.
- State files and .env are never in git. A git pull cannot overwrite them.
- sudoers limits: dryft user can only restart/stop/start/status the dryft-bot service

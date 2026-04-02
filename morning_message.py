"""
Dryft — Morning Message

Standalone script. Runs via cron at 8:15 AM MDT (14:15 UTC) on weekdays.
Sends a proactive morning briefing to the user via Telegram.

Synthesizes: weather, active projects, herd flags, user preferences.
Preferences accumulate in the grass layer under subject "morning_message_preferences".

Usage:
    python morning_message.py

Cron (VPS, UTC):
    15 14 * * 1-5 cd /opt/dryft && python morning_message.py >> /var/log/dryft-morning.log 2>&1
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project directory is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anthropic
from urllib.request import urlopen, Request


STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state")
MORNING_MODEL = "claude-sonnet-4-20250514"


def _load_json(path: str) -> dict | list:
    """Load a JSON file, returning empty dict/list on failure."""
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _atomic_json_write(path: str, data, indent: int = 2):
    """Write JSON atomically."""
    import tempfile
    dir_name = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def get_active_projects(state_dir: str, days: int = 7) -> list[str]:
    """Get memories activated in the last N days worth of queries."""
    herd = _load_json(os.path.join(state_dir, "herd_live.json"))
    meta = _load_json(os.path.join(state_dir, "herd_meta.json"))
    query_count = meta.get("query_count", 0)

    # Rough heuristic: ~10 queries/day
    recency_threshold = query_count - (days * 10)

    active = []
    if isinstance(herd, list):
        for mem in herd:
            if mem.get("status") != "active":
                continue
            last_active = mem.get("last_activated_at", 0)
            if last_active >= recency_threshold and last_active > 0:
                active.append({
                    "content": mem.get("content", ""),
                    "last_activated_at": last_active,
                    "activation_count": mem.get("activation_count", 0),
                })

    # Sort by recency
    active.sort(key=lambda x: x["last_activated_at"], reverse=True)
    return [m["content"][:150] for m in active[:10]]


def get_morning_preferences(state_dir: str) -> list[str]:
    """Pull grass layer entries with morning message preferences."""
    grass = _load_json(os.path.join(state_dir, "grass_live.json"))
    prefs = []
    if isinstance(grass, list):
        for entry in grass:
            entry_id = entry.get("id", "")
            content = entry.get("content", "")
            if "morning" in entry_id.lower() or "morning" in content.lower():
                prefs.append(content)
    return prefs


def get_watchlist_flags(state_dir: str) -> list[str]:
    """Check for unseen suspicious cull flags."""
    watchlist = _load_json(os.path.join(state_dir, "watchlist.json"))
    if not watchlist or not isinstance(watchlist, dict):
        return []
    unseen = watchlist.get("unseen_count", 0)
    if unseen == 0:
        return []
    entries = watchlist.get("entries", [])
    recent = entries[-unseen:] if unseen <= len(entries) else entries
    return [f"{e.get('content_summary', 'Unknown')} ({', '.join(e.get('flags', []))})" for e in recent]


def get_unresolved_conflicts(state_dir: str) -> list[str]:
    """Check for unresolved conflicts."""
    data = _load_json(os.path.join(state_dir, "conflicts_live.json"))
    unresolved = []
    conflict_list = data.get("conflicts", []) if isinstance(data, dict) else data if isinstance(data, list) else []
    for c in conflict_list:
        if isinstance(c, dict) and c.get("status") in ("pending", "deferred"):
            unresolved.append(c.get("description", c.get("subject", c.get("conflict_id", "unknown"))))
    return unresolved


def get_weather(api_key: str) -> str:
    """Fetch weather summary."""
    try:
        from weather import get_weather_summary
        return get_weather_summary(api_key, include_hourly=True)
    except Exception as e:
        return f"Weather unavailable: {e}"


def build_morning_prompt(weather: str, projects: list[str], flags: list[str],
                         conflicts: list[str], preferences: list[str]) -> str:
    """Build the synthesis prompt for the morning message."""
    sections = []

    sections.append(f"Today's weather:\n{weather}")

    if projects:
        sections.append("Recently active topics (from memory activity):\n" +
                        "\n".join(f"- {p}" for p in projects))
    else:
        sections.append("No recently active topics.")

    if flags:
        sections.append("Suspicious cull flags:\n" +
                        "\n".join(f"- {f}" for f in flags))

    if conflicts:
        sections.append("Unresolved memory conflicts:\n" +
                        "\n".join(f"- {c}" for c in conflicts))

    context = "\n\n".join(sections)

    pref_block = ""
    if preferences:
        pref_block = ("\n\nUser's morning message preferences (MUST follow these):\n" +
                      "\n".join(f"- {p}" for p in preferences))

    return (
        "Generate a brief morning briefing. Include: today's weather, "
        "currently active projects based on recent memory "
        "activity, any flagged herd issues. Keep it under 200 words. "
        "No headers. Conversational tone. Do not use bullet points unless "
        "the user has requested them."
        f"{pref_block}\n\n"
        f"Context:\n{context}"
    )


def send_telegram_message(bot_token: str, user_id: str, text: str) -> int:
    """Send a message via Telegram bot API. Returns the message_id."""
    import urllib.parse
    url = (
        f"https://api.telegram.org/bot{bot_token}/sendMessage?"
        f"chat_id={user_id}&text={urllib.parse.quote(text)}"
    )
    req = Request(url)
    with urlopen(req, timeout=10) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result.get("result", {}).get("message_id", 0)


def store_morning_message_id(state_dir: str, message_id: int):
    """Store the morning message ID in herd_meta for reply detection."""
    meta_path = os.path.join(state_dir, "herd_meta.json")
    meta = _load_json(meta_path)
    if not isinstance(meta, dict):
        meta = {}
    meta["last_morning_message_id"] = message_id
    _atomic_json_write(meta_path, meta)


def main():
    print(f"[{datetime.now(timezone.utc).isoformat()}] Morning message starting...")

    # Required env vars
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    user_id = os.environ.get("TELEGRAM_USER_ID")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    weather_key = os.environ.get("TOMORROW_API_KEY")

    if not bot_token or not user_id:
        print("ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_USER_ID required.")
        sys.exit(1)

    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY required.")
        sys.exit(1)

    # Gather context
    weather = get_weather(weather_key) if weather_key else "Weather not configured."
    projects = get_active_projects(STATE_DIR)
    flags = get_watchlist_flags(STATE_DIR)
    conflicts = get_unresolved_conflicts(STATE_DIR)
    preferences = get_morning_preferences(STATE_DIR)

    print(f"  Weather: {weather[:80]}...")
    print(f"  Active projects: {len(projects)}")
    print(f"  Flags: {len(flags)}")
    print(f"  Conflicts: {len(conflicts)}")
    print(f"  Preferences: {len(preferences)}")

    # Generate morning message
    prompt = build_morning_prompt(weather, projects, flags, conflicts, preferences)

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MORNING_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    message_text = response.content[0].text.strip()

    print(f"  Generated: {len(message_text)} chars")

    # Send via Telegram
    msg_id = send_telegram_message(bot_token, user_id, message_text)
    print(f"  Sent! message_id={msg_id}")

    # Store message_id for reply detection
    if msg_id:
        store_morning_message_id(STATE_DIR, msg_id)
        print(f"  Stored morning message ID in herd_meta.json")

    print(f"[{datetime.now(timezone.utc).isoformat()}] Morning message complete.")


if __name__ == "__main__":
    main()

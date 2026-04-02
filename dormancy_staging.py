"""
Dryft — Dormancy Staging (Signal Incubator)

Holds weak signals inferred from prompts until they are confirmed or expire.
A single mention of rice does not mean the user cooks regularly. Three
independent confirming queries within 14 days does.

Graduation routing:
- Behavioral pattern -> grass layer
- Identity-level signal -> foundational store
- Factual/project-level -> herd at fitness 0.5

Corrections are high-confidence and bypass staging entirely.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta


# Calibration parameters (starting values, not constants)
CONFIRMATIONS_REQUIRED = 3
EXPIRY_DAYS = 14


class StagedSignal:
    """A weak signal waiting for confirmation."""

    def __init__(self, data: dict):
        self.id = data["id"]
        self.content = data["content"]
        self.topic = data["topic"]           # topic category for matching
        self.signal_type = data["signal_type"]  # behavioral | identity | factual
        self.created_at = data["created_at"]    # ISO timestamp
        self.confirmations = data.get("confirmations", [])  # list of {timestamp, query_excerpt}
        self.graduated = data.get("graduated", False)
        self.expired = data.get("expired", False)

    @property
    def confirmation_count(self) -> int:
        return len(self.confirmations)

    @property
    def is_ready(self) -> bool:
        return self.confirmation_count >= CONFIRMATIONS_REQUIRED

    @property
    def is_expired(self) -> bool:
        if self.graduated or self.expired:
            return self.expired
        created = datetime.fromisoformat(self.created_at)
        return datetime.now() - created > timedelta(days=EXPIRY_DAYS)

    def add_confirmation(self, query_excerpt: str):
        self.confirmations.append({
            "timestamp": datetime.now().isoformat(),
            "query_excerpt": query_excerpt[:200],
        })

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "topic": self.topic,
            "signal_type": self.signal_type,
            "created_at": self.created_at,
            "confirmations": self.confirmations,
            "graduated": self.graduated,
            "expired": self.expired,
        }


class DormancyStaging:
    """
    The signal incubator. Holds unconfirmed signals, checks for
    confirmations on each prompt, graduates or expires them.
    """

    def __init__(self, path: str = "dormancy_staging.json"):
        self.path = Path(path)
        self.signals: dict[str, StagedSignal] = {}
        self.graduation_log: list[dict] = []
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("signals", []):
            sig = StagedSignal(item)
            self.signals[sig.id] = sig
        self.graduation_log = data.get("graduation_log", [])

    def save(self):
        data = {
            "signals": [s.to_dict() for s in self.signals.values()],
            "graduation_log": self.graduation_log,
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def stage_signal(self, signal_id: str, content: str, topic: str,
                     signal_type: str = "behavioral") -> StagedSignal:
        """Add a new signal to the staging area."""
        signal = StagedSignal({
            "id": signal_id,
            "content": content,
            "topic": topic,
            "signal_type": signal_type,
            "created_at": datetime.now().isoformat(),
        })
        self.signals[signal.id] = signal
        self.save()
        return signal

    def check_confirmations(self, topics_confirmed: list[str],
                            query_excerpt: str) -> list[StagedSignal]:
        """
        Check if any staged signals are confirmed by the given topics.
        Returns list of signals that just graduated.
        """
        graduated = []

        for signal in list(self.signals.values()):
            if signal.graduated or signal.expired:
                continue

            # Check expiry
            if signal.is_expired:
                signal.expired = True
                continue

            # Check if any confirmed topic matches this signal's topic
            if signal.topic.lower() in [t.lower() for t in topics_confirmed]:
                signal.add_confirmation(query_excerpt)

                if signal.is_ready:
                    signal.graduated = True
                    graduated.append(signal)
                    self.graduation_log.append({
                        "signal_id": signal.id,
                        "content": signal.content,
                        "signal_type": signal.signal_type,
                        "topic": signal.topic,
                        "graduated_at": datetime.now().isoformat(),
                        "confirmations": signal.confirmation_count,
                    })

        if graduated:
            self.save()

        return graduated

    def cleanup_expired(self) -> int:
        """Remove expired signals. Returns count removed."""
        expired_ids = [
            sid for sid, s in self.signals.items()
            if s.is_expired and not s.graduated
        ]
        for sid in expired_ids:
            del self.signals[sid]
        if expired_ids:
            self.save()
        return len(expired_ids)

    def get_active_signals(self) -> list[StagedSignal]:
        """Return all signals still in staging (not graduated, not expired)."""
        return [
            s for s in self.signals.values()
            if not s.graduated and not s.expired and not s.is_expired
        ]

    def get_pending_topics(self) -> list[str]:
        """Return topic categories of all active signals, for matching."""
        return [s.topic for s in self.get_active_signals()]

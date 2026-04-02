"""
Dryft — Conflict Resolver

Manages the lifecycle of detected conflicts: pending -> deferred -> resolved | retired.
Three resolution branches:
  1. confirm_a / confirm_b: user picks one side
  2. correct_both: user corrects both memories
  3. user_deferred: user declines to answer

Cascade logic weakens the loser and strengthens the winner.
Annoyance throttling: max 1 conflict question per session, max 2 per 24 hours.
Deferral retirement: 2 deferrals on the same pair = permanently retired.

Persists to state/conflicts_live.json.
"""

import json
import os
import sys
import time
import uuid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import anthropic


# ── Cascade parameters ──────────────────────────────────────────────────────
WINNER_CONFIDENCE_GAIN = 0.15
WINNER_FITNESS_GAIN = 0.10
LOSER_CONFIDENCE_PENALTY = 0.30
LOSER_FITNESS_PENALTY = 0.15
LOSER_DECAY_MULTIPLIER = 2.0
LOSER_DECAY_DURATION = 20         # query cycles of doubled decay
LOSER_PREDATOR_THRESHOLD = 0.20   # permanent raise
BOND_PENALTY = 0.10
BOND_CASCADE_CAP = 5              # max bonds affected per loser

# ── Throttle parameters ─────────────────────────────────────────────────────
MAX_PER_SESSION = 1
MAX_PER_24H = 2
DEFERRAL_RETRY_INTERVAL = 20      # queries before retry
MAX_DEFERRALS = 2                  # deferrals before permanent retirement
DEFERRED_ADJACENCY_WINDOW = 10    # queries to wait for topical adjacency


class ConflictResolver:
    """Manages conflict lifecycle, resolution, and cascade execution."""

    def __init__(self, state_dir: str = "state"):
        self.state_dir = state_dir
        self.conflicts: dict[str, dict] = {}  # conflict_id -> conflict data
        self.conflict_log: list[dict] = []     # audit trail
        self.session_surfaced = 0              # conflicts surfaced this session
        self.session_surfaced_pairs: set[tuple[str, str]] = set()  # pairs surfaced this session (dedup)
        self.last_24h_surfaced: list[float] = []  # timestamps of surfacing events
        self.resolutions: list[dict] = []      # persistent resolution records
        self._load_state()
        self._load_resolutions()

    def _load_state(self):
        """Load persisted conflict state."""
        path = os.path.join(self.state_dir, "conflicts_live.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            self.conflicts = {c["conflict_id"]: c for c in data.get("conflicts", [])}
            self.conflict_log = data.get("conflict_log", [])
            self.last_24h_surfaced = data.get("last_24h_surfaced", [])

    def _load_resolutions(self):
        """Load persistent resolution records."""
        path = os.path.join(self.state_dir, "resolutions.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                self.resolutions = json.load(f)
        else:
            self.resolutions = []

    def _save_resolutions(self):
        """Persist resolution records (atomic write)."""
        os.makedirs(self.state_dir, exist_ok=True)
        import tempfile
        path = os.path.join(self.state_dir, "resolutions.json")
        fd, tmp_path = tempfile.mkstemp(dir=self.state_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.resolutions, f, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _append_resolution(self, winner_id: str, loser_id: str, query_count: int,
                           resolution_type: str = "user_confirmed"):
        """Append a resolution record and persist."""
        from datetime import datetime, timezone
        self.resolutions.append({
            "winner_id": winner_id,
            "loser_id": loser_id,
            "resolved_at": datetime.now(timezone.utc).isoformat(),
            "query_count": query_count,
            "resolution_type": resolution_type,
        })
        self._save_resolutions()

    def save_state(self):
        """Persist conflict state (atomic write to prevent corruption on crash)."""
        os.makedirs(self.state_dir, exist_ok=True)
        data = {
            "conflicts": list(self.conflicts.values()),
            "conflict_log": self.conflict_log,
            "last_24h_surfaced": self.last_24h_surfaced,
        }
        import tempfile
        path = os.path.join(self.state_dir, "conflicts_live.json")
        fd, tmp_path = tempfile.mkstemp(dir=self.state_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def get_resolved_pairs(self) -> set[tuple[str, str]]:
        """Return set of (id_a, id_b) pairs that are resolved or retired.
        Merges conflict state with persistent resolution records."""
        pairs = set()
        # From conflict state
        for c in self.conflicts.values():
            if c["status"] in ("resolved", "retired"):
                pair = tuple(sorted([c["memory_a_id"], c["memory_b_id"]]))
                pairs.add(pair)
        # From persistent resolution records
        for r in self.resolutions:
            pair = tuple(sorted([r["winner_id"], r["loser_id"]]))
            pairs.add(pair)
        return pairs

    def register_conflict(
        self,
        memory_a_id: str,
        memory_a_herd: str,
        memory_b_id: str,
        memory_b_herd: str,
        subject: str,
        description: str,
        category: str,
        confidence: float,
        query_count: int,
    ) -> str:
        """
        Register a newly detected conflict. Returns the conflict_id.
        Skips if a conflict for this pair already exists and is not resolved.
        """
        pair = tuple(sorted([memory_a_id, memory_b_id]))

        # Check if this pair was already resolved (persistent records + conflict state)
        if pair in self.get_resolved_pairs():
            return None

        # Check for existing active conflict on this pair
        for c in self.conflicts.values():
            existing_pair = tuple(sorted([c["memory_a_id"], c["memory_b_id"]]))
            if existing_pair == pair and c["status"] in ("pending", "deferred"):
                return c["conflict_id"]

        conflict_id = f"conflict-{uuid.uuid4().hex[:8]}"
        conflict = {
            "conflict_id": conflict_id,
            "memory_a_id": memory_a_id,
            "memory_a_herd": memory_a_herd,
            "memory_b_id": memory_b_id,
            "memory_b_herd": memory_b_herd,
            "subject": subject,
            "description": description,
            "category": category,
            "confidence": confidence,
            "detected_at_query": query_count,
            "status": "pending",
            "deferral_count": 0,
            "last_surfaced_at_query": None,
            "resolution": None,
        }
        self.conflicts[conflict_id] = conflict
        return conflict_id

    # ── Throttle check ───────────────────────────────────────────────────────

    def _can_surface(self) -> bool:
        """Check annoyance throttle: max 1 per session, max 2 per 24h."""
        if self.session_surfaced >= MAX_PER_SESSION:
            return False

        now = time.time()
        cutoff = now - 86400
        self.last_24h_surfaced = [t for t in self.last_24h_surfaced if t > cutoff]
        if len(self.last_24h_surfaced) >= MAX_PER_24H:
            return False

        return True

    def _record_surfacing(self):
        """Record that a conflict was surfaced."""
        self.session_surfaced += 1
        self.last_24h_surfaced.append(time.time())

    # ── Surfacing decision ───────────────────────────────────────────────────

    def get_conflict_to_surface(
        self,
        query_count: int,
        activated_ids: list[str],
    ) -> dict | None:
        """
        Decide which conflict (if any) to surface right now.

        Priority:
        1. Pending conflicts relevant to current query (confidence >= 0.85)
        2. Pending conflicts relevant to current query (any confidence)
        3. Deferred conflicts that are topically adjacent and past retry interval
        """
        if not self._can_surface():
            return None

        activated_set = set(activated_ids)
        candidates = []

        # Build resolved set (persistent + session dedup)
        resolved = self.get_resolved_pairs()

        for c in self.conflicts.values():
            pair = tuple(sorted([c["memory_a_id"], c["memory_b_id"]]))
            # Skip resolved pairs (persistent) and already-surfaced-this-session pairs
            if pair in resolved or pair in self.session_surfaced_pairs:
                continue

            if c["status"] == "pending":
                # Check if relevant to current query
                if c["memory_a_id"] in activated_set or c["memory_b_id"] in activated_set:
                    candidates.append(c)

            elif c["status"] == "deferred":
                # Check retry interval
                last_surfaced = c.get("last_surfaced_at_query", 0) or 0
                if query_count - last_surfaced < DEFERRAL_RETRY_INTERVAL:
                    continue
                # Check topical adjacency
                if c["memory_a_id"] in activated_set or c["memory_b_id"] in activated_set:
                    candidates.append(c)

        if not candidates:
            return None

        # Sort by confidence descending, pick highest
        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        return candidates[0]

    def build_surfacing_instruction(self, conflict: dict, memory_a_content: str, memory_b_content: str) -> str:
        """
        Build a system instruction that tells the model to ask the user
        about this conflict naturally.
        """
        herd_a = "evaluation herd" if conflict["memory_a_herd"] == "evaluation" else "main memory"
        herd_b = "evaluation herd" if conflict["memory_b_herd"] == "evaluation" else "main memory"

        return (
            f"\n\n[CONFLICT RESOLUTION REQUEST]\n"
            f"Your memory system has detected a conflict about: {conflict['subject']}.\n"
            f"Memory A ({herd_a}): {memory_a_content}\n"
            f"Memory B ({herd_b}): {memory_b_content}\n\n"
            f"First respond to the user's actual message normally. Then add a brief aside like "
            f"'By the way, I have two different notes about {conflict['subject']}' and state both "
            f"claims plainly, asking which is current. Include an option that both might be wrong. "
            f"Keep the aside brief, no apology.\n"
            f"[END CONFLICT RESOLUTION REQUEST]"
        )

    # ── Response classification ──────────────────────────────────────────────

    def classify_response(self, user_message: str, conflict: dict) -> str:
        """
        Classify whether the user's response resolves the pending conflict.

        Returns: confirm_a, confirm_b, correct_both, user_deferred, not_a_response
        """
        client = anthropic.Anthropic()

        prompt = f"""A memory system asked the user to clarify a conflict about: {conflict['subject']}.

Memory A (ID: {conflict['memory_a_id']}): The system believes one thing.
Memory B (ID: {conflict['memory_b_id']}): The system believes something contradictory.

The user responded with: "{user_message}"

Classify this response as ONE of:
- "confirm_a": The user confirmed Memory A is correct
- "confirm_b": The user confirmed Memory B is correct
- "correct_both": The user said both are wrong and provided a correction
- "user_deferred": The user declined to answer, said "I don't know", asked to skip, or deferred
- "not_a_response": The user's message is not responding to the conflict question at all (they moved on to a new topic)

Return ONLY valid JSON: {{"resolution": "confirm_a|confirm_b|correct_both|user_deferred|not_a_response", "correction_text": "only if correct_both, the user's corrected version"}}"""

        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            result = json.loads(text)
            return result.get("resolution", "not_a_response"), result.get("correction_text")
        except (json.JSONDecodeError, anthropic.APIError, IndexError):
            return "not_a_response", None

    # ── Resolution execution ─────────────────────────────────────────────────

    def resolve(
        self,
        conflict_id: str,
        resolution: str,
        correction_text: str | None,
        winner_memory,
        loser_memory,
        query_count: int,
        engine=None,
    ) -> dict:
        """
        Execute a resolution branch and return the cascade log entry.

        Args:
            conflict_id: The conflict being resolved
            resolution: confirm_a, confirm_b, correct_both, user_deferred
            correction_text: User's correction if correct_both
            winner_memory: Memory object of the winner (or memory_a for correct_both)
            loser_memory: Memory object of the loser (or memory_b for correct_both)
            query_count: Current query count
            engine: HerdEngine for adding new memories (correct_both branch)
        """
        conflict = self.conflicts.get(conflict_id)
        if not conflict:
            return {}

        log_entry = {
            "conflict_id": conflict_id,
            "query_number": query_count,
            "resolution": resolution,
            "winner_id": None,
            "loser_id": None,
            "loser_herd": None,
            "score_changes": {},
            "bond_changes": [],
            "new_memory_id": None,
        }

        if resolution in ("confirm_a", "confirm_b"):
            # Winner gets small fitness bump, loser gets humane dispatch (immediate cull)
            log_entry["winner_id"] = winner_memory.id
            log_entry["loser_id"] = loser_memory.id
            log_entry["loser_herd"] = conflict[
                "memory_b_herd" if resolution == "confirm_a" else "memory_a_herd"
            ]

            # Winner boost (small reinforcement)
            old_w_conf = winner_memory.confidence_score
            old_w_fit = winner_memory.fitness_score
            winner_memory.confidence_score = min(
                winner_memory.confidence_score + WINNER_CONFIDENCE_GAIN, 1.0
            )
            winner_memory.fitness_score = min(
                winner_memory.fitness_score + 0.05, 1.0
            )

            # Humane dispatch: immediate cull, no decomposition
            loser_memory.status = "culled"
            log_entry["cull_type"] = "humane_dispatch"

            log_entry["score_changes"] = {
                "winner": {
                    "confidence_before": round(old_w_conf, 4),
                    "confidence_after": round(winner_memory.confidence_score, 4),
                    "fitness_before": round(old_w_fit, 4),
                    "fitness_after": round(winner_memory.fitness_score, 4),
                },
                "loser": {
                    "fitness_before": round(loser_memory.fitness_score, 4),
                    "fitness_after": 0.0,
                    "status": "culled",
                    "cull_type": "humane_dispatch",
                },
            }

            # Write resolution record
            self._append_resolution(winner_memory.id, loser_memory.id, query_count)

        elif resolution == "correct_both":
            # Humane dispatch both: user says both are wrong
            log_entry["loser_id"] = f"{winner_memory.id}, {loser_memory.id}"
            log_entry["cull_type"] = "humane_dispatch"

            # Cull both memories immediately
            old_a_fit = winner_memory.fitness_score
            old_b_fit = loser_memory.fitness_score
            winner_memory.status = "culled"
            loser_memory.status = "culled"

            log_entry["score_changes"] = {
                "memory_a": {
                    "fitness_before": round(old_a_fit, 4),
                    "fitness_after": 0.0,
                    "status": "culled",
                    "cull_type": "humane_dispatch",
                },
                "memory_b": {
                    "fitness_before": round(old_b_fit, 4),
                    "fitness_after": 0.0,
                    "status": "culled",
                    "cull_type": "humane_dispatch",
                },
            }

            # Write resolution records for both
            self._append_resolution("none", winner_memory.id, query_count,
                                    resolution_type="correct_both")
            self._append_resolution("none", loser_memory.id, query_count,
                                    resolution_type="correct_both")

            # Track both herds for embedding cleanup
            log_entry["loser_herd"] = conflict["memory_a_herd"]
            log_entry["loser_b_herd"] = conflict["memory_b_herd"]
            log_entry["loser_b_id"] = loser_memory.id

            # Create new memory from correction
            if correction_text and engine:
                from herd_engine import Memory
                new_id = f"conflict-correction-{conflict_id}"
                new_mem = Memory({
                    "id": new_id,
                    "content": correction_text,
                    "memory_type": "semantic",
                    "keywords": conflict.get("subject", "").lower().split(),
                    "fitness_score": 0.7,
                    "confidence_score": 0.95,
                })
                engine.memories[new_id] = new_mem
                log_entry["new_memory_id"] = new_id

        elif resolution == "user_deferred":
            conflict["deferral_count"] = conflict.get("deferral_count", 0) + 1
            conflict["last_surfaced_at_query"] = query_count

            if conflict["deferral_count"] >= MAX_DEFERRALS:
                conflict["status"] = "retired"
            else:
                conflict["status"] = "deferred"

            log_entry["resolution"] = "user_deferred"
            log_entry["deferral_count"] = conflict["deferral_count"]
            self.conflict_log.append(log_entry)
            self.save_state()
            return log_entry

        # Mark resolved for confirm/correct branches
        conflict["status"] = "resolved"
        conflict["resolution"] = resolution
        conflict["last_surfaced_at_query"] = query_count

        self.conflict_log.append(log_entry)
        self.save_state()
        return log_entry

    def _apply_loser_cascade(self, memory, query_count: int) -> dict:
        """Apply cascade penalties to a losing memory. Returns change log."""
        old_conf = memory.confidence_score
        old_fit = memory.fitness_score
        old_decay = memory.decay_rate
        old_pred = memory.predator_threshold

        memory.confidence_score = max(memory.confidence_score - LOSER_CONFIDENCE_PENALTY, 0.05)
        memory.fitness_score = max(memory.fitness_score - LOSER_FITNESS_PENALTY, 0.0)
        memory.predator_threshold = max(memory.predator_threshold, LOSER_PREDATOR_THRESHOLD)

        # Store decay override info for the engine to use
        memory._decay_override = memory.decay_rate * LOSER_DECAY_MULTIPLIER
        memory._decay_override_until = query_count + LOSER_DECAY_DURATION
        memory.decay_rate = memory._decay_override

        score_changes = {
            "confidence_before": round(old_conf, 4),
            "confidence_after": round(memory.confidence_score, 4),
            "fitness_before": round(old_fit, 4),
            "fitness_after": round(memory.fitness_score, 4),
            "decay_before": old_decay,
            "decay_after": memory.decay_rate,
            "predator_threshold_before": old_pred,
            "predator_threshold_after": memory.predator_threshold,
        }

        # Bond cascade: penalize top 5 strongest bonds
        bond_changes = []
        if memory.proximity_bonds:
            sorted_bonds = sorted(
                memory.proximity_bonds.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            for partner_id, old_bond in sorted_bonds[:BOND_CASCADE_CAP]:
                new_bond = max(old_bond - BOND_PENALTY, 0.0)
                memory.proximity_bonds[partner_id] = new_bond
                bond_changes.append({
                    "partner_id": partner_id,
                    "bond_before": round(old_bond, 4),
                    "bond_after": round(new_bond, 4),
                })

        return {"score_changes": score_changes, "bond_changes": bond_changes}

    def mark_surfaced(self, conflict_id: str, query_count: int):
        """Record that a conflict was surfaced to the user."""
        conflict = self.conflicts.get(conflict_id)
        if conflict:
            conflict["last_surfaced_at_query"] = query_count
            self._record_surfacing()
            # Session-level dedup: track this pair so it won't resurface this session
            pair = tuple(sorted([conflict["memory_a_id"], conflict["memory_b_id"]]))
            self.session_surfaced_pairs.add(pair)

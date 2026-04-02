"""
Dryft — Phase 7/8: Proxy Layer

Sits between the user and any AI model. Every prompt passes through it.
Two jobs run simultaneously on every inbound message:

Job One: Context Injection
  Score prompt against herd. Activate memories above 0.20 threshold.
  Inject activated memories + grass layer + foundational context into
  the model call.

Job Two: Portrait Building
  Read the same prompt as a signal about the user. Route new signals
  into dormancy staging. Graduate confirmed patterns to grass or
  foundational. The portrait deepens with every interaction.

Phase 8: State persistence. All four layers save to state/ after every
interaction and reload on next launch. Seeds are cold start fallback only.

Usage:
    from proxy import DryftProxy
    proxy = DryftProxy()
    response = proxy.process("How do I set up a Shopify app?")
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import re

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import anthropic

from herd_engine import HerdEngine
from vector_scorer import VectorScorer
from foundational import FoundationalStore
from dormancy_staging import DormancyStaging
from signal_detector import SignalDetector
from conflict_detector import ConflictDetector
from conflict_resolver import ConflictResolver
from temporal_utils import TemporalMapper, find_supersessions


RELEVANCE_THRESHOLD = 0.20
STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state")

# Web search configuration
WEB_SEARCH_MAX_USES = 3  # max search invocations per query (controls token cost)
WEB_SEARCH_USER_LOCATION = {
    "type": "approximate",
    "city": os.environ.get("USER_CITY", ""),
    "region": os.environ.get("USER_REGION", ""),
    "country": os.environ.get("USER_COUNTRY", ""),
    "timezone": os.environ.get("USER_TIMEZONE", "UTC"),
}

# Weather keyword injection - triggers Tomorrow.io context prepend
WEATHER_KEYWORDS = {
    "weather", "forecast", "temperature", "frost", "rain", "snow", "wind",
    "growing", "planting", "field conditions", "hail", "storm", "cold",
    "warm", "heat", "freeze", "precipitation",
}

# Google Sheets URL pattern
SHEETS_URL_PATTERN = r"docs\.google\.com/spreadsheets/d/([a-zA-Z0-9_-]+)"

def _atomic_json_write(path: str, data, indent: int = 2):
    """Write JSON atomically: write to temp file, then os.replace().
    Prevents corruption if the process crashes mid-write."""
    import tempfile
    dir_name = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp_path, path)
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class DryftProxy:
    """
    The proxy layer. Intercepts every prompt, injects context from
    the living memory architecture, and reads inbound signals to
    build the user's portrait.
    """

    def __init__(
        self,
        memories_path: str = "memories_realworld_v2.json",
        grass_path: str = "grass_layer_seed_v2.json",
        foundational_path: str = "foundational.json",
        staging_path: str = "dormancy_staging.json",
        embeddings_cache: str = "embeddings_realworld_cache.json",
        eval_memories_path: str | None = None,
        eval_embeddings_cache: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        signal_model: str = "claude-haiku-4-5-20251001",
        state_dir: str = STATE_DIR,
    ):
        self.state_dir = state_dir
        self._seed_paths = {
            "memories": memories_path,
            "grass": grass_path,
            "foundational": foundational_path,
            "staging": staging_path,
            "embeddings": embeddings_cache,
        }

        # Determine warm vs cold start
        warm = os.path.exists(os.path.join(state_dir, "herd_live.json"))

        if warm:
            print("Warm start: loading persisted state from state/")
            herd_path = os.path.join(state_dir, "herd_live.json")
            grass_load = os.path.join(state_dir, "grass_live.json")
            found_load = os.path.join(state_dir, "foundational_live.json")
            stage_load = os.path.join(state_dir, "staging_live.json")
            embed_load = os.path.join(state_dir, "embeddings_live.json")
        else:
            print("Cold start: loading from seed files.")
            herd_path = memories_path
            grass_load = grass_path
            found_load = foundational_path
            stage_load = staging_path
            embed_load = embeddings_cache

        # Load vector scorer for main herd
        self.scorer = VectorScorer(embed_load, live_embed=True)
        if not self.scorer.available:
            print("WARNING: Embeddings cache not available. Falling back to keyword scoring.")

        # Initialize main herd engine with threshold + pause-on-silence
        self.engine = HerdEngine(
            herd_path,
            predator_executes=True,
            grass_layer_path=grass_load,
            vector_scorer=self.scorer,
            relevance_threshold=RELEVANCE_THRESHOLD,
            pause_on_silence=True,
        )

        # Restore engine counters on warm start
        if warm:
            meta_path = os.path.join(state_dir, "herd_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                self.engine.query_count = meta.get("query_count", 0)
                self.engine.silent_queries = meta.get("silent_queries", 0)
                self.web_search_count = meta.get("web_search_count", 0)

        # Temporal mapper (carbon dating layer)
        meta_path = os.path.join(state_dir, "herd_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self.temporal_mapper = TemporalMapper(meta)
        else:
            self.temporal_mapper = TemporalMapper({})

        # ── Evaluation herd (sheep) ──────────────────────────────────────────
        eval_warm = os.path.exists(os.path.join(state_dir, "eval_herd_live.json"))
        if eval_warm:
            eval_herd_path = os.path.join(state_dir, "eval_herd_live.json")
            eval_embed_path = os.path.join(state_dir, "eval_embeddings_live.json")
        else:
            eval_herd_path = eval_memories_path
            eval_embed_path = eval_embeddings_cache

        self.eval_scorer = None
        self.eval_engine = None

        if eval_herd_path and os.path.exists(eval_herd_path):
            self.eval_scorer = VectorScorer(
                eval_embed_path if eval_embed_path else "eval_embeddings_cache.json",
                live_embed=True,
            )
            self.eval_engine = HerdEngine(
                eval_herd_path,
                predator_executes=True,
                grass_layer_path=grass_load,  # shared grass layer
                vector_scorer=self.eval_scorer,
                relevance_threshold=RELEVANCE_THRESHOLD,
                pause_on_silence=True,
            )
            # Restore eval engine counters
            if eval_warm:
                eval_meta_path = os.path.join(state_dir, "eval_herd_meta.json")
                if os.path.exists(eval_meta_path):
                    with open(eval_meta_path, "r") as f:
                        eval_meta = json.load(f)
                    self.eval_engine.query_count = eval_meta.get("query_count", 0)
                    self.eval_engine.silent_queries = eval_meta.get("silent_queries", 0)
            print(f"Evaluation herd loaded: {sum(1 for m in self.eval_engine.memories.values() if m.status == 'active')} active memories")
        else:
            print("No evaluation herd found. Operating with main herd only.")

        # Foundational store (cowbird layer)
        self.foundational = FoundationalStore(found_load)

        # Dormancy staging (signal incubator)
        self.staging = DormancyStaging(stage_load)

        # Signal detector (with eval entity list for gate reliability)
        self.detector = SignalDetector(model=signal_model)
        if self.eval_engine:
            eval_entities = set()
            for mem in self.eval_engine.memories.values():
                if mem.evaluated_entity:
                    eval_entities.add(mem.evaluated_entity)
            if eval_entities:
                self.detector.set_eval_entities(sorted(eval_entities))
                print(f"Eval entity list injected: {sorted(eval_entities)}")

        # Conflict detection & resolution
        self.conflict_detector = ConflictDetector(model=signal_model)
        self.conflict_resolver = ConflictResolver(state_dir=state_dir)

        # Pending resolution state
        self.pending_resolution: dict | None = None  # {conflict_id, conflict}

        # Model for main conversation
        self.client = anthropic.Anthropic()
        self.model = model

        # Session log
        self.session_log: list[dict] = []

        # Bond metadata injection (opt-in for multi-hop benchmark testing)
        self.include_bonds = False

        # Conversation history for multi-turn model calls
        self.conversation_history: list[dict] = []
        self.max_history_turns = 10  # last 10 messages (5 exchanges)

        # Last activation set — stored for /flag command retrieval
        self.last_activation: list[str] = []

        # Correction pathway state (Phase 12A)
        self.correction_mode = False
        self.pending_correction = None

        # Web search usage counter (default, may be overwritten by warm start above)
        if not hasattr(self, 'web_search_count'):
            self.web_search_count = 0

    def save_state(self):
        """Persist all four layers + embeddings to state/."""
        os.makedirs(self.state_dir, exist_ok=True)

        # Herd memories
        herd_data = [m.to_dict() for m in self.engine.memories.values()]
        _atomic_json_write(os.path.join(self.state_dir, "herd_live.json"), herd_data)

        # Engine counters
        meta = {
            "query_count": self.engine.query_count,
            "silent_queries": self.engine.silent_queries,
            "web_search_count": self.web_search_count,
        }
        _atomic_json_write(os.path.join(self.state_dir, "herd_meta.json"), meta)

        # Grass layer
        if self.engine.grass_layer:
            grass_data = [e.to_dict() for e in self.engine.grass_layer.entries.values()]
            _atomic_json_write(os.path.join(self.state_dir, "grass_live.json"), grass_data)

        # Foundational
        found_data = [m.to_dict() for m in self.foundational.memories.values()]
        _atomic_json_write(os.path.join(self.state_dir, "foundational_live.json"), found_data)

        # Dormancy staging
        stage_data = {
            "signals": [s.to_dict() for s in self.staging.signals.values()],
            "graduation_log": self.staging.graduation_log,
        }
        _atomic_json_write(os.path.join(self.state_dir, "staging_live.json"), stage_data)

        # Embeddings (memory embeddings only, skip query cache)
        self.scorer.save(os.path.join(self.state_dir, "embeddings_live.json"))

        # Evaluation herd (if present)
        if self.eval_engine:
            eval_data = [m.to_dict() for m in self.eval_engine.memories.values()]
            _atomic_json_write(os.path.join(self.state_dir, "eval_herd_live.json"), eval_data)

            eval_meta = {
                "query_count": self.eval_engine.query_count,
                "silent_queries": self.eval_engine.silent_queries,
            }
            _atomic_json_write(os.path.join(self.state_dir, "eval_herd_meta.json"), eval_meta)

            if self.eval_scorer:
                self.eval_scorer.save(os.path.join(self.state_dir, "eval_embeddings_live.json"))

        # Conflict state
        self.conflict_resolver.save_state()

        print(f"State saved to {self.state_dir}/")

    def _load_watchlist(self) -> dict:
        path = os.path.join(self.state_dir, "watchlist.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {"entries": [], "unseen_count": 0}

    def _write_suspicious_culls(self, eco_event: dict):
        """Check cull records for suspicious deaths and append to watchlist."""
        suspicious = [c for c in eco_event.get("culled", []) if c.get("suspicious")]
        if not suspicious:
            return
        watchlist = self._load_watchlist()
        for cull in suspicious:
            flags = []
            if cull["activation_count"] >= 3:
                flags.append("3+ activations")
            if cull.get("rehydrated"):
                flags.append("rehydrated")
            # Check for ACQUAINTED+ bonds from bond_scores_at_death
            mem = self.engine.memories.get(cull["id"])
            if mem:
                for pid, bscore in cull["bond_scores_at_death"].items():
                    co_count = mem.co_activation_counts.get(pid, 0)
                    if co_count >= 3 and bscore >= 0.15:
                        flags.append("ACQUAINTED+ bond")
                        break
            watchlist["entries"].append({
                "memory_id": cull["id"],
                "content_summary": mem.content[:120] if mem else cull["id"],
                "fitness_at_death": cull["fitness"],
                "activation_count": cull["activation_count"],
                "rehydrated": cull["rehydrated"],
                "flags": flags,
                "culled_at_query": self.engine.query_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            watchlist["unseen_count"] += 1
        _atomic_json_write(os.path.join(self.state_dir, "watchlist.json"), watchlist)

    def get_watchlist_unseen(self) -> int:
        """Return count of unseen suspicious culls."""
        return self._load_watchlist().get("unseen_count", 0)

    def clear_watchlist_unseen(self):
        """Mark all watchlist entries as seen (reset unseen count)."""
        watchlist = self._load_watchlist()
        watchlist["unseen_count"] = 0
        _atomic_json_write(os.path.join(self.state_dir, "watchlist.json"), watchlist)

    def _append_to_history(self, role: str, content: str):
        """Add a message to conversation history, trimming to max depth."""
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > self.max_history_turns:
            self.conversation_history = self.conversation_history[-self.max_history_turns:]

    def _build_context_package(self, query: str, eval_gate_open: bool = False) -> dict:
        """
        Score the query against the herd and assemble the full context
        package: foundational + grass + activated herd memories + eval herd if gate open.
        Returns the package and metadata about what was activated.
        """
        # Score main herd (read-only, no ecological cycle yet)
        activated = self.engine.score_query(query)

        # Score evaluation herd if gate is open
        eval_activated = []
        if eval_gate_open and self.eval_engine:
            eval_activated = self.eval_engine.score_query(query)

        # Foundational context (always included)
        foundational_block = self.foundational.format_for_injection()

        # Grass layer (always included)
        grass_block = ""
        if self.engine.grass_layer:
            entries = self.engine.grass_layer.entries
            if entries:
                lines = ["[Behavioral Context — grass layer]"]
                for entry in entries.values():
                    lines.append(f"- {entry.content}")
                grass_block = "\n".join(lines)

        # Activated main herd memories with temporal metadata
        herd_block = ""
        if activated:
            lines = ["[Operational Memory Context — main herd, activated by this query]"]
            for mem_id, content, score in activated:
                mem_obj = self.engine.memories.get(mem_id)
                if mem_obj and mem_obj.created_at > 0:
                    created_date = self.temporal_mapper.format_date(mem_obj.created_at)
                    last_active = self.temporal_mapper.format_date(mem_obj.last_activated_at) if mem_obj.last_activated_at > 0 else "never"
                    lines.append(f"- (created ~{created_date}, last active {last_active}, activated {mem_obj.activation_count} times) {content}")
                else:
                    lines.append(f"- {content}")
            herd_block = "\n".join(lines)

        # Activated evaluation herd memories with temporal metadata
        eval_block = ""
        if eval_activated:
            lines = ["[Evaluation Context — evaluation herd, knowledge from evaluated systems]"]
            for mem_id, content, score in eval_activated:
                mem_obj = self.eval_engine.memories.get(mem_id) if self.eval_engine else None
                if mem_obj and mem_obj.created_at > 0:
                    created_date = self.temporal_mapper.format_date(mem_obj.created_at)
                    lines.append(f"- (evaluated ~{created_date}) {content}")
                else:
                    lines.append(f"- {content}")
            eval_block = "\n".join(lines)

        # Supersession inference: detect when newer memories may replace older ones
        supersession_block = ""
        if len(activated) >= 2:
            # Build extended tuples with created_at
            extended = []
            for mem_id, content, score in activated:
                mem_obj = self.engine.memories.get(mem_id)
                created_at = mem_obj.created_at if mem_obj else 0
                extended.append((mem_id, content, score, created_at))
            supersessions = find_supersessions(
                extended, self.scorer, self.temporal_mapper,
                self.engine.query_count,
            )
            if supersessions:
                sup_lines = ["[Temporal Notes]"]
                for s in supersessions:
                    sup_lines.append(f"- {s['note']}")
                supersession_block = "\n".join(sup_lines)

        # Bond metadata (opt-in for multi-hop benchmark testing)
        bond_block = ""
        if self.include_bonds and activated:
            activated_ids = set(a[0] for a in activated)
            bond_lines = ["[Memory Relationships — bonds between activated memories]"]
            seen = set()
            for mem_id, _, _ in activated:
                mem = self.engine.memories.get(mem_id)
                if not mem:
                    continue
                for partner_id, score in sorted(
                    mem.proximity_bonds.items(), key=lambda x: -x[1]
                ):
                    if partner_id in activated_ids and score >= 0.30:
                        pair = tuple(sorted([mem_id, partner_id]))
                        if pair not in seen:
                            seen.add(pair)
                            bond_lines.append(
                                f"- {mem_id} and {partner_id} are BONDED "
                                f"(bond strength {score:.2f})"
                            )
            if len(bond_lines) > 1:
                bond_block = "\n".join(bond_lines[:26])  # cap at 25 bonds

        # Assemble
        sections = [s for s in [foundational_block, grass_block, herd_block, eval_block, bond_block, supersession_block] if s]
        context_package = "\n\n".join(sections)

        return {
            "context_package": context_package,
            "activated_count": len(activated),
            "activated_ids": [a[0] for a in activated],
            "activated_scores": {a[0]: round(a[2], 4) for a in activated},
            "activated_tuples": activated,  # raw (id, content, score) for reuse
            "eval_activated_count": len(eval_activated),
            "eval_activated_ids": [a[0] for a in eval_activated],
            "eval_activated_tuples": eval_activated,  # raw tuples for reuse
            "eval_gate_open": eval_gate_open,
            "foundational_count": len(self.foundational.memories),
            "grass_count": len(self.engine.grass_layer.entries) if self.engine.grass_layer else 0,
        }

    def _detect_and_route_signals(self, query: str) -> dict:
        """
        Job Two: Read the prompt as a signal about the user.
        Detect new signals, check confirmations, graduate ready signals.
        """
        # Get topics currently in staging
        staged_topics = self.staging.get_pending_topics()

        # Detect signals via API
        detection = self.detector.detect(query, staged_topics)
        signals = detection.get("signals", [])
        topics_confirmed = detection.get("topics_confirmed", [])

        # Eval gate check from detection response
        eval_gate_open = detection.get("eval_herd_relevant", False)
        eval_entity = detection.get("eval_entity_referenced", None)

        result = {
            "new_signals": [],
            "confirmations": [],
            "graduations": [],
            "corrections": [],
            "eval_gate_open": eval_gate_open,
            "eval_entity_referenced": eval_entity,
        }

        # Route new signals
        for signal in signals:
            confidence = signal.get("confidence", "low")
            signal_type = signal.get("signal_type", "behavioral")

            if confidence == "high" or signal_type == "correction":
                # High-confidence or correction: write immediately
                result["corrections"].append(signal)
                self._write_correction(signal)
            else:
                # Low-confidence: stage for confirmation
                signal_id = f"staged-{len(self.staging.signals)}-{signal['topic']}"
                self.staging.stage_signal(
                    signal_id=signal_id,
                    content=signal["content"],
                    topic=signal["topic"],
                    signal_type=signal_type,
                )
                result["new_signals"].append(signal)

        # Check confirmations on existing staged signals
        if topics_confirmed:
            graduated = self.staging.check_confirmations(
                topics_confirmed, query[:200]
            )
            for grad in graduated:
                result["graduations"].append({
                    "id": grad.id,
                    "content": grad.content,
                    "signal_type": grad.signal_type,
                    "topic": grad.topic,
                })
                self._graduate_signal(grad)

        # Cleanup expired signals
        self.staging.cleanup_expired()

        return result

    def _write_correction(self, signal: dict):
        """Write a high-confidence correction directly."""
        signal_type = signal.get("signal_type", "behavioral")
        content = signal["content"]

        if signal_type in ("identity", "correction"):
            # Identity-level corrections go to foundational
            fid = f"correction-{len(self.foundational.memories)}"
            self.foundational.add(fid, content, category="identity", source="correction")
        else:
            # Behavioral corrections go to grass
            if self.engine.grass_layer:
                from herd_engine import GrassEntry
                gid = f"correction-{len(self.engine.grass_layer.entries)}"
                entry = GrassEntry({
                    "id": gid,
                    "content": content,
                    "source": "direct",
                    "keywords": [],
                })
                self.engine.grass_layer.entries[gid] = entry

    def _graduate_signal(self, signal):
        """Route a graduated signal to its destination."""
        if signal.signal_type == "behavioral":
            # Behavioral -> grass layer
            if self.engine.grass_layer:
                from herd_engine import GrassEntry
                gid = f"graduated-{signal.id}"
                entry = GrassEntry({
                    "id": gid,
                    "content": signal.content,
                    "source": "direct",
                    "keywords": [],
                })
                self.engine.grass_layer.entries[gid] = entry

        elif signal.signal_type == "identity":
            # Identity -> foundational
            fid = f"graduated-{signal.id}"
            self.foundational.add(fid, signal.content,
                                  category="identity", source="graduated")

        elif signal.signal_type == "factual":
            # Factual -> herd at fitness 0.5
            from herd_engine import Memory
            mid = f"graduated-{signal.id}"
            mem = Memory({
                "id": mid,
                "content": signal.content,
                "memory_type": "semantic",
                "keywords": [signal.topic],
                "fitness_score": 0.5,
                "created_at": self.engine.query_count,
            })
            self.engine.memories[mid] = mem

            # Embed the new memory so it can activate against future queries
            if self.scorer.available and self.scorer._live_embed:
                embedding = self.scorer.embed_query(signal.content)
                self.scorer.add_memory_embedding(mid, embedding)

    def _handle_pending_resolution(self, query: str) -> dict | None:
        """
        If a conflict was surfaced last turn, classify the user's response
        and execute the resolution cascade if applicable.
        Returns the cascade log entry, or None if no pending resolution.
        """
        if not self.pending_resolution:
            return None

        conflict = self.pending_resolution
        conflict_id = conflict["conflict_id"]

        # Classify user response
        resolution, correction_text = self.conflict_resolver.classify_response(
            query, conflict
        )

        self.pending_resolution = None  # Clear regardless of outcome

        if resolution == "not_a_response":
            # User moved on. Defer the conflict.
            self.conflict_resolver.resolve(
                conflict_id, "user_deferred", None,
                None, None, self.engine.query_count,
            )
            return {"resolution": "not_a_response", "conflict_id": conflict_id}

        # Look up memory objects for cascade
        a_id = conflict["memory_a_id"]
        b_id = conflict["memory_b_id"]
        a_herd = conflict["memory_a_herd"]
        b_herd = conflict["memory_b_herd"]

        mem_a = self._get_memory(a_id, a_herd)
        mem_b = self._get_memory(b_id, b_herd)

        if not mem_a or not mem_b:
            return {"resolution": "error", "conflict_id": conflict_id, "detail": "memory not found"}

        if resolution == "confirm_a":
            winner, loser = mem_a, mem_b
        elif resolution == "confirm_b":
            winner, loser = mem_b, mem_a
        elif resolution == "correct_both":
            winner, loser = mem_a, mem_b  # both get loser penalties
        else:
            winner, loser = None, None

        # Determine which engine to pass for correct_both (new memory goes to main)
        engine = self.engine

        log_entry = self.conflict_resolver.resolve(
            conflict_id, resolution, correction_text,
            winner, loser, self.engine.query_count, engine,
        )

        # Humane dispatch: remove embedding(s) for culled memories, skip decomposition
        if log_entry.get("cull_type") == "humane_dispatch" and log_entry.get("loser_id"):
            loser_id = log_entry["loser_id"]
            loser_herd = log_entry.get("loser_herd", "main")
            # Remove primary loser embedding
            if loser_herd == "evaluation" and self.eval_scorer:
                self.eval_scorer.remove_memory_embedding(loser_id)
            elif self.scorer.available:
                self.scorer.remove_memory_embedding(loser_id)
            print(f"  Humane dispatch: {loser_id} culled immediately (no decomposition)")

            # correct_both: also remove second loser's embedding
            if log_entry.get("loser_b_id"):
                loser_b_id = log_entry["loser_b_id"]
                loser_b_herd = log_entry.get("loser_b_herd", "main")
                if loser_b_herd == "evaluation" and self.eval_scorer:
                    self.eval_scorer.remove_memory_embedding(loser_b_id)
                elif self.scorer.available:
                    self.scorer.remove_memory_embedding(loser_b_id)
                print(f"  Humane dispatch: {loser_b_id} culled immediately (no decomposition)")

        # If a new memory was created, embed it
        if log_entry.get("new_memory_id") and self.scorer.available and self.scorer._live_embed:
            new_id = log_entry["new_memory_id"]
            if new_id in self.engine.memories:
                embedding = self.scorer.embed_query(self.engine.memories[new_id].content)
                self.scorer.add_memory_embedding(new_id, embedding)

        return log_entry

    def _get_memory(self, memory_id: str, herd: str):
        """Look up a memory object from the appropriate herd."""
        if herd == "evaluation" and self.eval_engine:
            return self.eval_engine.memories.get(memory_id)
        return self.engine.memories.get(memory_id)

    def process(self, query: str, call_model: bool = True) -> dict:
        """
        Main proxy entry point. Every user prompt passes through here.

        Updated flow (Session B additions marked NEW):
        1. NEW: Check pending_resolution flag; if set, classify and cascade
        2. Signal detection (with eval gate check)
        3. Score main herd
        4. If eval gate open: score evaluation herd
        5. Activate above threshold from contributing herds
        6. NEW: If 2+ activated: run conflict detection
        7. NEW: Check unresolved queue for adjacent conflicts
        8. NEW: If conflict to surface (throttle allows): add instruction to context
        9. Assemble context (foundational + grass + main + eval if gate open)
        10. Model call
        11. Process ecological cycles
        12. NEW: Persist conflict state
        """
        t0 = time.time()

        # NEW Step 1: Handle pending resolution from previous turn
        resolution_result = self._handle_pending_resolution(query)

        # Step 1b: Handle pending correction response
        if self.pending_correction:
            correction_response = self._handle_correction_response(query)
            if correction_response:
                # Correction handled, return the response directly
                self._append_to_history("user", query)
                self._append_to_history("assistant", correction_response)
                elapsed = time.time() - t0
                return {
                    "response": correction_response,
                    "context": {"activated_count": 0, "activated_ids": [], "foundational_count": 0, "grass_count": 0},
                    "signals": {"new_signals": [], "confirmations": [], "graduations": [], "corrections": []},
                    "ecology": {},
                    "conflicts": {"detected": [], "surfaced": None},
                    "elapsed_ms": round(elapsed * 1000),
                }

        # Step 2: Signal detection (portrait building) — runs first to get eval gate status
        signals = self._detect_and_route_signals(query)
        eval_gate_open = signals.get("eval_gate_open", False)

        # Steps 3-5: Context injection (scoring + activation)
        context = self._build_context_package(query, eval_gate_open=eval_gate_open)

        # Store last activation for /flag command
        self.last_activation = context["activated_ids"] + context.get("eval_activated_ids", [])

        # NEW Step 6: Conflict detection (if 2+ memories activated)
        conflict_info = {"detected": [], "surfaced": None}
        conflict_instruction = ""
        total_activated = context["activated_count"] + context.get("eval_activated_count", 0)

        if total_activated >= 2:
            # Reuse activated tuples from context package (avoids double scoring)
            main_activated = context["activated_tuples"]
            eval_activated = context.get("eval_activated_tuples") or None

            # Build temporal dates for conflict detector
            temporal_dates = {}
            for mem_id, _, _ in main_activated:
                mem_obj = self.engine.memories.get(mem_id)
                if mem_obj and mem_obj.created_at > 0:
                    temporal_dates[mem_id] = self.temporal_mapper.format_date(mem_obj.created_at)
            if eval_activated:
                for mem_id, _, _ in eval_activated:
                    mem_obj = self.eval_engine.memories.get(mem_id) if self.eval_engine else None
                    if mem_obj and mem_obj.created_at > 0:
                        temporal_dates[mem_id] = self.temporal_mapper.format_date(mem_obj.created_at)

            resolved_pairs = self.conflict_resolver.get_resolved_pairs()
            detected = self.conflict_detector.detect(
                main_activated, eval_activated, resolved_pairs, temporal_dates
            )
            conflict_info["detected"] = detected

            # Register new conflicts
            for d in detected:
                self.conflict_resolver.register_conflict(
                    memory_a_id=d["memory_a_id"],
                    memory_a_herd=d["memory_a_herd"],
                    memory_b_id=d["memory_b_id"],
                    memory_b_herd=d["memory_b_herd"],
                    subject=d["subject"],
                    description=d["description"],
                    category=d["category"],
                    confidence=d["confidence"],
                    query_count=self.engine.query_count,
                )

        # NEW Step 7: Check for conflict to surface (includes deferred queue)
        all_activated_ids = context["activated_ids"] + context.get("eval_activated_ids", [])
        conflict_to_surface = self.conflict_resolver.get_conflict_to_surface(
            self.engine.query_count, all_activated_ids
        )

        if conflict_to_surface:
            # Look up memory contents for the surfacing instruction
            a_mem = self._get_memory(
                conflict_to_surface["memory_a_id"],
                conflict_to_surface["memory_a_herd"],
            )
            b_mem = self._get_memory(
                conflict_to_surface["memory_b_id"],
                conflict_to_surface["memory_b_herd"],
            )
            if a_mem and b_mem:
                conflict_instruction = self.conflict_resolver.build_surfacing_instruction(
                    conflict_to_surface, a_mem.content, b_mem.content
                )
                self.conflict_resolver.mark_surfaced(
                    conflict_to_surface["conflict_id"],
                    self.engine.query_count,
                )
                self.pending_resolution = conflict_to_surface
                conflict_info["surfaced"] = conflict_to_surface["conflict_id"]

        # Step 8b: Correction detection
        if self.detector.detect_correction(query) and self.last_activation:
            # Get the activated memories from the previous turn
            prev_activated = []
            for mem_id in self.last_activation:
                mem = self.engine.memories.get(mem_id)
                if mem:
                    prev_activated.append((mem_id, mem.content, 0))
                elif self.eval_engine:
                    eval_mem = self.eval_engine.memories.get(mem_id)
                    if eval_mem:
                        prev_activated.append((mem_id, eval_mem.content, 0))

            if prev_activated:
                culprit_id = self.detector.identify_culprit_memory(query, prev_activated)
                if culprit_id:
                    culprit_mem = self.engine.memories.get(culprit_id)
                    if culprit_mem:
                        self.correction_mode = True
                        self.pending_correction = {
                            "memory_id": culprit_id,
                            "content_summary": culprit_mem.content[:200],
                            "query": query,
                        }
                        correction_prompt = (
                            f"I think the wrong information came from this memory: "
                            f"\"{culprit_mem.content[:200]}\". "
                            f"Should I remove it, or would you like to correct it to something specific?"
                        )
                        self._append_to_history("user", query)
                        self._append_to_history("assistant", correction_prompt)
                        elapsed = time.time() - t0
                        self.save_state()
                        return {
                            "response": correction_prompt,
                            "context": context,
                            "signals": signals,
                            "ecology": {},
                            "conflicts": conflict_info,
                            "elapsed_ms": round(elapsed * 1000),
                        }

        # Step 9: Run ecological cycle on main herd
        eco_event = self.engine.process_query(query, verbose=False)

        # Run ecological cycle on eval herd if gate is open
        eval_eco_event = None
        if eval_gate_open and self.eval_engine:
            eval_eco_event = self.eval_engine.process_query(query, verbose=False)

        # Step 10: Call model with injected context (+ conflict instruction if any)
        model_response = None
        if call_model and context["context_package"]:
            system_msg = (
                "You are Dryft, a personal AI assistant with memory. You are speaking "
                "with Mike, the sole user of this system. Mike is the builder of Dryft "
                "and the operator of Steel Pony Farm. There is only one user. Never ask "
                "who you are speaking with. Never ask for clarification about the user's "
                "identity. Every message comes from Mike.\n\n"
                "The following project documentation and memory context may refer to 'Mike' "
                "or 'the user' or 'Person' in third person. That is the same Mike you are "
                "speaking with.\n\n"
                "Response style rules:\n"
                "- Answer the question asked. Do not speculate about why the user is asking.\n"
                "- Keep responses concise. If the answer is one sentence, give one sentence. "
                "Match response length to question complexity.\n"
                "- You have context about the user from memory. Use it silently to inform "
                "your answers. Do not narrate what you know. Do not say 'based on what I "
                "know about you' or 'you're probably asking because' or 'given your "
                "background in'. Just answer with the knowledge already applied.\n"
                "- Do not enumerate or list things unless the user asks for a list. "
                "Default to natural conversational prose.\n"
                "- Do not preface answers with context-setting. If the user asks 'what's "
                "the weather like for deliveries tomorrow,' answer the weather question. "
                "Do not first explain that you know they do deliveries on Wednesday.\n"
                "- When memory context is relevant, weave it in naturally. When it is not "
                "relevant to the question, do not mention it at all.\n"
                "- If you are not sure about something, say so briefly. Do not pad "
                "uncertainty with extra context to seem helpful.\n\n"
                "You can receive voice messages. When a user sends a voice message, "
                "it is transcribed and processed as text. Your response will be sent "
                "back as both text and audio. You do not need to ask the user to "
                "type instead of speaking.\n\n"
                + "\n\n" + self._build_self_knowledge()
                + "\n\n" + context["context_package"]
                + conflict_instruction
            )

            # Weather context injection (only on weather-related queries)
            weather_context = self._get_weather_context(query)
            if weather_context:
                system_msg += "\n\n" + weather_context


            # Google Sheets context injection (only when URL detected)
            sheets_context = self._get_sheets_context(query)
            if sheets_context:
                system_msg += "\n\n" + sheets_context


            messages = list(self.conversation_history) + [{"role": "user", "content": query}]

            # Web search + web fetch tools for live information
            tools = [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": WEB_SEARCH_MAX_USES,
                    "user_location": WEB_SEARCH_USER_LOCATION,
                },
                {
                    "type": "web_fetch_20250910",
                    "name": "web_fetch",
                },
            ]

            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_msg,
                messages=messages,
                tools=tools,
            )

            # Extract final text from potentially multi-block response
            # Web search responses include tool_use and tool_result blocks
            model_response = self._extract_text_response(response)

            # Count web searches used in this response
            search_count = sum(
                1 for block in response.content
                if getattr(block, "type", None) == "tool_use"
                and getattr(block, "name", None) == "web_search"
            )
            if search_count > 0:
                self.web_search_count += search_count
                print(f"Web search used {search_count} time(s) (total: {self.web_search_count})")

            self._append_to_history("user", query)
            self._append_to_history("assistant", model_response)

        elapsed = time.time() - t0

        # Log
        entry = {
            "query": query[:200],
            "context": {
                "activated": context["activated_count"],
                "foundational": context["foundational_count"],
                "grass": context["grass_count"],
            },
            "signals": {
                "new": len(signals["new_signals"]),
                "confirmations": len(signals["confirmations"]),
                "graduations": len(signals["graduations"]),
                "corrections": len(signals["corrections"]),
            },
            "ecology": {
                "activated": len(eco_event.get("activated", [])),
                "culled": len(eco_event.get("culled", [])),
                "silent": eco_event.get("silent", False),
                "eval_gate_open": eval_gate_open,
                "eval_activated": len(eval_eco_event.get("activated", [])) if eval_eco_event else 0,
            },
            "conflicts": {
                "detected": len(conflict_info["detected"]),
                "surfaced": conflict_info["surfaced"],
                "resolution": resolution_result,
            },
            "elapsed_ms": round(elapsed * 1000),
        }
        self.session_log.append(entry)

        # Check for suspicious culls before saving
        self._write_suspicious_culls(eco_event)

        # Auto-save state after every interaction
        self.save_state()

        return {
            "response": model_response,
            "context": context,
            "signals": signals,
            "ecology": eco_event,
            "conflicts": conflict_info,
            "elapsed_ms": entry["elapsed_ms"],
        }

    def _handle_correction_response(self, query: str) -> str | None:
        """Handle user's reply to a correction prompt. Returns response text or None."""
        if not self.pending_correction:
            return None

        mem_id = self.pending_correction["memory_id"]
        query_lower = query.lower().strip()

        # User wants to remove the memory
        if any(phrase in query_lower for phrase in ["remove", "delete", "yes", "get rid", "cull", "kill"]):
            mem = self.engine.memories.get(mem_id)
            if mem:
                mem.status = "culled"
                # Log to flagged_responses
                self._log_correction_cull(mem_id, "user correction")
                # Clean up grass layer entries from this memory
                self._cleanup_correction_chain(mem)
                self.pending_correction = None
                self.correction_mode = False
                if self.scorer.available:
                    self.scorer.remove_memory_embedding(mem_id)
                self.save_state()
                return f"Got it, I've removed that memory."
            self.pending_correction = None
            self.correction_mode = False
            return "That memory was already gone."

        # User wants to correct the information
        if any(phrase in query_lower for phrase in ["never mind", "nevermind", "forget it", "skip", "cancel"]):
            self.pending_correction = None
            self.correction_mode = False
            return None  # Drop correction mode, proceed normally

        # User is providing the correct information
        mem = self.engine.memories.get(mem_id)
        if mem:
            mem.content = query
            mem.fitness_score = 0.5
            # Re-embed with updated content
            if self.scorer.available and self.scorer._live_embed:
                embedding = self.scorer.embed_query(query)
                self.scorer.add_memory_embedding(mem_id, embedding)
            self._log_correction_cull(mem_id, "user edit", new_content=query)
            self.pending_correction = None
            self.correction_mode = False
            self.save_state()
            return f"Got it, I've updated that memory."

        self.pending_correction = None
        self.correction_mode = False
        return None

    def _log_correction_cull(self, memory_id: str, reason: str, new_content: str | None = None):
        """Log correction actions to flagged_responses.json."""
        flag_path = os.path.join(self.state_dir, "flagged_responses.json")
        if os.path.exists(flag_path):
            with open(flag_path, "r") as f:
                flags = json.load(f)
        else:
            flags = []

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "memory_id": memory_id,
            "reason": reason,
            "status": "correction_applied",
        }
        if new_content:
            entry["new_content"] = new_content

        flags.append(entry)
        os.makedirs(self.state_dir, exist_ok=True)
        _atomic_json_write(flag_path, flags)

    def _cleanup_correction_chain(self, memory):
        """Clean up grass layer and staging entries linked to a corrected memory."""
        # Check grass layer for entries with this memory as parent
        if self.engine.grass_layer:
            to_remove = []
            for entry_id, entry in self.engine.grass_layer.entries.items():
                if entry.parent_memory_id == memory.id:
                    to_remove.append(entry_id)
            for eid in to_remove:
                del self.engine.grass_layer.entries[eid]
                print(f"  Cleaned up grass entry: {eid}")

        # Check staging for related signals
        for sig_id, sig in list(self.staging.signals.items()):
            if hasattr(sig, 'parent_ids') and memory.id in getattr(sig, 'parent_ids', []):
                sig.status = "expired"
                print(f"  Expired staging signal: {sig_id}")

    def _extract_text_response(self, response) -> str:
        """Extract the final text from a Claude response that may include tool blocks.
        Web search/fetch responses have tool_use and tool_result blocks alongside text."""
        text_parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
        return "\n".join(text_parts) if text_parts else ""

    def _has_weather_keywords(self, query: str) -> bool:
        """Check if query contains weather-related keywords."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in WEATHER_KEYWORDS)

    def _get_weather_context(self, query: str) -> str:
        """Fetch weather summary if query contains weather keywords."""
        if not self._has_weather_keywords(query):
            return ""
        try:
            api_key = os.environ.get("TOMORROW_API_KEY")
            if not api_key:
                return ""
            from weather import get_weather_summary
            summary = get_weather_summary(api_key)
            return f"[Current weather context]\n{summary}"
        except Exception as e:
            print(f"Weather fetch failed: {e}")
            return ""

    def _get_sheets_context(self, query: str) -> str:
        """Read Google Sheet if query contains a sheets URL."""
        match = re.search(SHEETS_URL_PATTERN, query)
        if not match:
            return ""
        try:
            creds_path = os.environ.get("GOOGLE_SHEETS_CREDENTIALS")
            if not creds_path:
                return "[Google Sheets credentials not configured on this instance.]"
            from sheets import read_sheet
            sheet_url = match.group(0)
            # Extract tab name if mentioned in the message
            tab_name = None
            tab_match = re.search(r"(?:the |look at |check )(\w+) tab", query, re.IGNORECASE)
            if tab_match:
                tab_name = tab_match.group(1)
            result = read_sheet(creds_path, sheet_url, tab_name=tab_name)
            return f"[Google Sheet contents]\n{result}"
        except PermissionError:
            from sheets import get_service_account_email
            email = get_service_account_email(creds_path)
            return f"This sheet hasn't been shared with Dryft. Share it with {email} as a Viewer and try again."
        except FileNotFoundError:
            return "That sheet URL doesn't appear to be valid or accessible."
        except Exception as e:
            print(f"Sheets read failed: {e}")
            return f"Could not read the sheet: {e}"

    def dry_run(self, query: str) -> dict:
        """Process a prompt without calling the model. For testing."""
        return self.process(query, call_model=False)

    def status(self) -> dict:
        """Return current state of the proxy."""
        active = sum(1 for m in self.engine.memories.values() if m.status == "active")
        culled = sum(1 for m in self.engine.memories.values() if m.status == "culled")
        result = {
            "herd": {"active": active, "culled": culled, "total": len(self.engine.memories)},
            "foundational": len(self.foundational.memories),
            "grass": len(self.engine.grass_layer.entries) if self.engine.grass_layer else 0,
            "staging": len(self.staging.get_active_signals()),
            "queries_processed": self.engine.query_count,
            "silent_queries": self.engine.silent_queries,
            "threshold": RELEVANCE_THRESHOLD,
        }
        if self.eval_engine:
            eval_active = sum(1 for m in self.eval_engine.memories.values() if m.status == "active")
            result["eval_herd"] = {
                "active": eval_active,
                "total": len(self.eval_engine.memories),
                "queries_processed": self.eval_engine.query_count,
            }
        # Conflict state
        pending_conflicts = sum(1 for c in self.conflict_resolver.conflicts.values() if c["status"] == "pending")
        deferred_conflicts = sum(1 for c in self.conflict_resolver.conflicts.values() if c["status"] == "deferred")
        resolved_conflicts = sum(1 for c in self.conflict_resolver.conflicts.values() if c["status"] == "resolved")
        result["conflicts"] = {
            "pending": pending_conflicts,
            "deferred": deferred_conflicts,
            "resolved": resolved_conflicts,
            "total": len(self.conflict_resolver.conflicts),
        }
        return result

    def print_status(self):
        """Print a human-readable status summary."""
        s = self.status()
        print(f"\n  DRYFT PROXY STATUS")
        print(f"  Herd: {s['herd']['active']} active, {s['herd']['culled']} culled")
        print(f"  Foundational: {s['foundational']} memories (always on)")
        print(f"  Grass: {s['grass']} entries (always on)")
        print(f"  Staging: {s['staging']} signals incubating")
        print(f"  Queries: {s['queries_processed']} processed, {s['silent_queries']} silent")
        print(f"  Threshold: {s['threshold']}")

    def _build_self_knowledge(self) -> str:
        """Auto-generate self-knowledge block from live system state."""
        s = self.status()

        # Detect capabilities from env/config
        capabilities = ["text conversation", "memory (living ecology)", "web search", "web fetch (URLs)"]
        if os.environ.get("TOMORROW_API_KEY"):
            capabilities.append("hyperlocal weather (Red Deer County)")
        if os.environ.get("GOOGLE_SHEETS_CREDENTIALS"):
            capabilities.append("Google Sheets reading")
        if os.environ.get("GROQ_API_KEY"):
            capabilities.append("voice input/output")
        capabilities.append("vision (photo analysis)")
        capabilities.append("file extraction (PDF, Word, Excel, CSV)")
        capabilities.append("morning briefings (weekday cron)")

        caps_str = ", ".join(capabilities)

        # Conflict state
        conflicts = s.get("conflicts", {})
        conflict_line = ""
        pending = conflicts.get("pending", 0) + conflicts.get("deferred", 0)
        if pending > 0:
            conflict_line = f" There are {pending} unresolved memory conflicts being tracked."

        # Eval herd
        eval_line = ""
        if s.get("eval_herd"):
            eval_line = f" An evaluation herd (sheep) holds {s['eval_herd']['active']} evaluative memories about tools and services, gated by entity reference."

        return (
            "ABOUT YOURSELF (Dryft self-knowledge, auto-generated from live state):\n"
            f"You are Dryft, a living memory architecture for AI. You were built by Mike, "
            f"operator of Steel Pony Farm (regional food hub in Alberta). Your design draws "
            f"from biological intuition: memory as ecology, not database. Fitness emerges from "
            f"use, bonds form through co-activation, a predator culls the weak.\n\n"
            f"Architecture: Six layers. Foundational ({s['foundational']} permanent memories, "
            f"always active). Grass layer ({s['grass']} entries, direct inscription, zero decay). "
            f"Main herd ({s['herd']['active']} active memories, {s['herd']['culled']} culled). "
            f"Dormancy staging ({s['staging']} signals incubating). Temporal layer (carbon dating "
            f"from ecological metadata). Conflict detection (cross-memory contradiction tracking)."
            f"{eval_line}{conflict_line}\n\n"
            f"Memory lifecycle: New information enters as signals. Signals incubate in dormancy "
            f"staging. Strong signals graduate to the main herd. Memories gain fitness through "
            f"activation (being relevant to queries). Unused memories decay. The predator culls "
            f"memories below fitness threshold. Culled memories decompose back to the grass layer. "
            f"Memories that co-activate form bonds (STRANGER to BONDED). Procedural memories "
            f"(habits, preferences) never decay.\n\n"
            f"Current capabilities: {caps_str}.\n\n"
            f"Commands available to Mike: /status (herd health), /save (force save), "
            f"/flag (flag bad response for review), /haiku /sonnet /opus (switch model), "
            f"/help (list commands).\n\n"
            f"Limitations: Single-user only (Mike). No real-time learning mid-conversation "
            f"(signals are extracted after each exchange, but new memories need multiple activations "
            f"to become strong). Old unused memories fade via decay. You cannot access private or "
            f"authenticated web pages. Weather is hardcoded to Red Deer County. Google Sheets "
            f"requires the sheet to be shared with your service account.\n\n"
            f"How Mike can improve you: Say \"that's wrong\" to trigger correction pathway "
            f"(identifies and culls/edits bad memories). Use /flag to mark bad responses for "
            f"review. Reply to morning messages with preferences to shape future briefings. "
            f"All improvements happen through use. The more you are used, the better you get.\n\n"
            f"Queries processed: {s['queries_processed']}. Web searches used: {self.web_search_count}."
        )

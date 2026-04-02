"""
Dryft - HerdEngine
Phase 1: Proximity Prototype / Phase 6: Vector Scoring

The core of the living memory architecture. Manages a population of memory
objects — their fitness, decay, proximity bonds, and eventual cull eligibility.
Phase 4 Run B adds the grass layer: procedural substrate with decomposition loop.
Phase 6 adds vector scoring: cosine similarity on dense embeddings replaces keyword overlap.

No external dependencies required (vector scoring uses pre-computed cache).
"""

import json
import re
import math
from pathlib import Path
from copy import deepcopy


# ── Grass layer parameters ────────────────────────────────────────────────────
# Minimum richness score for a culled memory to synthesize a grass entry.
# A memory below this threshold decomposes to nothing (too sparse to leave a trace).
GRASS_RICHNESS_THRESHOLD = 0.20

# ── Bond status thresholds ────────────────────────────────────────────────────
# A pair of memories moves through states as co-activation accumulates.
# These are the floors. Both conditions (count AND bond score) must be met.

BOND_THRESHOLDS = {
    "acquainted": {"co_activations": 3,  "bond_score": 0.15},
    "bonded":     {"co_activations": 6,  "bond_score": 0.30},
    "dating":     {"co_activations": 10, "bond_score": 0.50},
}

# ── Fitness parameters ────────────────────────────────────────────────────────
FITNESS_GAIN_ON_ACTIVATION = 0.08
FITNESS_MAX = 1.0
FITNESS_MIN = 0.0

# ── Bond score parameters ─────────────────────────────────────────────────────
BOND_GAIN_ON_CO_ACTIVATION = 0.06
BOND_DRIFT_WHEN_APART = 0.008   # subtracted per query when pair not co-activated
BOND_MAX = 1.0
BOND_MIN = 0.0

# ── Predator ──────────────────────────────────────────────────────────────────
# Memories below this fitness AND with age > this threshold become cull-eligible.
PREDATOR_FITNESS_FLOOR = 0.1
PREDATOR_MIN_AGE = 20           # queries; young memories get a grace period
PREDATOR_CONSECUTIVE_TO_CULL = 5  # consecutive prey-eligible queries before cull executes


class Memory:
    """A single memory object with all its ecological properties."""

    def __init__(self, data: dict):
        self.id = data["id"]
        self.content = data["content"]
        self.memory_type = data["memory_type"]          # episodic | semantic | procedural
        self.keywords = [k.lower() for k in data.get("keywords", [])]
        self.fitness_score = data.get("fitness_score", 0.5)
        self.confidence_score = data.get("confidence_score", 0.8)
        self.decay_rate = data.get("decay_rate", 0.005)
        self.age = data.get("age", 0)                   # query cycles since creation
        self.activation_count = data.get("activation_count", 0)
        self.last_activated_at = data.get("last_activated_at", 0)
        self.parent_ids = data.get("parent_ids", [])
        self.child_ids = data.get("child_ids", [])
        self.proximity_bonds = data.get("proximity_bonds", {})      # {id: bond_score}
        self.co_activation_counts = data.get("co_activation_counts", {})  # {id: int}
        self.retrieval_pointer = data.get("retrieval_pointer", None)
        self.compression_history = data.get("compression_history", [])
        self.predator_threshold = data.get("predator_threshold", PREDATOR_FITNESS_FLOOR)
        self.status = data.get("status", "active")      # active | compressed | culled
        self.consecutive_prey_eligible = data.get("consecutive_prey_eligible", 0)
        self.rehydrated = data.get("rehydrated", False)  # True if ever recovered from prey-eligible
        # Phase 9: Retrieval integrity fields
        self.extraction_context = data.get("extraction_context", None)  # "declarative" | "evaluative" | None
        self.evaluated_entity = data.get("evaluated_entity", None)      # e.g. "Shopify" | None
        self.source_conversation_id = data.get("source_conversation_id", None)  # conversation name/index
        # Temporal metadata: query number when memory entered the herd (0 = seed/unknown)
        self.created_at = data.get("created_at", 0)

    def relevance_score(self, query_tokens: list[str]) -> float:
        """
        Score this memory's relevance to a query.
        Uses keyword overlap — a deliberate simplification for Phase 1.
        In Phase 2+, swap this for vector similarity via Chroma embeddings.

        Returns a float 0.0–1.0.
        """
        if not self.keywords or not query_tokens:
            return 0.0

        query_set = set(query_tokens)
        keyword_set = set(self.keywords)

        # Multi-word keywords (e.g. "overnight oats") need phrase matching
        full_query = " ".join(query_tokens)
        phrase_matches = sum(1 for kw in self.keywords if kw in full_query)

        # Single-word overlap
        single_matches = len(query_set & keyword_set)

        # Phrase matches worth double — they're more specific
        total_matches = phrase_matches + single_matches
        denominator = len(keyword_set) + len(query_set) - total_matches
        if denominator == 0:
            return 0.0

        # Jaccard-like score, boosted by phrase matches
        base = total_matches / denominator
        return min(base * 1.5, 1.0)

    def bond_status_with(self, other_id: str) -> str:
        """Return the current bond status between this memory and another."""
        co_count = self.co_activation_counts.get(other_id, 0)
        bond = self.proximity_bonds.get(other_id, 0.0)

        for status in ["dating", "bonded", "acquainted"]:
            t = BOND_THRESHOLDS[status]
            if co_count >= t["co_activations"] and bond >= t["bond_score"]:
                return status

        return "separate"

    def is_predator_eligible(self) -> bool:
        return (
            self.fitness_score <= self.predator_threshold
            and self.age >= PREDATOR_MIN_AGE
            and self.status == "active"
        )

    def apply_conflict_loss(
        self,
        fitness_penalty: float,
        confidence_penalty: float,
        decay_multiplier: float,
        new_predator_threshold: float,
        bond_penalty: float,
        decay_duration: int,
    ) -> dict:
        """
        Apply conflict cascade penalties. Called by ConflictResolver.
        Returns a dict of all changes for logging.
        """
        changes = {
            "fitness_before": round(self.fitness_score, 4),
            "confidence_before": round(self.confidence_score, 4),
            "decay_before": self.decay_rate,
            "predator_threshold_before": self.predator_threshold,
            "bond_changes": [],
        }

        self.fitness_score = max(self.fitness_score - fitness_penalty, FITNESS_MIN)
        self.confidence_score = max(self.confidence_score - confidence_penalty, 0.05)
        self.predator_threshold = max(self.predator_threshold, new_predator_threshold)
        self.decay_rate = self.decay_rate * decay_multiplier

        changes["fitness_after"] = round(self.fitness_score, 4)
        changes["confidence_after"] = round(self.confidence_score, 4)
        changes["decay_after"] = self.decay_rate
        changes["predator_threshold_after"] = self.predator_threshold

        # Bond cascade: penalize top 5 strongest bonds
        if self.proximity_bonds:
            sorted_bonds = sorted(
                self.proximity_bonds.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            for partner_id, old_bond in sorted_bonds[:5]:
                new_bond = max(old_bond - bond_penalty, BOND_MIN)
                self.proximity_bonds[partner_id] = new_bond
                changes["bond_changes"].append({
                    "partner_id": partner_id,
                    "bond_before": round(old_bond, 4),
                    "bond_after": round(new_bond, 4),
                })

        return changes

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "keywords": self.keywords,
            "fitness_score": round(self.fitness_score, 4),
            "confidence_score": round(self.confidence_score, 4),
            "decay_rate": self.decay_rate,
            "age": self.age,
            "activation_count": self.activation_count,
            "last_activated_at": self.last_activated_at,
            "parent_ids": self.parent_ids,
            "child_ids": self.child_ids,
            "proximity_bonds": {k: round(v, 4) for k, v in self.proximity_bonds.items()},
            "co_activation_counts": self.co_activation_counts,
            "retrieval_pointer": self.retrieval_pointer,
            "compression_history": self.compression_history,
            "predator_threshold": self.predator_threshold,
            "status": self.status,
            "consecutive_prey_eligible": self.consecutive_prey_eligible,
            "rehydrated": self.rehydrated,
            "extraction_context": self.extraction_context,
            "evaluated_entity": self.evaluated_entity,
            "source_conversation_id": self.source_conversation_id,
            "created_at": self.created_at,
        }


class GrassEntry:
    """
    A single entry in the procedural substrate (grass layer).
    No fitness, no decay, no predator eligibility. Always on.
    Two origins: direct inscription (user stated a preference) or
    emergent synthesis (decomposed from a culled herd memory).
    """

    def __init__(self, data: dict):
        self.id = data["id"]
        self.content = data["content"]
        self.source = data.get("source", "direct")       # 'direct' | 'emergent'
        self.created_at = data.get("created_at", 0)      # query number
        self.parent_memory_id = data.get("parent_memory_id", None)
        self.keywords = [k.lower() for k in data.get("keywords", [])]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "created_at": self.created_at,
            "parent_memory_id": self.parent_memory_id,
            "keywords": self.keywords,
        }


class GrassLayer:
    """
    The procedural substrate. Not part of the herd — the ground the herd stands on.
    Manages two kinds of entries:
      - Direct: inscribed immediately when a user states a preference.
      - Emergent: synthesized from a culled memory's lifetime stats (decomposition loop).
    """

    def __init__(self, seed_path: str = "grass_layer_seed.json"):
        self.entries: dict[str, GrassEntry] = {}
        self.decomposition_log: list[dict] = []
        self._load(seed_path)

    def _load(self, seed_path: str):
        path = Path(seed_path)
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        for item in data:
            # Seed entries may carry a full Memory schema; extract what GrassEntry needs.
            entry_data = {
                "id": item["id"],
                "content": item["content"],
                "source": item.get("source", "direct"),
                "created_at": item.get("created_at", 0),
                "parent_memory_id": item.get("parent_memory_id", None),
                "keywords": item.get("keywords", []),
            }
            entry = GrassEntry(entry_data)
            self.entries[entry.id] = entry

    def save(self, path: str = "grass_layer_state.json"):
        data = [e.to_dict() for e in self.entries.values()]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ── Decomposition loop ────────────────────────────────────────────────────

    def _richness(self, memory: "Memory") -> float:
        """
        Compute how rich a dying memory's decomposition will be.
        Proportional to lifetime activity: activations, bond depth,
        co-activation diversity, and rehydration bonus.
        """
        activation_score = min(memory.activation_count / 30, 1.0)
        bond_depth = max(memory.proximity_bonds.values(), default=0.0)
        diversity_score = min(len(memory.co_activation_counts) / 5, 1.0)
        rehydration_bonus = 0.10 if memory.rehydrated else 0.0

        return (
            activation_score  * 0.40
            + bond_depth      * 0.30
            + diversity_score * 0.20
            + rehydration_bonus
        )

    def _synthesize_content(self, memory: "Memory", richness: float) -> str:
        """
        Rule-based pattern extraction driven by bond topology.
        Richer memories produce more specific, behaviorally useful entries.
        Bond partners are the primary signal: if a memory died with strong bonds,
        the procedural trace captures the relational pattern, not just the topic.
        """
        bond_partners = sorted(
            [(oid, score) for oid, score in memory.proximity_bonds.items() if score > 0.05],
            key=lambda x: x[1], reverse=True,
        )
        top_keywords = memory.keywords[:3]
        topic = ", ".join(top_keywords) if top_keywords else memory.id.replace("-", " ")
        acts = memory.activation_count

        if bond_partners and richness >= 0.45:
            # Well-bonded, well-lived: relational pattern from bond topology
            partners = [p[0].replace("-", " ").replace("_", " ") for p in bond_partners[:2]]
            partner_str = " and ".join(partners)
            return (
                f"Recurring context cluster: topics around '{topic}' co-occur regularly "
                f"with '{partner_str}'. Treat as related context when either arises. "
                f"[Synthesized from {memory.id} — {acts} activations, "
                f"{len(bond_partners)} bond partner(s), richness {richness:.2f}]"
            )
        elif richness >= 0.30:
            # Moderate richness, solo domain: domain presence trace
            return (
                f"User has recurring context around: {topic}. "
                f"This topic appeared across multiple interactions. "
                f"[Synthesized from {memory.id} — {acts} activations, richness {richness:.2f}]"
            )
        else:
            # Just above threshold: thin trace, preserve summary of content
            summary = memory.content[:100].rstrip()
            if len(memory.content) > 100:
                summary += "..."
            return (
                f"Thin trace: {summary} "
                f"[Synthesized from {memory.id} — {acts} activation(s), richness {richness:.2f}]"
            )

    def decompose(self, memory: "Memory", query_count: int) -> "GrassEntry | None":
        """
        Decompose a culled memory into the grass layer.
        Logs the event regardless of outcome.
        Returns the new GrassEntry if one was synthesized, None if memory was too sparse.
        """
        richness = self._richness(memory)
        bond_depth = max(memory.proximity_bonds.values(), default=0.0)

        log_entry = {
            "query": query_count,
            "memory_id": memory.id,
            "memory_type": memory.memory_type,
            "activation_count": memory.activation_count,
            "bond_depth": round(bond_depth, 4),
            "co_activation_diversity": len(memory.co_activation_counts),
            "rehydrated": memory.rehydrated,
            "richness_score": round(richness, 4),
        }

        if richness < GRASS_RICHNESS_THRESHOLD:
            log_entry["outcome"] = "sparse"
            log_entry["grass_entry_id"] = None
            self.decomposition_log.append(log_entry)
            return None

        content = self._synthesize_content(memory, richness)
        entry_id = f"emergent-{memory.id}"
        entry = GrassEntry({
            "id": entry_id,
            "content": content,
            "source": "emergent",
            "created_at": query_count,
            "parent_memory_id": memory.id,
            "keywords": memory.keywords[:5],
        })
        self.entries[entry_id] = entry

        log_entry["outcome"] = "synthesized"
        log_entry["grass_entry_id"] = entry_id
        log_entry["grass_content"] = content
        self.decomposition_log.append(log_entry)

        return entry


class HerdEngine:
    """
    The core ecological engine. Manages the population of memories,
    processes queries, updates fitness and bonds, and runs the predator.
    """

    def __init__(self, memories_path: str = "memories.json",
                 predator_executes: bool = True,
                 grass_layer_path: str | None = None,
                 vector_scorer: "VectorScorer | None" = None,
                 top_n: int = 2,
                 relevance_threshold: float | None = None,
                 pause_on_silence: bool = False):
        self.memories_path = Path(memories_path)
        self.memories: dict[str, Memory] = {}
        self.query_count = 0
        self.event_log: list[dict] = []   # structured log of what happened each query
        self.predator_executes = predator_executes  # False = flag prey but never cull
        self.top_n = top_n                 # how many memories activate per query (Phase 1-6)
        self.relevance_threshold = relevance_threshold  # Phase 7: threshold-based activation
        self.pause_on_silence = pause_on_silence  # Phase 7: herd clock pauses on zero activations
        self.silent_queries = 0            # counter: queries where herd clock paused
        self.grass_layer: GrassLayer | None = None
        self.vector_scorer = vector_scorer  # Phase 6: pre-computed embedding scorer

        if grass_layer_path:
            self.grass_layer = GrassLayer(grass_layer_path)

        self._load_memories()

    def _load_memories(self):
        with open(self.memories_path) as f:
            data = json.load(f)
        for item in data:
            m = Memory(item)
            self.memories[m.id] = m

    def save_memories(self):
        """Persist current herd state back to JSON."""
        data = [m.to_dict() for m in self.memories.values()]
        with open(self.memories_path, "w") as f:
            json.dump(data, f, indent=2)

    def save_grass_layer(self, path: str = "grass_layer_state.json"):
        """Persist the grass layer (direct + emergent entries) to JSON."""
        if self.grass_layer is not None:
            self.grass_layer.save(path)

    # ── Query processing ──────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase, split on non-alpha, remove very short tokens."""
        tokens = re.findall(r"[a-z][a-z']+", text.lower())
        stopwords = {"how", "do", "i", "the", "a", "an", "to", "for", "is",
                     "are", "what", "my", "me", "in", "of", "and", "or", "can",
                     "should", "would", "on", "at", "with", "that", "this", "it",
                     "be", "we", "when", "if", "about", "have", "has", "was"}
        return [t for t in tokens if t not in stopwords and len(t) > 2]

    def process_query(self, query: str, verbose: bool = True) -> dict:
        """
        Core method. One query passes through the herd:
          1. Score all memories for relevance
          2. Activate top 2
          3. Update fitness (up for activated, decay for rest)
          4. Update co-activation counts and bond scores
          5. Age all memories
          6. Check predator eligibility
          7. Return event record
        """
        self.query_count += 1
        tokens = self._tokenize(query)

        # ── 1. Score relevance ──────────────────────────────────────────────
        active_memories = [m for m in self.memories.values() if m.status == "active"]
        if self.vector_scorer and self.vector_scorer.available:
            scored = [(m, self.vector_scorer.score(m.id, query)) for m in active_memories]
        else:
            scored = [(m, m.relevance_score(tokens)) for m in active_memories]
        scored.sort(key=lambda x: x[1], reverse=True)

        # ── Activation selection ──────────────────────────────────────────
        if self.relevance_threshold is not None:
            # Phase 7: threshold-based activation — all above threshold
            activated = [m for m, score in scored if score >= self.relevance_threshold]
            activated_scores = {m.id: score for m, score in scored if score >= self.relevance_threshold}
        else:
            # Phase 1-6: top-N activation
            activated = [m for m, score in scored[:self.top_n] if score > 0]
            activated_scores = {m.id: score for m, score in scored[:self.top_n]}

        top_n = activated  # keep variable name for bond logic compatibility
        top_two = top_n
        top_two_ids = [m.id for m in activated]
        top_two_scores = activated_scores

        event = {
            "query_number": self.query_count,
            "query": query,
            "tokens": tokens,
            "activated": [],
            "decayed": [],
            "bond_updates": [],
            "predator_eligible": [],
            "culled": [],
        }

        # ── Pause on silence: herd clock stops if nothing crossed threshold ──
        if self.pause_on_silence and len(activated) == 0:
            self.silent_queries += 1
            event["silent"] = True
            self.event_log.append(event)
            return event

        # ── 2 & 3. Update fitness ───────────────────────────────────────────
        for memory in active_memories:
            old_fitness = memory.fitness_score
            memory.age += 1

            if memory.id in top_two_ids:
                # Rehydration: memory was prey-eligible but is now being rescued by activation
                if memory.consecutive_prey_eligible > 0:
                    memory.rehydrated = True
                memory.activation_count += 1
                memory.last_activated_at = self.query_count
                memory.fitness_score = min(
                    memory.fitness_score + FITNESS_GAIN_ON_ACTIVATION, FITNESS_MAX
                )
                event["activated"].append({
                    "id": memory.id,
                    "fitness_before": round(old_fitness, 4),
                    "fitness_after": round(memory.fitness_score, 4),
                    "relevance_score": round(top_two_scores.get(memory.id, 0), 4),
                })
            else:
                memory.fitness_score = max(
                    memory.fitness_score - memory.decay_rate, FITNESS_MIN
                )
                event["decayed"].append({
                    "id": memory.id,
                    "fitness_before": round(old_fitness, 4),
                    "fitness_after": round(memory.fitness_score, 4),
                })

        # ── 4. Update bonds ─────────────────────────────────────────────────
        # All pairs among co-activated memories get a bond update
        if len(top_n) >= 2:
            for idx_a in range(len(top_n)):
                for idx_b in range(idx_a + 1, len(top_n)):
                    m_a, m_b = top_n[idx_a], top_n[idx_b]

                    # Co-activation count (stored on both sides)
                    m_a.co_activation_counts[m_b.id] = m_a.co_activation_counts.get(m_b.id, 0) + 1
                    m_b.co_activation_counts[m_a.id] = m_b.co_activation_counts.get(m_a.id, 0) + 1

                    co_count = m_a.co_activation_counts[m_b.id]

                    # Bond score: gain on co-activation
                    old_bond = m_a.proximity_bonds.get(m_b.id, 0.0)
                    new_bond = min(old_bond + BOND_GAIN_ON_CO_ACTIVATION, BOND_MAX)
                    m_a.proximity_bonds[m_b.id] = new_bond
                    m_b.proximity_bonds[m_a.id] = new_bond

                    old_status = m_a.bond_status_with(m_b.id)
                    # re-check after update
                    new_status = m_a.bond_status_with(m_b.id)

                    event["bond_updates"].append({
                        "pair": (m_a.id, m_b.id),
                        "co_activation_count": co_count,
                        "bond_score_before": round(old_bond, 4),
                        "bond_score_after": round(new_bond, 4),
                        "status_before": old_status,
                        "status_after": new_status,
                    })

        # Drift: bonds weaken only when BOTH memories in a pair were inactive this query.
        # A bond between A and B should not erode just because A was called without B —
        # that is normal retrieval. The bond only drifts when the topic goes entirely quiet.
        for memory in active_memories:
            for other_id in list(memory.proximity_bonds.keys()):
                if memory.id not in top_two_ids and other_id not in top_two_ids:
                    old = memory.proximity_bonds[other_id]
                    memory.proximity_bonds[other_id] = max(old - BOND_DRIFT_WHEN_APART, BOND_MIN)

        # ── 5. Predator check + cull execution ─────────────────────────────
        for memory in active_memories:
            if memory.is_predator_eligible():
                memory.consecutive_prey_eligible += 1
                event["predator_eligible"].append(memory.id)
                if self.predator_executes and memory.consecutive_prey_eligible >= PREDATOR_CONSECUTIVE_TO_CULL:
                    memory.status = "culled"
                    # Suspicious death detection
                    has_acquainted_bond = any(
                        memory.co_activation_counts.get(pid, 0) >= BOND_THRESHOLDS["acquainted"]["co_activations"]
                        and bscore >= BOND_THRESHOLDS["acquainted"]["bond_score"]
                        for pid, bscore in memory.proximity_bonds.items()
                    )
                    suspicious = (
                        memory.activation_count >= 3
                        or has_acquainted_bond
                        or memory.rehydrated
                    )

                    cull_record = {
                        "id": memory.id,
                        "fitness": round(memory.fitness_score, 4),
                        "consecutive_prey_eligible": memory.consecutive_prey_eligible,
                        "activation_count": memory.activation_count,
                        "bond_scores_at_death": {k: round(v, 4) for k, v in memory.proximity_bonds.items()},
                        "co_activation_diversity": len(memory.co_activation_counts),
                        "rehydrated": memory.rehydrated,
                        "suspicious": suspicious,
                    }
                    # ── Decomposition loop: feed the grass layer ────────────
                    if self.grass_layer is not None:
                        grass_entry = self.grass_layer.decompose(memory, self.query_count)
                        decomp_log = self.grass_layer.decomposition_log[-1]
                        cull_record["decomposition"] = {
                            "richness_score": decomp_log["richness_score"],
                            "outcome": decomp_log["outcome"],
                            "grass_entry_id": decomp_log.get("grass_entry_id"),
                        }
                        if grass_entry:
                            event.setdefault("grass_synthesized", []).append({
                                "entry_id": grass_entry.id,
                                "parent_memory": memory.id,
                                "richness": decomp_log["richness_score"],
                                "content": grass_entry.content,
                            })
                    event["culled"].append(cull_record)
            else:
                memory.consecutive_prey_eligible = 0

        self.event_log.append(event)
        return event

    # ── Herd queries (read-only) ───────────────────────────────────────────────

    def herd_status(self) -> list[dict]:
        """Return all memories sorted by fitness, descending."""
        return sorted(
            [m.to_dict() for m in self.memories.values()],
            key=lambda x: x["fitness_score"],
            reverse=True
        )

    def get_bonds(self) -> list[dict]:
        """Return all meaningful bond pairs (acquainted or stronger)."""
        seen = set()
        result = []
        for memory in self.memories.values():
            for other_id, bond_score in memory.proximity_bonds.items():
                pair_key = tuple(sorted([memory.id, other_id]))
                if pair_key in seen or bond_score < 0.01:
                    continue
                seen.add(pair_key)
                status = memory.bond_status_with(other_id)
                co_count = memory.co_activation_counts.get(other_id, 0)
                result.append({
                    "memory_a": memory.id,
                    "memory_b": other_id,
                    "bond_score": round(bond_score, 4),
                    "co_activations": co_count,
                    "status": status,
                })
        return sorted(result, key=lambda x: x["bond_score"], reverse=True)

    def get_predator_eligible(self) -> list[str]:
        return [m.id for m in self.memories.values() if m.is_predator_eligible()]

    def score_query(self, query: str) -> list[tuple[str, str, float]]:
        """
        Score a query against the herd without running the ecological cycle.
        Returns list of (memory_id, content, score) for memories above threshold
        (or top_n if no threshold set), sorted by score descending.
        Read-only: does not modify fitness, bonds, or age.
        """
        active_memories = [m for m in self.memories.values() if m.status == "active"]
        if self.vector_scorer and self.vector_scorer.available:
            scored = [(m, self.vector_scorer.score(m.id, query)) for m in active_memories]
        else:
            tokens = self._tokenize(query)
            scored = [(m, m.relevance_score(tokens)) for m in active_memories]
        scored.sort(key=lambda x: x[1], reverse=True)

        if self.relevance_threshold is not None:
            results = [(m.id, m.content, s) for m, s in scored if s >= self.relevance_threshold]
        else:
            results = [(m.id, m.content, s) for m, s in scored[:self.top_n] if s > 0]
        return results

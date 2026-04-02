"""
Dryft — Phase 10B: Temporal Inference (Carbon Dating Layer)

Reads the herd's own ecological metadata to infer temporal relationships.
No external calendar. The isotope signatures are:
  created_at, age, last_activated_at, activation_count, parent_ids

Query-to-calendar mapping converts query numbers to approximate dates.
Supersession inference detects when newer memories replace older ones.
Generational tracking walks parent/child lineage chains.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path


class TemporalMapper:
    """
    Converts between query numbers and approximate calendar dates.
    Parameters persist in herd_meta.json.
    """

    def __init__(self, meta: dict):
        self.proxy_start_date = datetime.fromisoformat(
            meta.get("proxy_start_date", "2026-02-20")
        )
        self.queries_per_day = meta.get("queries_per_day", 5.0)

    def query_to_date(self, query_number: int) -> datetime:
        """Convert a query number to an approximate calendar date."""
        if query_number <= 0:
            return self.proxy_start_date
        days_offset = query_number / self.queries_per_day
        return self.proxy_start_date + timedelta(days=days_offset)

    def date_to_query(self, dt: datetime) -> int:
        """Convert a calendar date to an approximate query number."""
        delta = (dt - self.proxy_start_date).total_seconds() / 86400
        return max(0, int(delta * self.queries_per_day))

    def format_date(self, query_number: int) -> str:
        """Return a human-readable date string for a query number."""
        dt = self.query_to_date(query_number)
        return dt.strftime("%b %d, %Y")

    def days_ago(self, query_number: int, current_query: int) -> int:
        """How many approximate days between two query numbers."""
        dt1 = self.query_to_date(query_number)
        dt2 = self.query_to_date(current_query)
        return max(0, (dt2 - dt1).days)


def find_supersessions(activated_memories, scorer, mapper, current_query,
                       similarity_threshold=0.80, age_gap_threshold=50):
    """
    Detect potential supersession pairs among activated memories.

    Two memories supersede when:
    - They're about the same subject (vector similarity > threshold)
    - One was created significantly later than the other (age gap > threshold queries)

    Returns list of supersession notes to inject into context.
    Advisory only — does not modify memories.
    """
    if len(activated_memories) < 2:
        return []

    supersessions = []

    for i in range(len(activated_memories)):
        for j in range(i + 1, len(activated_memories)):
            tup_a = activated_memories[i]
            tup_b = activated_memories[j]
            id_a = tup_a[0]
            id_b = tup_b[0]

            # Get vector similarity between the two memories
            sim = scorer.similarity(id_a, id_b)
            if sim is None or sim < similarity_threshold:
                continue

            # Get created_at from the 4th element (extended tuples)
            created_a = tup_a[3] if len(tup_a) > 3 else 0
            created_b = tup_b[3] if len(tup_b) > 3 else 0

            age_gap = abs(created_a - created_b)
            if age_gap < age_gap_threshold:
                continue

            # Determine which is newer
            if created_a > created_b:
                newer_id, older_id = id_a, id_b
                newer_date = mapper.format_date(created_a)
                older_date = mapper.format_date(created_b)
            else:
                newer_id, older_id = id_b, id_a
                newer_date = mapper.format_date(created_b)
                older_date = mapper.format_date(created_a)

            supersessions.append({
                "newer_id": newer_id,
                "older_id": older_id,
                "similarity": round(sim, 3),
                "age_gap_queries": age_gap,
                "note": (
                    f"Note: Memory '{newer_id}' (from ~{newer_date}) may supersede "
                    f"'{older_id}' (from ~{older_date}) on overlapping subject matter. "
                    f"The newer memory is more likely to reflect current state."
                ),
            })

    return supersessions


def get_lineage(memory_id, memories_dict):
    """
    Walk the full ancestor and descendant chain for a memory.
    Returns {ancestors: [...], descendants: [...]}.
    """
    ancestors = []
    descendants = []

    # Walk up: parent_ids
    visited = set()
    queue = [memory_id]
    while queue:
        current = queue.pop(0)
        if current in visited or current == memory_id:
            visited.add(current)
            mem = memories_dict.get(current)
            if mem and current != memory_id:
                ancestors.append(current)
            if mem:
                for pid in mem.parent_ids:
                    if pid not in visited:
                        queue.append(pid)
            continue
        visited.add(current)
        mem = memories_dict.get(current)
        if mem:
            ancestors.append(current)
            for pid in mem.parent_ids:
                if pid not in visited:
                    queue.append(pid)

    # Walk down: child_ids
    visited = set()
    queue = [memory_id]
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        mem = memories_dict.get(current)
        if mem and current != memory_id:
            descendants.append(current)
        if mem:
            for cid in mem.child_ids:
                if cid not in visited:
                    queue.append(cid)

    return {"ancestors": ancestors, "descendants": descendants}


def get_generation(memory_id, memories_dict):
    """
    How many synthesis events separate this memory from its oldest ancestor.
    Generation 0 = original extraction (no parents).
    Generation 1 = synthesized from generation-0 memories.
    """
    mem = memories_dict.get(memory_id)
    if not mem or not mem.parent_ids:
        return 0

    max_parent_gen = 0
    for pid in mem.parent_ids:
        parent_gen = get_generation(pid, memories_dict)
        max_parent_gen = max(max_parent_gen, parent_gen)

    return max_parent_gen + 1

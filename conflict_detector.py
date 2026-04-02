"""
Dryft — Conflict Detector

Identifies genuine contradictions between activated memories. Runs after
scoring, before context injection. Uses Haiku to classify activated memory
sets for pairwise conflicts across four categories:

  identity:  Two memories disagree about what tool/platform/system is used
  temporal:  Two memories disagree about when something happened or current timing
  factual:   Two memories disagree about a measurable fact (count, price, name)
  status:    Two memories disagree about the current state of something

Only fires when 2+ herd memories activate. Confidence >= 0.70 proceeds.
Cross-herd awareness: evaluation memories are labeled as evaluative context,
not operational claims.
"""

import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import anthropic


CONFIDENCE_THRESHOLD = 0.70


def build_detection_prompt(memories: list[dict]) -> str:
    """Build the conflict detection prompt for Haiku."""
    memory_block = []
    for m in memories:
        herd_label = m.get("herd", "main")
        label = "[EVALUATION CONTEXT]" if herd_label == "evaluation" else "[OPERATIONAL]"
        created_str = m.get("created_date", "")
        temporal_line = f"  Created: ~{created_str}\n" if created_str else ""
        memory_block.append(
            f"  ID: {m['id']}\n"
            f"  Herd: {label}\n"
            f"{temporal_line}"
            f"  Content: {m['content']}"
        )
    memories_text = "\n\n".join(memory_block)

    return f"""You are a conflict detector for a living memory system. You are given a set of memories that all activated for the same query. Your job is to find GENUINE CONTRADICTIONS: two memories that make claims about the same subject that cannot both be true simultaneously.

CONFLICT CATEGORIES:
- identity: Two memories disagree about what tool, platform, or system is used for a task
- temporal: Two memories disagree about when something happened or current timing/schedule
- factual: Two memories disagree about a measurable fact (count, price, percentage, name)
- status: Two memories disagree about the current state of something (active vs cancelled, etc.)

WHAT IS NOT A CONFLICT:
- Different details about the same topic that complement each other
- A general memory and a more specific memory about the same topic (refinement)
- Two episodic memories from different time periods that are both historically accurate
- Memories that discuss different aspects of the same subject
- A memory stating a fact and another memory providing more detail about the same fact

CROSS-HERD AWARENESS:
- Memories labeled [OPERATIONAL] are declarative facts about the user's current operations
- Memories labeled [EVALUATION CONTEXT] are knowledge from systems the user evaluated but may not currently use
- An evaluation memory saying "Shopify costs $79/month" does NOT conflict with an operational memory saying "Local Line costs $2700/year" because they describe different systems
- A genuine cross-herd conflict would be: operational says "I use Local Line" and evaluation says "I switched to Shopify"

TEMPORAL AWARENESS:
- Each memory includes an approximate creation date. When two memories conflict and one was created significantly later than the other, this is likely a temporal conflict: the newer memory supersedes the older one.
- Flag temporal conflicts with the category "temporal" and include the approximate dates of both memories in the description.
- The newer memory should generally be treated as more current, but both could contain historically accurate information.

Analyze the following activated memories for pairwise conflicts. For each genuine conflict found, return the pair of memory IDs, the subject they disagree about, a brief description, the category, and your confidence (0.0 to 1.0).

Be conservative. False positives are worse than missed conflicts. If you are not sure, do not report it.

Return ONLY valid JSON in this format:
{{
  "conflicts": [
    {{
      "memory_a_id": "id-of-first-memory",
      "memory_b_id": "id-of-second-memory",
      "subject": "what they disagree about",
      "description": "brief description of the contradiction",
      "category": "identity|temporal|factual|status",
      "confidence": 0.85
    }}
  ]
}}

If no conflicts are found, return: {{"conflicts": []}}

MEMORIES:

{memories_text}"""


class ConflictDetector:
    """Detects genuine contradictions in activated memory sets."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.client = anthropic.Anthropic()
        self.model = model

    def detect(
        self,
        activated_main: list[tuple[str, str, float]],
        activated_eval: list[tuple[str, str, float]] | None = None,
        resolved_pairs: set[tuple[str, str]] | None = None,
        temporal_dates: dict[str, str] | None = None,
    ) -> list[dict]:
        """
        Detect conflicts in the activated memory set.

        Args:
            activated_main: [(memory_id, content, score)] from main herd
            activated_eval: [(memory_id, content, score)] from eval herd (if gate open)
            resolved_pairs: set of (id_a, id_b) tuples to skip (already resolved)
            temporal_dates: {memory_id: "Feb 20, 2026"} approximate creation dates

        Returns:
            List of conflict dicts with confidence >= CONFIDENCE_THRESHOLD.
            Each has: memory_a_id, memory_b_id, subject, description,
                      category, confidence, memory_a_herd, memory_b_herd
        """
        dates = temporal_dates or {}

        # Build memory list with herd labels
        memories = []
        for mem_id, content, score in activated_main:
            memories.append({
                "id": mem_id,
                "content": content,
                "herd": "main",
                "score": score,
                "created_date": dates.get(mem_id, ""),
            })
        if activated_eval:
            for mem_id, content, score in activated_eval:
                memories.append({
                    "id": mem_id,
                    "content": content,
                    "herd": "evaluation",
                    "score": score,
                    "created_date": dates.get(mem_id, ""),
                })

        # Need at least 2 memories to have a conflict
        if len(memories) < 2:
            return []

        prompt = build_detection_prompt(memories)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()

            # Handle markdown wrapping
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            # Handle extra text after JSON (Haiku sometimes appends explanation)
            # Find the outermost JSON object
            brace_depth = 0
            json_end = -1
            for i, ch in enumerate(text):
                if ch == '{':
                    brace_depth += 1
                elif ch == '}':
                    brace_depth -= 1
                    if brace_depth == 0:
                        json_end = i + 1
                        break
            if json_end > 0:
                text = text[:json_end]

            result = json.loads(text)
            raw_conflicts = result.get("conflicts", [])
        except (json.JSONDecodeError, anthropic.APIError, IndexError) as e:
            print(f"ConflictDetector error: {e}")
            return []

        # Filter by confidence threshold and resolved pairs
        resolved = resolved_pairs or set()
        memory_herd_map = {m["id"]: m["herd"] for m in memories}
        filtered = []

        for c in raw_conflicts:
            conf = c.get("confidence", 0.0)
            if conf < CONFIDENCE_THRESHOLD:
                continue

            a_id = c.get("memory_a_id", "")
            b_id = c.get("memory_b_id", "")

            # Skip resolved pairs (check both orderings)
            pair = tuple(sorted([a_id, b_id]))
            if pair in resolved:
                continue

            # Verify both IDs exist in our activated set
            if a_id not in memory_herd_map or b_id not in memory_herd_map:
                continue

            filtered.append({
                "memory_a_id": a_id,
                "memory_b_id": b_id,
                "memory_a_herd": memory_herd_map.get(a_id, "main"),
                "memory_b_herd": memory_herd_map.get(b_id, "main"),
                "subject": c.get("subject", ""),
                "description": c.get("description", ""),
                "category": c.get("category", "factual"),
                "confidence": conf,
            })

        return filtered

"""
Dryft — Foundational Memory Store (Cowbird Layer)

Memories that are load-bearing context for almost every interaction.
Always injected into model calls. No fitness, no decay, no predator.
Not scored against the herd. They are the ground truth about who
this person is.

Test: would losing this memory make the system materially worse at
understanding the user across three or more distinct topic areas?
"""

import json
from pathlib import Path


class FoundationalMemory:
    """A single foundational memory. Always on, never competes."""

    def __init__(self, data: dict):
        self.id = data["id"]
        self.content = data["content"]
        self.category = data.get("category", "identity")  # identity | context | background
        self.created_at = data.get("created_at", None)
        self.source = data.get("source", "bootstrap")  # bootstrap | graduated | manual

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "created_at": self.created_at,
            "source": self.source,
        }


class FoundationalStore:
    """
    The cowbird layer. Foundational memories attend to the herd without
    competing in it. Always injected, never scored, never culled.
    """

    def __init__(self, path: str = "foundational.json"):
        self.path = Path(path)
        self.memories: dict[str, FoundationalMemory] = {}
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            mem = FoundationalMemory(item)
            self.memories[mem.id] = mem

    def save(self):
        data = [m.to_dict() for m in self.memories.values()]
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add(self, memory_id: str, content: str,
            category: str = "identity", source: str = "graduated") -> FoundationalMemory:
        """Add a new foundational memory."""
        mem = FoundationalMemory({
            "id": memory_id,
            "content": content,
            "category": category,
            "source": source,
        })
        self.memories[mem.id] = mem
        self.save()
        return mem

    def remove(self, memory_id: str) -> bool:
        """Remove a foundational memory. Returns True if found and removed."""
        if memory_id in self.memories:
            del self.memories[memory_id]
            self.save()
            return True
        return False

    def get_all_content(self) -> list[str]:
        """Return all foundational memory contents for context injection."""
        return [m.content for m in self.memories.values()]

    def format_for_injection(self) -> str:
        """Format all foundational memories as a context block for injection."""
        if not self.memories:
            return ""
        lines = ["[Foundational Context — always active]"]
        for mem in self.memories.values():
            lines.append(f"- {mem.content}")
        return "\n".join(lines)

"""
Dryft — Phase 6/7: Vector Scorer

Replaces keyword-overlap scoring with cosine similarity on dense embeddings.
Pre-computed embeddings are loaded from a cache file (produced by embed_herd.py).

Phase 7 addition: runtime embedding for live queries not in cache.
Uses sentence-transformers locally (10-30ms per query).
"""

import json
import math
from pathlib import Path


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity. No numpy required."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorScorer:
    """
    Scores query-to-memory relevance using pre-computed embeddings.
    Loaded from a cache file keyed by memory ID (for memories) and
    by exact query text (for queries).

    Cache structure:
    {
        "model": "all-MiniLM-L6-v2",
        "memories": {"memory-id": [float, ...], ...},
        "queries": {"query text": [float, ...], ...}
    }
    """

    def __init__(self, cache_path: str = "embeddings_cache.json",
                 live_embed: bool = False):
        self.cache_path = Path(cache_path)
        self.memory_embeddings: dict[str, list[float]] = {}
        self.query_embeddings: dict[str, list[float]] = {}
        self.model_name: str = ""
        self._loaded = False
        self._embed_model = None  # lazy-loaded for live embedding
        self._live_embed = live_embed
        self._load()

    def _load(self):
        if not self.cache_path.exists():
            return
        with open(self.cache_path) as f:
            data = json.load(f)
        self.model_name = data.get("model", "unknown")
        self.memory_embeddings = data.get("memories", {})
        self.query_embeddings = data.get("queries", {})
        self._loaded = True

    @property
    def available(self) -> bool:
        return self._loaded and len(self.memory_embeddings) > 0

    def _get_embed_model(self):
        """Lazy-load the embedding model for live queries."""
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            model_name = self.model_name if self.model_name else "all-MiniLM-L6-v2"
            self._embed_model = SentenceTransformer(model_name)
        return self._embed_model

    def embed_query(self, query_text: str) -> list[float]:
        """Embed a query at runtime and cache the result."""
        if query_text in self.query_embeddings:
            return self.query_embeddings[query_text]
        model = self._get_embed_model()
        vec = model.encode([query_text], show_progress_bar=False)[0]
        embedding = [round(float(v), 6) for v in vec]
        self.query_embeddings[query_text] = embedding
        return embedding

    def score(self, memory_id: str, query_text: str) -> float:
        """
        Return cosine similarity between a memory's embedding and
        a query's embedding. If live_embed is enabled, embeds unknown
        queries on the fly. Returns 0.0 if memory is missing from cache.
        """
        mem_vec = self.memory_embeddings.get(memory_id)
        if mem_vec is None:
            return 0.0
        q_vec = self.query_embeddings.get(query_text)
        if q_vec is None:
            if self._live_embed:
                q_vec = self.embed_query(query_text)
            else:
                return 0.0
        return max(cosine_similarity(mem_vec, q_vec), 0.0)

    def similarity(self, memory_id_a: str, memory_id_b: str) -> float | None:
        """Cosine similarity between two memory embeddings. Returns None if either is missing."""
        vec_a = self.memory_embeddings.get(memory_id_a)
        vec_b = self.memory_embeddings.get(memory_id_b)
        if vec_a is None or vec_b is None:
            return None
        return cosine_similarity(vec_a, vec_b)

    def add_memory_embedding(self, memory_id: str, embedding: list[float]):
        """Add or update a memory embedding at runtime (for injected memories)."""
        self.memory_embeddings[memory_id] = embedding

    def remove_memory_embedding(self, memory_id: str):
        """Remove a memory embedding (e.g., on user-directed cull)."""
        self.memory_embeddings.pop(memory_id, None)

    def save(self, path: str | None = None):
        """Persist embeddings cache to disk."""
        save_path = Path(path) if path else self.cache_path
        data = {
            "model": self.model_name,
            "memories": self.memory_embeddings,
            "queries": {},  # don't persist query cache, it grows unbounded
        }
        with open(save_path, "w") as f:
            json.dump(data, f)

    def stats(self) -> dict:
        return {
            "model": self.model_name,
            "memories_cached": len(self.memory_embeddings),
            "queries_cached": len(self.query_embeddings),
            "available": self.available,
        }

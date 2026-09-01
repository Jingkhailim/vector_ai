"""
vector_db.py
============
A minimal, dependency-light vector database built from scratch so you can
see exactly what a "real" vector DB (Pinecone, Weaviate, Chroma, FAISS...)
is doing conceptually under the hood.

A vector database does three jobs:
  1. STORE vectors (embeddings) + the original content + metadata
  2. SEARCH by similarity (usually cosine similarity or dot product)
  3. RETURN the top-k closest matches to a query vector

Everything below is "brute-force" (compares the query to every stored
vector). That's O(n) per search - totally fine for thousands of vectors,
and exactly how you'd prototype before reaching for an Approximate
Nearest Neighbor (ANN) index like HNSW or IVF that real vector DBs use
to stay fast at millions/billions of vectors.
"""

import json
import pickle
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Record:
    id: str
    vector: np.ndarray
    text: str
    metadata: dict = field(default_factory=dict)


class SimpleVectorDB:
    def __init__(self):
        self._records: dict[str, Record] = {}

    # ---------- CORE OPERATIONS ----------

    def add(self, id: str, vector: np.ndarray, text: str, metadata: Optional[dict] = None):
        """Insert (or overwrite) a vector + its source text + metadata."""
        vector = np.asarray(vector, dtype=np.float32)
        self._records[id] = Record(id=id, vector=vector, text=text, metadata=metadata or {})

    def add_batch(self, ids, vectors, texts, metadatas=None):
        metadatas = metadatas or [{}] * len(ids)
        for id, vec, text, meta in zip(ids, vectors, texts, metadatas):
            self.add(id, vec, text, meta)

    def delete(self, id: str):
        self._records.pop(id, None)

    def __len__(self):
        return len(self._records)

    # ---------- SIMILARITY MATH ----------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        # cosine similarity = dot(a, b) / (||a|| * ||b||)
        # Ranges from -1 (opposite) to 1 (identical direction).
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    # ---------- SEARCH ----------

    def search(self, query_vector: np.ndarray, top_k: int = 5, filter_fn=None):
        """
        Compare query_vector against every stored vector, return the
        top_k most similar records.

        filter_fn: optional function(metadata) -> bool, to pre-filter
                   records before scoring (like a metadata "WHERE" clause
                   in real vector DBs).
        """
        query_vector = np.asarray(query_vector, dtype=np.float32)
        scored = []

        for record in self._records.values():
            if filter_fn and not filter_fn(record.metadata):
                continue
            score = self._cosine_similarity(query_vector, record.vector)
            scored.append((score, record))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, record in scored[:top_k]:
            results.append({
                "id": record.id,
                "score": round(score, 4),
                "text": record.text,
                "metadata": record.metadata,
            })
        return results

    # ---------- PERSISTENCE ----------

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self._records, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self._records = pickle.load(f)

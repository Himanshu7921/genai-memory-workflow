from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

from sentence_transformers import SentenceTransformer

from app.retrieval.embeddings import EmbeddingProvider


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


class HFEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.model = _load_model()

    def embed_text(self, text: str) -> List[float]:
        vectors = self.embed_texts([text])
        return vectors[0] if vectors else []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(texts)
        arr = np.array(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        # Keep vectors on the unit sphere for cosine similarity consistency.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized = arr / norms
        return normalized.tolist()
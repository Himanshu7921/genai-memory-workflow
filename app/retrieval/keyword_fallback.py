from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.models.memory import CorpusChunkRecord


@dataclass(slots=True)
class KeywordSearchResult:
    chunk: CorpusChunkRecord
    score: float
    reason: str = "keyword"
    metadata: dict[str, Any] = field(default_factory=dict)


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token}


def keyword_search(
    query: str,
    chunks: list[CorpusChunkRecord],
    *,
    top_k: int = 5,
    min_score: float = 0.0,
) -> list[KeywordSearchResult]:
    query_tokens = _tokenize(query)
    if not query_tokens or not chunks:
        return []

    results: list[KeywordSearchResult] = []
    for chunk in chunks:
        chunk_tokens = _tokenize(chunk.content)
        if not chunk_tokens:
            continue
        overlap = query_tokens & chunk_tokens
        if not overlap:
            continue
        coverage = len(overlap) / max(len(query_tokens), 1)
        density = len(overlap) / max(len(chunk_tokens), 1)
        score = round((coverage * 0.7) + (density * 0.3), 4)
        if score < min_score:
            continue
        results.append(
            KeywordSearchResult(
                chunk=chunk,
                score=score,
                metadata={"overlap_tokens": sorted(overlap)},
            )
        )

    results.sort(key=lambda item: (item.score, len(item.chunk.content)), reverse=True)
    return results[:top_k]

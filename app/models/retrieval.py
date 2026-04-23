from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievalChunk:
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    score: float = 0.0
    source: str = "vector"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalRequest:
    query: str
    document_ids: list[str] = field(default_factory=list)
    user_id: str | None = None
    top_k: int = 5
    min_score: float = 0.2
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalContext:
    query: str
    chunks: list[RetrievalChunk] = field(default_factory=list)
    used_keyword_fallback: bool = False
    empty: bool = False


@dataclass(slots=True)
class RetrievalAudit:
    requested_top_k: int
    returned_chunks: int
    vector_hits: int
    keyword_hits: int
    fallback_used: bool
    filtered_out: int = 0


@dataclass(slots=True)
class RetrievalResult:
    context: RetrievalContext
    audit: RetrievalAudit

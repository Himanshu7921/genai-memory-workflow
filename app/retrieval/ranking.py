from __future__ import annotations

from collections.abc import Iterable

from app.models.memory import CorpusChunkRecord
from app.models.retrieval import RetrievalChunk


def rerank_chunks(
    chunks: Iterable[RetrievalChunk],
    *,
    query: str,
    min_score: float = 0.2,
    max_chunks: int = 5,
) -> list[RetrievalChunk]:
    query_terms = {part for part in query.lower().split() if part}
    scored: list[RetrievalChunk] = []
    for chunk in chunks:
        if chunk.score < min_score:
            continue
        bonus = 0.0
        if query_terms:
            content_terms = {part for part in chunk.content.lower().split() if part}
            coverage = len(query_terms & content_terms) / max(len(query_terms), 1)
            bonus = min(0.2, coverage * 0.2)
        chunk.score = round(min(1.0, chunk.score + bonus), 4)
        scored.append(chunk)

    scored.sort(key=lambda item: (item.score, len(item.content)), reverse=True)
    deduped: list[RetrievalChunk] = []
    seen: set[str] = set()
    for chunk in scored:
        dedupe_key = chunk.chunk_id or f"{chunk.document_id}:{chunk.chunk_index}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(chunk)
        if len(deduped) >= max_chunks:
            break
    return deduped


def to_retrieval_chunks(chunks: Iterable[CorpusChunkRecord], *, source: str = "vector") -> list[RetrievalChunk]:
    return [
        RetrievalChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            content=chunk.content,
            chunk_index=chunk.chunk_index,
            score=0.0,
            source=source,
            metadata={**chunk.metadata, "status": chunk.status},
        )
        for chunk in chunks
    ]

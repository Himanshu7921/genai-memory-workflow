from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any

from app.core.config import DEFAULT_MEMORY_CONFIG
from app.models.memory import CorpusChunkRecord
from app.models.retrieval import RetrievalAudit, RetrievalChunk, RetrievalContext, RetrievalRequest, RetrievalResult
from app.memory.corpus import CorpusMemoryService
from app.retrieval.chunking import ChunkPlan, chunk_document_content
from app.retrieval.keyword_fallback import keyword_search
from app.retrieval.ranking import rerank_chunks, to_retrieval_chunks
from app.retrieval.vector_store import ChromaVectorStore, InMemoryVectorStore, VectorStore


@dataclass(slots=True)
class RetrievalIndexResult:
    document_id: str
    chunk_count: int
    content_hash: str


class RetrievalService:
    def __init__(
        self,
        corpus: CorpusMemoryService,
        vector_store: VectorStore | None = None,
        *,
        chunk_plan: ChunkPlan | None = None,
    ) -> None:
        self.corpus = corpus
        self.vector_store = vector_store or self._build_default_store()
        self.chunk_plan = chunk_plan or ChunkPlan()

    def _build_default_store(self) -> VectorStore:
        chroma = ChromaVectorStore()
        if chroma._load_collection() is not None:
            return chroma
        return InMemoryVectorStore()

    def index_document(
        self,
        *,
        document_id: str,
        content: str,
        user_id: str | None = None,
        source_uri: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RetrievalIndexResult:
        content_hash = sha256(content.encode("utf-8")).hexdigest()
        self.corpus.upsert_document(
            document_id=document_id,
            user_id=user_id,
            source_uri=source_uri,
            title=title,
            content_hash=content_hash,
            metadata=metadata or {},
        )

        chunked = chunk_document_content(content, plan=self.chunk_plan)
        chunk_records: list[CorpusChunkRecord] = []
        for chunk_index, chunk_text in chunked:
            chunk = CorpusChunkRecord(
                chunk_id=f"{document_id}_chunk_{chunk_index}",
                document_id=document_id,
                user_id=user_id,
                chunk_index=chunk_index,
                content=chunk_text,
                content_hash=sha256(chunk_text.encode("utf-8")).hexdigest(),
                metadata={
                    "source_uri": source_uri,
                    "title": title,
                    "indexed_at": datetime.utcnow().isoformat(),
                    **(metadata or {}),
                },
            )
            self.corpus.upsert_chunk(chunk)
            chunk_records.append(chunk)

        self.vector_store.upsert_chunks(chunk_records)
        return RetrievalIndexResult(document_id=document_id, chunk_count=len(chunk_records), content_hash=content_hash)

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        vector_hits = self.vector_store.search(
            request.query,
            top_k=request.top_k,
            document_ids=request.document_ids or None,
            user_id=request.user_id,
            filters=request.filters or None,
        )

        corpus_chunks = (
            self.corpus.list_chunks_for_documents(request.document_ids)
            if request.document_ids
            else self.corpus.repository.list_all_chunks(user_id=request.user_id)
        )
        keyword_hits = keyword_search(request.query, corpus_chunks, top_k=request.top_k, min_score=request.min_score)

        retrieval_chunks = [
            RetrievalChunk(
                chunk_id=hit.chunk.chunk_id,
                document_id=hit.chunk.document_id,
                content=hit.chunk.content,
                chunk_index=hit.chunk.chunk_index,
                score=hit.score,
                source=hit.source,
                metadata=hit.metadata or {},
            )
            for hit in vector_hits
        ]

        retrieval_chunks.extend(
            RetrievalChunk(
                chunk_id=hit.chunk.chunk_id,
                document_id=hit.chunk.document_id,
                content=hit.chunk.content,
                chunk_index=hit.chunk.chunk_index,
                score=hit.score,
                source=hit.reason,
                metadata=hit.metadata,
            )
            for hit in keyword_hits
        )

        initial_count = len(retrieval_chunks)
        reranked = rerank_chunks(
            retrieval_chunks,
            query=request.query,
            min_score=request.min_score,
            max_chunks=request.top_k,
        )

        # Final safety gate: treat sub-threshold results as unusable grounded context.
        effective_min_score = max(request.min_score, 0.5)
        grounded = [chunk for chunk in reranked if chunk.score >= effective_min_score]
        filtered_out = max(initial_count - len(grounded), 0)

        used_fallback = (not vector_hits and bool(keyword_hits)) or not grounded
        context = RetrievalContext(
            query=request.query,
            chunks=grounded,
            used_keyword_fallback=used_fallback,
            empty=not grounded,
        )
        audit = RetrievalAudit(
            requested_top_k=request.top_k,
            returned_chunks=len(grounded),
            vector_hits=len(vector_hits),
            keyword_hits=len(keyword_hits),
            fallback_used=used_fallback,
            filtered_out=filtered_out,
        )
        return RetrievalResult(context=context, audit=audit)

    def retrieve_safe_context(self, request: RetrievalRequest) -> dict[str, Any]:
        result = self.retrieve(request)
        if result.context.empty:
            return {
                "query": request.query,
                "chunks": [],
                "instruction": "No relevant source chunks were found. Answer only from conversation state or ask for more detail.",
                "audit": result.audit,
            }
        return {
            "query": request.query,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "score": chunk.score,
                    "source": chunk.source,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                }
                for chunk in result.context.chunks
            ],
            "audit": result.audit,
        }

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from app.models.memory import CorpusChunkRecord
from app.retrieval.embeddings import EmbeddingProvider, cosine_similarity
from app.retrieval.embeddings_hf import HFEmbeddingProvider


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VectorSearchHit:
    chunk: CorpusChunkRecord
    score: float
    source: str = "vector"
    metadata: dict[str, Any] = None


class VectorStore(Protocol):
    def upsert_chunks(self, chunks: list[CorpusChunkRecord]) -> None: ...

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_ids: list[str] | None = None,
        user_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchHit]: ...


class InMemoryVectorStore:
    def __init__(self, embedding_provider: EmbeddingProvider | None = None) -> None:
        self.embedding_provider = embedding_provider or HFEmbeddingProvider()
        self._records: list[tuple[CorpusChunkRecord, list[float]]] = []

    def upsert_chunks(self, chunks: list[CorpusChunkRecord]) -> None:
        if not chunks:
            return
        embeddings = self.embedding_provider.embed_texts([chunk.content for chunk in chunks])
        for chunk, embedding in zip(chunks, embeddings):
            self._records = [record for record in self._records if record[0].chunk_id != chunk.chunk_id]
            self._records.append((chunk, embedding))

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_ids: list[str] | None = None,
        user_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchHit]:
        query_embedding = self.embedding_provider.embed_text(query)
        logger.info(
            "vector_query_embedding_shape store=in_memory dims=%s top_k=%s",
            len(query_embedding),
            top_k,
        )
        candidates: list[VectorSearchHit] = []
        for chunk, embedding in self._records:
            if document_ids and chunk.document_id not in document_ids:
                continue
            if user_id and chunk.user_id not in (None, user_id):
                continue
            if filters:
                matched = True
                for key, value in filters.items():
                    if chunk.metadata.get(key) != value:
                        matched = False
                        break
                if not matched:
                    continue
            raw_score = cosine_similarity(query_embedding, embedding)
            score = max(0.0, min(1.0, (raw_score + 1.0) / 2.0))
            candidates.append(
                VectorSearchHit(
                    chunk=chunk,
                    score=round(score, 4),
                    metadata={"store": "in_memory"},
                )
            )
        candidates.sort(key=lambda item: (item.score, len(item.chunk.content)), reverse=True)
        logger.info(
            "vector_candidates store=in_memory count=%s top_scores=%s",
            len(candidates),
            [hit.score for hit in candidates[: min(5, len(candidates))]],
        )
        return candidates[:top_k]


class ChromaVectorStore:
    def __init__(self, *, persist_directory: str = "data/chroma", collection_name: str = "corpus", embedding_provider: EmbeddingProvider | None = None) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider or HFEmbeddingProvider()
        self._collection = None

    def _load_collection(self):
        if self._collection is not None:
            return self._collection
        try:
            import chromadb
        except Exception:
            return None

        client = chromadb.PersistentClient(path=self.persist_directory)
        self._collection = client.get_or_create_collection(
            self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def upsert_chunks(self, chunks: list[CorpusChunkRecord]) -> None:
        collection = self._load_collection()
        if collection is None:
            return
        if not chunks:
            return
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        embeddings = self.embedding_provider.embed_texts(documents)
        metadatas = [
            {
                "document_id": chunk.document_id,
                "user_id": chunk.user_id,
                "chunk_index": chunk.chunk_index,
                **chunk.metadata,
            }
            for chunk in chunks
        ]
        collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_ids: list[str] | None = None,
        user_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchHit]:
        collection = self._load_collection()
        if collection is None:
            return []

        query_embedding = self.embedding_provider.embed_text(query)
        logger.info(
            "vector_query_embedding_shape store=chroma dims=%s top_k=%s",
            len(query_embedding),
            top_k,
        )
        where: dict[str, Any] = {}
        if document_ids:
            where["document_id"] = {"$in": document_ids}
        if filters:
            where.update(filters)

        try:
            response = collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where or None)
        except Exception:
            return []

        ids = (response.get("ids") or [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]

        hits: list[VectorSearchHit] = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            metadata_user_id = metadata.get("user_id")
            if user_id and metadata_user_id not in (None, "", user_id):
                continue
            chunk = CorpusChunkRecord(
                chunk_id=chunk_id,
                document_id=metadata.get("document_id", ""),
                user_id=metadata_user_id,
                chunk_index=int(metadata.get("chunk_index", 0)),
                content=documents[index] if index < len(documents) else "",
                content_hash=metadata.get("content_hash", ""),
                metadata={k: v for k, v in metadata.items() if k not in {"document_id", "user_id", "chunk_index", "content_hash"}},
            )
            distance = distances[index] if index < len(distances) else 1.0
            score = round(1.0 / (1.0 + (0.75 * float(distance))), 4)
            hits.append(VectorSearchHit(chunk=chunk, score=score, source="chroma", metadata={"distance": distance}))
        logger.info(
            "vector_candidates store=chroma count=%s top_scores=%s",
            len(hits),
            [hit.score for hit in hits[: min(5, len(hits))]],
        )
        return hits[:top_k]

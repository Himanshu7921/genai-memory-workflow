from __future__ import annotations

from app.models.memory import CorpusChunkRecord
from app.storage.memory_repo import MemoryRepository


class CorpusMemoryService:
    def __init__(self, repository: MemoryRepository) -> None:
        self.repository = repository

    def upsert_document(
        self,
        *,
        document_id: str,
        user_id: str | None,
        source_uri: str | None,
        title: str | None,
        content_hash: str,
        metadata: dict | None = None,
    ) -> None:
        self.repository.upsert_document(
            document_id=document_id,
            user_id=user_id,
            source_uri=source_uri,
            title=title,
            content_hash=content_hash,
            metadata=metadata or {},
        )

    def upsert_chunk(self, chunk: CorpusChunkRecord) -> None:
        self.repository.upsert_chunk(chunk)

    def list_chunks_for_documents(self, document_ids: list[str]) -> list[CorpusChunkRecord]:
        return self.repository.list_chunks_for_documents(document_ids)

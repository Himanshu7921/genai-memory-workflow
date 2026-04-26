from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    document_id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    user_id: str | None = None


class IngestResponse(BaseModel):
    document_id: str
    chunk_count: int
    content_hash: str

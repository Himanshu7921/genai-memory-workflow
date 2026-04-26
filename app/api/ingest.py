from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_retrieval_service
from app.models.ingest import IngestRequest, IngestResponse
from app.retrieval.service import RetrievalService


router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    payload: IngestRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> IngestResponse:
    result = retrieval_service.index_document(
        document_id=payload.document_id,
        content=payload.content,
        user_id=payload.user_id,
        source_uri="api:/ingest",
        title=payload.document_id,
        metadata={"ingested_from": "api"},
    )
    return IngestResponse(
        document_id=result.document_id,
        chunk_count=result.chunk_count,
        content_hash=result.content_hash,
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.models.retrieval import RetrievalRequest
from app.retrieval.service import RetrievalService
from app.tools.base import ToolContext, ToolExecutionError


@dataclass(slots=True)
class RetrievalTool:
    name: str = "retrieval_tool"
    retrieval_service: RetrievalService | None = None

    def run(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        if self.retrieval_service is None:
            raise ToolExecutionError(self.name, "retrieval service unavailable", retryable=True)

        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ToolExecutionError(self.name, "missing query", retryable=False)

        document_ids = arguments.get("document_ids") or []
        top_k = int(arguments.get("top_k", 5))
        min_score = float(arguments.get("min_score", 0.5))

        result = self.retrieval_service.retrieve(
            RetrievalRequest(
                query=query,
                document_ids=list(document_ids),
                user_id=context.user_id,
                top_k=top_k,
                min_score=min_score,
                filters=dict(arguments.get("filters") or {}),
            )
        )
        return {
            "query": query,
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

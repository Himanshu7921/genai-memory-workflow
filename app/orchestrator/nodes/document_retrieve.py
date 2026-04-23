from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from app.models.orchestration import OrchestrationState
from app.models.retrieval import RetrievalRequest
from app.retrieval.service import RetrievalService
from app.orchestrator.nodes.base import NodeResult


@dataclass(slots=True)
class DocumentRetrievalNode:
    name: str = "document_retrieval"
    retrieval_service: RetrievalService | None = None

    def run(self, state: OrchestrationState) -> NodeResult:
        if self.retrieval_service is None:
            state.errors.append("retrieval_service_unavailable")
            state.retrieval_context = None
            return NodeResult(state=state)

        if state.intent and state.intent.intent not in {"grounded_qa", "general_qa", "calculation"}:
            state.retrieval_context = None
            return NodeResult(state=state)

        start = perf_counter()
        request = RetrievalRequest(
            query=state.turn.message,
            user_id=state.turn.user_id,
            document_ids=state.turn.document_ids,
            top_k=5,
            min_score=0.5,
        )
        result = self.retrieval_service.retrieve(request)
        state.retrieval_context = result.context
        state.retrieval_audit = result.audit
        state.metadata["document_retrieval_duration_ms"] = int((perf_counter() - start) * 1000)
        return NodeResult(state=state)

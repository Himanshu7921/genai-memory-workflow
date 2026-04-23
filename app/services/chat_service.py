from __future__ import annotations

from dataclasses import asdict

from app.core.context import RequestContext
from app.models.chat import BudgetAuditResponse, ChatRequest, ChatResponse, SourceItem, TraceEventResponse
from app.orchestrator.graph import OrchestrationGraph


class ChatService:
    def __init__(self, graph: OrchestrationGraph) -> None:
        self.graph = graph

    def handle_chat(self, request: ChatRequest, context: RequestContext) -> ChatResponse:
        result = self.graph.run(
            user_id=request.user_id,
            session_id=request.session_id,
            message=request.message,
            document_ids=request.document_ids,
            trace_id=context.trace_id,
        )

        return ChatResponse(
            final_answer=result.answer,
            used_llm=result.used_llm,
            sources=[SourceItem(**source) for source in result.sources],
            budget_audit=BudgetAuditResponse(**asdict(result.budget_audit)) if result.budget_audit else None,
            request_id=context.request_id,
            trace_id=result.trace.trace_id,
            user_id=request.user_id,
            session_id=request.session_id,
            trace_events=[TraceEventResponse(**asdict(event)) for event in result.trace.events],
            warnings=list(result.trace.warnings),
            metadata=dict(result.metadata),
        )

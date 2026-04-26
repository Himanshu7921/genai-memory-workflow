from __future__ import annotations

from dataclasses import asdict
from typing import Any

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
        budget_audit_payload = self._build_budget_audit_payload(result.budget_audit, result.metadata)

        return ChatResponse(
            final_answer=result.answer,
            used_llm=result.used_llm,
            sources=[SourceItem(**source) for source in result.sources],
            budget_audit=BudgetAuditResponse(**budget_audit_payload) if budget_audit_payload else None,
            request_id=context.request_id,
            trace_id=result.trace.trace_id,
            user_id=request.user_id,
            session_id=request.session_id,
            trace_events=[TraceEventResponse(**asdict(event)) for event in result.trace.events],
            warnings=list(result.trace.warnings),
            metadata=dict(result.metadata),
        )

    def _build_budget_audit_payload(self, memory_budget_audit: Any, metadata: dict[str, Any]) -> dict[str, Any] | None:
        if memory_budget_audit is None and "context_budget_audit" not in metadata:
            return None

        payload: dict[str, Any]
        if memory_budget_audit is not None:
            payload = asdict(memory_budget_audit)
        else:
            payload = {
                "session_recent_turns_used": 0,
                "session_summary_chars": 0,
                "user_facts_used": 0,
                "pinned_facts_used": 0,
                "corpus_items_used": 0,
                "evicted_items": [],
            }

        context_budget = metadata.get("context_budget_audit")
        if not isinstance(context_budget, dict):
            return payload

        allocation_detail = context_budget.get("allocation") if isinstance(context_budget.get("allocation"), dict) else {}
        allocation_tokens: dict[str, int] = {}
        for key, value in allocation_detail.items():
            if isinstance(value, dict):
                tokens = int(value.get("tokens", 0))
            else:
                tokens = int(value or 0)
            allocation_tokens[key] = max(0, tokens)

        budget_total = int(context_budget.get("budget_total", 0))
        used_tokens = max(0, budget_total - allocation_tokens.get("unused", 0)) if budget_total else sum(allocation_tokens.values())
        remaining_tokens = max(0, budget_total - used_tokens) if budget_total else allocation_tokens.get("unused", 0)
        eviction_events = [str(item) for item in context_budget.get("eviction_events", []) if str(item).strip()]

        payload.update(
            {
                "model": context_budget.get("model"),
                "budget_total": budget_total or None,
                "total_tokens": budget_total or None,
                "used_tokens": used_tokens,
                "remaining_tokens": remaining_tokens,
                "allocation": allocation_tokens,
                "evictions": eviction_events,
                "eviction_events": eviction_events,
                "evicted_items": list(dict.fromkeys([*payload.get("evicted_items", []), *eviction_events])),
            }
        )
        return payload

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from app.memory.service import MemoryService
from app.models.orchestration import OrchestrationState
from app.orchestrator.nodes.base import NodeResult


@dataclass(slots=True)
class MemoryRetrievalNode:
    name: str = "memory_retrieval"
    memory_service: MemoryService | None = None

    def run(self, state: OrchestrationState) -> NodeResult:
        if self.memory_service is None:
            state.errors.append("memory_service_unavailable")
            return NodeResult(state=state)

        start = perf_counter()
        snapshot = self.memory_service.load(
            user_id=state.turn.user_id,
            session_id=state.turn.session_id,
            document_ids=state.turn.document_ids,
        )
        state.memory_snapshot = snapshot.snapshot
        state.memory_budget_audit = snapshot.audit
        state.turn.retrieved_memory = {
            "summary": snapshot.snapshot.summary.summary if snapshot.snapshot.summary else "",
            "pinned_facts": [
                {
                    "canonical_key": fact.canonical_key,
                    "value": fact.value,
                    "reason": fact.reason,
                }
                for fact in snapshot.snapshot.pinned_facts
            ],
            "user_facts": [
                {
                    "canonical_key": fact.canonical_key,
                    "value": fact.value,
                    "status": fact.status.value,
                }
                for fact in snapshot.snapshot.user_facts
            ],
        }
        state.metadata["memory_retrieval_duration_ms"] = int((perf_counter() - start) * 1000)
        return NodeResult(state=state)

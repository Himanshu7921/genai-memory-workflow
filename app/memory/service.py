from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.core.config import DEFAULT_MEMORY_CONFIG, MemoryConfig
from app.memory.corpus import CorpusMemoryService
from app.memory.session import SessionMemoryService
from app.memory.user import UserMemoryService, UserMemoryWriteResult
from app.models.memory import CorpusChunkRecord, MemoryBudgetAudit, MemorySnapshot, PinnedFact, SessionTurn, WorkingMemoryTurn
from app.storage.memory_repo import MemoryRepository


@dataclass(slots=True)
class MemoryWritePlan:
    write_session_summary: bool
    write_user_memory: bool
    reason: str
    fact_delta_count: int = 0
    pinned_changed: bool = False


@dataclass(slots=True)
class MemoryBuildResult:
    snapshot: MemorySnapshot
    audit: MemoryBudgetAudit


class MemoryService:
    def __init__(self, repository: MemoryRepository, config: MemoryConfig = DEFAULT_MEMORY_CONFIG) -> None:
        self.repository = repository
        self.config = config
        self.session = SessionMemoryService(repository, config)
        self.user = UserMemoryService(repository, config)
        self.corpus = CorpusMemoryService(repository)

    def capture_turn(self, *, turn: WorkingMemoryTurn, role: str = "user") -> None:
        self.session.append_turn(
            SessionTurn(
                turn_id=self.repository.new_id("turn"),
                user_id=turn.user_id,
                session_id=turn.session_id,
                role=role,
                content=turn.message,
                metadata={"document_ids": turn.document_ids, "trace_id": turn.trace_id},
            )
        )

    def load(self, *, user_id: str, session_id: str, document_ids: list[str] | None = None) -> MemoryBuildResult:
        snapshot = self.session.snapshot(user_id=user_id, session_id=session_id)
        corpus_chunks = self.corpus.list_chunks_for_documents(document_ids or [])
        total_turns = self.repository.count_session_turns(user_id=user_id, session_id=session_id)
        evicted_count = max(total_turns - len(snapshot.session_turns), 0)
        evicted_items = [f"session_turns_evicted:{evicted_count}"] if evicted_count else []
        audit = MemoryBudgetAudit(
            session_recent_turns_used=len(snapshot.session_turns),
            session_summary_chars=len(snapshot.summary.summary) if snapshot.summary else 0,
            user_facts_used=len(snapshot.user_facts),
            pinned_facts_used=len(snapshot.pinned_facts),
            corpus_items_used=len(corpus_chunks),
            evicted_items=evicted_items,
        )
        snapshot.corpus_chunks = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "user_id": chunk.user_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "content_hash": chunk.content_hash,
                "metadata": chunk.metadata,
                "status": chunk.status,
            }
            for chunk in corpus_chunks
        ]
        snapshot.budget_audit = audit
        return MemoryBuildResult(snapshot=snapshot, audit=audit)

    def plan_write_back(
        self,
        *,
        turns_since_write: int,
        last_write_at: datetime | None,
        fact_delta_count: int = 0,
        pinned_changed: bool = False,
    ) -> MemoryWritePlan:
        if last_write_at is None:
            return MemoryWritePlan(True, True, "initial_write", fact_delta_count, pinned_changed)
        elapsed = (datetime.utcnow() - last_write_at).total_seconds()
        if pinned_changed:
            return MemoryWritePlan(True, True, "pinned_changed", fact_delta_count, pinned_changed)
        if fact_delta_count >= 2:
            return MemoryWritePlan(True, True, "fact_delta_threshold", fact_delta_count, pinned_changed)
        if turns_since_write >= self.config.session_summary_trigger_turns or elapsed >= self.config.session_write_throttle_seconds:
            return MemoryWritePlan(True, True, "throttle_elapsed_or_turns", fact_delta_count, pinned_changed)
        return MemoryWritePlan(False, False, "throttled", fact_delta_count, pinned_changed)

    def write_session_memory(
        self,
        *,
        user_id: str,
        session_id: str,
        summary_text: str,
        pinned_facts: list[PinnedFact],
        turn_count_reset: bool = True,
    ) -> None:
        self.session.write_back_state(
            user_id=user_id,
            session_id=session_id,
            summary_text=summary_text,
            pinned_facts=pinned_facts,
            turn_count_reset=turn_count_reset,
        )

    def write_user_memory(self, *, user_id: str, incoming_facts: list, source: str) -> UserMemoryWriteResult:
        return self.user.write_facts(user_id=user_id, incoming=incoming_facts, source=source)

    def build_working_memory(self, *, user_id: str, session_id: str, message: str, document_ids: list[str] | None = None, trace_id: str | None = None) -> WorkingMemoryTurn:
        return WorkingMemoryTurn(
            user_id=user_id,
            session_id=session_id,
            message=message,
            document_ids=document_ids or [],
            trace_id=trace_id,
        )

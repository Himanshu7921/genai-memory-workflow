from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from app.core.config import DEFAULT_MEMORY_CONFIG, MemoryConfig
from app.models.memory import MemorySnapshot, PinnedFact, SessionSummary, SessionTurn
from app.storage.memory_repo import MemoryRepository


@dataclass(slots=True)
class SessionMemoryState:
    session_id: str
    user_id: str
    summary: SessionSummary | None
    recent_turns: list[SessionTurn]
    pinned_facts: list[PinnedFact]
    last_write_at: datetime | None
    turns_since_write: int


class SessionMemoryService:
    def __init__(self, repository: MemoryRepository, config: MemoryConfig = DEFAULT_MEMORY_CONFIG) -> None:
        self.repository = repository
        self.config = config

    def load_state(self, *, user_id: str, session_id: str) -> SessionMemoryState:
        summary = self.repository.get_session_summary(user_id=user_id, session_id=session_id)
        recent_turns = self.repository.list_recent_turns(
            user_id=user_id,
            session_id=session_id,
            limit=self.config.session_recent_turns,
        )
        pinned_facts = self.repository.list_session_pinned_facts(user_id=user_id, session_id=session_id)
        session_row = self.repository.get_session_row(user_id=user_id, session_id=session_id)
        return SessionMemoryState(
            session_id=session_id,
            user_id=user_id,
            summary=summary,
            recent_turns=recent_turns,
            pinned_facts=pinned_facts,
            last_write_at=datetime.fromisoformat(session_row["last_write_at"]) if session_row and session_row["last_write_at"] else None,
            turns_since_write=session_row["turn_count"] if session_row else 0,
        )

    def append_turn(self, turn: SessionTurn) -> None:
        self.repository.append_turn(turn)
        self.repository.bump_session_turn_count(user_id=turn.user_id, session_id=turn.session_id)

    def update_summary(self, *, user_id: str, session_id: str, summary_text: str, pinned_facts: Iterable[PinnedFact]) -> None:
        self.repository.upsert_session_summary(
            user_id=user_id,
            session_id=session_id,
            summary=summary_text,
            pinned_facts=list(pinned_facts),
            updated_at=datetime.utcnow(),
        )

    def write_back_state(
        self,
        *,
        user_id: str,
        session_id: str,
        summary_text: str,
        pinned_facts: Iterable[PinnedFact],
        turn_count_reset: bool = True,
    ) -> None:
        self.update_summary(user_id=user_id, session_id=session_id, summary_text=summary_text, pinned_facts=pinned_facts)
        if turn_count_reset:
            self.repository.reset_session_write_counters(user_id=user_id, session_id=session_id)

    def snapshot(self, *, user_id: str, session_id: str) -> MemorySnapshot:
        state = self.load_state(user_id=user_id, session_id=session_id)
        user_facts = self.repository.list_user_facts(user_id=user_id)
        return MemorySnapshot(
            session_turns=state.recent_turns,
            summary=state.summary,
            user_facts=user_facts,
            pinned_facts=state.pinned_facts,
            corpus_chunks=[],
        )

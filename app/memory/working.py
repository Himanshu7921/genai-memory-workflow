from __future__ import annotations

from dataclasses import dataclass, field

from app.models.memory import WorkingMemoryTurn


@dataclass(slots=True)
class WorkingMemory:
    turn: WorkingMemoryTurn
    intent: str | None = None
    retrieved_session_memory: dict = field(default_factory=dict)
    retrieved_user_memory: dict = field(default_factory=dict)
    retrieved_corpus_memory: dict = field(default_factory=dict)
    tool_plan: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)

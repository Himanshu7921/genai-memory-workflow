from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class FactScope(str, Enum):
    SESSION = "session"
    USER = "user"
    CORPUS = "corpus"


class FactStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


@dataclass(slots=True)
class WorkingMemoryTurn:
    user_id: str
    session_id: str
    message: str
    document_ids: list[str] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    retrieved_memory: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None


@dataclass(slots=True)
class MemoryFact:
    fact_id: str
    user_id: str
    scope: FactScope
    canonical_key: str
    value: str
    source: str
    confidence: float = 1.0
    priority: float = 1.0
    status: FactStatus = FactStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed_at: datetime = field(default_factory=datetime.utcnow)
    supersedes_fact_id: str | None = None
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PinnedFact:
    fact_id: str
    user_id: str
    session_id: str
    canonical_key: str
    value: str
    reason: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SessionTurn:
    turn_id: str
    user_id: str
    session_id: str
    role: str
    content: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SessionSummary:
    session_id: str
    user_id: str
    summary: str
    pinned_facts: list[PinnedFact] = field(default_factory=list)
    turn_count: int = 0
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class MemoryBudgetAudit:
    session_recent_turns_used: int
    session_summary_chars: int
    user_facts_used: int
    pinned_facts_used: int
    corpus_items_used: int
    evicted_items: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MemorySnapshot:
    session_turns: list[SessionTurn]
    summary: SessionSummary | None
    user_facts: list[MemoryFact]
    pinned_facts: list[PinnedFact]
    corpus_chunks: list[dict[str, Any]]
    budget_audit: MemoryBudgetAudit | None = None


@dataclass(slots=True)
class CorpusChunkRecord:
    chunk_id: str
    document_id: str
    user_id: str | None
    chunk_index: int
    content: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "active"

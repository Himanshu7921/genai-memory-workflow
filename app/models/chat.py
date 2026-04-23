from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    document_ids: list[str] = Field(default_factory=list)


class SourceItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    type: str
    name: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    score: float | None = None
    latency_ms: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BudgetAuditResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_recent_turns_used: int
    session_summary_chars: int
    user_facts_used: int
    pinned_facts_used: int
    corpus_items_used: int
    evicted_items: list[str] = Field(default_factory=list)


class TraceEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    event_type: str
    node_name: str
    trace_id: str
    session_id: str
    user_id: str
    started_at: datetime
    finished_at: datetime | None = None
    success: bool | None = None
    attempt: int = 0
    duration_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    final_answer: str
    used_llm: bool
    sources: list[SourceItem] = Field(default_factory=list)
    budget_audit: BudgetAuditResponse | None = None
    request_id: str
    trace_id: str
    user_id: str
    session_id: str
    trace_events: list[TraceEventResponse] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

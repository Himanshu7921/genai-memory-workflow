from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.models.memory import MemoryBudgetAudit, MemorySnapshot, WorkingMemoryTurn
from app.models.retrieval import RetrievalAudit, RetrievalContext
from app.models.tools import ToolExecutionSummary


@dataclass(slots=True)
class OrchestrationEvent:
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
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OrchestrationTrace:
    trace_id: str
    user_id: str
    session_id: str
    events: list[OrchestrationEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ModelSelection:
    model_name: str
    fallback_used: bool = False
    reason: str = "primary"


@dataclass(slots=True)
class IntentResult:
    intent: str
    confidence: float
    labels: list[str] = field(default_factory=list)
    model_selection: ModelSelection | None = None


@dataclass(slots=True)
class ToolPlanResult:
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""


@dataclass(slots=True)
class GeneratedResponse:
    final_answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    model_selection: ModelSelection | None = None
    incomplete: bool = False


@dataclass(slots=True)
class OrchestrationState:
    turn: WorkingMemoryTurn
    trace: OrchestrationTrace
    memory_snapshot: MemorySnapshot | None = None
    retrieval_context: RetrievalContext | None = None
    retrieval_audit: RetrievalAudit | None = None
    intent: IntentResult | None = None
    tool_plan: ToolPlanResult | None = None
    tool_results: ToolExecutionSummary | None = None
    response: GeneratedResponse | None = None
    memory_budget_audit: MemoryBudgetAudit | None = None
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OrchestrationResult:
    answer: str
    sources: list[dict[str, Any]]
    budget_audit: MemoryBudgetAudit | None
    trace: OrchestrationTrace
    used_llm: bool
    metadata: dict[str, Any] = field(default_factory=dict)

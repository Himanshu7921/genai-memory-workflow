from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolCall:
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None


@dataclass(slots=True)
class ToolResult:
    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None
    retry_count: int = 0
    latency_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolPlanItem:
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    required: bool = True


@dataclass(slots=True)
class ToolExecutionSummary:
    requested: int
    succeeded: int
    failed: int
    results: list[ToolResult] = field(default_factory=list)

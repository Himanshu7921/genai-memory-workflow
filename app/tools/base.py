from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class ToolContext:
    user_id: str
    session_id: str
    trace_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolExecutionError(RuntimeError):
    tool_name: str
    message: str
    retryable: bool = True

    def __str__(self) -> str:
        return f"{self.tool_name}: {self.message}"


class Tool(Protocol):
    name: str

    def run(self, arguments: dict[str, Any], context: ToolContext) -> Any: ...


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    tool: Tool
    retryable: bool = True


@dataclass(slots=True)
class ToolRegistry:
    tools: dict[str, ToolDefinition] = field(default_factory=dict)

    def register(self, definition: ToolDefinition) -> None:
        self.tools[definition.name] = definition

    def get(self, name: str) -> ToolDefinition:
        if name not in self.tools:
            raise KeyError(f"unknown tool: {name}")
        return self.tools[name]

    def list_tools(self) -> list[ToolDefinition]:
        return list(self.tools.values())

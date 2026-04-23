from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.models.orchestration import OrchestrationState


@dataclass(slots=True)
class NodeResult:
    state: OrchestrationState
    next_node: str | None = None


class Node(Protocol):
    name: str

    def run(self, state: OrchestrationState) -> NodeResult: ...

from __future__ import annotations

from dataclasses import dataclass
from random import random
from typing import Any

from app.tools.base import ToolContext, ToolExecutionError


@dataclass(slots=True)
class FlakyTool:
    name: str = "flaky_tool"
    failure_rate: float = 0.1

    def run(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        payload = arguments.get("payload", {})
        if random() < self.failure_rate:
            raise ToolExecutionError(self.name, "transient failure", retryable=True)
        return {
            "payload": payload,
            "status": "ok",
            "session_id": context.session_id,
        }

from __future__ import annotations

from dataclasses import dataclass
from time import sleep
from typing import Any

from app.tools.base import ToolContext, ToolExecutionError


@dataclass(slots=True)
class SlowEchoTool:
    name: str = "slow_tool"
    delay_seconds: float = 3.0

    def run(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        message = str(arguments.get("message", "")).strip()
        if not message:
            raise ToolExecutionError(self.name, "missing message", retryable=False)
        sleep(self.delay_seconds)
        return {
            "message": message,
            "delay_seconds": self.delay_seconds,
            "session_id": context.session_id,
            "user_id": context.user_id,
        }

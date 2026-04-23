from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from app.models.orchestration import OrchestrationState
from app.models.tools import ToolCall
from app.tools.base import ToolContext
from app.tools.executor import ToolExecutor
from app.orchestrator.nodes.base import NodeResult


@dataclass(slots=True)
class ToolExecutionNode:
    name: str = "tool_execution"
    tool_executor: ToolExecutor | None = None

    def run(self, state: OrchestrationState) -> NodeResult:
        if self.tool_executor is None or not state.tool_plan or not state.tool_plan.tool_calls:
            return NodeResult(state=state)

        start = perf_counter()
        tool_calls = [ToolCall(tool_name=item["tool_name"], arguments=item.get("arguments", {})) for item in state.tool_plan.tool_calls]
        context = ToolContext(user_id=state.turn.user_id, session_id=state.turn.session_id, trace_id=state.trace.trace_id)
        state.tool_results = self.tool_executor.execute(tool_calls, context)
        state.metadata["tool_execution_duration_ms"] = int((perf_counter() - start) * 1000)
        return NodeResult(state=state)

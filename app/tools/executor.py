from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from app.core.retry import RetryExhaustedError, RetryPolicy, run_with_retry
from app.tools.base import ToolContext, ToolDefinition, ToolExecutionError, ToolRegistry
from app.models.tools import ToolCall, ToolExecutionSummary, ToolResult


@dataclass(slots=True)
class ToolExecutor:
    registry: ToolRegistry
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    max_tool_calls_per_turn: int = 4

    def execute(self, calls: list[ToolCall], context: ToolContext) -> ToolExecutionSummary:
        limited_calls = calls[: self.max_tool_calls_per_turn]
        results: list[ToolResult] = []

        for call in limited_calls:
            results.append(self._execute_single(call, context))

        succeeded = sum(1 for result in results if result.success)
        return ToolExecutionSummary(requested=len(calls), succeeded=succeeded, failed=len(results) - succeeded, results=results)

    def _execute_single(self, call: ToolCall, context: ToolContext) -> ToolResult:
        definition = self.registry.get(call.tool_name)
        start = perf_counter()
        try:
            output, retry_count = self._run_with_retries(definition, call.arguments, context)
            latency_ms = int((perf_counter() - start) * 1000)
            return ToolResult(
                tool_name=definition.name,
                success=True,
                output=output,
                retry_count=retry_count,
                latency_ms=latency_ms,
            )
        except ToolExecutionError as exc:
            latency_ms = int((perf_counter() - start) * 1000)
            return ToolResult(
                tool_name=definition.name,
                success=False,
                error=str(exc),
                retry_count=self.retry_policy.max_attempts - 1 if exc.retryable else 0,
                latency_ms=latency_ms,
                metadata={"retryable": exc.retryable},
            )
        except RetryExhaustedError as exc:
            latency_ms = int((perf_counter() - start) * 1000)
            return ToolResult(
                tool_name=definition.name,
                success=False,
                error=str(exc),
                retry_count=self.retry_policy.max_attempts,
                latency_ms=latency_ms,
                metadata={"retryable": True},
            )

    def _run_with_retries(self, definition: ToolDefinition, arguments: dict[str, Any], context: ToolContext) -> tuple[Any, int]:
        def operation() -> Any:
            return definition.tool.run(arguments, context)

        if not definition.retryable:
            return operation(), 0

        output, attempt_index = run_with_retry(
            operation,
            policy=self.retry_policy,
            should_retry=lambda error: isinstance(error, ToolExecutionError) and error.retryable,
        )
        return output, attempt_index

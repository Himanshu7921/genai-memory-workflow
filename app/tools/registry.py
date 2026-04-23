from __future__ import annotations

from dataclasses import dataclass, field

from app.retrieval.service import RetrievalService
from app.tools.base import ToolDefinition, ToolRegistry
from app.tools.calculator import CalculatorTool
from app.tools.flaky_tool import FlakyTool
from app.tools.retrieval_tool import RetrievalTool
from app.tools.slow_tool import SlowEchoTool


@dataclass(slots=True)
class DefaultToolRegistryFactory:
    retrieval_service: RetrievalService | None = None

    def build(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="calculator", description="Evaluate safe arithmetic expressions.", tool=CalculatorTool(), retryable=False))
        registry.register(ToolDefinition(name="slow_tool", description="Simulate a slow external dependency.", tool=SlowEchoTool(), retryable=False))
        registry.register(ToolDefinition(name="flaky_tool", description="Simulate a transiently failing dependency.", tool=FlakyTool(), retryable=True))
        registry.register(ToolDefinition(name="retrieval_tool", description="Retrieve grounded context from the corpus.", tool=RetrievalTool(retrieval_service=self.retrieval_service), retryable=True))
        return registry

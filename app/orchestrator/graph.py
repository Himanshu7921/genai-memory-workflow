from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

from app.memory.service import MemoryService
from app.models.orchestration import OrchestrationEvent, OrchestrationResult, OrchestrationState, OrchestrationTrace
from app.observability.logging import TraceLogger
from app.orchestrator.nodes.base import Node
from app.orchestrator.nodes.document_retrieve import DocumentRetrievalNode
from app.orchestrator.nodes.generate import ResponseGenerationNode
from app.orchestrator.nodes.intent import IntentClassificationNode
from app.orchestrator.nodes.memory_retrieve import MemoryRetrievalNode
from app.orchestrator.nodes.tool_execute import ToolExecutionNode
from app.orchestrator.nodes.tool_plan import ToolPlanningNode
from app.orchestrator.nodes.write_back import MemoryWriteBackNode
from app.tools.executor import ToolExecutor
from app.retrieval.service import RetrievalService


@dataclass(slots=True)
class NodeConfig:
    retries: int = 2
    retry_backoff_seconds: float = 0.1


@dataclass(slots=True)
class OrchestrationGraph:
    memory_service: MemoryService
    retrieval_service: RetrievalService
    tool_executor: ToolExecutor
    nodes: dict[str, Node] = field(init=False)
    node_config: NodeConfig = field(default_factory=NodeConfig)

    def __post_init__(self) -> None:
        self.nodes = {
            "intent_classification": IntentClassificationNode(),
            "memory_retrieval": MemoryRetrievalNode(memory_service=self.memory_service),
            "document_retrieval": DocumentRetrievalNode(retrieval_service=self.retrieval_service),
            "tool_planning": ToolPlanningNode(),
            "tool_execution": ToolExecutionNode(tool_executor=self.tool_executor),
            "response_generation": ResponseGenerationNode(),
            "memory_write_back": MemoryWriteBackNode(memory_service=self.memory_service),
        }

    def run(self, *, user_id: str, session_id: str, message: str, document_ids: list[str] | None = None, trace_id: str | None = None) -> OrchestrationResult:
        trace_id = trace_id or uuid4().hex
        turn = self.memory_service.build_working_memory(
            user_id=user_id,
            session_id=session_id,
            message=message,
            document_ids=document_ids or [],
            trace_id=trace_id,
        )
        state = OrchestrationState(
            turn=turn,
            trace=OrchestrationTrace(trace_id=trace_id, user_id=user_id, session_id=session_id),
            metadata={"started_at": datetime.utcnow().isoformat()},
        )
        logger = TraceLogger(trace_id=trace_id, user_id=user_id, session_id=session_id)

        state = self._run_node_with_retry(node_name="intent_classification", node=self.nodes["intent_classification"], state=state, logger=logger)
        state = self._run_parallel_retrieval(state, logger)
        for node_name in ["tool_planning", "tool_execution", "response_generation", "memory_write_back"]:
            state = self._run_node_with_retry(node_name=node_name, node=self.nodes[node_name], state=state, logger=logger)

        self._persist_turn(state)
        used_llm = bool(
            state.response
            and state.response.model_selection
            and not state.response.model_selection.fallback_used
            and state.response.model_selection.reason == "llm_generator"
        )
        return OrchestrationResult(
            answer=state.response.final_answer if state.response else state.turn.message,
            used_llm=used_llm,
            sources=state.response.sources if state.response else [],
            budget_audit=state.memory_budget_audit,
            trace=state.trace,
            metadata=state.metadata,
        )

    def _run_node_with_retry(self, *, node_name: str, node: Node, state: OrchestrationState, logger: TraceLogger) -> OrchestrationState:
        last_error: Exception | None = None
        for attempt in range(self.node_config.retries + 1):
            started = perf_counter()
            logger.log_node_start(node_name, attempt)
            event = OrchestrationEvent(
                event_type="node",
                node_name=node_name,
                trace_id=state.trace.trace_id,
                session_id=state.turn.session_id,
                user_id=state.turn.user_id,
                started_at=datetime.utcnow(),
                attempt=attempt,
            )
            try:
                result = node.run(state)
                state = result.state
                duration_ms = int((perf_counter() - started) * 1000)
                event.finished_at = datetime.utcnow()
                event.success = True
                event.duration_ms = duration_ms
                state.trace.events.append(event)
                logger.log_node_end(node_name, True, duration_ms)
                return state
            except Exception as exc:
                last_error = exc
                duration_ms = int((perf_counter() - started) * 1000)
                event.finished_at = datetime.utcnow()
                event.success = False
                event.duration_ms = duration_ms
                event.metadata["error"] = str(exc)
                state.trace.events.append(event)
                logger.log_node_end(node_name, False, duration_ms, {"error": str(exc)})
                if attempt < self.node_config.retries:
                    continue
                state.errors.append(f"{node_name}:{exc}")
                logger.log_warning(f"node_failed_after_retries:{node_name}", {"error": str(exc)})
                return state

        if last_error is not None:
            state.errors.append(f"{node_name}:{last_error}")
        return state

    def _run_parallel_retrieval(self, state: OrchestrationState, logger: TraceLogger) -> OrchestrationState:
        node_names = ["memory_retrieval", "document_retrieval"]

        def forked_state() -> OrchestrationState:
            return OrchestrationState(
                turn=state.turn,
                trace=OrchestrationTrace(trace_id=state.trace.trace_id, user_id=state.trace.user_id, session_id=state.trace.session_id),
                memory_snapshot=state.memory_snapshot,
                retrieval_context=state.retrieval_context,
                retrieval_audit=state.retrieval_audit,
                intent=state.intent,
                tool_plan=state.tool_plan,
                tool_results=state.tool_results,
                response=state.response,
                memory_budget_audit=state.memory_budget_audit,
                errors=list(state.errors),
                metadata=dict(state.metadata),
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                name: executor.submit(self._run_node_with_retry, node_name=name, node=self.nodes[name], state=forked_state(), logger=logger)
                for name in node_names
            }
            memory_state = futures["memory_retrieval"].result()
            document_state = futures["document_retrieval"].result()

        state.memory_snapshot = memory_state.memory_snapshot
        state.memory_budget_audit = memory_state.memory_budget_audit
        state.retrieval_context = document_state.retrieval_context
        state.retrieval_audit = document_state.retrieval_audit
        state.metadata.update(memory_state.metadata)
        state.metadata.update(document_state.metadata)
        state.errors.extend(memory_state.errors)
        state.errors.extend(document_state.errors)
        state.trace.events.extend(memory_state.trace.events)
        state.trace.events.extend(document_state.trace.events)
        return state

    def _persist_turn(self, state: OrchestrationState) -> None:
        self.memory_service.capture_turn(turn=state.turn)

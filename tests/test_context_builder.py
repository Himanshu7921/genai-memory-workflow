from __future__ import annotations

from datetime import datetime

from app.models.memory import FactScope, FactStatus, MemoryFact, MemorySnapshot, ProtectedFact, SessionSummary, SessionTurn, WorkingMemoryTurn
from app.models.orchestration import OrchestrationState, OrchestrationTrace
from app.models.retrieval import RetrievalChunk, RetrievalContext
from app.models.tools import ToolExecutionSummary, ToolResult
from app.orchestrator.context_builder import ContextBuilder


def _fact(*, key: str, value: str, idx: int) -> MemoryFact:
    now = datetime.utcnow()
    return MemoryFact(
        fact_id=f"fact_{idx}",
        user_id="u_test",
        scope=FactScope.USER,
        canonical_key=key,
        value=value,
        source="test",
        status=FactStatus.ACTIVE,
        created_at=now,
        updated_at=now,
        last_accessed_at=now,
    )


def _state() -> OrchestrationState:
    turns = [
        SessionTurn(
            turn_id=f"turn_{idx}",
            user_id="u_test",
            session_id="s_test",
            role="user" if idx % 2 == 0 else "assistant",
            content=("long historical turn " * 40) + str(idx),
        )
        for idx in range(10)
    ]
    protected = [
        ProtectedFact(
            fact_id="pf_1",
            user_id="u_test",
            session_id="s_test",
            canonical_key="resolution_time_critical",
            value="4 hours",
            source="conversation",
            reason="duration_value",
        )
    ]
    summary = SessionSummary(
        session_id="s_test",
        user_id="u_test",
        summary=("summary text " * 300),
        protected_facts=protected,
    )
    snapshot = MemorySnapshot(
        session_turns=turns,
        summary=summary,
        user_facts=[_fact(key="sla", value="critical in 4 hours", idx=1), _fact(key="owner", value="platform team", idx=2)],
        pinned_facts=[],
        corpus_chunks=[],
    )
    retrieval = RetrievalContext(
        query="test",
        chunks=[
            RetrievalChunk(
                chunk_id=f"chunk_{idx}",
                document_id="doc_policy",
                chunk_index=idx,
                score=1.0 - (idx * 0.1),
                content=("retrieved content " * 200) + str(idx),
            )
            for idx in range(6)
        ],
    )
    tools = ToolExecutionSummary(
        requested=2,
        succeeded=2,
        failed=0,
        results=[
            ToolResult(tool_name="calculator", success=True, output={"result": "42"}),
            ToolResult(tool_name="retrieval_tool", success=True, output={"chunks": [{"content": "tool context" * 200}]})
        ],
    )
    return OrchestrationState(
        turn=WorkingMemoryTurn(user_id="u_test", session_id="s_test", message="What is the resolution time?"),
        trace=OrchestrationTrace(trace_id="trace_1", user_id="u_test", session_id="s_test"),
        memory_snapshot=snapshot,
        retrieval_context=retrieval,
        tool_results=tools,
        metadata={},
    )


def test_overflow_scenario_triggers_eviction() -> None:
    builder = ContextBuilder(
        model_budgets={"primary": 450, "fallback": 250},
        default_allocation={
            "system_prompt": 120,
            "l3_user_facts": 200,
            "l2_summary": 150,
            "recent_turns": 220,
            "retrieved_docs": 260,
            "tools": 120,
            "current_turn": 80,
            "response_headroom": 120,
        },
    )
    state = _state()
    result = builder.build(
        state=state,
        system_prompt="You are a production assistant.",
        include_memory_context=True,
        model_tier="primary",
    )

    assert result.audit["budget_total"] == 450
    assert len(result.audit["eviction_events"]) > 0


def test_priority_protection_keeps_current_turn_and_l3() -> None:
    builder = ContextBuilder(
        model_budgets={"primary": 1500, "fallback": 700},
        default_allocation={
            "system_prompt": 150,
            "l3_user_facts": 240,
            "l2_summary": 180,
            "recent_turns": 280,
            "retrieved_docs": 320,
            "tools": 120,
            "current_turn": 120,
            "response_headroom": 250,
        },
    )
    state = _state()
    result = builder.build(
        state=state,
        system_prompt="You are a production assistant.",
        include_memory_context=True,
        model_tier="primary",
    )

    assert result.scoped_state.turn.message == "What is the resolution time?"
    assert result.scoped_state.memory_snapshot is not None
    assert len(result.scoped_state.memory_snapshot.user_facts) >= 1


def test_budget_audit_sum_is_within_budget() -> None:
    builder = ContextBuilder()
    state = _state()
    result = builder.build(
        state=state,
        system_prompt="You are a production assistant.",
        include_memory_context=True,
        model_tier="primary",
    )

    allocation = result.audit["allocation"]
    total_allocated = sum(item["tokens"] for item in allocation.values())
    assert total_allocated <= result.audit["budget_total"]


def test_context_builder_is_deterministic() -> None:
    builder = ContextBuilder()
    state = _state()

    first = builder.build(
        state=state,
        system_prompt="You are a production assistant.",
        include_memory_context=True,
        model_tier="primary",
    )
    second = builder.build(
        state=state,
        system_prompt="You are a production assistant.",
        include_memory_context=True,
        model_tier="primary",
    )

    assert first.audit == second.audit
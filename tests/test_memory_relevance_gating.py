from __future__ import annotations

from datetime import datetime

from app.llm.prompts import build_response_prompt_bundle
from app.models.memory import MemorySnapshot, SessionSummary, WorkingMemoryTurn
from app.models.orchestration import OrchestrationState, OrchestrationTrace
from app.orchestrator.nodes.generate import ResponseGenerationNode, memory_relevance_score
from app.retrieval.embeddings import EmbeddingProvider


def _build_state(message: str, *, summary_text: str = "Critical issues are resolved in 4 hours; high-priority issues are resolved in 12 hours.") -> OrchestrationState:
    turn = WorkingMemoryTurn(user_id="u_test", session_id="s_test", message=message)
    trace = OrchestrationTrace(trace_id="trace_1", user_id="u_test", session_id="s_test")
    snapshot = MemorySnapshot(
        session_turns=[],
        summary=SessionSummary(session_id="s_test", user_id="u_test", summary=summary_text),
        user_facts=[],
        pinned_facts=[],
        corpus_chunks=[],
    )
    return OrchestrationState(turn=turn, trace=trace, memory_snapshot=snapshot, metadata={"created_at": datetime.utcnow().isoformat()})


class _NoLLMNode(ResponseGenerationNode):
    def _generate_with_llm(self, state: OrchestrationState, memory_answer: str | None = None, include_memory_context: bool = True) -> str | None:
        return None


class _FakeSemanticEmbeddingProvider(EmbeddingProvider):
    def embed_text(self, text: str) -> list[float]:
        normalized = text.lower()
        urgency = any(token in normalized for token in ["urgent", "critical", "priority", "high-priority"])
        speed = any(token in normalized for token in ["quick", "quickly", "fast", "speed", "time", "handled", "resolved", "resolution"])
        greeting = any(token in normalized for token in ["hello", "hi", "ignore", "random"])
        return [1.0 if urgency else 0.0, 1.0 if speed else 0.0, 1.0 if greeting else 0.0]


def test_is_memory_relevant_false_for_noise() -> None:
    query = "ignore this"
    context = "Critical issues are resolved in 4 hours; high-priority issues are resolved in 12 hours."
    relevant, similarity, _ = memory_relevance_score(query, context, threshold=0.58, embedding_provider=_FakeSemanticEmbeddingProvider())
    assert relevant is False
    assert similarity < 0.58


def test_semantic_query_matches_memory_chunk() -> None:
    query = "How quickly are urgent issues handled?"
    context = "Billing code: INV-9001. Critical issues are resolved in 4 hours."
    relevant, similarity, matched_chunk = memory_relevance_score(
        query,
        context,
        threshold=0.58,
        embedding_provider=_FakeSemanticEmbeddingProvider(),
    )
    assert relevant is True
    assert similarity >= 0.58
    assert matched_chunk is not None
    assert "Critical issues" in matched_chunk


def test_partial_resolution_query_still_matches() -> None:
    query = "resolution time?"
    context = "Critical issues are resolved in 4 hours; high-priority issues are resolved in 12 hours."
    relevant, similarity, _ = memory_relevance_score(
        query,
        context,
        threshold=0.58,
        embedding_provider=_FakeSemanticEmbeddingProvider(),
    )
    assert relevant is True
    assert similarity >= 0.58


def test_prompt_omits_memory_context_when_irrelevant() -> None:
    state = _build_state("ignore this")
    prompt = build_response_prompt_bundle(state, include_memory_context=False)
    assert "omitted due to irrelevance gate" in prompt.user_prompt


def test_noise_query_does_not_force_memory_answer_on_fallback() -> None:
    state = _build_state("ignore this")
    node = _NoLLMNode()
    result = node.run(state)

    assert result.state.metadata["response_memory_relevant"] is False
    assert "response_memory_match" not in result.state.metadata
    assert result.state.response is not None
    assert result.state.response.final_answer == "ignore this"
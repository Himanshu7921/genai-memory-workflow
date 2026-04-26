from __future__ import annotations

from datetime import datetime

from app.models.memory import FactScope, FactStatus, MemoryFact, MemorySnapshot, SessionSummary, WorkingMemoryTurn
from app.models.orchestration import OrchestrationState, OrchestrationTrace
from app.orchestrator.nodes import memory_retrieve as memory_retrieve_module
from app.orchestrator.nodes.generate import ResponseGenerationNode
from app.orchestrator.nodes.memory_retrieve import MemoryRetrievalNode
from app.retrieval.embeddings import EmbeddingProvider


class _SemanticTestEmbeddingProvider(EmbeddingProvider):
    def embed_text(self, text: str) -> list[float]:
        normalized = text.lower()
        like_intent = any(token in normalized for token in ["like", "enjoy", "preference"])
        football = "football" in normalized
        name = "name" in normalized
        return [1.0 if like_intent else 0.0, 1.0 if football else 0.0, 1.0 if name else 0.0]


class _StubMemoryLoadResult:
    def __init__(self, snapshot: MemorySnapshot) -> None:
        self.snapshot = snapshot
        self.audit = None


class _StubMemoryService:
    def __init__(self, snapshot: MemorySnapshot) -> None:
        self._snapshot = snapshot

    def load(self, *, user_id: str, session_id: str, document_ids: list[str] | None = None):
        return _StubMemoryLoadResult(self._snapshot)


class _NoLLMNode(ResponseGenerationNode):
    def _generate_with_llm(self, state: OrchestrationState, memory_answer: str | None = None, include_memory_context: bool = True) -> str | None:
        return None


def _build_state(message: str, fact: MemoryFact) -> OrchestrationState:
    turn = WorkingMemoryTurn(user_id="u_test", session_id="s_test", message=message)
    trace = OrchestrationTrace(trace_id="trace_semantic", user_id="u_test", session_id="s_test")
    snapshot = MemorySnapshot(
        session_turns=[],
        summary=SessionSummary(session_id="s_test", user_id="u_test", summary=""),
        user_facts=[fact],
        pinned_facts=[],
        corpus_chunks=[],
    )
    return OrchestrationState(turn=turn, trace=trace, memory_snapshot=snapshot, metadata={"created_at": datetime.utcnow().isoformat()})


def test_semantic_memory_retrieval_matches_preference(monkeypatch):
    provider = _SemanticTestEmbeddingProvider()
    preference_fact = MemoryFact(
        fact_id="fact_pref",
        user_id="u_test",
        scope=FactScope.USER,
        canonical_key="preference",
        value="I enjoy football",
        source="conversation",
        status=FactStatus.ACTIVE,
        embedding=provider.embed_text("preference: I enjoy football"),
    )
    state = _build_state("What do I like?", preference_fact)

    node = MemoryRetrievalNode(memory_service=_StubMemoryService(state.memory_snapshot), min_similarity_threshold=0.6)
    monkeypatch.setattr(memory_retrieve_module, "_get_query_embedding_provider", lambda: provider)

    result = node.run(state)

    semantic = result.state.turn.retrieved_memory.get("semantic_user_facts", [])
    assert semantic
    assert semantic[0]["value"] == "I enjoy football"
    assert semantic[0]["score"] >= 0.6
    assert result.state.metadata["memory_similarity_scores"]


def test_preference_query_returns_semantic_memory_answer(monkeypatch):
    provider = _SemanticTestEmbeddingProvider()
    preference_fact = MemoryFact(
        fact_id="fact_pref",
        user_id="u_test",
        scope=FactScope.USER,
        canonical_key="preference",
        value="I enjoy football",
        source="conversation",
        status=FactStatus.ACTIVE,
        embedding=provider.embed_text("preference: I enjoy football"),
    )
    state = _build_state("What do I like?", preference_fact)

    memory_node = MemoryRetrievalNode(memory_service=_StubMemoryService(state.memory_snapshot), min_similarity_threshold=0.6)
    monkeypatch.setattr(memory_retrieve_module, "_get_query_embedding_provider", lambda: provider)
    state = memory_node.run(state).state

    response_node = _NoLLMNode()
    result = response_node.run(state)

    assert result.state.response is not None
    assert result.state.response.final_answer == "You enjoy football."

from __future__ import annotations

from app.orchestrator.nodes.write_back import MemoryWriteBackNode
from app.models.memory import WorkingMemoryTurn
from app.models.orchestration import OrchestrationState, OrchestrationTrace


def _state_for_message(message: str) -> OrchestrationState:
    turn = WorkingMemoryTurn(user_id="u_test", session_id="s_test", message=message)
    trace = OrchestrationTrace(trace_id="trace_1", user_id="u_test", session_id="s_test")
    return OrchestrationState(turn=turn, trace=trace)


def test_does_not_extract_favorite_from_question_text() -> None:
    node = MemoryWriteBackNode(memory_service=None)
    state = _state_for_message("Who is my favorite anime character?")
    facts = node._extract_user_facts(state)
    favorite_facts = [fact for fact in facts if fact.canonical_key == "favorite"]
    assert favorite_facts == []


def test_extracts_declared_favorite_and_interest() -> None:
    node = MemoryWriteBackNode(memory_service=None)
    state = _state_for_message("I like driving f1 car in sports and my fav anime character is Itachi from Naruto.")
    facts = node._extract_user_facts(state)

    keys = {fact.canonical_key for fact in facts}
    values = {fact.value.lower() for fact in facts}

    assert "interest" in keys
    assert "favorite" in keys
    assert "driving f1 car" in values
    assert "itachi" in values

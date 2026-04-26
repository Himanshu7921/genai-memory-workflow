from __future__ import annotations

from app.core.config import MemoryConfig
from app.memory.protected_facts import ProtectedFactCandidate, build_protected_fact, extract_protected_fact_candidates
from app.memory.service import MemoryService
from app.storage.memory_repo import MemoryRepository
from app.storage.sqlite import SQLiteStore


def make_service(tmp_path) -> MemoryService:
    store = SQLiteStore(db_path=tmp_path / "memory.sqlite3")
    return MemoryService(MemoryRepository(store), MemoryConfig())


def test_extract_protected_facts_keeps_numeric_truth() -> None:
    text = "Critical issues resolved in 4 hours on 2026-04-26 with invoice: INV-1234 and payment $1,250."

    candidates = extract_protected_fact_candidates(text)
    values = {candidate.value for candidate in candidates}

    assert "4 hours" in values
    assert "2026-04-26" in values
    assert "INV-1234" in values
    assert "$1,250" in values


def test_protected_facts_survive_repeated_summary_writes(tmp_path) -> None:
    service = make_service(tmp_path)
    candidate = ProtectedFactCandidate(
        canonical_key="duration",
        value="4 hours",
        source="conversation",
        reason="duration_value",
    )
    fact = build_protected_fact(user_id="user-1", session_id="session-1", candidate=candidate)

    for _ in range(10):
        service.write_session_memory(
            user_id="user-1",
            session_id="session-1",
            summary_text="Critical issues resolved in 4 hours.",
            pinned_facts=[],
            protected_facts=[fact],
        )

    state = service.session.load_state(user_id="user-1", session_id="session-1")
    assert state.summary is not None
    assert len(state.summary.protected_facts) == 1
    assert state.summary.protected_facts[0].value == "4 hours"


def test_latest_protected_fact_overrides_conflicting_value(tmp_path) -> None:
    service = make_service(tmp_path)

    age_25 = extract_protected_fact_candidates("My age is 25.")
    age_30 = extract_protected_fact_candidates("My age is 30.")

    fact_25 = build_protected_fact(
        user_id="user-1",
        session_id="session-1",
        candidate=next(candidate for candidate in age_25 if candidate.canonical_key == "age"),
    )
    fact_30 = build_protected_fact(
        user_id="user-1",
        session_id="session-1",
        candidate=next(candidate for candidate in age_30 if candidate.canonical_key == "age"),
    )

    service.write_session_memory(
        user_id="user-1",
        session_id="session-1",
        summary_text="My age is 25.",
        pinned_facts=[],
        protected_facts=[fact_25],
    )
    service.write_session_memory(
        user_id="user-1",
        session_id="session-1",
        summary_text="My age is 30.",
        pinned_facts=[],
        protected_facts=[fact_30],
    )

    state = service.session.load_state(user_id="user-1", session_id="session-1")
    assert state.summary is not None
    assert [fact.value for fact in state.summary.protected_facts] == ["30"]

    rows = service.repository.store.read_only()
    with rows as connection:
        persisted = connection.execute(
            """
            SELECT value, status
            FROM session_protected_facts
            WHERE user_id = ? AND session_id = ?
            ORDER BY created_at ASC
            """,
            ("user-1", "session-1"),
        ).fetchall()

    assert [row["value"] for row in persisted] == ["25", "30"]
    assert persisted[0]["status"] == "superseded"
    assert persisted[1]["status"] == "active"
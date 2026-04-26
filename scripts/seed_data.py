from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.api.dependencies import build_container
from app.memory.policies import normalize_fact_key
from app.models.memory import FactStatus, PinnedFact, ProtectedFact, SessionTurn


DEMO_USER_ID = "u_demo"
DEMO_SESSION_ID = "s_demo"
DEMO_DOCUMENT_ID = "policy_demo"


@dataclass(slots=True)
class SeedResult:
    facts_added: int
    turns_added: int
    docs_indexed: int


def seed_user_facts(container, *, user_id: str) -> int:
    memory_service = container.memory_service
    existing = memory_service.user.load_facts(user_id=user_id)
    existing_active = {
        (fact.canonical_key, fact.value.strip().lower())
        for fact in existing
        if fact.status == FactStatus.ACTIVE
    }

    desired_facts = [
        ("name", "Himanshu Singh", 1.0),
        ("age", "25", 0.95),
        ("risk_tolerance", "low", 0.95),
        ("interest", "football", 0.8),
    ]

    incoming = []
    for key, value, priority in desired_facts:
        canonical_key = normalize_fact_key(key)
        if (canonical_key, value.strip().lower()) in existing_active:
            continue
        incoming.append(
            memory_service.user.build_fact(
                user_id=user_id,
                key=key,
                value=value,
                source="seed_script",
                priority=priority,
                metadata={"seed": True},
            )
        )

    if not incoming:
        return 0

    write_result = memory_service.write_user_memory(
        user_id=user_id,
        incoming_facts=incoming,
        source="seed_script",
    )
    return len(write_result.written)


def _pair(turn: SessionTurn) -> tuple[str, str]:
    return (turn.role.strip().lower(), turn.content.strip())


def seed_session_turns_and_summary(container, *, user_id: str, session_id: str) -> int:
    memory_service = container.memory_service

    desired_turns: list[tuple[str, str]] = [
        ("user", "My name is Himanshu"),
        ("assistant", "Noted."),
        ("user", "I enjoy football"),
        ("assistant", "Got it."),
        ("user", "I am 25 years old"),
        ("assistant", "Updated."),
    ]

    turn_count = memory_service.repository.count_session_turns(user_id=user_id, session_id=session_id)
    recent = memory_service.repository.list_recent_turns(
        user_id=user_id,
        session_id=session_id,
        limit=max(turn_count, 100),
    )
    existing_pairs = {_pair(turn) for turn in recent}

    inserted = 0
    for role, content in desired_turns:
        if (role, content) in existing_pairs:
            continue
        working_turn = memory_service.build_working_memory(
            user_id=user_id,
            session_id=session_id,
            message=content,
            trace_id="seed_script",
        )
        memory_service.capture_turn(turn=working_turn, role=role)
        existing_pairs.add((role, content))
        inserted += 1

    summary_text = (
        "User profile snapshot: name is Himanshu Singh, age 25, risk tolerance low, "
        "and prefers football."
    )
    state = memory_service.session.load_state(user_id=user_id, session_id=session_id)
    current_summary = state.summary.summary.strip() if state.summary else ""
    if current_summary != summary_text:
        pinned: Iterable[PinnedFact] = state.pinned_facts if state else []
        protected: Iterable[ProtectedFact] = state.summary.protected_facts if state and state.summary else []
        memory_service.write_session_memory(
            user_id=user_id,
            session_id=session_id,
            summary_text=summary_text,
            pinned_facts=list(pinned),
            protected_facts=list(protected),
            turn_count_reset=False,
        )

    return inserted


def seed_documents(container, *, user_id: str) -> int:
    retrieval_service = container.retrieval_service
    existing_chunks = container.repository.list_chunks_for_documents([DEMO_DOCUMENT_ID])
    if existing_chunks:
        return 0

    policy_text = """
Demo Policy Handbook

Warranty Policy
- All hardware accessories include a 24-month limited warranty.
- Manufacturing defects are eligible for free replacement within warranty coverage.
- Accidental or liquid damage is excluded from warranty claims.

SLA Resolution Times
- Critical priority incidents: first response within 30 minutes, target resolution in 4 hours.
- High priority incidents: first response within 2 hours, target resolution in 12 hours.
- Medium priority incidents: first response within 8 hours, target resolution in 2 business days.
- Low priority incidents: best effort support with target resolution in 5 business days.

Refund Policy
- Full refund is available within 14 days of purchase for unused subscriptions.
- Pro-rated refund is available between day 15 and day 30 for annual plans.
- Refund requests after 30 days are not eligible unless required by local law.
- Approved refunds are processed to the original payment method within 7 business days.
""".strip()

    retrieval_service.index_document(
        document_id=DEMO_DOCUMENT_ID,
        content=policy_text,
        user_id=user_id,
        source_uri="seed://policy_demo",
        title="Demo Policy",
        metadata={"seed": True, "category": "policy"},
    )
    return 1


def run_seed() -> SeedResult:
    container = build_container()
    facts_added = seed_user_facts(container, user_id=DEMO_USER_ID)
    turns_added = seed_session_turns_and_summary(container, user_id=DEMO_USER_ID, session_id=DEMO_SESSION_ID)
    docs_indexed = seed_documents(container, user_id=DEMO_USER_ID)
    return SeedResult(facts_added=facts_added, turns_added=turns_added, docs_indexed=docs_indexed)


def main() -> None:
    result = run_seed()
    print("✔ Seed data created")
    print(f"✔ User: {DEMO_USER_ID}")
    print(f"✔ Session: {DEMO_SESSION_ID}")
    print(f"✔ Documents indexed: {DEMO_DOCUMENT_ID}")
    print(f"  Facts added: {result.facts_added}")
    print(f"  Turns added: {result.turns_added}")
    print(f"  Documents indexed this run: {result.docs_indexed}")


if __name__ == "__main__":
    main()

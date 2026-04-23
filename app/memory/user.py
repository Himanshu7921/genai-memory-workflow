from __future__ import annotations

from dataclasses import dataclass

from app.core.config import DEFAULT_MEMORY_CONFIG, MemoryConfig
from app.memory.policies import compute_fact_decay_score, resolve_fact_contradictions, should_prune_fact
from app.memory.resolver import build_fact
from app.models.memory import FactScope, FactStatus, MemoryFact
from app.storage.memory_repo import MemoryRepository


@dataclass(slots=True)
class UserMemoryWriteResult:
    written: list[MemoryFact]
    superseded: list[MemoryFact]
    pruned: list[str]


class UserMemoryService:
    def __init__(self, repository: MemoryRepository, config: MemoryConfig = DEFAULT_MEMORY_CONFIG) -> None:
        self.repository = repository
        self.config = config

    def load_facts(self, *, user_id: str) -> list[MemoryFact]:
        facts = self.repository.list_user_facts(user_id=user_id)
        return facts

    def write_facts(
        self,
        *,
        user_id: str,
        incoming: list[MemoryFact],
        source: str,
    ) -> UserMemoryWriteResult:
        existing = self.repository.list_user_facts(user_id=user_id)
        resolved, _ = resolve_fact_contradictions(existing, incoming)
        written: list[MemoryFact] = []
        superseded: list[MemoryFact] = []
        pruned: list[str] = []

        for fact in resolved:
            if fact.status == FactStatus.SUPERSEDED:
                superseded.append(fact)
            else:
                written.append(fact)
            self.repository.upsert_user_fact(fact)

        for fact in self.repository.list_user_facts(user_id=user_id):
            score = compute_fact_decay_score(fact)
            if should_prune_fact(fact, config=self.config) and score < 0.5:
                pruned.append(fact.fact_id)
                self.repository.archive_user_fact(fact.fact_id)

        return UserMemoryWriteResult(written=written, superseded=superseded, pruned=pruned)

    def build_fact(
        self,
        *,
        user_id: str,
        key: str,
        value: str,
        source: str,
        confidence: float = 1.0,
        priority: float = 1.0,
        metadata: dict | None = None,
    ) -> MemoryFact:
        return build_fact(
            user_id=user_id,
            scope=FactScope.USER,
            key=key,
            value=value,
            source=source,
            confidence=confidence,
            priority=priority,
            metadata=metadata,
        )

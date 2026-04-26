from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging

from app.core.config import DEFAULT_MEMORY_CONFIG, MemoryConfig
from app.memory.policies import compute_fact_decay_score, resolve_fact_contradictions, should_prune_fact
from app.memory.resolver import build_fact
from app.models.memory import FactScope, FactStatus, MemoryFact
from app.retrieval.embeddings import EmbeddingProvider, HashEmbeddingProvider
from app.retrieval.embeddings_hf import HFEmbeddingProvider
from app.storage.memory_repo import MemoryRepository


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_user_fact_embedding_provider() -> EmbeddingProvider:
    try:
        return HFEmbeddingProvider()
    except Exception:
        logger.warning("user_fact_embedding_provider_fallback", exc_info=True)
        return HashEmbeddingProvider()


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
        self._ensure_embeddings(incoming)
        existing = self.repository.list_user_facts(user_id=user_id)
        resolved, _ = resolve_fact_contradictions(existing, incoming)
        self._ensure_embeddings(resolved)
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

    def backfill_missing_embeddings(self, *, user_id: str) -> int:
        facts = self.repository.list_user_facts(user_id=user_id)
        missing = [fact for fact in facts if not fact.embedding]
        if not missing:
            return 0
        self._ensure_embeddings(missing)
        backfilled = 0
        for fact in missing:
            if not fact.embedding:
                continue
            self.repository.upsert_user_fact(fact)
            backfilled += 1
        if backfilled:
            logger.info("Backfilled user fact embeddings: user_id=%s count=%s", user_id, backfilled)
        return backfilled

    def _ensure_embeddings(self, facts: list[MemoryFact]) -> None:
        to_embed: list[MemoryFact] = [fact for fact in facts if not fact.embedding and fact.value.strip()]
        if not to_embed:
            return
        provider = _get_user_fact_embedding_provider()
        texts = [f"{fact.canonical_key}: {fact.value}" for fact in to_embed]
        try:
            vectors = provider.embed_texts(texts)
        except Exception:
            logger.warning("user_fact_embedding_batch_failed", exc_info=True)
            vectors = []
        for fact, vector in zip(to_embed, vectors):
            if vector:
                fact.embedding = vector

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

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
from time import perf_counter

from app.memory.service import MemoryService
from app.models.memory import FactStatus, MemoryFact
from app.models.orchestration import OrchestrationState
from app.orchestrator.nodes.base import NodeResult
from app.retrieval.embeddings import EmbeddingProvider, HashEmbeddingProvider, cosine_similarity
from app.retrieval.embeddings_hf import HFEmbeddingProvider


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_query_embedding_provider() -> EmbeddingProvider:
    try:
        return HFEmbeddingProvider()
    except Exception:
        logger.warning("memory_query_embedding_provider_fallback", exc_info=True)
        return HashEmbeddingProvider()


@dataclass(slots=True)
class MemoryRetrievalNode:
    name: str = "memory_retrieval"
    memory_service: MemoryService | None = None
    min_similarity_threshold: float = 0.35
    min_fallback_similarity: float = 0.1
    semantic_top_k: int = 5

    def run(self, state: OrchestrationState) -> NodeResult:
        if self.memory_service is None:
            state.errors.append("memory_service_unavailable")
            return NodeResult(state=state)

        start = perf_counter()
        backfilled = 0
        user_service = getattr(self.memory_service, "user", None)
        if user_service is not None and hasattr(user_service, "backfill_missing_embeddings"):
            backfilled = user_service.backfill_missing_embeddings(user_id=state.turn.user_id)
        state.metadata["memory_embedding_backfilled"] = backfilled
        snapshot = self.memory_service.load(
            user_id=state.turn.user_id,
            session_id=state.turn.session_id,
            document_ids=state.turn.document_ids,
        )
        state.memory_snapshot = snapshot.snapshot
        state.memory_budget_audit = snapshot.audit

        active_facts = [fact for fact in snapshot.snapshot.user_facts if fact.status == FactStatus.ACTIVE]
        semantic_matches, threshold_used = self._semantic_retrieve_facts(state.turn.message, active_facts)
        selected_facts = [match[0] for match in semantic_matches]

        similarity_scores = [
            {
                "fact_id": fact.fact_id,
                "canonical_key": fact.canonical_key,
                "value": fact.value,
                "score": round(score, 4),
            }
            for fact, score in semantic_matches
        ]
        state.metadata["memory_similarity_scores"] = similarity_scores
        state.metadata["memory_query_mode"] = "semantic_only"
        state.metadata["memory_similarity_top_k"] = self.semantic_top_k
        state.metadata["memory_similarity_threshold"] = self.min_similarity_threshold
        state.metadata["memory_similarity_threshold_used"] = threshold_used

        state.turn.retrieved_memory = {
            "protected_facts": [
                {
                    "canonical_key": fact.canonical_key,
                    "value": fact.value,
                    "source": fact.source,
                    "reason": fact.reason,
                }
                for fact in snapshot.snapshot.summary.protected_facts
            ] if snapshot.snapshot.summary else [],
            "summary": snapshot.snapshot.summary.summary if snapshot.snapshot.summary else "",
            "pinned_facts": [
                {
                    "canonical_key": fact.canonical_key,
                    "value": fact.value,
                    "reason": fact.reason,
                }
                for fact in snapshot.snapshot.pinned_facts
            ],
            "user_facts": [
                {
                    "canonical_key": fact.canonical_key,
                    "value": fact.value,
                    "status": fact.status.value,
                }
                for fact in selected_facts
            ],
            "semantic_user_facts": [
                {
                    "canonical_key": fact.canonical_key,
                    "value": fact.value,
                    "score": round(score, 4),
                }
                for fact, score in semantic_matches
            ],
        }
        retrieved_facts = state.turn.retrieved_memory.get("user_facts", [])
        state.metadata["retrieved_facts"] = retrieved_facts
        logger.info("Retrieved facts: %s", retrieved_facts)
        state.metadata["memory_retrieval_duration_ms"] = int((perf_counter() - start) * 1000)
        return NodeResult(state=state)

    def _semantic_retrieve_facts(self, query: str, facts: list[MemoryFact]) -> tuple[list[tuple[MemoryFact, float]], float]:
        embedder = _get_query_embedding_provider()
        query_embedding = embedder.embed_text(query)
        print("Query embedding generated:", query_embedding[:5])
        logger.info("Query embedding generated: %s", query_embedding[:5])
        if not facts:
            print("Top retrieved facts:", [])
            logger.info("Top retrieved facts: []")
            return [], self.min_similarity_threshold
        scored: list[tuple[MemoryFact, float]] = []
        for fact in facts:
            if not fact.embedding:
                continue
            if len(fact.embedding) != len(query_embedding):
                continue
            score = cosine_similarity(query_embedding, fact.embedding)
            scored.append((fact, score))

        if not scored:
            print("Top retrieved facts:", [])
            logger.info("Top retrieved facts: []")
            return [], self.min_similarity_threshold

        matches = [item for item in scored if item[1] >= self.min_similarity_threshold]
        threshold_used = self.min_similarity_threshold
        if not matches:
            matches = [item for item in scored if item[1] >= self.min_fallback_similarity]
            threshold_used = self.min_fallback_similarity

        matches.sort(key=lambda item: item[1], reverse=True)
        matches = matches[: self.semantic_top_k]
        log_rows = [{"fact": fact.value, "score": round(score, 4)} for fact, score in matches]
        print("Top retrieved facts:", log_rows)
        logger.info("Top retrieved facts (threshold=%s): %s", threshold_used, log_rows)
        return matches, threshold_used

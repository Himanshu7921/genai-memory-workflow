from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import logging
import re

from app.memory.service import MemoryService
from app.memory.resolver import build_fact
from app.memory.protected_facts import build_protected_fact, extract_protected_fact_candidates, merge_protected_facts
from app.models.memory import FactScope, PinnedFact, ProtectedFact
from app.models.orchestration import OrchestrationState
from app.orchestrator.nodes.base import NodeResult
from app.retrieval.embeddings import EmbeddingProvider, HashEmbeddingProvider
from app.retrieval.embeddings_hf import HFEmbeddingProvider


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_fact_embedding_provider() -> EmbeddingProvider:
    try:
        return HFEmbeddingProvider()
    except Exception:
        logger.warning("fact_embedding_provider_fallback", exc_info=True)
        return HashEmbeddingProvider()


@dataclass(slots=True)
class MemoryWriteBackNode:
    name: str = "memory_write_back"
    memory_service: MemoryService | None = None
    max_facts_per_turn: int = 2

    def run(self, state: OrchestrationState) -> NodeResult:
        if state.memory_snapshot is None or self.memory_service is None:
            return NodeResult(state=state)

        newly_pinned = self._extract_pinned_facts(state)
        pinned_facts = self._merge_pinned_facts(
            existing=list(state.memory_snapshot.pinned_facts),
            incoming=newly_pinned,
        )
        newly_protected = self._extract_protected_facts(state)
        existing_protected = list(state.memory_snapshot.summary.protected_facts) if state.memory_snapshot.summary else []
        protected_facts = merge_protected_facts(
            existing_protected,
            newly_protected,
            max_items=self.memory_service.config.session_protected_facts_max,
        )

        facts_to_write = self._extract_user_facts(state)
        if facts_to_write and state.turn.user_id:
            write_result = self.memory_service.write_user_memory(
                user_id=state.turn.user_id,
                incoming_facts=facts_to_write,
                source="conversation",
            )
            state.metadata["facts_written"] = len(write_result.written)
            state.metadata["facts_superseded"] = len(write_result.superseded)
            state.metadata["facts_pruned"] = len(write_result.pruned)
            stored_facts = [
                {
                    "canonical_key": fact.canonical_key,
                    "value": fact.value,
                    "status": fact.status.value,
                }
                for fact in write_result.written
            ]
            state.metadata["stored_facts"] = stored_facts
            logger.info("Stored facts: %s", stored_facts)

        if state.turn.user_id and state.turn.session_id:
            session_state = self.memory_service.session.load_state(
                user_id=state.turn.user_id,
                session_id=state.turn.session_id,
            )
            turn_count_trigger = session_state.turns_since_write + 1
            context_chars = self._estimate_context_chars(state)
            budget_triggered = context_chars >= self.memory_service.config.session_context_char_budget
            plan = self.memory_service.plan_write_back(
                turns_since_write=turn_count_trigger,
                last_write_at=session_state.last_write_at,
                fact_delta_count=len(facts_to_write),
                pinned_changed=bool(newly_pinned),
            )
            write_summary = plan.write_session_summary or budget_triggered
            summary_text = state.memory_snapshot.summary.summary if state.memory_snapshot.summary else ""
            should_write_memory = write_summary or newly_pinned or newly_protected
            if should_write_memory:
                if write_summary:
                    summary_source = state.response.final_answer if state.response else state.turn.message
                    summary_text = self._build_summary(state, summary_source)
                self.memory_service.write_session_memory(
                    user_id=state.turn.user_id,
                    session_id=state.turn.session_id,
                    summary_text=summary_text,
                    pinned_facts=pinned_facts,
                    protected_facts=protected_facts,
                    turn_count_reset=write_summary,
                )

            refreshed = self.memory_service.load(
                user_id=state.turn.user_id,
                session_id=state.turn.session_id,
                document_ids=state.turn.document_ids,
            )
            state.memory_snapshot = refreshed.snapshot
            state.memory_budget_audit = refreshed.audit
            state.metadata["summary_updated"] = True
            state.metadata["summary_write_reason"] = "budget_trigger" if budget_triggered and not plan.write_session_summary else plan.reason
            state.metadata["summary_context_chars"] = context_chars
            state.metadata["summary_write_triggered"] = write_summary
            state.metadata["protected_facts_extracted"] = [fact.value for fact in newly_protected]
            state.metadata["protected_facts_count"] = len(protected_facts)

        state.metadata["write_back_snapshot_at"] = datetime.utcnow().isoformat()
        state.metadata["write_back_summary"] = state.memory_snapshot.summary.summary if state.memory_snapshot and state.memory_snapshot.summary else ""
        state.metadata["write_back_pinned_count"] = len(pinned_facts)
        state.metadata["write_back_protected_count"] = len(state.memory_snapshot.summary.protected_facts) if state.memory_snapshot and state.memory_snapshot.summary else 0
        state.metadata["write_back_facts"] = [fact.fact_id for fact in facts_to_write]
        return NodeResult(state=state)

    def _extract_user_facts(self, state: OrchestrationState):
        facts = []
        text = state.turn.message.strip()
        lower = text.lower()
        extracted_risk = self._extract_risk_tolerance(text)
        extracted_name = self._extract_name(text)
        extracted_age = self._extract_age(text)
        if extracted_name:
            facts.append(self._build_fact_with_embedding(state=state, key="name", value=extracted_name, priority=1.0))
        if extracted_risk:
            facts.append(self._build_fact_with_embedding(state=state, key="risk_tolerance", value=extracted_risk, priority=0.95))
        if extracted_age is not None:
            facts.append(self._build_fact_with_embedding(state=state, key="age", value=str(extracted_age), priority=0.95))
        if any(token in lower for token in ["my name is", "i am", "i'm"]):
            if len(facts) < self.max_facts_per_turn:
                facts.append(self._build_fact_with_embedding(state=state, key="self_identifier", value=text, priority=0.7))
        if any(token in lower for token in ["i prefer", "my preference is", "i like"]):
            facts.append(self._build_fact_with_embedding(state=state, key="preference", value=text, priority=0.8))
        return facts[: self.max_facts_per_turn]

    def _build_fact_with_embedding(self, *, state: OrchestrationState, key: str, value: str, priority: float):
        fact = build_fact(
            user_id=state.turn.user_id,
            scope=FactScope.USER,
            key=key,
            value=value,
            source="conversation",
            priority=priority,
        )
        try:
            embedding_text = f"{fact.canonical_key}: {fact.value}"
            fact.embedding = _get_fact_embedding_provider().embed_text(embedding_text)
        except Exception:
            logger.warning("fact_embedding_generation_failed", exc_info=True)
            fact.embedding = None
        return fact

    def _extract_name(self, text: str) -> str | None:
        match = re.search(
            r"\bmy\s+name\s+is\s+([A-Za-z][A-Za-z\s'.-]{0,80}?)(?=\s+(?:and\b|but\b|my\b)|[,.!?;]|$)",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None
        return match.group(1).strip(" .,!?:;\"'")

    def _extract_risk_tolerance(self, text: str) -> str | None:
        level_pattern = r"(low|medium|high|conservative|moderate|aggressive)"
        risk_term_pattern = r"(?:tolerance|tolerence|tollerance|profile)"
        normalized = re.sub(r"\s+", " ", text.strip().lower())

        patterns = [
            rf"\brisk\s+{risk_term_pattern}\s*(?:is|=|:)?\s*{level_pattern}\b",
            rf"\b{risk_term_pattern}\s+risk\s*(?:is|=|:)?\s*{level_pattern}\b",
            rf"\b(?:i\s+am|i'm|im)\s+{level_pattern}\s+on\s+risk\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, normalized, re.IGNORECASE)
            if match:
                return match.group(1).lower()

        return None

    def _extract_age(self, text: str) -> int | None:
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        patterns = [
            r"\b(?:i\s+am|i'm|im)\s+(\d{1,3})\s*(?:years?\s+old|yrs?\s+old|yo)?\b",
            r"\bmy\s+age\s+(?:is|=|:)\s*(\d{1,3})\b",
            r"\bage\s+(?:is|=|:)\s*(\d{1,3})\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, normalized, re.IGNORECASE)
            if not match:
                continue
            age = int(match.group(1))
            if 0 < age < 125:
                return age

        return None

    def _build_summary(self, state: OrchestrationState, response_text: str) -> str:
        base = state.memory_snapshot.summary.summary if state.memory_snapshot and state.memory_snapshot.summary else ""
        recent_turns = state.memory_snapshot.session_turns[-self.memory_service.config.session_recent_turns :] if state.memory_snapshot else []

        turn_lines = [
            f"{turn.role[:1].upper()}: {turn.content.strip()}"
            for turn in recent_turns
            if turn.content.strip()
        ]
        if response_text.strip():
            turn_lines.append(f"A: {response_text.strip()}")

        incremental = "\n".join(turn_lines)
        if not base:
            return incremental[-self.memory_service.config.session_summary_max_chars :]

        if incremental and incremental in base:
            return base[-self.memory_service.config.session_summary_max_chars :]

        merged = f"{base}\n{incremental}" if incremental else base
        return merged[-self.memory_service.config.session_summary_max_chars :]

    def _estimate_context_chars(self, state: OrchestrationState) -> int:
        summary_chars = len(state.memory_snapshot.summary.summary) if state.memory_snapshot and state.memory_snapshot.summary else 0
        turn_chars = sum(len(turn.content) for turn in state.memory_snapshot.session_turns) if state.memory_snapshot else 0
        pinned_chars = sum(len(fact.value) for fact in state.memory_snapshot.pinned_facts) if state.memory_snapshot else 0
        protected_chars = sum(len(fact.value) for fact in state.memory_snapshot.summary.protected_facts) if state.memory_snapshot and state.memory_snapshot.summary else 0
        return summary_chars + turn_chars + pinned_chars + protected_chars + len(state.turn.message)

    def _extract_protected_facts(self, state: OrchestrationState) -> list[ProtectedFact]:
        now = datetime.utcnow()
        extracted: list[ProtectedFact] = []
        seen: set[str] = set()

        source_texts: list[tuple[str, str]] = [("conversation", state.turn.message)]
        if state.retrieval_context and state.retrieval_context.chunks:
            for chunk in state.retrieval_context.chunks:
                source_texts.append(("retrieval", chunk.content))

        for source, text in source_texts:
            for candidate in extract_protected_fact_candidates(text, source=source):
                fact = build_protected_fact(user_id=state.turn.user_id, session_id=state.turn.session_id, candidate=candidate, now=now)
                marker = f"{fact.canonical_key}:{fact.value.lower()}"
                if marker in seen:
                    continue
                seen.add(marker)
                extracted.append(fact)

        return extracted

    def _extract_pinned_facts(self, state: OrchestrationState) -> list[PinnedFact]:
        text = state.turn.message.strip()
        normalized = re.sub(r"\s+", " ", text)
        now = datetime.utcnow()
        pinned: list[PinnedFact] = []

        # Explicit identifiers like account IDs, customer IDs, reference numbers.
        for match in re.finditer(
            r"\b(account|acct|customer\s*id|identifier|id|reference|ref)\b\s*(?:number|no\.?|id)?\s*(?:is|=|:)?\s*([A-Za-z0-9\-]{4,})",
            normalized,
            re.IGNORECASE,
        ):
            label = re.sub(r"\s+", "_", match.group(1).lower())
            value = match.group(2).strip()
            pinned.append(self._build_pinned_fact(state, f"identifier_{label}", value, "explicit_identifier", now))

        # Dates in ISO, slash, and dash formats.
        date_pattern = r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}-\d{1,2}-\d{2,4})\b"
        for match in re.finditer(date_pattern, normalized):
            pinned.append(self._build_pinned_fact(state, "date", match.group(0), "date_value", now))

        # Long numeric values likely to be account/reference numbers.
        for match in re.finditer(r"\b\d{6,}\b", normalized):
            pinned.append(self._build_pinned_fact(state, "numeric_identifier", match.group(0), "long_number", now))

        # Structured key-value entries: key: value.
        for match in re.finditer(r"\b([A-Za-z][A-Za-z0-9_\- ]{1,40})\s*:\s*([A-Za-z0-9\-_/]{2,80})", normalized):
            key = re.sub(r"\s+", "_", match.group(1).strip().lower())
            value = match.group(2).strip()
            pinned.append(self._build_pinned_fact(state, f"kv_{key}", value, "structured_value", now))

        return pinned

    def _build_pinned_fact(self, state: OrchestrationState, canonical_key: str, value: str, reason: str, now: datetime) -> PinnedFact:
        from hashlib import sha256

        digest = sha256(f"{state.turn.user_id}:{state.turn.session_id}:{canonical_key}:{value.lower()}".encode("utf-8")).hexdigest()[:24]
        return PinnedFact(
            fact_id=f"pinned_{digest}",
            user_id=state.turn.user_id,
            session_id=state.turn.session_id,
            canonical_key=canonical_key,
            value=value,
            reason=reason,
            created_at=now,
            updated_at=now,
            metadata={"source": "conversation"},
        )

    def _merge_pinned_facts(self, *, existing: list[PinnedFact], incoming: list[PinnedFact]) -> list[PinnedFact]:
        merged: dict[str, PinnedFact] = {f"{fact.canonical_key}:{fact.value}": fact for fact in existing}
        for fact in incoming:
            merged[f"{fact.canonical_key}:{fact.value}"] = fact
        return list(merged.values())

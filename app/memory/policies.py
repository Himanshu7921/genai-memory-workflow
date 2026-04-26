from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Iterable

from app.core.config import DEFAULT_MEMORY_CONFIG, MemoryConfig
from app.models.memory import FactStatus, MemoryFact, PinnedFact


@dataclass(frozen=True, slots=True)
class ResolutionEvent:
    canonical_key: str
    previous_fact_id: str | None
    new_fact_id: str
    action: str


@dataclass(frozen=True, slots=True)
class ThrottleDecision:
    should_write: bool
    reason: str


def normalize_fact_key(text: str) -> str:
    normalized = " ".join(text.lower().strip().split())
    return normalized.replace(" ", "_")


def make_fact_fingerprint(canonical_key: str, value: str) -> str:
    digest = sha256(f"{canonical_key}:{value.strip().lower()}".encode("utf-8")).hexdigest()
    return digest[:24]


def resolve_fact_contradictions(
    existing: Iterable[MemoryFact],
    incoming: Iterable[MemoryFact],
) -> tuple[list[MemoryFact], list[ResolutionEvent]]:
    resolved: list[MemoryFact] = []
    events: list[ResolutionEvent] = []
    active_by_key = {fact.canonical_key: fact for fact in existing if fact.status == FactStatus.ACTIVE}

    for fact in incoming:
        prior = active_by_key.get(fact.canonical_key)
        if prior is None:
            resolved.append(fact)
            active_by_key[fact.canonical_key] = fact
            events.append(ResolutionEvent(fact.canonical_key, None, fact.fact_id, "insert"))
            continue

        if prior.value.strip() == fact.value.strip():
            preserved = MemoryFact(
                fact_id=prior.fact_id,
                user_id=prior.user_id,
                scope=prior.scope,
                canonical_key=prior.canonical_key,
                value=prior.value,
                source=prior.source,
                confidence=max(prior.confidence, fact.confidence),
                priority=max(prior.priority, fact.priority),
                status=FactStatus.ACTIVE,
                created_at=prior.created_at,
                updated_at=datetime.utcnow(),
                last_accessed_at=datetime.utcnow(),
                supersedes_fact_id=prior.supersedes_fact_id,
                expires_at=prior.expires_at,
                metadata={**prior.metadata, **fact.metadata},
                embedding=prior.embedding or fact.embedding,
            )
            resolved.append(preserved)
            active_by_key[fact.canonical_key] = preserved
            events.append(ResolutionEvent(fact.canonical_key, prior.fact_id, prior.fact_id, "refresh"))
            continue

        prior.status = FactStatus.SUPERSEDED
        prior.updated_at = datetime.utcnow()
        prior.expires_at = datetime.utcnow()
        resolved.append(prior)
        replacement = MemoryFact(
                fact_id=fact.fact_id,
                user_id=fact.user_id,
                scope=fact.scope,
                canonical_key=fact.canonical_key,
                value=fact.value,
                source=fact.source,
                confidence=fact.confidence,
                priority=max(fact.priority, prior.priority),
                status=FactStatus.ACTIVE,
                created_at=fact.created_at,
                updated_at=datetime.utcnow(),
                last_accessed_at=datetime.utcnow(),
                supersedes_fact_id=prior.fact_id,
                expires_at=fact.expires_at,
                metadata={**prior.metadata, **fact.metadata},
                embedding=fact.embedding or prior.embedding,
            )
        resolved.append(replacement)
        active_by_key[fact.canonical_key] = replacement
        events.append(ResolutionEvent(fact.canonical_key, prior.fact_id, fact.fact_id, "supersede"))

    return resolved, events


def should_write_back(
    turns_since_write: int,
    elapsed_since_write: float,
    *,
    config: MemoryConfig = DEFAULT_MEMORY_CONFIG,
    pinned_changed: bool = False,
    fact_delta_count: int = 0,
) -> ThrottleDecision:
    if pinned_changed:
        return ThrottleDecision(True, "pinned_facts_changed")
    if fact_delta_count >= 2:
        return ThrottleDecision(True, "fact_delta_threshold")
    if turns_since_write >= config.session_summary_trigger_turns:
        return ThrottleDecision(True, "turn_threshold")
    if elapsed_since_write >= config.session_write_throttle_seconds:
        return ThrottleDecision(True, "elapsed_threshold")
    return ThrottleDecision(False, "throttled")


def compute_fact_decay_score(fact: MemoryFact, *, now: datetime | None = None) -> float:
    now = now or datetime.utcnow()
    age_days = max((now - fact.last_accessed_at).total_seconds() / 86400.0, 0.0)
    recency_component = max(0.0, 1.0 - age_days / 30.0)
    return round((0.5 * fact.priority) + (0.3 * fact.confidence) + (0.2 * recency_component), 4)


def should_prune_fact(fact: MemoryFact, *, config: MemoryConfig = DEFAULT_MEMORY_CONFIG) -> bool:
    now = datetime.utcnow()
    age = now - fact.last_accessed_at
    if fact.status != FactStatus.ACTIVE:
        return False
    if fact.priority <= config.prune_low_priority_below and age > timedelta(days=config.fact_decay_days):
        return True
    if fact.scope.name == "SESSION" and age > timedelta(days=config.fact_decay_days):
        return True
    if fact.scope.name == "USER" and age > timedelta(days=config.fact_decay_days * 3):
        return fact.priority < 0.4
    return False


def should_keep_pinned_fact(pinned_fact: PinnedFact, *, config: MemoryConfig = DEFAULT_MEMORY_CONFIG) -> bool:
    age = datetime.utcnow() - pinned_fact.updated_at
    return age <= timedelta(days=config.pinned_fact_decay_days)

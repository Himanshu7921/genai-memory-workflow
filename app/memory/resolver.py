from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from app.memory.policies import make_fact_fingerprint, normalize_fact_key, resolve_fact_contradictions
from app.models.memory import FactScope, MemoryFact


def build_fact(
    *,
    user_id: str,
    scope: FactScope,
    key: str,
    value: str,
    source: str,
    confidence: float = 1.0,
    priority: float = 1.0,
    metadata: dict | None = None,
) -> MemoryFact:
    canonical_key = normalize_fact_key(key)
    return MemoryFact(
        fact_id=make_fact_fingerprint(canonical_key, value) + "_" + uuid4().hex[:8],
        user_id=user_id,
        scope=scope,
        canonical_key=canonical_key,
        value=value.strip(),
        source=source,
        confidence=confidence,
        priority=priority,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        last_accessed_at=datetime.utcnow(),
        metadata=metadata or {},
    )


def resolve_facts(existing: list[MemoryFact], incoming: list[MemoryFact]) -> tuple[list[MemoryFact], list]:
    return resolve_fact_contradictions(existing, incoming)

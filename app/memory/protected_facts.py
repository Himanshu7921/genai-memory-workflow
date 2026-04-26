from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
import re
from typing import Any

from app.models.memory import FactStatus, ProtectedFact


@dataclass(slots=True)
class ProtectedFactCandidate:
    canonical_key: str
    value: str
    source: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)
    fact_id: str | None = None


_DATE_PATTERN = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}-\d{1,2}-\d{2,4})\b")
_DURATION_PATTERN = re.compile(r"\b\d+\s*(?:hours?|days?|months?|years?|weeks?|mins?|minutes?|hrs?)\b", re.IGNORECASE)
_MONEY_PATTERN = re.compile(
    r"(?:[$€£]\s?\d[\d,]*(?:\.\d{1,2})?|\b\d[\d,]*(?:\.\d{1,2})?\s?(?:usd|eur|gbp|dollars?|euros?|pounds?)\b)",
    re.IGNORECASE,
)
_IDENTIFIER_PATTERN = re.compile(
    r"\b(?:account|acct|customer\s*id|client\s*id|identifier|id|reference|ref|order|invoice|ticket|case)\b\s*(?:number|no\.?|#|id)?\s*(?:is|=|:)?\s*([A-Za-z0-9\-]{4,})",
    re.IGNORECASE,
)
_STRUCTURED_PATTERN = re.compile(r"\b([A-Za-z][A-Za-z0-9_\- ]{1,40})\s*:\s*([^\n;,]{2,120})")
_AGE_PATTERN = re.compile(
    r"\b(?:my\s+age\s+(?:is|=|:)?|age\s+(?:is|=|:)?|i\s+am|i'm|im)\s*(\d{1,3})\b",
    re.IGNORECASE,
)


def extract_protected_facts(text: str) -> list[str]:
    return [candidate.value for candidate in extract_protected_fact_candidates(text)]


def extract_protected_fact_candidates(text: str, *, source: str = "conversation") -> list[ProtectedFactCandidate]:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []

    candidates: list[ProtectedFactCandidate] = []
    occupied_spans: list[tuple[int, int]] = []
    seen: set[tuple[str, str]] = set()

    def add_candidate(*, canonical_key: str, value: str, reason: str, metadata: dict[str, Any] | None = None) -> None:
        cleaned = value.strip()
        if not cleaned:
            return
        marker = (canonical_key, cleaned.lower())
        if marker in seen:
            return
        seen.add(marker)
        candidates.append(
            ProtectedFactCandidate(
                canonical_key=canonical_key,
                value=cleaned,
                source=source,
                reason=reason,
                metadata=metadata or {},
            )
        )

    for match in _STRUCTURED_PATTERN.finditer(normalized):
        label = _normalize_key(match.group(1))
        value = match.group(2).strip().rstrip(".?!")
        occupied_spans.append(match.span())
        add_candidate(
            canonical_key=f"kv_{label}",
            value=value,
            reason="structured_value",
            metadata={"label": label},
        )

    for match in _IDENTIFIER_PATTERN.finditer(normalized):
        value = match.group(1).strip()
        occupied_spans.append(match.span())
        add_candidate(
            canonical_key="identifier",
            value=value,
            reason="identifier_value",
        )

    for match in _DATE_PATTERN.finditer(normalized):
        value = match.group(0).strip()
        occupied_spans.append(match.span())
        add_candidate(canonical_key="date", value=value, reason="date_value")

    for match in _DURATION_PATTERN.finditer(normalized):
        value = match.group(0).strip()
        occupied_spans.append(match.span())
        add_candidate(canonical_key="duration", value=value, reason="duration_value")

    for match in _MONEY_PATTERN.finditer(normalized):
        value = match.group(0).strip()
        occupied_spans.append(match.span())
        add_candidate(canonical_key="money", value=value, reason="monetary_value")

    for match in _AGE_PATTERN.finditer(normalized):
        age = match.group(1).strip()
        if 0 < int(age) < 125:
            occupied_spans.append(match.span())
            add_candidate(canonical_key="age", value=age, reason="explicit_user_fact")

    for match in re.finditer(r"\b\d+\b", normalized):
        if any(start <= match.start() and match.end() <= end for start, end in occupied_spans):
            continue
        value = match.group(0).strip()
        add_candidate(canonical_key=f"number_{value}", value=value, reason="number_value")

    return candidates


def build_protected_fact(*, user_id: str, session_id: str, candidate: ProtectedFactCandidate, existing_fact_id: str | None = None, supersedes_fact_id: str | None = None, now: datetime | None = None) -> ProtectedFact:
    now = now or datetime.utcnow()
    digest = sha256(f"{user_id}:{session_id}:{candidate.canonical_key}:{candidate.value.lower()}".encode("utf-8")).hexdigest()[:24]
    return ProtectedFact(
        fact_id=existing_fact_id or f"protected_{digest}",
        user_id=user_id,
        session_id=session_id,
        canonical_key=candidate.canonical_key,
        value=candidate.value,
        source=candidate.source,
        reason=candidate.reason,
        supersedes_fact_id=supersedes_fact_id,
        status=FactStatus.ACTIVE,
        created_at=now,
        updated_at=now,
        metadata=candidate.metadata,
    )


def merge_protected_facts(existing: list[ProtectedFact], incoming: list[ProtectedFact], *, max_items: int | None = None) -> list[ProtectedFact]:
    # Canonical-key merge with latest fact winning, preserving deterministic ordering.
    merged_by_key: dict[str, ProtectedFact] = {}

    for fact in existing:
        if fact.status != FactStatus.ACTIVE:
            continue
        merged_by_key[fact.canonical_key] = fact

    for fact in incoming:
        if fact.status != FactStatus.ACTIVE:
            continue
        prior = merged_by_key.get(fact.canonical_key)
        if prior is not None and prior.value.strip() == fact.value.strip():
            merged_by_key[fact.canonical_key] = ProtectedFact(
                fact_id=prior.fact_id,
                user_id=prior.user_id,
                session_id=prior.session_id,
                canonical_key=prior.canonical_key,
                value=prior.value,
                source=fact.source,
                reason=fact.reason,
                supersedes_fact_id=prior.supersedes_fact_id,
                status=FactStatus.ACTIVE,
                created_at=prior.created_at,
                updated_at=fact.updated_at,
                metadata={**prior.metadata, **fact.metadata},
            )
            continue

        merged_by_key[fact.canonical_key] = ProtectedFact(
            fact_id=fact.fact_id,
            user_id=fact.user_id,
            session_id=fact.session_id,
            canonical_key=fact.canonical_key,
            value=fact.value,
            source=fact.source,
            reason=fact.reason,
            supersedes_fact_id=prior.fact_id if prior else fact.supersedes_fact_id,
            status=FactStatus.ACTIVE,
            created_at=fact.created_at,
            updated_at=fact.updated_at,
            metadata={**(prior.metadata if prior else {}), **fact.metadata},
        )

    merged = sorted(merged_by_key.values(), key=lambda item: item.updated_at, reverse=True)
    if max_items is not None and max_items > 0:
        return merged[:max_items]
    return merged


def _normalize_key(text: str) -> str:
    normalized = " ".join(text.lower().strip().split())
    return normalized.replace(" ", "_")
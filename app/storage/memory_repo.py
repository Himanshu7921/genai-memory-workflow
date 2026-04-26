from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import uuid4

from app.models.memory import CorpusChunkRecord, FactScope, FactStatus, MemoryFact, PinnedFact, ProtectedFact, SessionSummary, SessionTurn
from app.storage.sqlite import SQLiteStore


def _dump_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _load_json(raw: str | None) -> dict[str, Any]:
    return json.loads(raw or "{}")


def _parse_dt(raw: str | None) -> datetime | None:
    if raw in (None, ""):
        return None
    return datetime.fromisoformat(raw)


class MemoryRepository:
    def __init__(self, store: SQLiteStore | None = None) -> None:
        self.store = store or SQLiteStore()

    def new_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid4().hex}"

    def get_session_row(self, *, user_id: str, session_id: str) -> dict[str, Any] | None:
        with self.store.read_only() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            ).fetchone()
            return dict(row) if row else None

    def get_session_summary(self, *, user_id: str, session_id: str) -> SessionSummary | None:
        row = self.get_session_row(user_id=user_id, session_id=session_id)
        if not row:
            return None
        pinned = self.list_session_pinned_facts(user_id=user_id, session_id=session_id)
        protected = self.list_session_protected_facts(user_id=user_id, session_id=session_id)
        return SessionSummary(
            session_id=session_id,
            user_id=user_id,
            summary=row["summary"],
            pinned_facts=pinned,
            protected_facts=protected,
            turn_count=row["turn_count"],
            updated_at=_parse_dt(row["updated_at"]) or datetime.utcnow(),
        )

    def list_recent_turns(self, *, user_id: str, session_id: str, limit: int) -> list[SessionTurn]:
        with self.store.read_only() as connection:
            rows = connection.execute(
                """
                SELECT * FROM session_turns
                WHERE user_id = ? AND session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, session_id, limit),
            ).fetchall()
        turns = [
            SessionTurn(
                turn_id=row["turn_id"],
                user_id=row["user_id"],
                session_id=row["session_id"],
                role=row["role"],
                content=row["content"],
                created_at=_parse_dt(row["created_at"]) or datetime.utcnow(),
                metadata=_load_json(row["metadata_json"]),
            )
            for row in rows
        ]
        return list(reversed(turns))

    def count_session_turns(self, *, user_id: str, session_id: str) -> int:
        with self.store.read_only() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM session_turns
                WHERE user_id = ? AND session_id = ?
                """,
                (user_id, session_id),
            ).fetchone()
        return int(row["count"] if row else 0)

    def prune_session_turns(self, *, user_id: str, session_id: str, keep_latest: int) -> list[str]:
        total = self.count_session_turns(user_id=user_id, session_id=session_id)
        to_delete = max(total - max(keep_latest, 0), 0)
        if to_delete <= 0:
            return []

        with self.store.transaction() as connection:
            rows = connection.execute(
                """
                SELECT turn_id
                FROM session_turns
                WHERE user_id = ? AND session_id = ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (user_id, session_id, to_delete),
            ).fetchall()
            turn_ids = [row["turn_id"] for row in rows]
            if turn_ids:
                placeholders = ",".join("?" for _ in turn_ids)
                connection.execute(
                    f"DELETE FROM session_turns WHERE turn_id IN ({placeholders})",
                    tuple(turn_ids),
                )
        return turn_ids

    def append_turn(self, turn: SessionTurn) -> None:
        with self.store.transaction() as connection:
            connection.execute(
                """
                INSERT INTO session_turns(turn_id, user_id, session_id, role, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn.turn_id,
                    turn.user_id,
                    turn.session_id,
                    turn.role,
                    turn.content,
                    _dump_json(turn.metadata),
                    turn.created_at.isoformat(),
                ),
            )
            connection.execute(
                """
                INSERT INTO sessions(session_id, user_id, summary, summary_updated_at, turn_count, last_write_at, created_at, updated_at)
                VALUES(?, ?, '', ?, 0, ?, ?, ?)
                ON CONFLICT(session_id, user_id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (
                    turn.session_id,
                    turn.user_id,
                    turn.created_at.isoformat(),
                    turn.created_at.isoformat(),
                    turn.created_at.isoformat(),
                    turn.created_at.isoformat(),
                ),
            )

    def bump_session_turn_count(self, *, user_id: str, session_id: str) -> None:
        with self.store.transaction() as connection:
            connection.execute(
                """
                UPDATE sessions
                SET turn_count = turn_count + 1,
                    updated_at = ?
                WHERE user_id = ? AND session_id = ?
                """,
                (datetime.utcnow().isoformat(), user_id, session_id),
            )

    def reset_session_write_counters(self, *, user_id: str, session_id: str) -> None:
        now = datetime.utcnow().isoformat()
        with self.store.transaction() as connection:
            connection.execute(
                """
                UPDATE sessions
                SET turn_count = 0,
                    last_write_at = ?,
                    updated_at = ?
                WHERE user_id = ? AND session_id = ?
                """,
                (now, now, user_id, session_id),
            )

    def list_session_pinned_facts(self, *, user_id: str, session_id: str) -> list[PinnedFact]:
        with self.store.read_only() as connection:
            rows = connection.execute(
                """
                SELECT * FROM session_pinned_facts
                WHERE user_id = ? AND session_id = ?
                ORDER BY updated_at DESC
                """,
                (user_id, session_id),
            ).fetchall()
        return [
            PinnedFact(
                fact_id=row["fact_id"],
                user_id=row["user_id"],
                session_id=row["session_id"],
                canonical_key=row["canonical_key"],
                value=row["value"],
                reason=row["reason"],
                created_at=_parse_dt(row["created_at"]) or datetime.utcnow(),
                updated_at=_parse_dt(row["updated_at"]) or datetime.utcnow(),
                metadata=_load_json(row["metadata_json"]),
            )
            for row in rows
        ]

    def list_session_protected_facts(self, *, user_id: str, session_id: str) -> list[ProtectedFact]:
        with self.store.read_only() as connection:
            rows = connection.execute(
                """
                SELECT * FROM session_protected_facts
                WHERE user_id = ? AND session_id = ? AND status = ?
                ORDER BY updated_at DESC
                """,
                (user_id, session_id, FactStatus.ACTIVE.value),
            ).fetchall()
        return [self._row_to_protected_fact(row) for row in rows]

    def upsert_session_summary(self, *, user_id: str, session_id: str, summary: str, pinned_facts: list[PinnedFact], protected_facts: list[ProtectedFact] | None = None, updated_at: datetime) -> None:
        with self.store.transaction() as connection:
            connection.execute(
                """
                INSERT INTO sessions(session_id, user_id, summary, summary_updated_at, turn_count, last_write_at, created_at, updated_at)
                VALUES(?, ?, ?, ?, 0, ?, ?, ?)
                ON CONFLICT(session_id, user_id) DO UPDATE SET
                    summary = excluded.summary,
                    summary_updated_at = excluded.summary_updated_at,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    user_id,
                    summary,
                    updated_at.isoformat(),
                    updated_at.isoformat(),
                    updated_at.isoformat(),
                    updated_at.isoformat(),
                ),
            )
            for pinned_fact in pinned_facts:
                connection.execute(
                    """
                    INSERT INTO session_pinned_facts(fact_id, user_id, session_id, canonical_key, value, reason, metadata_json, created_at, updated_at)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(fact_id) DO UPDATE SET
                        canonical_key = excluded.canonical_key,
                        value = excluded.value,
                        reason = excluded.reason,
                        metadata_json = excluded.metadata_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        pinned_fact.fact_id,
                        pinned_fact.user_id,
                        pinned_fact.session_id,
                        pinned_fact.canonical_key,
                        pinned_fact.value,
                        pinned_fact.reason,
                        _dump_json(pinned_fact.metadata),
                        pinned_fact.created_at.isoformat(),
                        pinned_fact.updated_at.isoformat(),
                    ),
                )
            self._upsert_session_protected_facts(
                connection,
                user_id=user_id,
                session_id=session_id,
                protected_facts=protected_facts or [],
                updated_at=updated_at,
            )

    def _upsert_session_protected_facts(self, connection: Any, *, user_id: str, session_id: str, protected_facts: list[ProtectedFact], updated_at: datetime) -> None:
        if not protected_facts:
            return

        existing_rows = connection.execute(
            """
            SELECT * FROM session_protected_facts
            WHERE user_id = ? AND session_id = ? AND status = ?
            ORDER BY updated_at DESC
            """,
            (user_id, session_id, FactStatus.ACTIVE.value),
        ).fetchall()
        active_by_key = {row["canonical_key"]: row for row in existing_rows}

        for protected_fact in protected_facts:
            prior_row = active_by_key.get(protected_fact.canonical_key)
            if prior_row is not None and prior_row["value"].strip() != protected_fact.value.strip():
                connection.execute(
                    """
                    UPDATE session_protected_facts
                    SET status = ?,
                        updated_at = ?
                    WHERE fact_id = ?
                    """,
                    (FactStatus.SUPERSEDED.value, updated_at.isoformat(), prior_row["fact_id"]),
                )
            elif prior_row is not None:
                connection.execute(
                    """
                    UPDATE session_protected_facts
                    SET source = ?,
                        reason = ?,
                        metadata_json = ?,
                        updated_at = ?
                    WHERE fact_id = ?
                    """,
                    (
                        protected_fact.source,
                        protected_fact.reason,
                        _dump_json(protected_fact.metadata),
                        updated_at.isoformat(),
                        prior_row["fact_id"],
                    ),
                )
                active_by_key[protected_fact.canonical_key] = prior_row
                continue

            connection.execute(
                """
                INSERT INTO session_protected_facts(fact_id, user_id, session_id, canonical_key, value, source, reason, status, supersedes_fact_id, metadata_json, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fact_id) DO UPDATE SET
                    canonical_key = excluded.canonical_key,
                    value = excluded.value,
                    source = excluded.source,
                    reason = excluded.reason,
                    status = excluded.status,
                    supersedes_fact_id = excluded.supersedes_fact_id,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    protected_fact.fact_id,
                    protected_fact.user_id,
                    protected_fact.session_id,
                    protected_fact.canonical_key,
                    protected_fact.value,
                    protected_fact.source,
                    protected_fact.reason,
                    protected_fact.status.value,
                    protected_fact.supersedes_fact_id,
                    _dump_json(protected_fact.metadata),
                    protected_fact.created_at.isoformat(),
                    protected_fact.updated_at.isoformat(),
                ),
            )
            active_by_key[protected_fact.canonical_key] = {
                "fact_id": protected_fact.fact_id,
                "value": protected_fact.value,
            }

    def list_user_facts(self, *, user_id: str) -> list[MemoryFact]:
        with self.store.read_only() as connection:
            rows = connection.execute(
                """
                SELECT * FROM user_facts
                WHERE user_id = ?
                ORDER BY updated_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [self._row_to_user_fact(row) for row in rows]

    def upsert_user_fact(self, fact: MemoryFact) -> None:
        with self.store.transaction() as connection:
            connection.execute(
                """
                INSERT INTO user_facts(
                    fact_id, user_id, scope, canonical_key, value, source,
                    confidence, priority, status, supersedes_fact_id, metadata_json,
                    created_at, updated_at, last_accessed_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fact_id) DO UPDATE SET
                    value = excluded.value,
                    source = excluded.source,
                    confidence = excluded.confidence,
                    priority = excluded.priority,
                    status = excluded.status,
                    supersedes_fact_id = excluded.supersedes_fact_id,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at,
                    last_accessed_at = excluded.last_accessed_at,
                    expires_at = excluded.expires_at
                """,
                (
                    fact.fact_id,
                    fact.user_id,
                    fact.scope.value,
                    fact.canonical_key,
                    fact.value,
                    fact.source,
                    fact.confidence,
                    fact.priority,
                    fact.status.value,
                    fact.supersedes_fact_id,
                    _dump_json(fact.metadata),
                    fact.created_at.isoformat(),
                    fact.updated_at.isoformat(),
                    fact.last_accessed_at.isoformat(),
                    fact.expires_at.isoformat() if fact.expires_at else None,
                ),
            )

    def archive_user_fact(self, fact_id: str) -> None:
        now = datetime.utcnow().isoformat()
        with self.store.transaction() as connection:
            connection.execute(
                """
                UPDATE user_facts
                SET status = ?,
                    updated_at = ?,
                    expires_at = ?
                WHERE fact_id = ?
                """,
                (FactStatus.ARCHIVED.value, now, now, fact_id),
            )

    def _row_to_user_fact(self, row: Any) -> MemoryFact:
        return MemoryFact(
            fact_id=row["fact_id"],
            user_id=row["user_id"],
            scope=FactScope(row["scope"]),
            canonical_key=row["canonical_key"],
            value=row["value"],
            source=row["source"],
            confidence=row["confidence"],
            priority=row["priority"],
            status=FactStatus(row["status"]),
            supersedes_fact_id=row["supersedes_fact_id"],
            metadata=_load_json(row["metadata_json"]),
            created_at=_parse_dt(row["created_at"]) or datetime.utcnow(),
            updated_at=_parse_dt(row["updated_at"]) or datetime.utcnow(),
            last_accessed_at=_parse_dt(row["last_accessed_at"]) or datetime.utcnow(),
            expires_at=_parse_dt(row["expires_at"]),
        )

    def _row_to_protected_fact(self, row: Any) -> ProtectedFact:
        return ProtectedFact(
            fact_id=row["fact_id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            canonical_key=row["canonical_key"],
            value=row["value"],
            source=row["source"],
            reason=row["reason"],
            supersedes_fact_id=row["supersedes_fact_id"],
            status=FactStatus(row["status"]),
            created_at=_parse_dt(row["created_at"]) or datetime.utcnow(),
            updated_at=_parse_dt(row["updated_at"]) or datetime.utcnow(),
            metadata=_load_json(row["metadata_json"]),
        )

    def upsert_document(self, *, document_id: str, user_id: str | None, source_uri: str | None, title: str | None, content_hash: str, metadata: dict[str, Any]) -> None:
        now = datetime.utcnow().isoformat()
        with self.store.transaction() as connection:
            connection.execute(
                """
                INSERT INTO corpus_documents(document_id, user_id, source_uri, title, content_hash, metadata_json, created_at, updated_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active')
                ON CONFLICT(document_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    source_uri = excluded.source_uri,
                    title = excluded.title,
                    content_hash = excluded.content_hash,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at,
                    status = excluded.status
                """,
                (document_id, user_id, source_uri, title, content_hash, _dump_json(metadata), now, now),
            )

    def upsert_chunk(self, chunk: CorpusChunkRecord) -> None:
        with self.store.transaction() as connection:
            connection.execute(
                """
                INSERT INTO corpus_chunks(
                    chunk_id, document_id, user_id, chunk_index, content, content_hash,
                    embedding_model, metadata_json, created_at, updated_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    document_id = excluded.document_id,
                    user_id = excluded.user_id,
                    chunk_index = excluded.chunk_index,
                    content = excluded.content,
                    content_hash = excluded.content_hash,
                    embedding_model = excluded.embedding_model,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at,
                    status = excluded.status
                """,
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.user_id,
                    chunk.chunk_index,
                    chunk.content,
                    chunk.content_hash,
                    None,
                    _dump_json(chunk.metadata),
                    chunk.created_at.isoformat(),
                    chunk.updated_at.isoformat(),
                    chunk.status,
                ),
            )

    def list_chunks_for_documents(self, document_ids: list[str]) -> list[CorpusChunkRecord]:
        if not document_ids:
            return []
        placeholders = ",".join("?" for _ in document_ids)
        with self.store.read_only() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM corpus_chunks
                WHERE document_id IN ({placeholders})
                ORDER BY document_id, chunk_index
                """,
                tuple(document_ids),
            ).fetchall()
        return [
            CorpusChunkRecord(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                user_id=row["user_id"],
                chunk_index=row["chunk_index"],
                content=row["content"],
                content_hash=row["content_hash"],
                metadata=_load_json(row["metadata_json"]),
                created_at=_parse_dt(row["created_at"]) or datetime.utcnow(),
                updated_at=_parse_dt(row["updated_at"]) or datetime.utcnow(),
                status=row["status"],
            )
            for row in rows
        ]

    def list_all_chunks(self, *, user_id: str | None = None) -> list[CorpusChunkRecord]:
        with self.store.read_only() as connection:
            if user_id is None:
                rows = connection.execute(
                    """
                    SELECT * FROM corpus_chunks
                    ORDER BY document_id, chunk_index
                    """
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM corpus_chunks
                    WHERE user_id = ? OR user_id IS NULL
                    ORDER BY document_id, chunk_index
                    """,
                    (user_id,),
                ).fetchall()
        return [
            CorpusChunkRecord(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                user_id=row["user_id"],
                chunk_index=row["chunk_index"],
                content=row["content"],
                content_hash=row["content_hash"],
                metadata=_load_json(row["metadata_json"]),
                created_at=_parse_dt(row["created_at"]) or datetime.utcnow(),
                updated_at=_parse_dt(row["updated_at"]) or datetime.utcnow(),
                status=row["status"],
            )
            for row in rows
        ]

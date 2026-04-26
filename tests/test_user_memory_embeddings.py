from __future__ import annotations

from app.core.config import MemoryConfig
from app.memory.service import MemoryService
from app.memory.resolver import build_fact
from app.models.memory import FactScope
from app.storage.memory_repo import MemoryRepository
from app.storage.sqlite import SQLiteStore


def make_service(tmp_path) -> MemoryService:
    store = SQLiteStore(db_path=tmp_path / "memory.sqlite3")
    return MemoryService(MemoryRepository(store), MemoryConfig())


def test_backfill_missing_embeddings_for_legacy_facts(tmp_path) -> None:
    service = make_service(tmp_path)
    fact = build_fact(
        user_id="u_test",
        scope=FactScope.USER,
        key="favorite",
        value="Itachi",
        source="conversation",
        priority=0.9,
    )
    fact.embedding = None
    service.repository.upsert_user_fact(fact)

    backfilled = service.user.backfill_missing_embeddings(user_id="u_test")
    assert backfilled == 1

    loaded = service.repository.list_user_facts(user_id="u_test")
    assert loaded
    assert loaded[0].embedding is not None
    assert len(loaded[0].embedding) > 0

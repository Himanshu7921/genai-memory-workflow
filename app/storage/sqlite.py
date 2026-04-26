from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.core.errors import StorageError
from app.storage.schema import SCHEMA_STATEMENTS


class SQLiteStore:
    def __init__(self, db_path: str | Path = "data/memory.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            for statement in SCHEMA_STATEMENTS:
                connection.execute(statement)
            self._migrate_schema(connection)
            connection.commit()
        except sqlite3.Error as exc:  # pragma: no cover - defensive wrapper
            raise StorageError(f"failed to initialize sqlite store: {exc}") from exc
        finally:
            connection.close()

    def _migrate_schema(self, connection: sqlite3.Connection) -> None:
        columns = connection.execute("PRAGMA table_info(user_facts)").fetchall()
        column_names = {str(column[1]) for column in columns}
        if "embedding" not in column_names:
            connection.execute("ALTER TABLE user_facts ADD COLUMN embedding BLOB")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except sqlite3.Error as exc:
            connection.rollback()
            raise StorageError(f"sqlite transaction failed: {exc}") from exc
        finally:
            connection.close()

    @contextmanager
    def read_only(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

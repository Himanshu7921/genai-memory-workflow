from __future__ import annotations


SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        summary TEXT NOT NULL DEFAULT '',
        summary_updated_at TEXT,
        turn_count INTEGER NOT NULL DEFAULT 0,
        last_write_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (session_id, user_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS session_turns (
        turn_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS session_pinned_facts (
        fact_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        canonical_key TEXT NOT NULL,
        value TEXT NOT NULL,
        reason TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS session_protected_facts (
        fact_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        canonical_key TEXT NOT NULL,
        value TEXT NOT NULL,
        source TEXT NOT NULL,
        reason TEXT NOT NULL,
        status TEXT NOT NULL,
        supersedes_fact_id TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_session_protected_facts_lookup
    ON session_protected_facts(user_id, session_id, status, canonical_key)
    """,
    """
    CREATE TABLE IF NOT EXISTS user_facts (
        fact_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        scope TEXT NOT NULL,
        canonical_key TEXT NOT NULL,
        value TEXT NOT NULL,
        embedding BLOB,
        source TEXT NOT NULL,
        confidence REAL NOT NULL,
        priority REAL NOT NULL,
        status TEXT NOT NULL,
        supersedes_fact_id TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        last_accessed_at TEXT NOT NULL,
        expires_at TEXT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_user_facts_user_status
    ON user_facts(user_id, status, canonical_key)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_session_turns_lookup
    ON session_turns(user_id, session_id, created_at DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS corpus_documents (
        document_id TEXT PRIMARY KEY,
        user_id TEXT,
        source_uri TEXT,
        title TEXT,
        content_hash TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS corpus_chunks (
        chunk_id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL,
        user_id TEXT,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        embedding_model TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active'
    )
    """,
)

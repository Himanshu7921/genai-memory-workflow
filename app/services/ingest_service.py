from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import logging
from pathlib import Path
from typing import Any

from app.retrieval.chunking import ChunkPlan
from app.retrieval.service import RetrievalService
from app.retrieval.vector_store import InMemoryVectorStore


logger = logging.getLogger(__name__)

_ROOT_DIR = Path(__file__).resolve().parents[2]
_RAG_DOCS_DIR = _ROOT_DIR / "rag_docs"
_INGEST_STATE_PATH = _RAG_DOCS_DIR / ".ingest_state.json"

_SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}
_DEFAULT_CHUNK_PLAN = ChunkPlan(max_chars=700, overlap_chars=100, min_chunk_chars=120)

_retrieval_service: RetrievalService | None = None


@dataclass(slots=True)
class IngestOutcome:
    document_id: str
    file_name: str
    indexed: bool
    chunk_count: int = 0
    content_hash: str = ""
    reason: str = ""


def configure_ingest_service(retrieval_service: RetrievalService) -> None:
    global _retrieval_service
    _retrieval_service = retrieval_service


def ingest_all_documents() -> dict[str, Any]:
    service = _require_retrieval_service()
    _RAG_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    force_reindex = isinstance(service.vector_store, InMemoryVectorStore)

    previous_state = _load_ingest_state()
    next_state: dict[str, str] = {}
    outcomes: list[IngestOutcome] = []

    for path in sorted(_RAG_DOCS_DIR.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            outcomes.append(
                IngestOutcome(
                    document_id=path.name,
                    file_name=path.name,
                    indexed=False,
                    reason=f"unsupported_file_type:{path.suffix.lower()}",
                )
            )
            continue

        raw_bytes = path.read_bytes()
        file_hash = sha256(raw_bytes).hexdigest()
        next_state[path.name] = file_hash

        if (not force_reindex) and previous_state.get(path.name) == file_hash:
            outcomes.append(
                IngestOutcome(
                    document_id=path.name,
                    file_name=path.name,
                    indexed=False,
                    content_hash=file_hash,
                    reason="unchanged",
                )
            )
            continue

        outcome = ingest_file(str(path), retrieval_service=service)
        outcomes.append(outcome)

    _save_ingest_state(next_state)
    indexed_count = sum(1 for item in outcomes if item.indexed)
    logger.info("ingest_all_documents_completed", extra={"indexed": indexed_count, "total": len(outcomes)})
    return {
        "rag_docs_path": str(_RAG_DOCS_DIR),
        "indexed": indexed_count,
        "total": len(outcomes),
        "documents": [asdict(item) for item in outcomes],
    }


def ingest_file(path: str, *, retrieval_service: RetrievalService | None = None, user_id: str | None = None) -> IngestOutcome:
    service = retrieval_service or _require_retrieval_service()
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(path)

    suffix = file_path.suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        return IngestOutcome(
            document_id=file_path.name,
            file_name=file_path.name,
            indexed=False,
            reason=f"unsupported_file_type:{suffix}",
        )

    content = _read_file_content(file_path)
    if not content.strip():
        return IngestOutcome(
            document_id=file_path.name,
            file_name=file_path.name,
            indexed=False,
            reason="empty_content",
        )

    original_plan = service.chunk_plan
    service.chunk_plan = _DEFAULT_CHUNK_PLAN
    try:
        result = service.index_document(
            document_id=file_path.name,
            content=content,
            user_id=user_id,
            source_uri=str(file_path),
            title=file_path.stem,
            metadata={"ingested_from": "rag_docs", "file_name": file_path.name},
        )
    finally:
        service.chunk_plan = original_plan

    logger.info(
        "ingest_file_completed",
        extra={"document_id": result.document_id, "chunk_count": result.chunk_count},
    )
    return IngestOutcome(
        document_id=result.document_id,
        file_name=file_path.name,
        indexed=True,
        chunk_count=result.chunk_count,
        content_hash=result.content_hash,
        reason="indexed",
    )


def _require_retrieval_service() -> RetrievalService:
    if _retrieval_service is None:
        raise RuntimeError("ingest service not configured")
    return _retrieval_service


def _load_ingest_state() -> dict[str, str]:
    if not _INGEST_STATE_PATH.exists():
        return {}
    try:
        data = json.loads(_INGEST_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def _save_ingest_state(state: dict[str, str]) -> None:
    _RAG_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    _INGEST_STATE_PATH.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")


def _read_file_content(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception:
            logger.warning("pdf_ingestion_dependency_missing", extra={"file": str(path)})
            return ""
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return ""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    max_chars: int = 1200
    overlap_chars: int = 150
    min_chunk_chars: int = 200


def split_text_into_chunks(text: str, *, plan: ChunkPlan | None = None) -> list[str]:
    plan = plan or ChunkPlan()
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        paragraphs = [normalized]

    chunks: list[str] = []
    buffer = ""

    def flush_buffer(*, final: bool = False) -> None:
        nonlocal buffer
        candidate = buffer.strip()
        if candidate and (final or len(candidate) >= plan.min_chunk_chars or not chunks):
            chunks.append(candidate)
        buffer = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > plan.max_chars:
            flush_buffer()
            start = 0
            while start < len(paragraph):
                end = min(start + plan.max_chars, len(paragraph))
                piece = paragraph[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end >= len(paragraph):
                    break
                start = max(end - plan.overlap_chars, start + 1)
            continue

        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= plan.max_chars:
            buffer = candidate
            continue

        flush_buffer()
        buffer = paragraph

    flush_buffer(final=True)
    return chunks


def chunk_document_content(text: str, *, plan: ChunkPlan | None = None) -> list[tuple[int, str]]:
    chunks = split_text_into_chunks(text, plan=plan)
    return [(index, chunk) for index, chunk in enumerate(chunks)]

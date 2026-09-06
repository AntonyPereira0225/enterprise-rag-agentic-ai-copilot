from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChunkingConfig:
    """Word-based chunking settings for the transparent baseline."""

    size_words: int = 60
    overlap_words: int = 10

    def __post_init__(self) -> None:
        if self.size_words <= 0:
            raise ValueError("Chunk size must be greater than zero")
        if self.overlap_words < 0:
            raise ValueError("Chunk overlap cannot be negative")
        if self.overlap_words >= self.size_words:
            raise ValueError("Chunk overlap must be smaller than chunk size")


def chunk_document(document: dict[str, Any], config: ChunkingConfig) -> list[dict[str, Any]]:
    """Split one cleaned document while preserving searchable source metadata."""
    words = document["content"].split()
    if not words:
        return []

    step = config.size_words - config.overlap_words
    chunks: list[dict[str, Any]] = []
    metadata_fields = (
        "document_type",
        "title",
        "department",
        "region",
        "product",
        "version",
        "effective_date",
        "status",
        "sensitivity",
        "tags",
        "source_uri",
    )

    for chunk_index, start in enumerate(range(0, len(words), step)):
        chunk_words = words[start : start + config.size_words]
        chunk = {
            "chunk_id": f"{document['document_id']}::chunk-{chunk_index:04d}",
            "document_id": document["document_id"],
            "chunk_index": chunk_index,
            "word_start": start,
            "word_end": start + len(chunk_words),
            "word_count": len(chunk_words),
            "content_sha256": document["content_sha256"],
            "content": " ".join(chunk_words),
        }
        chunk.update({field: document[field] for field in metadata_fields})
        chunks.append(chunk)

        if start + config.size_words >= len(words):
            break

    return chunks


def chunk_documents(
    documents: list[dict[str, Any]], config: ChunkingConfig
) -> list[dict[str, Any]]:
    """Chunk documents in input order so repeated runs are deterministic."""
    return [chunk for document in documents for chunk in chunk_document(document, config)]

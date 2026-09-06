from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from enterprise_copilot.ingestion.chunking import ChunkingConfig, chunk_documents
from enterprise_copilot.ingestion.cleaning import clean_and_deduplicate
from enterprise_copilot.ingestion.loaders import load_jsonl, validate_documents


@dataclass(frozen=True)
class IngestionConfig:
    source_path: str
    chunks_path: str
    manifest_path: str
    chunk_size_words: int
    chunk_overlap_words: int
    included_statuses: list[str]

    @classmethod
    def from_json(cls, path: Path) -> IngestionConfig:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload)

    @property
    def chunking(self) -> ChunkingConfig:
        return ChunkingConfig(self.chunk_size_words, self.chunk_overlap_words)


@dataclass(frozen=True)
class IngestionResult:
    source_document_count: int
    included_document_count: int
    excluded_status_count: int
    duplicate_content_count: int
    chunk_count: int
    duplicate_document_ids: list[str]
    chunks_path: Path
    manifest_path: Path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary_path.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def run_ingestion(project_root: Path, config: IngestionConfig) -> IngestionResult:
    """Run loading, validation, status filtering, cleaning, deduplication, and chunking."""
    source_path = project_root / config.source_path
    chunks_path = project_root / config.chunks_path
    manifest_path = project_root / config.manifest_path

    source_documents = validate_documents(load_jsonl(source_path))
    included_statuses = set(config.included_statuses)
    status_filtered = [
        document for document in source_documents if document["status"] in included_statuses
    ]
    excluded_status_count = len(source_documents) - len(status_filtered)
    cleaned_documents, duplicate_ids = clean_and_deduplicate(status_filtered)
    chunks = chunk_documents(cleaned_documents, config.chunking)

    _write_jsonl(chunks_path, chunks)
    manifest = {
        "pipeline_version": 1,
        "source_path": config.source_path,
        "source_sha256": _sha256(source_path),
        "chunks_path": config.chunks_path,
        "chunks_sha256": _sha256(chunks_path),
        "source_document_count": len(source_documents),
        "included_document_count": len(cleaned_documents),
        "excluded_status_count": excluded_status_count,
        "duplicate_content_count": len(duplicate_ids),
        "duplicate_document_ids": duplicate_ids,
        "chunk_count": len(chunks),
        "chunking": asdict(config.chunking),
        "included_statuses": config.included_statuses,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )

    return IngestionResult(
        source_document_count=len(source_documents),
        included_document_count=len(cleaned_documents),
        excluded_status_count=excluded_status_count,
        duplicate_content_count=len(duplicate_ids),
        chunk_count=len(chunks),
        duplicate_document_ids=duplicate_ids,
        chunks_path=chunks_path,
        manifest_path=manifest_path,
    )

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REQUIRED_DOCUMENT_FIELDS = frozenset(
    {
        "document_id",
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
        "content",
    }
)


class DocumentLoadError(ValueError):
    """Raised when a source file cannot be converted into valid documents."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSON objects from a UTF-8 JSON Lines file."""
    if not path.is_file():
        raise FileNotFoundError(f"Source JSONL file does not exist: {path}")

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise DocumentLoadError(
                    f"Invalid JSON in {path} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise DocumentLoadError(f"Expected a JSON object in {path} at line {line_number}")
            rows.append(row)
    return rows


def validate_documents(documents: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate the minimum schema and uniqueness needed by ingestion."""
    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for row_number, document in enumerate(documents, start=1):
        missing = REQUIRED_DOCUMENT_FIELDS.difference(document)
        if missing:
            fields = ", ".join(sorted(missing))
            raise DocumentLoadError(f"Document row {row_number} is missing fields: {fields}")

        document_id = document["document_id"]
        if not isinstance(document_id, str) or not document_id.strip():
            raise DocumentLoadError(f"Document row {row_number} has an invalid document_id")
        if document_id in seen_ids:
            raise DocumentLoadError(f"Duplicate document_id: {document_id}")
        if not isinstance(document["content"], str) or not document["content"].strip():
            raise DocumentLoadError(f"Document {document_id} has empty content")
        if not isinstance(document["tags"], list):
            raise DocumentLoadError(f"Document {document_id} must contain a tags list")

        seen_ids.add(document_id)
        validated.append(document)

    return validated

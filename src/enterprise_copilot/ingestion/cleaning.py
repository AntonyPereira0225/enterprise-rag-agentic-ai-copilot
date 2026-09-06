from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from typing import Any

_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Apply deterministic Unicode and whitespace normalisation."""
    normalised = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [_HORIZONTAL_WHITESPACE.sub(" ", line).strip() for line in normalised.split("\n")]
    return _EXCESS_BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def content_hash(text: str) -> str:
    """Return a stable fingerprint used for lineage and duplicate detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_and_deduplicate(
    documents: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Clean content and remove later documents with identical cleaned text."""
    cleaned_documents: list[dict[str, Any]] = []
    duplicate_ids: list[str] = []
    seen_hashes: set[str] = set()

    for source in documents:
        document = dict(source)
        document["content"] = clean_text(document["content"])
        fingerprint = content_hash(document["content"])
        document["content_sha256"] = fingerprint

        if fingerprint in seen_hashes:
            duplicate_ids.append(document["document_id"])
            continue

        seen_hashes.add(fingerprint)
        cleaned_documents.append(document)

    return cleaned_documents, duplicate_ids

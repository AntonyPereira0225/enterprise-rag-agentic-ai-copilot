from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enterprise_copilot.retrieval.vector_index import (
    REQUIRED_CHUNK_FIELDS,
    SearchResult,
    chunk_embedding_text,
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenise(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.casefold())


@dataclass(frozen=True)
class BM25Record:
    chunk: dict[str, Any]
    term_frequencies: dict[str, int]
    document_length: int


class BM25Index:
    """Dependency-free Okapi BM25 keyword index with deterministic persistence."""

    schema_version = 1

    def __init__(
        self,
        records: list[BM25Record],
        inverse_document_frequency: dict[str, float],
        average_document_length: float,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.records = records
        self.inverse_document_frequency = inverse_document_frequency
        self.average_document_length = average_document_length
        self.k1 = k1
        self.b = b
        self.metadata = metadata or {}

    @classmethod
    def build(
        cls,
        chunks: list[dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        metadata: dict[str, Any] | None = None,
    ) -> BM25Index:
        if not chunks:
            raise ValueError("At least one chunk is required to build a BM25 index")
        if k1 <= 0:
            raise ValueError("BM25 k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("BM25 b must be between zero and one")

        records: list[BM25Record] = []
        document_frequency: Counter[str] = Counter()
        seen_ids: set[str] = set()
        for row_number, chunk in enumerate(chunks, start=1):
            missing = REQUIRED_CHUNK_FIELDS.difference(chunk)
            if missing:
                fields = ", ".join(sorted(missing))
                raise ValueError(f"Chunk row {row_number} is missing fields: {fields}")
            if chunk["chunk_id"] in seen_ids:
                raise ValueError(f"Duplicate chunk_id: {chunk['chunk_id']}")
            seen_ids.add(chunk["chunk_id"])

            terms = tokenise(chunk_embedding_text(chunk))
            frequencies = dict(Counter(terms))
            document_frequency.update(frequencies.keys())
            records.append(BM25Record(chunk, frequencies, len(terms)))

        count = len(records)
        average_length = sum(record.document_length for record in records) / count
        inverse_document_frequency = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        return cls(
            records,
            inverse_document_frequency,
            average_length,
            k1=k1,
            b=b,
            metadata=metadata,
        )

    def _score(self, query_terms: set[str], record: BM25Record) -> float:
        score = 0.0
        length_ratio = record.document_length / self.average_document_length
        for term in sorted(query_terms):
            frequency = record.term_frequencies.get(term, 0)
            if not frequency:
                continue
            denominator = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
            score += self.inverse_document_frequency.get(term, 0.0) * (
                frequency * (self.k1 + 1) / denominator
            )
        return score

    @staticmethod
    def _matches_filters(chunk: dict[str, Any], filters: dict[str, Any]) -> bool:
        return all(chunk.get(field) == expected for field, expected in filters.items())

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        query_terms = set(tokenise(query)).intersection(self.inverse_document_frequency)
        if not query_terms:
            return []

        active_filters = filters or {}
        results = [
            SearchResult(score=self._score(query_terms, record), chunk=record.chunk)
            for record in self.records
            if self._matches_filters(record.chunk, active_filters)
        ]
        results.sort(key=lambda result: (-result.score, result.chunk["chunk_id"]))
        return results[:top_k]

    def to_state(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metadata": self.metadata,
            "k1": self.k1,
            "b": self.b,
            "average_document_length": self.average_document_length,
            "inverse_document_frequency": self.inverse_document_frequency,
            "records": [
                {
                    "chunk": record.chunk,
                    "term_frequencies": record.term_frequencies,
                    "document_length": record.document_length,
                }
                for record in self.records
            ],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> BM25Index:
        if state.get("schema_version") != cls.schema_version:
            raise ValueError(f"Unsupported BM25 index schema: {state.get('schema_version')}")
        records = [
            BM25Record(
                chunk=row["chunk"],
                term_frequencies={
                    term: int(frequency) for term, frequency in row["term_frequencies"].items()
                },
                document_length=int(row["document_length"]),
            )
            for row in state["records"]
        ]
        return cls(
            records,
            {term: float(value) for term, value in state["inverse_document_frequency"].items()},
            float(state["average_document_length"]),
            k1=float(state["k1"]),
            b=float(state["b"]),
            metadata=state.get("metadata", {}),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(self.to_state(), ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary_path.replace(path)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        if not path.is_file():
            raise FileNotFoundError(f"BM25 index does not exist: {path}")
        return cls.from_state(json.loads(path.read_text(encoding="utf-8")))

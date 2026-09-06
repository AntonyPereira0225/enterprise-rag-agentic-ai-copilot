from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enterprise_copilot.embeddings.base import SparseVector
from enterprise_copilot.embeddings.tfidf import TfidfEmbeddingModel

REQUIRED_CHUNK_FIELDS = frozenset(
    {
        "chunk_id",
        "document_id",
        "document_type",
        "title",
        "department",
        "region",
        "product",
        "status",
        "tags",
        "source_uri",
        "content",
    }
)


def chunk_embedding_text(chunk: dict[str, Any]) -> str:
    """Combine searchable metadata with content before embedding."""
    values = [
        chunk["document_type"].replace("_", " "),
        chunk["title"],
        chunk["department"],
        chunk["region"],
        chunk["product"],
        " ".join(chunk["tags"]),
        chunk["content"],
    ]
    return "\n".join(values)


@dataclass(frozen=True)
class VectorRecord:
    chunk: dict[str, Any]
    vector: SparseVector


@dataclass(frozen=True)
class SearchResult:
    score: float
    chunk: dict[str, Any]
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "score": self.score,
            "chunk_id": self.chunk["chunk_id"],
            "document_id": self.chunk["document_id"],
            "content": self.chunk["content"],
            "source_uri": self.chunk["source_uri"],
            "document_type": self.chunk["document_type"],
            "region": self.chunk["region"],
            "product": self.chunk["product"],
        }
        if self.details:
            payload["retrieval_details"] = self.details
        return payload


class VectorIndex:
    """Small persisted cosine-similarity index for sparse local vectors."""

    schema_version = 1

    def __init__(
        self,
        model: TfidfEmbeddingModel,
        records: list[VectorRecord],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.records = records
        self.metadata = metadata or {}

    @classmethod
    def build(
        cls,
        chunks: list[dict[str, Any]],
        model: TfidfEmbeddingModel,
        metadata: dict[str, Any] | None = None,
    ) -> VectorIndex:
        if not chunks:
            raise ValueError("At least one chunk is required to build an index")

        seen_chunk_ids: set[str] = set()
        for row_number, chunk in enumerate(chunks, start=1):
            missing = REQUIRED_CHUNK_FIELDS.difference(chunk)
            if missing:
                fields = ", ".join(sorted(missing))
                raise ValueError(f"Chunk row {row_number} is missing fields: {fields}")
            chunk_id = chunk["chunk_id"]
            if chunk_id in seen_chunk_ids:
                raise ValueError(f"Duplicate chunk_id: {chunk_id}")
            seen_chunk_ids.add(chunk_id)

        texts = [chunk_embedding_text(chunk) for chunk in chunks]
        vectors = model.fit_transform(texts)
        records = [
            VectorRecord(chunk=chunk, vector=vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        return cls(model=model, records=records, metadata=metadata)

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

        query_vector = self.model.transform(query)
        if not query_vector:
            return []

        active_filters = filters or {}
        results = []
        for record in self.records:
            if not self._matches_filters(record.chunk, active_filters):
                continue
            score = sum(
                query_vector.get(index, 0.0) * value for index, value in record.vector.items()
            )
            results.append(SearchResult(score=score, chunk=record.chunk))

        results.sort(key=lambda result: (-result.score, result.chunk["chunk_id"]))
        return results[:top_k]

    def to_state(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metadata": self.metadata,
            "embedding": self.model.to_state(),
            "records": [
                {
                    "chunk": record.chunk,
                    "vector": [[index, value] for index, value in sorted(record.vector.items())],
                }
                for record in self.records
            ],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> VectorIndex:
        if state.get("schema_version") != cls.schema_version:
            raise ValueError(f"Unsupported vector index schema: {state.get('schema_version')}")

        model = TfidfEmbeddingModel.from_state(state["embedding"])
        records = [
            VectorRecord(
                chunk=record["chunk"],
                vector={int(index): float(value) for index, value in record["vector"]},
            )
            for record in state["records"]
        ]
        return cls(model=model, records=records, metadata=state.get("metadata", {}))

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
    def load(cls, path: Path) -> VectorIndex:
        if not path.is_file():
            raise FileNotFoundError(f"Vector index does not exist: {path}")
        state = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_state(state)

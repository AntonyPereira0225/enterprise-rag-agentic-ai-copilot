from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from enterprise_copilot.embeddings.sentence_transformer import (
    DenseDependencyError,
    SentenceTransformerEmbeddingModel,
)
from enterprise_copilot.retrieval.vector_index import (
    REQUIRED_CHUNK_FIELDS,
    SearchResult,
    chunk_embedding_text,
)


def _dense_dependencies(
    faiss_module: Any | None = None, numpy_module: Any | None = None
) -> tuple[Any, Any]:
    if faiss_module is not None and numpy_module is not None:
        return faiss_module, numpy_module
    try:
        import faiss
        import numpy
    except ImportError as exc:
        raise DenseDependencyError(
            "FAISS indexing requires numpy and faiss-cpu from requirements-dense.txt."
        ) from exc
    return faiss, numpy


class FaissVectorIndex:
    """FAISS inner-product index for normalised Sentence Transformer vectors."""

    schema_version = 1

    def __init__(
        self,
        model: SentenceTransformerEmbeddingModel,
        chunks: list[dict[str, Any]],
        backend: Any,
        *,
        faiss_module: Any,
        numpy_module: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.chunks = chunks
        self._backend = backend
        self._faiss = faiss_module
        self._numpy = numpy_module
        self.metadata = metadata or {}

    @classmethod
    def build(
        cls,
        chunks: list[dict[str, Any]],
        model: SentenceTransformerEmbeddingModel,
        *,
        faiss_module: Any | None = None,
        numpy_module: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FaissVectorIndex:
        if not chunks:
            raise ValueError("At least one chunk is required to build a FAISS index")
        seen_ids: set[str] = set()
        for row_number, chunk in enumerate(chunks, start=1):
            missing = REQUIRED_CHUNK_FIELDS.difference(chunk)
            if missing:
                fields = ", ".join(sorted(missing))
                raise ValueError(f"Chunk row {row_number} is missing fields: {fields}")
            if chunk["chunk_id"] in seen_ids:
                raise ValueError(f"Duplicate chunk_id: {chunk['chunk_id']}")
            seen_ids.add(chunk["chunk_id"])

        faiss, numpy = _dense_dependencies(faiss_module, numpy_module)
        texts = [chunk_embedding_text(chunk) for chunk in chunks]
        matrix = numpy.asarray(model.transform_many(texts), dtype="float32")
        if matrix.ndim != 2 or matrix.shape[1] != model.dimension:
            raise ValueError("Dense embedding matrix has an unexpected shape")
        faiss.normalize_L2(matrix)
        backend = faiss.IndexFlatIP(model.dimension)
        backend.add(matrix)
        return cls(
            model,
            chunks,
            backend,
            faiss_module=faiss,
            numpy_module=numpy,
            metadata=metadata,
        )

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
        if not self.chunks:
            return []

        query_matrix = self._numpy.asarray([self.model.transform(query)], dtype="float32")
        self._faiss.normalize_L2(query_matrix)
        candidate_count = len(self.chunks) if filters else min(top_k, len(self.chunks))
        distances, positions = self._backend.search(query_matrix, candidate_count)
        active_filters = filters or {}
        results = [
            SearchResult(score=float(score), chunk=self.chunks[int(position)])
            for score, position in zip(distances[0], positions[0], strict=True)
            if int(position) >= 0
            and self._matches_filters(self.chunks[int(position)], active_filters)
        ]
        return results[:top_k]

    def save(self, index_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._backend, str(index_path))
        state = {
            "schema_version": self.schema_version,
            "embedding": self.model.to_state(),
            "metadata": self.metadata,
            "chunks": self.chunks,
        }
        self.metadata_path(index_path).write_text(
            json.dumps(state, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    @staticmethod
    def metadata_path(index_path: Path) -> Path:
        return index_path.with_suffix(index_path.suffix + ".metadata.json")

    @classmethod
    def load(
        cls,
        index_path: Path,
        *,
        faiss_module: Any | None = None,
        numpy_module: Any | None = None,
        model_backend: Any | None = None,
    ) -> FaissVectorIndex:
        faiss, numpy = _dense_dependencies(faiss_module, numpy_module)
        metadata_path = cls.metadata_path(index_path)
        if not index_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"FAISS index or metadata does not exist: {index_path}")
        state = json.loads(metadata_path.read_text(encoding="utf-8"))
        if state.get("schema_version") != cls.schema_version:
            raise ValueError(f"Unsupported FAISS index schema: {state.get('schema_version')}")
        model = SentenceTransformerEmbeddingModel.from_state(
            state["embedding"], backend=model_backend
        )
        backend = faiss.read_index(str(index_path))
        if int(backend.ntotal) != len(state["chunks"]):
            raise ValueError("FAISS index and chunk metadata counts do not match")
        return cls(
            model,
            state["chunks"],
            backend,
            faiss_module=faiss,
            numpy_module=numpy,
            metadata=state.get("metadata", {}),
        )

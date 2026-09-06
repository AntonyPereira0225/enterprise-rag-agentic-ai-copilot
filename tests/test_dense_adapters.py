import json
from math import sqrt
from pathlib import Path
from typing import Any, ClassVar

from enterprise_copilot.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingModel,
)
from enterprise_copilot.embeddings.tfidf import TfidfEmbeddingModel
from enterprise_copilot.retrieval import dense_pipeline
from enterprise_copilot.retrieval.faiss_index import FaissVectorIndex
from enterprise_copilot.retrieval.pipeline import RetrievalConfig
from enterprise_copilot.retrieval.vector_index import VectorIndex


class FakeSentenceBackend:
    def get_sentence_embedding_dimension(self) -> int:
        return 2

    def encode(self, texts: list[str], **_: Any) -> list[list[float]]:
        vectors = []
        for text in texts:
            raw = [float(text.casefold().count("refund")), float(text.casefold().count("service"))]
            norm = sqrt(sum(value * value for value in raw)) or 1.0
            vectors.append([value / norm for value in raw])
        return vectors


class FakeMatrix:
    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows
        self.ndim = 2
        self.shape = (len(rows), len(rows[0]) if rows else 0)


class FakeNumpy:
    @staticmethod
    def asarray(rows: list[list[float]], dtype: str) -> FakeMatrix:
        assert dtype == "float32"
        return FakeMatrix(rows)


class FakeFlatIndex:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.rows: list[list[float]] = []

    def add(self, matrix: FakeMatrix) -> None:
        self.rows.extend(matrix.rows)

    @property
    def ntotal(self) -> int:
        return len(self.rows)

    def search(self, query: FakeMatrix, top_k: int) -> tuple[list[list[float]], list[list[int]]]:
        scores = [
            (sum(left * right for left, right in zip(query.rows[0], row, strict=True)), index)
            for index, row in enumerate(self.rows)
        ]
        ranked = sorted(scores, reverse=True)[:top_k]
        return [[score for score, _ in ranked]], [[index for _, index in ranked]]


class FakeFaiss:
    saved: ClassVar[dict[str, FakeFlatIndex]] = {}

    @staticmethod
    def normalize_L2(_: FakeMatrix) -> None:
        return None

    @staticmethod
    def IndexFlatIP(dimension: int) -> FakeFlatIndex:
        return FakeFlatIndex(dimension)

    @classmethod
    def write_index(cls, backend: FakeFlatIndex, path: str) -> None:
        cls.saved[path] = backend
        Path(path).write_bytes(b"fake-faiss-index")

    @classmethod
    def read_index(cls, path: str) -> FakeFlatIndex:
        return cls.saved[path]


class FakePipelineIndex:
    saved: ClassVar[dict[str, "FakePipelineIndex"]] = {}

    def __init__(self, chunks: list[dict[str, Any]], model: Any, delegate: VectorIndex) -> None:
        self.chunks = chunks
        self.model = model
        self.delegate = delegate

    @classmethod
    def build(cls, chunks: list[dict[str, Any]], model: Any, **_: Any) -> "FakePipelineIndex":
        delegate = VectorIndex.build(chunks, TfidfEmbeddingModel())
        return cls(chunks, model, delegate)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-index")
        self.saved[str(path)] = self

    @classmethod
    def load(cls, path: Path) -> "FakePipelineIndex":
        return cls.saved[str(path)]

    def search(self, query: str, *, top_k: int = 5) -> list[Any]:
        return self.delegate.search(query, top_k=top_k)


def _chunk(chunk_id: str, content: str, region: str) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "document_id": chunk_id,
        "document_type": "policy",
        "title": content,
        "department": "Support",
        "region": region,
        "product": "NSG Connect",
        "status": "active",
        "tags": [],
        "source_uri": f"nsg://{chunk_id}",
        "content": content,
    }


def test_sentence_transformer_and_faiss_adapters_with_in_memory_backends() -> None:
    model = SentenceTransformerEmbeddingModel(backend=FakeSentenceBackend())
    chunks = [
        _chunk("refund", "refund guidance", "Ireland"),
        _chunk("service", "service activation", "Germany"),
    ]
    index = FaissVectorIndex.build(
        chunks,
        model,
        faiss_module=FakeFaiss(),
        numpy_module=FakeNumpy(),
    )

    results = index.search("refund", top_k=1)
    filtered = index.search("service", top_k=1, filters={"region": "Germany"})

    assert model.dimension == 2
    assert model.to_state()["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert results[0].chunk["chunk_id"] == "refund"
    assert filtered[0].chunk["chunk_id"] == "service"


def test_dense_adapter_state_and_faiss_files_round_trip(tmp_path: Path) -> None:
    backend = FakeSentenceBackend()
    model = SentenceTransformerEmbeddingModel(backend=backend)
    restored_model = SentenceTransformerEmbeddingModel.from_state(model.to_state(), backend=backend)
    chunks = [_chunk("refund", "refund guidance", "Ireland")]
    index = FaissVectorIndex.build(
        chunks,
        restored_model,
        faiss_module=FakeFaiss(),
        numpy_module=FakeNumpy(),
        metadata={"purpose": "test"},
    )
    index_path = tmp_path / "dense.faiss"
    index.save(index_path)
    restored_index = FaissVectorIndex.load(
        index_path,
        faiss_module=FakeFaiss(),
        numpy_module=FakeNumpy(),
        model_backend=backend,
    )

    assert model.transform_many([]) == []
    assert restored_index.metadata == {"purpose": "test"}
    assert restored_index.search("refund", top_k=1)[0].chunk["chunk_id"] == "refund"
    assert FaissVectorIndex.metadata_path(index_path).is_file()


def test_dense_pipeline_orchestration_with_in_memory_index(
    tmp_path: Path, monkeypatch: Any
) -> None:
    chunks_path = tmp_path / "data" / "chunks.jsonl"
    questions_path = tmp_path / "data" / "questions.jsonl"
    chunks_path.parent.mkdir(parents=True)
    chunks_path.write_text(
        json.dumps(_chunk("refund", "refund guidance seven days", "Ireland")) + "\n",
        encoding="utf-8",
    )
    questions_path.write_text(
        json.dumps(
            {
                "question_id": "Q1",
                "question": "refund guidance",
                "expected_answer": "seven days",
                "expected_document_ids": ["refund"],
                "answerable": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = RetrievalConfig(
        chunks_path="data/chunks.jsonl",
        evaluation_path="data/questions.jsonl",
        index_path="data/index.faiss",
        results_path="data/results.json",
        embedding={"type": "sentence_transformer", "model_name": "fake"},
        top_k_values=[1],
    )
    fake_model = SentenceTransformerEmbeddingModel(backend=FakeSentenceBackend())
    monkeypatch.setattr(dense_pipeline, "_model", lambda _: fake_model)
    monkeypatch.setattr(dense_pipeline, "FaissVectorIndex", FakePipelineIndex)

    build = dense_pipeline.build_dense_retrieval_index(tmp_path, config)
    results = dense_pipeline.run_dense_retrieval_evaluation(tmp_path, config)

    assert build.chunk_count == 1
    assert build.embedding_dimension == 2
    assert results["summary"]["evidence_recall_at_k"]["1"] == 1.0
    assert (tmp_path / config.results_path).is_file()

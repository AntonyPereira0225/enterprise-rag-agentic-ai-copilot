from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enterprise_copilot.embeddings.tfidf import TfidfEmbeddingModel
from enterprise_copilot.evaluation.retrieval import evaluate_retrieval
from enterprise_copilot.ingestion.loaders import load_jsonl
from enterprise_copilot.retrieval.vector_index import VectorIndex


@dataclass(frozen=True)
class RetrievalConfig:
    chunks_path: str
    evaluation_path: str
    index_path: str
    results_path: str
    embedding: dict[str, Any]
    top_k_values: list[int]
    minimum_score: float = 0.0

    @classmethod
    def from_json(cls, path: Path) -> RetrievalConfig:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload)


@dataclass(frozen=True)
class IndexBuildResult:
    chunk_count: int
    embedding_dimension: int
    index_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _create_embedding_model(settings: dict[str, Any]) -> TfidfEmbeddingModel:
    model_type = settings.get("type")
    if model_type != TfidfEmbeddingModel.model_type:
        raise ValueError(f"Unsupported embedding model type: {model_type}")
    return TfidfEmbeddingModel(
        ngram_min=settings.get("ngram_min", 1),
        ngram_max=settings.get("ngram_max", 2),
        min_document_frequency=settings.get("min_document_frequency", 1),
    )


def build_retrieval_index(project_root: Path, config: RetrievalConfig) -> IndexBuildResult:
    chunks_path = project_root / config.chunks_path
    index_path = project_root / config.index_path
    chunks = load_jsonl(chunks_path)
    model = _create_embedding_model(config.embedding)
    index = VectorIndex.build(
        chunks,
        model,
        metadata={
            "source_chunks_path": config.chunks_path,
            "source_chunks_sha256": _sha256(chunks_path),
        },
    )
    index.save(index_path)
    return IndexBuildResult(
        chunk_count=len(index.records),
        embedding_dimension=index.model.dimension,
        index_path=index_path,
    )


def run_retrieval_evaluation(project_root: Path, config: RetrievalConfig) -> dict[str, Any]:
    index_path = project_root / config.index_path
    evaluation_path = project_root / config.evaluation_path
    results_path = project_root / config.results_path
    index = VectorIndex.load(index_path)
    questions = load_jsonl(evaluation_path)
    results = evaluate_retrieval(
        index,
        questions,
        config.top_k_values,
        minimum_score=config.minimum_score,
    )

    results["configuration"] = {
        "index_path": config.index_path,
        "evaluation_path": config.evaluation_path,
        "embedding": config.embedding,
        "minimum_score": config.minimum_score,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return results

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from enterprise_copilot.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingModel,
)
from enterprise_copilot.evaluation.retrieval import evaluate_retrieval
from enterprise_copilot.ingestion.loaders import load_jsonl
from enterprise_copilot.retrieval.faiss_index import FaissVectorIndex
from enterprise_copilot.retrieval.pipeline import IndexBuildResult, RetrievalConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _model(settings: dict[str, Any]) -> SentenceTransformerEmbeddingModel:
    if settings.get("type") != SentenceTransformerEmbeddingModel.model_type:
        raise ValueError(f"Unsupported dense embedding type: {settings.get('type')}")
    return SentenceTransformerEmbeddingModel(
        model_name=settings["model_name"],
        device=settings.get("device", "cpu"),
        batch_size=settings.get("batch_size", 32),
    )


def build_dense_retrieval_index(project_root: Path, config: RetrievalConfig) -> IndexBuildResult:
    chunks_path = project_root / config.chunks_path
    chunks = load_jsonl(chunks_path)
    index = FaissVectorIndex.build(
        chunks,
        _model(config.embedding),
        metadata={
            "source_chunks_path": config.chunks_path,
            "source_chunks_sha256": _sha256(chunks_path),
        },
    )
    index_path = project_root / config.index_path
    index.save(index_path)
    return IndexBuildResult(
        chunk_count=len(index.chunks),
        embedding_dimension=index.model.dimension,
        index_path=index_path,
    )


def run_dense_retrieval_evaluation(project_root: Path, config: RetrievalConfig) -> dict[str, Any]:
    index = FaissVectorIndex.load(project_root / config.index_path)
    questions = load_jsonl(project_root / config.evaluation_path)
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
    results_path = project_root / config.results_path
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return results

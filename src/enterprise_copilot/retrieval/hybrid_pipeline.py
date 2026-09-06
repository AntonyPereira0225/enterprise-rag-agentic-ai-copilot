from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enterprise_copilot.evaluation.retrieval import evaluate_retrieval
from enterprise_copilot.ingestion.loaders import load_jsonl
from enterprise_copilot.retrieval.bm25_index import BM25Index
from enterprise_copilot.retrieval.hybrid import HybridRetriever
from enterprise_copilot.retrieval.reranking import MetadataAwareReranker, RerankerWeights
from enterprise_copilot.retrieval.vector_index import VectorIndex


@dataclass(frozen=True)
class HybridRetrievalConfig:
    evaluation_path: str
    vector_index_path: str
    bm25_index_path: str
    results_path: str
    top_k_values: list[int]
    minimum_score: float
    candidate_pool_size: int
    rrf_constant: int
    vector_weight: float
    bm25_weight: float
    max_chunks_per_document: int
    reranker_weights: dict[str, float]

    @classmethod
    def from_json(cls, path: Path) -> HybridRetrievalConfig:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def build_bm25_index(chunks: list[dict[str, Any]], path: Path, *, source_path: str) -> BM25Index:
    index = BM25Index.build(chunks, metadata={"source_chunks_path": source_path})
    index.save(path)
    return index


def load_hybrid_retriever(project_root: Path, config: HybridRetrievalConfig) -> HybridRetriever:
    vector_index = VectorIndex.load(project_root / config.vector_index_path)
    bm25_index = BM25Index.load(project_root / config.bm25_index_path)
    reranker = MetadataAwareReranker(RerankerWeights.from_dict(config.reranker_weights))
    return HybridRetriever(
        vector_index,
        bm25_index,
        reranker,
        candidate_pool_size=config.candidate_pool_size,
        rrf_constant=config.rrf_constant,
        vector_weight=config.vector_weight,
        bm25_weight=config.bm25_weight,
        max_chunks_per_document=config.max_chunks_per_document,
    )


def run_hybrid_evaluation(project_root: Path, config: HybridRetrievalConfig) -> dict[str, Any]:
    retriever = load_hybrid_retriever(project_root, config)
    questions = load_jsonl(project_root / config.evaluation_path)
    results = evaluate_retrieval(
        retriever,
        questions,
        config.top_k_values,
        minimum_score=config.minimum_score,
    )
    results["configuration"] = {
        "vector_index_path": config.vector_index_path,
        "bm25_index_path": config.bm25_index_path,
        "candidate_pool_size": config.candidate_pool_size,
        "rrf_constant": config.rrf_constant,
        "vector_weight": config.vector_weight,
        "bm25_weight": config.bm25_weight,
        "max_chunks_per_document": config.max_chunks_per_document,
        "reranker_weights": config.reranker_weights,
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

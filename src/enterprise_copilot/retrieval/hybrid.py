from __future__ import annotations

from typing import Any

from enterprise_copilot.retrieval.bm25_index import BM25Index
from enterprise_copilot.retrieval.reranking import MetadataAwareReranker
from enterprise_copilot.retrieval.vector_index import SearchResult, VectorIndex


class HybridRetriever:
    """Fuse vector and BM25 ranks, then rerank and diversify the candidates."""

    def __init__(
        self,
        vector_index: VectorIndex,
        bm25_index: BM25Index,
        reranker: MetadataAwareReranker,
        *,
        candidate_pool_size: int = 30,
        rrf_constant: int = 60,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
        max_chunks_per_document: int = 1,
    ) -> None:
        if candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be greater than zero")
        if rrf_constant < 0:
            raise ValueError("rrf_constant cannot be negative")
        if vector_weight < 0 or bm25_weight < 0 or vector_weight + bm25_weight == 0:
            raise ValueError("At least one non-negative retrieval weight is required")
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.reranker = reranker
        self.candidate_pool_size = candidate_pool_size
        self.rrf_constant = rrf_constant
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.max_chunks_per_document = max_chunks_per_document

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        pool_size = max(top_k, self.candidate_pool_size)
        vector_results = self.vector_index.search(query, top_k=pool_size, filters=filters)
        bm25_results = self.bm25_index.search(query, top_k=pool_size, filters=filters)

        fused: dict[str, dict[str, Any]] = {}
        for source, weight, results in (
            ("vector", self.vector_weight, vector_results),
            ("bm25", self.bm25_weight, bm25_results),
        ):
            for rank, result in enumerate(results, start=1):
                chunk_id = result.chunk["chunk_id"]
                entry = fused.setdefault(
                    chunk_id,
                    {"chunk": result.chunk, "rrf_score": 0.0, "details": {}},
                )
                entry["rrf_score"] += weight / (self.rrf_constant + rank)
                entry["details"][f"{source}_rank"] = rank
                entry["details"][f"{source}_score"] = result.score

        if not fused:
            return []
        maximum_rrf = (self.vector_weight + self.bm25_weight) / (self.rrf_constant + 1)
        candidates = []
        for entry in fused.values():
            normalised_score = entry["rrf_score"] / maximum_rrf
            details = {
                **entry["details"],
                "rrf_score": entry["rrf_score"],
                "normalised_rrf_score": normalised_score,
            }
            candidates.append(
                SearchResult(score=normalised_score, chunk=entry["chunk"], details=details)
            )

        return self.reranker.rerank(
            query,
            candidates,
            top_k=top_k,
            max_chunks_per_document=self.max_chunks_per_document,
        )

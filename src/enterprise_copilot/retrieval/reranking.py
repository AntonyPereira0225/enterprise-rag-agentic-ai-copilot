from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from enterprise_copilot.retrieval.bm25_index import tokenise
from enterprise_copilot.retrieval.vector_index import SearchResult

_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "for",
        "how",
        "in",
        "is",
        "of",
        "the",
        "to",
        "what",
        "with",
    }
)


@dataclass(frozen=True)
class RerankerWeights:
    vector_similarity: float = 0.45
    content_overlap: float = 0.25
    metadata_overlap: float = 0.20
    fused_rank: float = 0.10

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> RerankerWeights:
        weights = cls(**values)
        total = sum(
            (
                weights.vector_similarity,
                weights.content_overlap,
                weights.metadata_overlap,
                weights.fused_rank,
            )
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Reranker weights must sum to 1.0")
        if min(values.values(), default=0.0) < 0:
            raise ValueError("Reranker weights cannot be negative")
        return weights


class MetadataAwareReranker:
    """Transparent reranker using similarity plus text and metadata overlap."""

    def __init__(self, weights: RerankerWeights) -> None:
        self.weights = weights

    @staticmethod
    def _overlap(query_terms: set[str], candidate_terms: set[str]) -> float:
        return (
            len(query_terms.intersection(candidate_terms)) / len(query_terms)
            if query_terms
            else 0.0
        )

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        *,
        top_k: int,
        max_chunks_per_document: int,
    ) -> list[SearchResult]:
        if max_chunks_per_document <= 0:
            raise ValueError("max_chunks_per_document must be greater than zero")
        query_terms = set(tokenise(query)).difference(_STOP_WORDS)
        rescored: list[SearchResult] = []

        for candidate in candidates:
            chunk = candidate.chunk
            details = dict(candidate.details or {})
            content_terms = set(tokenise(chunk["content"]))
            metadata_text = " ".join(
                [
                    chunk["document_type"].replace("_", " "),
                    chunk["title"],
                    chunk["department"],
                    chunk["region"],
                    chunk["product"],
                    " ".join(chunk["tags"]),
                ]
            )
            content_overlap = self._overlap(query_terms, content_terms)
            metadata_overlap = self._overlap(query_terms, set(tokenise(metadata_text)))
            vector_similarity = max(0.0, min(float(details.get("vector_score", 0.0)), 1.0))
            fused_rank = max(0.0, min(float(details.get("normalised_rrf_score", 0.0)), 1.0))
            score = (
                self.weights.vector_similarity * vector_similarity
                + self.weights.content_overlap * content_overlap
                + self.weights.metadata_overlap * metadata_overlap
                + self.weights.fused_rank * fused_rank
            )
            details["reranker"] = {
                "vector_similarity": vector_similarity,
                "content_overlap": content_overlap,
                "metadata_overlap": metadata_overlap,
                "fused_rank": fused_rank,
            }
            rescored.append(SearchResult(score=score, chunk=chunk, details=details))

        rescored.sort(key=lambda result: (-result.score, result.chunk["chunk_id"]))
        selected: list[SearchResult] = []
        document_counts: dict[str, int] = {}
        for result in rescored:
            document_id = result.chunk["document_id"]
            if document_counts.get(document_id, 0) >= max_chunks_per_document:
                continue
            selected.append(result)
            document_counts[document_id] = document_counts.get(document_id, 0) + 1
            if len(selected) == top_k:
                break
        return selected

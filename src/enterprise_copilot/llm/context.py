from __future__ import annotations

from dataclasses import dataclass

from enterprise_copilot.llm.schemas import ContextPackage, Evidence
from enterprise_copilot.retrieval.vector_index import SearchResult


@dataclass(frozen=True)
class ContextBuilderConfig:
    top_k: int = 5
    minimum_query_score: float = 0.30
    minimum_evidence_score: float = 0.0
    max_context_words: int = 300

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self.minimum_query_score < 0 or self.minimum_evidence_score < 0:
            raise ValueError("Minimum scores cannot be negative")
        if self.max_context_words <= 0:
            raise ValueError("max_context_words must be greater than zero")


class ContextBuilder:
    """Apply the answerability gate and a strict context word budget."""

    def __init__(self, config: ContextBuilderConfig) -> None:
        self.config = config

    def build(self, question: str, results: list[SearchResult]) -> ContextPackage:
        top_score = results[0].score if results else 0.0
        if not results:
            return self._refusal(question, top_score, "No relevant passages were retrieved.")
        if top_score < self.config.minimum_query_score:
            return self._refusal(
                question,
                top_score,
                "The best passage was below the calibrated confidence threshold.",
            )

        evidence: list[Evidence] = []
        words_used = 0
        for result in results[: self.config.top_k]:
            if result.score < self.config.minimum_evidence_score:
                continue
            word_count = len(result.chunk["content"].split())
            if words_used + word_count > self.config.max_context_words:
                continue
            citation_id = f"C{len(evidence) + 1}"
            chunk = result.chunk
            evidence.append(
                Evidence(
                    citation_id=citation_id,
                    score=result.score,
                    chunk_id=chunk["chunk_id"],
                    document_id=chunk["document_id"],
                    document_type=chunk["document_type"],
                    title=chunk["title"],
                    region=chunk["region"],
                    product=chunk["product"],
                    source_uri=chunk["source_uri"],
                    content=chunk["content"],
                )
            )
            words_used += word_count

        if not evidence:
            return self._refusal(
                question,
                top_score,
                "No passage satisfied the evidence threshold and context budget.",
            )
        return ContextPackage(
            question=question,
            query_score=top_score,
            minimum_query_score=self.config.minimum_query_score,
            word_count=words_used,
            evidence=tuple(evidence),
        )

    def _refusal(self, question: str, score: float, reason: str) -> ContextPackage:
        return ContextPackage(
            question=question,
            query_score=score,
            minimum_query_score=self.config.minimum_query_score,
            word_count=0,
            evidence=(),
            refusal_reason=reason,
        )

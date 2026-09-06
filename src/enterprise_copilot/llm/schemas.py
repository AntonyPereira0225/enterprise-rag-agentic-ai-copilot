from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class Evidence:
    """A retrieval result made safe and explicit for answer generation."""

    citation_id: str
    score: float
    chunk_id: str
    document_id: str
    document_type: str
    title: str
    region: str
    product: str
    source_uri: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextPackage:
    question: str
    query_score: float
    minimum_query_score: float
    word_count: int
    evidence: tuple[Evidence, ...]
    refusal_reason: str | None = None

    @property
    def can_answer(self) -> bool:
        return bool(self.evidence) and self.refusal_reason is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "query_score": self.query_score,
            "minimum_query_score": self.minimum_query_score,
            "word_count": self.word_count,
            "evidence": [item.to_dict() for item in self.evidence],
            "refusal_reason": self.refusal_reason,
            "can_answer": self.can_answer,
        }


@dataclass(frozen=True)
class Citation:
    citation_id: str
    document_id: str
    chunk_id: str
    title: str
    source_uri: str
    quote: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class GroundedAnswer:
    status: Literal["answered", "insufficient_evidence"]
    answer: str
    confidence: float
    citations: tuple[Citation, ...]
    reason: str | None = None
    generator: str = "extractive"
    model: str | None = None
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "answer": self.answer,
            "confidence": self.confidence,
            "citations": [citation.to_dict() for citation in self.citations],
            "reason": self.reason,
            "generator": self.generator,
            "model": self.model,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class CitationVerification:
    valid: bool
    errors: tuple[str, ...]
    checked_citations: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroundedAnswerRun:
    context: ContextPackage
    response: GroundedAnswer
    verification: CitationVerification

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.context.to_dict(),
            "response": self.response.to_dict(),
            "verification": self.verification.to_dict(),
        }

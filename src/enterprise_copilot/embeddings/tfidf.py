from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from enterprise_copilot.embeddings.base import SparseVector

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class TfidfEmbeddingModel:
    """Dependency-free, L2-normalised word and phrase TF-IDF embeddings."""

    model_type = "tfidf"

    def __init__(
        self,
        *,
        ngram_min: int = 1,
        ngram_max: int = 2,
        min_document_frequency: int = 1,
    ) -> None:
        if ngram_min <= 0:
            raise ValueError("ngram_min must be greater than zero")
        if ngram_max < ngram_min:
            raise ValueError("ngram_max must be greater than or equal to ngram_min")
        if min_document_frequency <= 0:
            raise ValueError("min_document_frequency must be greater than zero")

        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.min_document_frequency = min_document_frequency
        self.document_count = 0
        self.terms: list[str] = []
        self.idf: list[float] = []
        self._term_indexes: dict[str, int] = {}

    @property
    def dimension(self) -> int:
        return len(self.terms)

    def _extract_terms(self, text: str) -> list[str]:
        tokens = _TOKEN_PATTERN.findall(text.casefold())
        terms: list[str] = []
        for size in range(self.ngram_min, self.ngram_max + 1):
            terms.extend(
                " ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)
            )
        return terms

    def fit(self, texts: list[str]) -> TfidfEmbeddingModel:
        if not texts:
            raise ValueError("At least one text is required to fit embeddings")

        document_frequency: Counter[str] = Counter()
        for text in texts:
            document_frequency.update(set(self._extract_terms(text)))

        self.document_count = len(texts)
        self.terms = sorted(
            term
            for term, frequency in document_frequency.items()
            if frequency >= self.min_document_frequency
        )
        self._term_indexes = {term: index for index, term in enumerate(self.terms)}
        self.idf = [
            math.log((1 + self.document_count) / (1 + document_frequency[term])) + 1
            for term in self.terms
        ]
        return self

    def transform(self, text: str) -> SparseVector:
        if not self._term_indexes:
            raise RuntimeError("The embedding model must be fitted before transform")

        counts = Counter(term for term in self._extract_terms(text) if term in self._term_indexes)
        weights = {
            self._term_indexes[term]: (1 + math.log(count)) * self.idf[self._term_indexes[term]]
            for term, count in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in weights.values()))
        if norm == 0:
            return {}
        return {index: value / norm for index, value in weights.items()}

    def transform_many(self, texts: list[str]) -> list[SparseVector]:
        return [self.transform(text) for text in texts]

    def fit_transform(self, texts: list[str]) -> list[SparseVector]:
        self.fit(texts)
        return self.transform_many(texts)

    def to_state(self) -> dict[str, Any]:
        if not self._term_indexes:
            raise RuntimeError("Cannot serialise an unfitted embedding model")
        return {
            "type": self.model_type,
            "ngram_min": self.ngram_min,
            "ngram_max": self.ngram_max,
            "min_document_frequency": self.min_document_frequency,
            "document_count": self.document_count,
            "terms": self.terms,
            "idf": self.idf,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> TfidfEmbeddingModel:
        if state.get("type") != cls.model_type:
            raise ValueError(f"Unsupported embedding model type: {state.get('type')}")

        model = cls(
            ngram_min=state["ngram_min"],
            ngram_max=state["ngram_max"],
            min_document_frequency=state["min_document_frequency"],
        )
        model.document_count = state["document_count"]
        model.terms = list(state["terms"])
        model.idf = list(state["idf"])
        if len(model.terms) != len(model.idf):
            raise ValueError("Embedding terms and IDF values have different lengths")
        model._term_indexes = {term: index for index, term in enumerate(model.terms)}
        return model

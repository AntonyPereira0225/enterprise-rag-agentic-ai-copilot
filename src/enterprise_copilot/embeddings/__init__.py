"""Replaceable text-embedding implementations."""

from enterprise_copilot.embeddings.sentence_transformer import (
    DenseDependencyError,
    SentenceTransformerEmbeddingModel,
)
from enterprise_copilot.embeddings.tfidf import TfidfEmbeddingModel

__all__ = [
    "DenseDependencyError",
    "SentenceTransformerEmbeddingModel",
    "TfidfEmbeddingModel",
]

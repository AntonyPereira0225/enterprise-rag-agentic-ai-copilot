from __future__ import annotations

from typing import Any, Protocol

SparseVector = dict[int, float]
DenseVector = list[float]


class EmbeddingModel(Protocol):
    """Small interface that future local or hosted embedding models can implement."""

    @property
    def dimension(self) -> int: ...

    def transform(self, text: str) -> SparseVector: ...

    def transform_many(self, texts: list[str]) -> list[SparseVector]: ...

    def to_state(self) -> dict[str, Any]: ...

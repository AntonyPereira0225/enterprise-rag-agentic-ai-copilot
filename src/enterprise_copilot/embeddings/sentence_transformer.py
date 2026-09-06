from __future__ import annotations

from typing import Any

from enterprise_copilot.embeddings.base import DenseVector


class DenseDependencyError(ImportError):
    """Raised when optional dense-retrieval packages are unavailable."""


class SentenceTransformerEmbeddingModel:
    """Lazy Sentence Transformers adapter that produces normalised dense vectors."""

    model_type = "sentence_transformer"

    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        backend: Any | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._backend = backend or self._load_backend()

    def _load_backend(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise DenseDependencyError(
                "Dense retrieval requires the optional packages in requirements-dense.txt. "
                "Install them, then rerun the dense-index command."
            ) from exc
        return SentenceTransformer(self.model_name, device=self.device)

    @property
    def dimension(self) -> int:
        dimension = self._backend.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError("The sentence-transformer backend did not report a dimension")
        return int(dimension)

    def transform_many(self, texts: list[str]) -> list[DenseVector]:
        if not texts:
            return []
        encoded = self._backend.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        rows = encoded.tolist() if hasattr(encoded, "tolist") else encoded
        return [[float(value) for value in row] for row in rows]

    def transform(self, text: str) -> DenseVector:
        return self.transform_many([text])[0]

    def to_state(self) -> dict[str, Any]:
        return {
            "type": self.model_type,
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "dimension": self.dimension,
        }

    @classmethod
    def from_state(
        cls, state: dict[str, Any], *, backend: Any | None = None
    ) -> SentenceTransformerEmbeddingModel:
        if state.get("type") != cls.model_type:
            raise ValueError(f"Unsupported dense embedding type: {state.get('type')}")
        model = cls(
            model_name=state["model_name"],
            device=state.get("device", "cpu"),
            batch_size=state.get("batch_size", 32),
            backend=backend,
        )
        if model.dimension != state["dimension"]:
            raise ValueError("Saved and loaded embedding dimensions do not match")
        return model

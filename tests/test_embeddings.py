import math

import pytest

from enterprise_copilot.embeddings.tfidf import TfidfEmbeddingModel


def _dot(left: dict[int, float], right: dict[int, float]) -> float:
    return sum(left.get(index, 0.0) * value for index, value in right.items())


def test_tfidf_embeddings_are_normalised_and_relevant() -> None:
    texts = [
        "refund policy for NSG Connect Ireland",
        "service outage playbook for NSG Home Germany",
    ]
    model = TfidfEmbeddingModel(ngram_min=1, ngram_max=2)
    vectors = model.fit_transform(texts)
    query = model.transform("NSG Connect refund policy Ireland")

    assert math.sqrt(sum(value * value for value in query.values())) == pytest.approx(1.0)
    assert _dot(query, vectors[0]) > _dot(query, vectors[1])


def test_tfidf_state_round_trip_preserves_vectors() -> None:
    model = TfidfEmbeddingModel()
    model.fit(["one two three", "three four five"])
    expected = model.transform("two three")

    restored = TfidfEmbeddingModel.from_state(model.to_state())

    assert restored.transform("two three") == expected
    assert restored.dimension == model.dimension


def test_tfidf_rejects_invalid_ngram_range() -> None:
    with pytest.raises(ValueError, match="greater than or equal"):
        TfidfEmbeddingModel(ngram_min=2, ngram_max=1)

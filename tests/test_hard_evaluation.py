from pathlib import Path

import pytest

from enterprise_copilot.evaluation.hard_questions import (
    HardEvaluationConfig,
    generate_hard_questions,
)
from enterprise_copilot.evaluation.retrieval import evaluate_retrieval
from enterprise_copilot.ingestion.synthetic_corpus import CorpusConfig, generate_corpus
from enterprise_copilot.retrieval.vector_index import SearchResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StubRetriever:
    def __init__(self, results: dict[str, list[SearchResult]]) -> None:
        self.results = results

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        return self.results[query][:top_k]


def _result(score: float, document_id: str, content: str) -> SearchResult:
    return SearchResult(
        score=score,
        chunk={
            "chunk_id": f"{document_id}-chunk",
            "document_id": document_id,
            "content": content,
            "source_uri": f"nsg://{document_id}",
            "document_type": "policy",
            "region": "Ireland",
            "product": "NSG Connect",
        },
    )


def test_hard_question_generation_is_deterministic_and_balanced() -> None:
    corpus = generate_corpus(
        CorpusConfig.from_json(PROJECT_ROOT / "configs" / "corpus_config.json")
    )
    config = HardEvaluationConfig(
        source_path="unused",
        output_path="unused",
        paraphrased_count=72,
        multi_source_count=12,
        unanswerable_count=12,
    )

    first = generate_hard_questions(corpus.knowledge_documents, config)
    second = generate_hard_questions(corpus.knowledge_documents, config)

    assert first == second
    assert len(first) == 96
    assert len({row["question"] for row in first}) == 96
    assert sum(row["difficulty"] == "multi_source" for row in first) == 12
    assert sum(not row["answerable"] for row in first) == 12


def test_evaluator_measures_all_evidence_and_abstention() -> None:
    questions = [
        {
            "question_id": "A",
            "question": "answerable",
            "expected_answer": "seven days and two checks",
            "expected_document_ids": ["D1", "D2"],
            "expected_evidence": [
                {"document_id": "D1", "answer": "seven days"},
                {"document_id": "D2", "answer": "two checks"},
            ],
            "answerable": True,
        },
        {
            "question_id": "U",
            "question": "unsupported",
            "expected_answer": None,
            "expected_document_ids": [],
            "expected_evidence": [],
            "answerable": False,
        },
    ]
    retriever = StubRetriever(
        {
            "answerable": [
                _result(0.9, "D1", "The requirement is seven days."),
                _result(0.8, "D2", "The requirement is two checks."),
            ],
            "unsupported": [_result(0.1, "D3", "Unrelated material")],
        }
    )

    results = evaluate_retrieval(retriever, questions, [1, 2], minimum_score=0.2)
    summary = results["summary"]

    assert summary["evidence_recall_at_k"]["1"] == pytest.approx(0.5)
    assert summary["all_evidence_recalled_at_k"]["1"] == 0.0
    assert summary["all_evidence_recalled_at_k"]["2"] == 1.0
    assert summary["unanswerable_abstention_accuracy"] == 1.0
    assert summary["answerability_accuracy"] == 1.0

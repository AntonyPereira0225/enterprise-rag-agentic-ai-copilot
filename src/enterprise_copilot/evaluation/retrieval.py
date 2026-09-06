from __future__ import annotations

from typing import Any

from enterprise_copilot.retrieval.vector_index import VectorIndex


def _first_rank(values: list[bool]) -> int | None:
    return next((index for index, is_relevant in enumerate(values, start=1) if is_relevant), None)


def evaluate_retrieval(
    index: VectorIndex,
    questions: list[dict[str, Any]],
    top_k_values: list[int],
    minimum_score: float = 0.0,
) -> dict[str, Any]:
    """Measure evidence coverage and answerability at configured cut-offs."""
    if not questions:
        raise ValueError("At least one evaluation question is required")
    if not top_k_values or any(value <= 0 for value in top_k_values):
        raise ValueError("top_k_values must contain positive integers")

    cutoffs = sorted(set(top_k_values))
    maximum_k = max(cutoffs)
    question_results: list[dict[str, Any]] = []

    for question in questions:
        answerable = question.get("answerable", True)
        expected_document_ids = set(question["expected_document_ids"])
        expected_evidence = question.get("expected_evidence") or [
            {"document_id": document_id, "answer": question["expected_answer"]}
            for document_id in expected_document_ids
        ]
        results = index.search(question["question"], top_k=maximum_k)
        top_score = results[0].score if results else 0.0
        predicted_answerable = bool(results) and top_score >= minimum_score

        document_ranks = {
            document_id: _first_rank(
                [result.chunk["document_id"] == document_id for result in results]
            )
            for document_id in expected_document_ids
        }
        evidence_ranks = [
            {
                **evidence,
                "rank": _first_rank(
                    [
                        result.chunk["document_id"] == evidence["document_id"]
                        and evidence["answer"].casefold() in result.chunk["content"].casefold()
                        for result in results
                    ]
                ),
            }
            for evidence in expected_evidence
        ]
        found_document_ranks = [rank for rank in document_ranks.values() if rank is not None]
        found_evidence_ranks = [
            evidence["rank"] for evidence in evidence_ranks if evidence["rank"] is not None
        ]

        question_results.append(
            {
                "question_id": question["question_id"],
                "question": question["question"],
                "difficulty": question.get("difficulty", "unspecified"),
                "answerable": answerable,
                "predicted_answerable": predicted_answerable,
                "top_score": top_score,
                "expected_document_ids": sorted(expected_document_ids),
                "expected_answer": question["expected_answer"],
                "document_ranks": document_ranks,
                "evidence_ranks": evidence_ranks,
                "first_document_rank": min(found_document_ranks, default=None),
                "first_evidence_rank": min(found_evidence_ranks, default=None),
                "retrieved": [result.to_dict() for result in results],
            }
        )

    answerable_results = [row for row in question_results if row["answerable"]]
    unanswerable_results = [row for row in question_results if not row["answerable"]]
    count = len(answerable_results)
    if count == 0:
        raise ValueError("At least one answerable evaluation question is required")

    document_recall = {
        str(k): sum(
            sum(rank is not None and rank <= k for rank in row["document_ranks"].values())
            / len(row["document_ranks"])
            for row in answerable_results
        )
        / count
        for k in cutoffs
    }
    evidence_recall = {
        str(k): sum(
            sum(item["rank"] is not None and item["rank"] <= k for item in row["evidence_ranks"])
            / len(row["evidence_ranks"])
            for row in answerable_results
        )
        / count
        for k in cutoffs
    }
    all_document_recall = {
        str(k): sum(
            all(rank is not None and rank <= k for rank in row["document_ranks"].values())
            for row in answerable_results
        )
        / count
        for k in cutoffs
    }
    all_evidence_recall = {
        str(k): sum(
            all(item["rank"] is not None and item["rank"] <= k for item in row["evidence_ranks"])
            for row in answerable_results
        )
        / count
        for k in cutoffs
    }
    mean_reciprocal_rank = (
        sum(
            0.0 if row["first_evidence_rank"] is None else 1.0 / row["first_evidence_rank"]
            for row in answerable_results
        )
        / count
    )
    answerability_accuracy = sum(
        row["answerable"] == row["predicted_answerable"] for row in question_results
    ) / len(question_results)
    answerable_acceptance = sum(row["predicted_answerable"] for row in answerable_results) / count
    unanswerable_abstention = (
        sum(not row["predicted_answerable"] for row in unanswerable_results)
        / len(unanswerable_results)
        if unanswerable_results
        else None
    )

    return {
        "summary": {
            "evaluated_questions": len(question_results),
            "answerable_questions": count,
            "unanswerable_questions": len(unanswerable_results),
            "top_k_values": cutoffs,
            "minimum_score": minimum_score,
            "document_recall_at_k": document_recall,
            "evidence_recall_at_k": evidence_recall,
            "all_documents_recalled_at_k": all_document_recall,
            "all_evidence_recalled_at_k": all_evidence_recall,
            "evidence_mean_reciprocal_rank": mean_reciprocal_rank,
            "answerability_accuracy": answerability_accuracy,
            "answerable_acceptance_accuracy": answerable_acceptance,
            "unanswerable_abstention_accuracy": unanswerable_abstention,
        },
        "questions": question_results,
    }

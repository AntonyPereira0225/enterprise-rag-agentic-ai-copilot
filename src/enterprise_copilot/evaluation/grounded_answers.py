from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from enterprise_copilot.ingestion.loaders import load_jsonl
from enterprise_copilot.llm.pipeline import (
    GroundedAnswerConfig,
    GroundedAnswerPipeline,
    load_grounded_answer_pipeline,
)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate_grounded_answers(
    pipeline: GroundedAnswerPipeline,
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    if not questions:
        raise ValueError("At least one evaluation question is required")

    question_results: list[dict[str, Any]] = []
    for question in questions:
        run = pipeline.ask(question["question"])
        response = run.response
        expected_evidence = question.get("expected_evidence", [])
        evidence_checks = []
        for expected in expected_evidence:
            matching_citations = [
                citation
                for citation in response.citations
                if citation.document_id == expected["document_id"]
            ]
            covered = any(
                expected["answer"].casefold() in citation.quote.casefold()
                for citation in matching_citations
            )
            evidence_checks.append({**expected, "covered": covered})

        expected_document_ids = set(question["expected_document_ids"])
        predicted_answerable = response.status == "answered"
        correct_sources = sum(
            citation.document_id in expected_document_ids for citation in response.citations
        )
        question_results.append(
            {
                "question_id": question["question_id"],
                "question": question["question"],
                "difficulty": question.get("difficulty", "unspecified"),
                "answerable": question.get("answerable", True),
                "predicted_answerable": predicted_answerable,
                "answerability_correct": (question.get("answerable", True) == predicted_answerable),
                "expected_answer": question.get("expected_answer"),
                "expected_document_ids": sorted(expected_document_ids),
                "evidence_checks": evidence_checks,
                "covered_evidence_count": sum(item["covered"] for item in evidence_checks),
                "all_expected_evidence_covered": bool(evidence_checks)
                and all(item["covered"] for item in evidence_checks),
                "correct_source_citations": correct_sources,
                "citation_count": len(response.citations),
                "citation_verification": run.verification.to_dict(),
                "response": response.to_dict(),
                "context": run.context.to_dict(),
            }
        )

    answerable_rows = [row for row in question_results if row["answerable"]]
    unanswerable_rows = [row for row in question_results if not row["answerable"]]
    evidence_total = sum(len(row["evidence_checks"]) for row in answerable_rows)
    evidence_covered = sum(row["covered_evidence_count"] for row in answerable_rows)
    citation_total = sum(row["citation_count"] for row in answerable_rows)
    correct_source_citations = sum(row["correct_source_citations"] for row in answerable_rows)

    by_difficulty: dict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "correct": 0, "verified": 0, "all_evidence": 0}
    )
    for row in question_results:
        group = by_difficulty[row["difficulty"]]
        group["count"] += 1
        group["correct"] += int(row["answerability_correct"])
        group["verified"] += int(row["citation_verification"]["valid"])
        group["all_evidence"] += int(row["all_expected_evidence_covered"])

    difficulty_summary = {
        name: {
            "questions": values["count"],
            "answerability_accuracy": values["correct"] / values["count"],
            "citation_verification_accuracy": values["verified"] / values["count"],
            "all_expected_evidence_covered": _safe_ratio(
                values["all_evidence"],
                sum(row["answerable"] and row["difficulty"] == name for row in question_results),
            ),
        }
        for name, values in sorted(by_difficulty.items())
    }

    return {
        "summary": {
            "evaluated_questions": len(question_results),
            "answerable_questions": len(answerable_rows),
            "unanswerable_questions": len(unanswerable_rows),
            "answerability_accuracy": sum(row["answerability_correct"] for row in question_results)
            / len(question_results),
            "answerable_acceptance_accuracy": _safe_ratio(
                sum(row["predicted_answerable"] for row in answerable_rows),
                len(answerable_rows),
            ),
            "unanswerable_refusal_accuracy": _safe_ratio(
                sum(not row["predicted_answerable"] for row in unanswerable_rows),
                len(unanswerable_rows),
            ),
            "citation_verification_accuracy": sum(
                row["citation_verification"]["valid"] for row in question_results
            )
            / len(question_results),
            "expected_evidence_coverage": _safe_ratio(evidence_covered, evidence_total),
            "all_expected_evidence_covered": _safe_ratio(
                sum(row["all_expected_evidence_covered"] for row in answerable_rows),
                len(answerable_rows),
            ),
            "citation_source_precision": _safe_ratio(
                correct_source_citations,
                citation_total,
            ),
            "by_difficulty": difficulty_summary,
        },
        "questions": question_results,
    }


def run_grounded_answer_evaluation(
    project_root: Path,
    config: GroundedAnswerConfig,
) -> dict[str, Any]:
    pipeline = load_grounded_answer_pipeline(project_root, config)
    questions = load_jsonl(project_root / config.evaluation_path)
    results = evaluate_grounded_answers(pipeline, questions)
    results["configuration"] = {
        "retrieval_config_path": config.retrieval_config_path,
        "top_k": config.top_k,
        "minimum_query_score": config.minimum_query_score,
        "minimum_evidence_score": config.minimum_evidence_score,
        "max_context_words": config.max_context_words,
        "max_answer_citations": config.max_answer_citations,
    }
    results_path = project_root / config.results_path
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return results

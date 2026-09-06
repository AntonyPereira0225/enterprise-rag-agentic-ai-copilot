from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from enterprise_copilot.evaluation.retrieval import evaluate_retrieval
from enterprise_copilot.ingestion.loaders import load_jsonl
from enterprise_copilot.retrieval.bm25_index import BM25Index
from enterprise_copilot.retrieval.hybrid import HybridRetriever
from enterprise_copilot.retrieval.hybrid_pipeline import HybridRetrievalConfig
from enterprise_copilot.retrieval.reranking import MetadataAwareReranker, RerankerWeights
from enterprise_copilot.retrieval.vector_index import VectorIndex


@dataclass(frozen=True)
class HybridTuningConfig:
    calibration_stride: int
    rrf_constants: list[int]
    vector_weights: list[float]
    bm25_weight: float
    minimum_scores: list[float]
    reranker_weight_profiles: list[dict[str, float]]
    output_path: str

    @classmethod
    def from_json(cls, path: Path) -> HybridTuningConfig:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def stratified_calibration_split(
    questions: list[dict[str, Any]], stride: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select every Nth item within each difficulty for calibration."""
    if stride < 2:
        raise ValueError("calibration_stride must be at least 2")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for question in questions:
        grouped.setdefault(question.get("difficulty", "unspecified"), []).append(question)

    calibration: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for difficulty in sorted(grouped):
        ordered = sorted(grouped[difficulty], key=lambda row: row["question_id"])
        for index, question in enumerate(ordered, start=1):
            target = calibration if index % stride == 0 else holdout
            target.append(question)
    return calibration, holdout


def _objective(summary: dict[str, Any]) -> float:
    abstention = summary["unanswerable_abstention_accuracy"]
    balanced_answerability = (
        summary["answerable_acceptance_accuracy"] + (abstention if abstention is not None else 1.0)
    ) / 2
    return (
        0.45 * summary["evidence_mean_reciprocal_rank"]
        + 0.25 * summary["evidence_recall_at_k"]["1"]
        + 0.30 * balanced_answerability
    )


def _threshold_margin(results: dict[str, Any], minimum_score: float) -> float:
    """Prefer a threshold centred away from nearby calibration scores."""
    return min(
        (abs(row["top_score"] - minimum_score) for row in results["questions"]),
        default=0.0,
    )


def _retriever(
    vector_index: VectorIndex,
    bm25_index: BM25Index,
    base: HybridRetrievalConfig,
    *,
    rrf_constant: int,
    vector_weight: float,
    reranker_weights: dict[str, float],
) -> HybridRetriever:
    return HybridRetriever(
        vector_index,
        bm25_index,
        MetadataAwareReranker(RerankerWeights.from_dict(reranker_weights)),
        candidate_pool_size=base.candidate_pool_size,
        rrf_constant=rrf_constant,
        vector_weight=vector_weight,
        bm25_weight=base.bm25_weight,
        max_chunks_per_document=base.max_chunks_per_document,
    )


def tune_hybrid_retrieval(
    vector_index: VectorIndex,
    bm25_index: BM25Index,
    questions: list[dict[str, Any]],
    base: HybridRetrievalConfig,
    tuning: HybridTuningConfig,
) -> dict[str, Any]:
    calibration, holdout = stratified_calibration_split(questions, tuning.calibration_stride)
    trials: list[dict[str, Any]] = []

    combinations = itertools.product(
        tuning.rrf_constants,
        tuning.vector_weights,
        tuning.reranker_weight_profiles,
        tuning.minimum_scores,
    )
    for rrf_constant, vector_weight, reranker_weights, minimum_score in combinations:
        retriever = _retriever(
            vector_index,
            bm25_index,
            base,
            rrf_constant=rrf_constant,
            vector_weight=vector_weight,
            reranker_weights=reranker_weights,
        )
        results = evaluate_retrieval(
            retriever,
            calibration,
            base.top_k_values,
            minimum_score=minimum_score,
        )
        trials.append(
            {
                "objective": _objective(results["summary"]),
                "threshold_margin": _threshold_margin(results, minimum_score),
                "rrf_constant": rrf_constant,
                "vector_weight": vector_weight,
                "bm25_weight": tuning.bm25_weight,
                "minimum_score": minimum_score,
                "reranker_weights": reranker_weights,
                "summary": results["summary"],
            }
        )

    trials.sort(
        key=lambda row: (
            -row["objective"],
            -row["threshold_margin"],
            row["rrf_constant"],
            row["vector_weight"],
            row["minimum_score"],
            json.dumps(row["reranker_weights"], sort_keys=True),
        )
    )
    best = trials[0]
    best_retriever = _retriever(
        vector_index,
        bm25_index,
        base,
        rrf_constant=best["rrf_constant"],
        vector_weight=best["vector_weight"],
        reranker_weights=best["reranker_weights"],
    )
    holdout_results = evaluate_retrieval(
        best_retriever,
        holdout,
        base.top_k_values,
        minimum_score=best["minimum_score"],
    )

    baseline_trials = []
    for minimum_score in tuning.minimum_scores:
        baseline_results = evaluate_retrieval(
            vector_index,
            calibration,
            base.top_k_values,
            minimum_score=minimum_score,
        )
        baseline_trials.append(
            {
                "objective": _objective(baseline_results["summary"]),
                "threshold_margin": _threshold_margin(baseline_results, minimum_score),
                "minimum_score": minimum_score,
                "summary": baseline_results["summary"],
            }
        )
    baseline_trials.sort(
        key=lambda row: (
            -row["objective"],
            -row["threshold_margin"],
            row["minimum_score"],
        )
    )
    best_baseline = baseline_trials[0]
    baseline_holdout = evaluate_retrieval(
        vector_index,
        holdout,
        base.top_k_values,
        minimum_score=best_baseline["minimum_score"],
    )["summary"]
    hybrid_holdout = holdout_results["summary"]
    comparison = {
        **{
            f"evidence_recall_at_{k}_delta": (
                hybrid_holdout["evidence_recall_at_k"][str(k)]
                - baseline_holdout["evidence_recall_at_k"][str(k)]
            )
            for k in base.top_k_values
        },
        "evidence_mrr_delta": (
            hybrid_holdout["evidence_mean_reciprocal_rank"]
            - baseline_holdout["evidence_mean_reciprocal_rank"]
        ),
        "answerability_accuracy_delta": (
            hybrid_holdout["answerability_accuracy"] - baseline_holdout["answerability_accuracy"]
        ),
    }
    return {
        "objective": (
            "0.45 * evidence MRR + 0.25 * Evidence Recall@1 + "
            "0.30 * balanced answerability accuracy"
        ),
        "split": {
            "calibration_stride": tuning.calibration_stride,
            "calibration_questions": len(calibration),
            "holdout_questions": len(holdout),
        },
        "best_parameters": {
            key: best[key]
            for key in (
                "rrf_constant",
                "vector_weight",
                "bm25_weight",
                "minimum_score",
                "reranker_weights",
            )
        },
        "calibration_summary": best["summary"],
        "holdout_summary": hybrid_holdout,
        "baseline": {
            "minimum_score": best_baseline["minimum_score"],
            "calibration_summary": best_baseline["summary"],
            "holdout_summary": baseline_holdout,
        },
        "holdout_delta_vs_baseline": comparison,
        "top_trials": trials[:10],
        "tuning_config": asdict(tuning),
    }


def run_hybrid_tuning(
    project_root: Path,
    base: HybridRetrievalConfig,
    tuning: HybridTuningConfig,
) -> dict[str, Any]:
    vector_index = VectorIndex.load(project_root / base.vector_index_path)
    bm25_index = BM25Index.load(project_root / base.bm25_index_path)
    questions = load_jsonl(project_root / base.evaluation_path)
    results = tune_hybrid_retrieval(vector_index, bm25_index, questions, base, tuning)
    output_path = project_root / tuning.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return results

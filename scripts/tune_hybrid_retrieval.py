from pathlib import Path

from enterprise_copilot.retrieval.hybrid_pipeline import HybridRetrievalConfig
from enterprise_copilot.retrieval.tuning import HybridTuningConfig, run_hybrid_tuning


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    base = HybridRetrievalConfig.from_json(
        project_root / "configs" / "hybrid_retrieval_config.json"
    )
    tuning = HybridTuningConfig.from_json(project_root / "configs" / "hybrid_tuning_config.json")
    results = run_hybrid_tuning(project_root, base, tuning)
    best = results["best_parameters"]
    holdout = results["holdout_summary"]

    print("Hybrid retrieval calibration completed successfully")
    print(
        f"Calibration questions: {results['split']['calibration_questions']}; "
        f"holdout questions: {results['split']['holdout_questions']}"
    )
    print(f"Recommended RRF constant: {best['rrf_constant']}")
    print(f"Recommended vector weight: {best['vector_weight']}")
    print(f"Recommended minimum score: {best['minimum_score']}")
    print(f"Holdout Evidence Recall@1: {holdout['evidence_recall_at_k']['1']:.1%}")
    print(f"Holdout Evidence Recall@3: {holdout['evidence_recall_at_k']['3']:.1%}")
    print(f"Holdout Evidence MRR: {holdout['evidence_mean_reciprocal_rank']:.3f}")
    print(f"Holdout answerability accuracy: {holdout['answerability_accuracy']:.1%}")
    baseline = results["baseline"]["holdout_summary"]
    print(
        f"TF-IDF holdout Recall@1/@3: {baseline['evidence_recall_at_k']['1']:.1%} / "
        f"{baseline['evidence_recall_at_k']['3']:.1%}"
    )
    print(f"Detailed tuning results: {tuning.output_path}")


if __name__ == "__main__":
    main()

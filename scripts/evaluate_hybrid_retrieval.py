from pathlib import Path

from enterprise_copilot.retrieval.hybrid_pipeline import (
    HybridRetrievalConfig,
    run_hybrid_evaluation,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = HybridRetrievalConfig.from_json(
        project_root / "configs" / "hybrid_retrieval_config.json"
    )
    results = run_hybrid_evaluation(project_root, config)
    summary = results["summary"]

    print("Hybrid retrieval evaluation completed successfully")
    print(f"Questions evaluated: {summary['evaluated_questions']}")
    for k in summary["top_k_values"]:
        print(
            f"Evidence Recall@{k}: {summary['evidence_recall_at_k'][str(k)]:.1%}; "
            f"all evidence: {summary['all_evidence_recalled_at_k'][str(k)]:.1%}"
        )
    print(f"Evidence MRR: {summary['evidence_mean_reciprocal_rank']:.3f}")
    print(f"Answerability accuracy: {summary['answerability_accuracy']:.1%}")
    print(f"Unanswerable abstention accuracy: {summary['unanswerable_abstention_accuracy']:.1%}")
    print(f"Detailed results: {config.results_path}")


if __name__ == "__main__":
    main()

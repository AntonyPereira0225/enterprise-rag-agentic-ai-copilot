from pathlib import Path

from enterprise_copilot.retrieval.pipeline import RetrievalConfig, run_retrieval_evaluation


def _percentage(value: float) -> str:
    return f"{value:.1%}"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = RetrievalConfig.from_json(project_root / "configs" / "retrieval_config.json")
    results = run_retrieval_evaluation(project_root, config)
    summary = results["summary"]

    print("Retrieval evaluation completed successfully")
    print(f"Questions evaluated: {summary['evaluated_questions']}")
    for k in summary["top_k_values"]:
        print(
            f"Recall@{k}: document={_percentage(summary['document_recall_at_k'][str(k)])}, "
            f"evidence={_percentage(summary['evidence_recall_at_k'][str(k)])}"
        )
    print(f"Evidence MRR: {summary['evidence_mean_reciprocal_rank']:.3f}")
    if summary["unanswerable_questions"]:
        print(f"Answerability accuracy: {_percentage(summary['answerability_accuracy'])}")
        print(
            "Unanswerable abstention accuracy: "
            f"{_percentage(summary['unanswerable_abstention_accuracy'])}"
        )
    print(f"Detailed results: {config.results_path}")


if __name__ == "__main__":
    main()

from pathlib import Path

from enterprise_copilot.embeddings.sentence_transformer import DenseDependencyError
from enterprise_copilot.retrieval.dense_pipeline import run_dense_retrieval_evaluation
from enterprise_copilot.retrieval.pipeline import RetrievalConfig


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = RetrievalConfig.from_json(project_root / "configs" / "dense_retrieval_config.json")
    results = run_dense_retrieval_evaluation(project_root, config)
    summary = results["summary"]
    print("Dense retrieval evaluation completed successfully")
    print(f"Questions evaluated: {summary['evaluated_questions']}")
    for k in summary["top_k_values"]:
        print(f"Evidence Recall@{k}: {summary['evidence_recall_at_k'][str(k)]:.1%}")
    print(f"Evidence MRR: {summary['evidence_mean_reciprocal_rank']:.3f}")
    print(f"Answerability accuracy: {summary['answerability_accuracy']:.1%}")
    print(f"Detailed results: {config.results_path}")


if __name__ == "__main__":
    try:
        main()
    except DenseDependencyError as exc:
        raise SystemExit(str(exc)) from exc

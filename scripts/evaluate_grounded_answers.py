from pathlib import Path

from enterprise_copilot.evaluation.grounded_answers import run_grounded_answer_evaluation
from enterprise_copilot.llm.pipeline import GroundedAnswerConfig


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = GroundedAnswerConfig.from_json(
        project_root / "configs" / "grounded_answer_config.json"
    )
    results = run_grounded_answer_evaluation(project_root, config)
    summary = results["summary"]

    print("Grounded-answer evaluation completed successfully")
    print(f"Questions evaluated: {summary['evaluated_questions']}")
    print(f"Answerability accuracy: {summary['answerability_accuracy']:.1%}")
    print(f"Unanswerable refusal accuracy: {summary['unanswerable_refusal_accuracy']:.1%}")
    print(f"Expected evidence coverage: {summary['expected_evidence_coverage']:.1%}")
    print(f"All expected evidence covered: {summary['all_expected_evidence_covered']:.1%}")
    print(f"Citation verification accuracy: {summary['citation_verification_accuracy']:.1%}")
    print(f"Detailed results: {config.results_path}")


if __name__ == "__main__":
    main()

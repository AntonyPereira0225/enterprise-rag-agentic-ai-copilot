from pathlib import Path

from enterprise_copilot.evaluation.model_integration import (
    run_model_integration_validation,
)
from enterprise_copilot.llm.pipeline import GroundedAnswerConfig


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = GroundedAnswerConfig.from_json(
        project_root / "configs" / "grounded_answer_config.json"
    )
    results = run_model_integration_validation(project_root, config)
    summary = results["summary"]

    print("Optional local-model integration validation completed")
    print(f"Checks passed: {summary['passed']}/{summary['checks']}")
    print(f"Pass rate: {summary['pass_rate']:.1%}")
    print(f"Live Ollama required: {summary['live_ollama_required']}")
    print(f"Default backend: {summary['default_backend']}")
    print(f"Detailed results: {config.model_validation_results_path}")

    if summary["pass_rate"] != 1.0:
        raise SystemExit("Model integration validation failed")


if __name__ == "__main__":
    main()

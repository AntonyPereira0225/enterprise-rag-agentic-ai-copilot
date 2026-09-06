from pathlib import Path

from enterprise_copilot.api.service import ApiConfig
from enterprise_copilot.evaluation.api_validation import run_api_validation


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = ApiConfig.from_json(project_root / "configs" / "api_config.json")
    results = run_api_validation(project_root, config)
    summary = results["summary"]

    print("API and demo validation completed successfully")
    print(f"Checks passed: {summary['passed']}/{summary['checks']}")
    print(f"Pass rate: {summary['pass_rate']:.1%}")
    print(f"Structured log events: {summary['structured_log_events']}")
    print(f"Raw questions logged: {summary['raw_questions_logged']}")
    print(f"Detailed results: {config.validation_results_path}")


if __name__ == "__main__":
    main()

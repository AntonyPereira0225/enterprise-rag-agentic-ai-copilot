from pathlib import Path

from enterprise_copilot.evaluation.mlops_validation import run_mlops_validation
from enterprise_copilot.monitoring.experiment import ObservabilityConfig


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = ObservabilityConfig.from_json(project_root / "configs" / "observability_config.json")
    results = run_mlops_validation(project_root, config)
    summary = results["summary"]

    print("Observability and delivery validation completed")
    print(f"Checks passed: {summary['passed']}/{summary['checks']}")
    print(f"Pass rate: {summary['pass_rate']:.1%}")
    print(f"Docker available locally: {summary['docker_available']}")
    print(f"Container validation: {summary['container_definition_validation']}")
    print(f"Detailed results: {config.validation_results_path}")

    if summary["pass_rate"] != 1.0:
        raise SystemExit("Observability and delivery validation failed")


if __name__ == "__main__":
    main()

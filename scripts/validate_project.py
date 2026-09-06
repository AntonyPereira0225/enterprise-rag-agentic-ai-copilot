from pathlib import Path

from enterprise_copilot.evaluation.project_completion import (
    ProjectCompletionConfig,
    run_project_completion_validation,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = ProjectCompletionConfig.from_json(
        project_root / "configs" / "project_completion_config.json"
    )
    results = run_project_completion_validation(project_root, config)
    summary = results["summary"]

    print("Final project acceptance validation completed")
    print(f"Checks passed: {summary['passed']}/{summary['checks']}")
    print(f"Pass rate: {summary['pass_rate']:.1%}")
    print(
        "Requirements covered: "
        f"{summary['functional_requirements_covered']}/"
        f"{summary['functional_requirements_total']} functional, "
        f"{summary['nonfunctional_requirements_covered']}/"
        f"{summary['nonfunctional_requirements_total']} non-functional"
    )
    print(f"Release version: {summary['release_version']}")
    print(f"Detailed results: {config.results_path}")

    if summary["pass_rate"] != 1.0:
        raise SystemExit("Final project acceptance validation failed")


if __name__ == "__main__":
    main()

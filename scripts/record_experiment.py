from pathlib import Path

from enterprise_copilot.monitoring.experiment import (
    ObservabilityConfig,
    record_project_experiment,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = ObservabilityConfig.from_json(project_root / "configs" / "observability_config.json")
    record = record_project_experiment(project_root, config)

    print("Release experiment recorded successfully")
    print(f"Backend: {record.backend}")
    print(f"Experiment: {record.experiment_name}")
    print(f"Run ID: {record.run_id}")
    print(f"Metrics recorded: {len(record.metrics)}")
    if record.record_path is not None:
        print(f"Record: {record.record_path.relative_to(project_root)}")
    else:
        print("Record: configured MLflow tracking store")


if __name__ == "__main__":
    main()

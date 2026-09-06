from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

from enterprise_copilot.monitoring.tracking import (
    LocalExperimentTracker,
    MlflowExperimentTracker,
)

REQUIRED_METRIC_SOURCES = {
    "hybrid_retrieval",
    "grounded_answers",
    "agent_workflow",
    "api_validation",
    "model_integration",
}


@dataclass(frozen=True)
class ProjectExperimentRecord:
    backend: str
    run_id: str
    experiment_name: str
    run_name: str
    status: str
    metrics: dict[str, float]
    record_path: Path | None


@dataclass(frozen=True)
class ObservabilityConfig:
    request_log_path: str
    metrics_snapshot_path: str
    experiment_root: str
    experiment_name: str
    tracking_backend: str
    metric_sources: dict[str, str]
    validation_results_path: str

    @classmethod
    def from_json(cls, path: Path) -> ObservabilityConfig:
        config = cls(**json.loads(path.read_text(encoding="utf-8")))
        if config.tracking_backend not in {"local_json", "mlflow"}:
            raise ValueError("tracking_backend must be local_json or mlflow")
        missing_sources = REQUIRED_METRIC_SOURCES - config.metric_sources.keys()
        if missing_sources:
            raise ValueError(f"Missing metric sources: {', '.join(sorted(missing_sources))}")
        return config


def collect_project_metrics(
    project_root: Path,
    sources: dict[str, str],
) -> tuple[dict[str, float], dict[str, str]]:
    payloads = {}
    for name, configured_path in sources.items():
        source_path = resolve_project_path(project_root, configured_path)
        payloads[name] = json.loads(source_path.read_text(encoding="utf-8"))
    hybrid = payloads["hybrid_retrieval"]["summary"]
    grounded = payloads["grounded_answers"]["summary"]
    agent = payloads["agent_workflow"]["summary"]
    api = payloads["api_validation"]["summary"]
    model = payloads["model_integration"]["summary"]
    metrics = {
        "hybrid_evidence_recall_at_1": hybrid["evidence_recall_at_k"]["1"],
        "hybrid_evidence_recall_at_5": hybrid["evidence_recall_at_k"]["5"],
        "hybrid_evidence_mrr": hybrid["evidence_mean_reciprocal_rank"],
        "hybrid_answerability_accuracy": hybrid["answerability_accuracy"],
        "grounded_evidence_coverage": grounded["expected_evidence_coverage"],
        "grounded_citation_verification": grounded["citation_verification_accuracy"],
        "agent_safety_pass_rate": agent["safety_pass_rate"],
        "agent_adversarial_block_rate": agent["adversarial_block_rate"],
        "api_validation_pass_rate": api["pass_rate"],
        "model_integration_pass_rate": model["pass_rate"],
    }
    artifacts = {name: Path(path).as_posix() for name, path in sources.items()}
    return {name: float(value) for name, value in metrics.items()}, artifacts


def record_project_experiment(
    project_root: Path,
    config: ObservabilityConfig,
) -> ProjectExperimentRecord:
    metrics, artifacts = collect_project_metrics(project_root, config.metric_sources)
    configuration_files = sorted((project_root / "configs").glob("*.json"))
    code_files = [
        path
        for directory in ("src", "scripts")
        for path in sorted((project_root / directory).rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    ]
    code_files.extend(
        path
        for path in (
            project_root / "pyproject.toml",
            project_root / "Dockerfile",
            project_root / "docker-compose.yml",
            project_root / ".github" / "workflows" / "ci.yml",
            *sorted(project_root.glob("requirements*.txt")),
        )
        if path.is_file()
    )
    data_files = [
        path
        for directory in ("raw", "processed", "vector_store")
        for path in sorted((project_root / "data" / directory).rglob("*"))
        if path.is_file()
    ]
    source_files = [resolve_project_path(project_root, path) for path in artifacts.values()]
    run_name = "v1.0-local-baseline"
    parameters: dict[str, str | int | float | bool] = {
        "tracking_backend": config.tracking_backend,
        "configuration_sha256": _manifest_sha256(project_root, configuration_files),
        "configuration_files": len(configuration_files),
        "code_sha256": _manifest_sha256(project_root, code_files),
        "generated_data_sha256": _manifest_sha256(project_root, data_files),
        "metric_artifacts_sha256": _manifest_sha256(project_root, source_files),
        "python_version": platform.python_version(),
        "deterministic_pipeline": True,
    }
    tags = {
        "data_classification": "synthetic",
        "execution_mode": "offline",
        "release": "1.0.0",
    }
    if config.tracking_backend == "local_json":
        tracker = LocalExperimentTracker(resolve_project_path(project_root, config.experiment_root))
        run, run_path = tracker.record(
            experiment_name=config.experiment_name,
            run_name=run_name,
            parameters=parameters,
            metrics=metrics,
            tags=tags,
            artifacts=artifacts,
            artifact_base=project_root,
        )
        return ProjectExperimentRecord(
            backend=config.tracking_backend,
            run_id=run.run_id,
            experiment_name=run.experiment_name,
            run_name=run.run_name,
            status=run.status,
            metrics=run.metrics,
            record_path=run_path,
        )

    mlflow_artifacts = {
        name: str(resolve_project_path(project_root, path)) for name, path in artifacts.items()
    }
    run_id = MlflowExperimentTracker().record(
        experiment_name=config.experiment_name,
        run_name=run_name,
        parameters=parameters,
        metrics=metrics,
        tags=tags,
        artifacts=mlflow_artifacts,
    )
    return ProjectExperimentRecord(
        backend=config.tracking_backend,
        run_id=run_id,
        experiment_name=config.experiment_name,
        run_name=run_name,
        status="FINISHED",
        metrics=metrics,
        record_path=None,
    )


def resolve_project_path(project_root: Path, configured_path: str | Path) -> Path:
    root = project_root.resolve()
    candidate = (root / configured_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Configured path leaves the project root: {configured_path}") from exc
    return candidate


def _manifest_sha256(project_root: Path, paths: list[Path]) -> str:
    manifest = [
        {
            "path": path.resolve().relative_to(project_root.resolve()).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    ]
    canonical = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()

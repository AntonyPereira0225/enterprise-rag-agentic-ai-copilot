from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class ExperimentRun:
    run_id: str
    experiment_name: str
    run_name: str
    recorded_at: str
    status: str
    parameters: dict[str, str | int | float | bool]
    metrics: dict[str, float]
    tags: dict[str, str]
    artifacts: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalExperimentTracker:
    """Dependency-free experiment records with an MLflow-like logical contract."""

    def __init__(
        self,
        root: Path,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.root = root
        self.now = now

    def record(
        self,
        *,
        experiment_name: str,
        run_name: str,
        parameters: dict[str, str | int | float | bool],
        metrics: dict[str, float],
        tags: dict[str, str] | None = None,
        artifacts: dict[str, str] | None = None,
        artifact_base: Path | None = None,
    ) -> tuple[ExperimentRun, Path]:
        if not experiment_name.strip() or not run_name.strip():
            raise ValueError("experiment_name and run_name cannot be empty")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._ -]{0,127}", experiment_name) is None:
            raise ValueError("experiment_name contains unsupported path characters")
        invalid_metrics = [
            name
            for name, value in metrics.items()
            if isinstance(value, bool) or not math.isfinite(float(value))
        ]
        if invalid_metrics:
            raise ValueError(f"Metrics must be finite numbers: {', '.join(invalid_metrics)}")

        recorded_at = self.now().astimezone(UTC).isoformat()
        fingerprint = json.dumps(
            {
                "experiment": experiment_name,
                "run": run_name,
                "recorded_at": recorded_at,
                "parameters": parameters,
                "metrics": metrics,
                "artifacts": artifacts or {},
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        run_id = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
        run_directory = self.root / experiment_name / run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        stored_artifacts = dict(artifacts or {})
        if artifact_base is not None:
            artifact_root = artifact_base.resolve()
            stored_artifacts = {}
            for label, configured_path in (artifacts or {}).items():
                if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", label) is None:
                    raise ValueError(f"Invalid artifact label: {label}")
                source = (artifact_root / configured_path).resolve()
                try:
                    source.relative_to(artifact_root)
                except ValueError as exc:
                    raise ValueError(
                        f"Artifact path leaves artifact_base: {configured_path}"
                    ) from exc
                if not source.is_file():
                    raise FileNotFoundError(source)
                destination = run_directory / "artifacts" / f"{label}{source.suffix}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                stored_artifacts[label] = destination.relative_to(self.root).as_posix()

        run = ExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            run_name=run_name,
            recorded_at=recorded_at,
            status="FINISHED",
            parameters=parameters,
            metrics={name: float(value) for name, value in metrics.items()},
            tags=tags or {},
            artifacts=stored_artifacts,
        )
        run_path = run_directory / "run.json"
        _write_json_atomic(run_path, run.to_dict())
        _write_json_atomic(
            self.root / experiment_name / "latest.json",
            {"run_id": run_id, "run_path": run_path.relative_to(self.root).as_posix()},
        )
        return run, run_path


class MlflowExperimentTracker:
    """Optional adapter preserving the local tracker's parameter/metric contract."""

    def record(
        self,
        *,
        experiment_name: str,
        run_name: str,
        parameters: dict[str, str | int | float | bool],
        metrics: dict[str, float],
        tags: dict[str, str] | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> str:
        try:
            import mlflow
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MLflow is not installed. Use the local_json backend or install "
                "requirements-mlops.txt."
            ) from exc

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as active_run:
            mlflow.log_params(parameters)
            mlflow.log_metrics(metrics)
            if tags:
                mlflow.set_tags(tags)
            for artifact_path in (artifacts or {}).values():
                mlflow.log_artifact(artifact_path)
            return active_run.info.run_id


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

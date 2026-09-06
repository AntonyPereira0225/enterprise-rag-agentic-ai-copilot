from __future__ import annotations

import io
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from enterprise_copilot.api.logging import JsonRequestLogger
from enterprise_copilot.api.metrics import ServiceMetrics
from enterprise_copilot.monitoring.experiment import (
    ObservabilityConfig,
    collect_project_metrics,
    resolve_project_path,
)
from enterprise_copilot.monitoring.observability import JsonlFileStream, TeeTextStream
from enterprise_copilot.monitoring.tracking import LocalExperimentTracker


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML is required for delivery validation. Install requirements-ci.txt."
        ) from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a YAML object in {path}")
    return payload


def _check(check_id: str, name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"check_id": check_id, "name": name, "passed": passed, **details}


def _contains_all(path: Path, required_fragments: tuple[str, ...]) -> tuple[bool, list[str]]:
    content = path.read_text(encoding="utf-8") if path.is_file() else ""
    missing = [fragment for fragment in required_fragments if fragment not in content]
    return not missing, missing


def run_mlops_validation(
    project_root: Path,
    config: ObservabilityConfig,
    *,
    results_path: Path | None = None,
) -> dict[str, Any]:
    """Exercise local observability and statically validate delivery contracts."""

    checks: list[dict[str, Any]] = []
    with TemporaryDirectory(prefix="enterprise-copilot-delivery-") as temporary_directory:
        temporary_root = Path(temporary_directory)

        memory_stream = io.StringIO()
        log_path = temporary_root / "requests.jsonl"
        logger = JsonRequestLogger(TeeTextStream((memory_stream, JsonlFileStream(log_path))))
        logger.log(
            request_id="validation-request",
            conversation_hash="b" * 64,
            question_hash="a" * 64,
            route="support",
            status="completed",
            latency_ms=2.5,
        )
        log_entries = [
            json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
        ]
        checks.append(
            _check(
                "OBS-001",
                "durable privacy-preserving JSON request log",
                len(log_entries) == 1
                and log_entries[0]["question_sha256"] == "a" * 64
                and "question" not in log_entries[0],
                events=len(log_entries),
            )
        )

        metrics_path = temporary_root / "service_metrics.json"
        metrics = ServiceMetrics(metrics_path)
        metrics.record(
            status="completed",
            route="support",
            latency_ms=10.0,
        )
        persisted_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        checks.append(
            _check(
                "OBS-002",
                "durable service metrics snapshot",
                persisted_metrics == metrics.snapshot()
                and persisted_metrics["requests_total"] == 1,
                requests=persisted_metrics["requests_total"],
            )
        )

        project_metrics, source_artifacts = collect_project_metrics(
            project_root,
            config.metric_sources,
        )
        checks.append(
            _check(
                "EXP-001",
                "evaluation metrics collected for experiment tracking",
                len(project_metrics) == 10
                and all(isinstance(value, (int, float)) for value in project_metrics.values())
                and all((project_root / path).is_file() for path in source_artifacts.values()),
                metrics=len(project_metrics),
                source_artifacts=len(source_artifacts),
            )
        )

        tracker = LocalExperimentTracker(temporary_root / "experiments")
        run, run_path = tracker.record(
            experiment_name=config.experiment_name,
            run_name="phase-9-validation",
            parameters={"tracking_backend": "local_json"},
            metrics=project_metrics,
            tags={"phase": "9", "purpose": "validation"},
            artifacts=source_artifacts,
        )
        stored_run = json.loads(run_path.read_text(encoding="utf-8"))
        latest_path = run_path.parents[1] / "latest.json"
        checks.append(
            _check(
                "EXP-002",
                "atomic local experiment record",
                stored_run["run_id"] == run.run_id
                and stored_run["status"] == "FINISHED"
                and latest_path.is_file(),
                run_id=run.run_id,
            )
        )

    docker_passed, docker_missing = _contains_all(
        project_root / "Dockerfile",
        (
            "FROM python:3.12-slim",
            "USER appuser",
            "EXPOSE 8000",
            "HEALTHCHECK",
            'CMD ["python", "scripts/serve_api.py"]',
        ),
    )
    checks.append(
        _check(
            "DEL-001",
            "container image contract",
            docker_passed,
            missing=docker_missing,
        )
    )

    compose_passed, compose_missing = _contains_all(
        project_root / "docker-compose.yml",
        (
            "read_only: true",
            "no-new-privileges:true",
            "copilot_artifacts:/app/artifacts",
            '"127.0.0.1:8000:8000"',
        ),
    )
    compose_payload = _load_yaml(project_root / "docker-compose.yml")
    compose_service = compose_payload.get("services", {}).get("copilot", {})
    compose_structure_passed = (
        compose_service.get("read_only") is True
        and "127.0.0.1:8000:8000" in compose_service.get("ports", [])
        and "copilot_artifacts:/app/artifacts" in compose_service.get("volumes", [])
        and "no-new-privileges:true" in compose_service.get("security_opt", [])
    )
    checks.append(
        _check(
            "DEL-002",
            "least-privilege local Compose contract",
            compose_passed and compose_structure_passed,
            missing=compose_missing,
            yaml_parsed=isinstance(compose_payload, dict),
        )
    )

    quality_passed, quality_missing = _contains_all(
        project_root / ".github" / "workflows" / "ci.yml",
        (
            "ruff check .",
            "ruff format --check .",
            "pytest --cov=enterprise_copilot --cov-report=xml:artifacts/coverage.xml ",
            "python scripts/validate_api.py",
            "python scripts/validate_model_integration.py",
            "python scripts/validate_mlops.py",
            "python scripts/validate_project.py",
            "actions/upload-artifact@v7",
        ),
    )
    workflow_payload = _load_yaml(project_root / ".github" / "workflows" / "ci.yml")
    workflow_jobs = workflow_payload.get("jobs", {})
    workflow_structure_passed = (
        isinstance(workflow_jobs.get("quality", {}).get("steps"), list)
        and isinstance(workflow_jobs.get("container", {}).get("steps"), list)
        and workflow_jobs.get("container", {}).get("needs") == "quality"
    )
    checks.append(
        _check(
            "CI-001",
            "continuous-integration quality gates",
            quality_passed and workflow_structure_passed,
            missing=quality_missing,
            yaml_parsed=isinstance(workflow_payload, dict),
        )
    )

    container_ci_passed, container_ci_missing = _contains_all(
        project_root / ".github" / "workflows" / "ci.yml",
        (
            "docker build --tag enterprise-copilot:ci .",
            "docker compose config --quiet",
            "docker run --detach --publish 127.0.0.1:18000:8000",
            "curl --fail --silent http://127.0.0.1:18000/health",
        ),
    )
    checks.append(
        _check(
            "CI-002",
            "container build and health-check gate",
            container_ci_passed,
            missing=container_ci_missing,
        )
    )

    passed = sum(check["passed"] for check in checks)
    docker_available = shutil.which("docker") is not None
    results = {
        "summary": {
            "checks": len(checks),
            "passed": passed,
            "pass_rate": passed / len(checks),
            "docker_available": docker_available,
            "local_container_build_executed": False,
            "container_definition_validation": "static",
        },
        "checks": checks,
    }
    output_path = results_path or resolve_project_path(project_root, config.validation_results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return results

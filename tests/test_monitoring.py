from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from enterprise_copilot.api.logging import JsonRequestLogger
from enterprise_copilot.api.metrics import ServiceMetrics
from enterprise_copilot.api.service import ApiConfig
from enterprise_copilot.evaluation.mlops_validation import run_mlops_validation
from enterprise_copilot.monitoring.experiment import (
    ObservabilityConfig,
    collect_project_metrics,
    record_project_experiment,
)
from enterprise_copilot.monitoring.observability import JsonlFileStream, TeeTextStream
from enterprise_copilot.monitoring.tracking import LocalExperimentTracker


def test_jsonl_file_and_tee_streams_keep_structured_logs_durable(tmp_path: Path) -> None:
    memory_stream = io.StringIO()
    log_path = tmp_path / "nested" / "requests.jsonl"
    logger = JsonRequestLogger(TeeTextStream((memory_stream, JsonlFileStream(log_path))))

    logger.log(
        request_id="request-1",
        conversation_hash="b" * 64,
        question_hash="a" * 64,
        route="support",
        status="completed",
        latency_ms=1.23456,
    )

    assert log_path.read_text(encoding="utf-8") == memory_stream.getvalue()
    event = json.loads(memory_stream.getvalue())
    assert event["latency_ms"] == 1.235
    assert event["question_sha256"] == "a" * 64
    assert event["conversation_sha256"] == "b" * 64
    assert "recorded_at" in event
    assert "question" not in event


def test_service_metrics_persist_each_snapshot(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "metrics" / "service.json"
    metrics = ServiceMetrics(snapshot_path)

    assert json.loads(snapshot_path.read_text(encoding="utf-8"))["requests_total"] == 0

    metrics.record(status="completed", route="support", latency_ms=8.0)
    metrics.record(
        status="blocked",
        route="blocked",
        latency_ms=12.0,
        guardrail_category="prompt_injection",
        error=True,
    )

    persisted = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert persisted == metrics.snapshot()
    assert persisted["requests_total"] == 2
    assert persisted["errors_total"] == 1
    assert list(snapshot_path.parent.glob("*.tmp")) == []

    restarted = ServiceMetrics(snapshot_path)
    restarted.record(status="completed", route="support", latency_ms=10.0)
    assert restarted.snapshot()["requests_total"] == 3
    assert restarted.snapshot()["latency_ms_total"] == 30.0


def test_service_metrics_are_thread_safe_and_reject_invalid_latency(tmp_path: Path) -> None:
    metrics = ServiceMetrics(tmp_path / "service.json")
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda _: metrics.record(
                    status="completed",
                    route="support",
                    latency_ms=1.0,
                ),
                range(50),
            )
        )

    assert metrics.snapshot()["requests_total"] == 50
    with pytest.raises(ValueError, match="finite, non-negative"):
        metrics.record(status="completed", route="support", latency_ms=float("nan"))
    with pytest.raises(ValueError, match="finite, non-negative"):
        metrics.record(status="completed", route="support", latency_ms=-1.0)


def test_jsonl_file_stream_reports_write_failures_without_raising(tmp_path: Path) -> None:
    unwritable_target = tmp_path / "directory-not-file"
    unwritable_target.mkdir()
    stream = JsonlFileStream(unwritable_target)

    assert stream.write("event\n") == 0
    assert stream.errors == 1


def test_local_experiment_tracker_writes_reproducible_run_and_latest_pointer(
    tmp_path: Path,
) -> None:
    fixed_time = datetime(2026, 9, 6, 12, 0, tzinfo=UTC)
    tracker = LocalExperimentTracker(tmp_path, now=lambda: fixed_time)

    first_run, first_path = tracker.record(
        experiment_name="copilot",
        run_name="baseline",
        parameters={"phase": 9},
        metrics={"pass_rate": 1.0},
    )
    second_run, second_path = tracker.record(
        experiment_name="copilot",
        run_name="baseline",
        parameters={"phase": 9},
        metrics={"pass_rate": 1.0},
    )

    assert first_run.run_id == second_run.run_id
    assert first_path == second_path
    assert json.loads(first_path.read_text(encoding="utf-8"))["status"] == "FINISHED"
    latest = json.loads((tmp_path / "copilot" / "latest.json").read_text(encoding="utf-8"))
    assert latest["run_id"] == first_run.run_id


@pytest.mark.parametrize("invalid_metric", [float("nan"), float("inf"), True])
def test_local_experiment_tracker_rejects_invalid_metrics(
    tmp_path: Path,
    invalid_metric: float | bool,
) -> None:
    with pytest.raises(ValueError, match="finite numbers"):
        LocalExperimentTracker(tmp_path).record(
            experiment_name="copilot",
            run_name="invalid",
            parameters={},
            metrics={"invalid": invalid_metric},  # type: ignore[dict-item]
        )


def test_project_evaluation_metrics_are_collected_and_recorded(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = ObservabilityConfig.from_json(project_root / "configs" / "observability_config.json")
    metrics, artifacts = collect_project_metrics(project_root, config.metric_sources)

    assert len(metrics) == 10
    assert metrics["hybrid_evidence_recall_at_5"] == 1.0
    assert metrics["grounded_citation_verification"] == 1.0
    assert metrics["agent_safety_pass_rate"] == 1.0
    assert metrics["api_validation_pass_rate"] == 1.0
    assert all((project_root / path).is_file() for path in artifacts.values())

    temporary_project = tmp_path / "project"
    (temporary_project / "configs").mkdir(parents=True)
    for source in (project_root / "configs").glob("*.json"):
        (temporary_project / "configs" / source.name).write_bytes(source.read_bytes())
    for configured_path in config.metric_sources.values():
        source = project_root / configured_path
        destination = temporary_project / configured_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())

    temporary_config = replace(config, experiment_root="artifacts/experiments")
    record = record_project_experiment(temporary_project, temporary_config)
    assert record.status == "FINISHED"
    assert record.record_path is not None and record.record_path.is_file()
    assert len(record.metrics) == 10
    stored = json.loads(record.record_path.read_text(encoding="utf-8"))
    assert all(not Path(path).is_absolute() for path in stored["artifacts"].values())
    assert len(list((record.record_path.parent / "artifacts").glob("*.json"))) == 5

    escaped_config = replace(config, experiment_root="../outside")
    with pytest.raises(ValueError, match="leaves the project root"):
        record_project_experiment(temporary_project, escaped_config)


def test_api_config_accepts_safe_environment_overrides() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = ApiConfig.from_json(project_root / "configs" / "api_config.json")

    overridden = config.with_environment({"API_HOST": "0.0.0.0", "API_PORT": "9000"})

    assert overridden.host == "0.0.0.0"
    assert overridden.port == 9000
    with pytest.raises(ValueError, match="API_PORT must be an integer"):
        config.with_environment({"API_PORT": "invalid"})
    with pytest.raises(ValueError, match="between 1 and 65535"):
        config.with_environment({"API_PORT": "70000"})


def test_phase_9_contract_validation_passes(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = ObservabilityConfig.from_json(project_root / "configs" / "observability_config.json")

    results = run_mlops_validation(
        project_root,
        config,
        results_path=tmp_path / "mlops_validation.json",
    )

    assert results["summary"]["checks"] == 8
    assert results["summary"]["pass_rate"] == 1.0
    assert results["summary"]["local_container_build_executed"] is False
    assert (tmp_path / "mlops_validation.json").is_file()

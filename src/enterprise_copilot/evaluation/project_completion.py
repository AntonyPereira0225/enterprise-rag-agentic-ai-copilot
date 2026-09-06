from __future__ import annotations

import hashlib
import json
import re
import shutil
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enterprise_copilot import __version__
from enterprise_copilot.ingestion.loaders import load_jsonl
from enterprise_copilot.monitoring.experiment import resolve_project_path

_METRIC_NAMES = {
    "hybrid_retrieval",
    "grounded_answers",
    "agent_workflow",
    "api_validation",
    "mlops_validation",
    "model_integration",
}
_FR_IDS = tuple(f"FR-{number:02d}" for number in range(1, 19))
_NFR_IDS = tuple(f"NFR-{number:02d}" for number in range(1, 9))
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


@dataclass(frozen=True)
class ProjectCompletionConfig:
    corpus_config_path: str
    hard_evaluation_config_path: str
    knowledge_documents_path: str
    support_cases_path: str
    baseline_questions_path: str
    hard_questions_path: str
    guardrail_questions_path: str
    chunks_path: str
    ingestion_manifest_path: str
    vector_index_paths: list[str]
    metric_paths: dict[str, str]
    expected_guardrail_cases: int
    minimum_quality_rate: float
    required_release_files: list[str]
    results_path: str

    def __post_init__(self) -> None:
        missing_metrics = _METRIC_NAMES - self.metric_paths.keys()
        if missing_metrics:
            raise ValueError(
                f"Missing completion metric paths: {', '.join(sorted(missing_metrics))}"
            )
        if not 0 <= self.minimum_quality_rate <= 1:
            raise ValueError("minimum_quality_rate must be between zero and one")
        if self.expected_guardrail_cases <= 0:
            raise ValueError("expected_guardrail_cases must be greater than zero")

    @classmethod
    def from_json(cls, path: Path) -> ProjectCompletionConfig:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def _check(
    check_id: str,
    name: str,
    passed: bool,
    requirements: tuple[str, ...],
    **details: Any,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "name": name,
        "passed": passed,
        "requirements": list(requirements),
        **details,
    }


def _load_json(project_root: Path, configured_path: str) -> dict[str, Any]:
    path = resolve_project_path(project_root, configured_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {configured_path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique_count(records: list[dict[str, Any]], field: str) -> int:
    return len({str(record[field]) for record in records})


def _find_embedded_secrets(project_root: Path) -> list[str]:
    candidates = [
        path
        for directory in ("src", "scripts", "configs", "docs", ".github")
        for path in (project_root / directory).rglob("*")
        if path.is_file()
    ]
    candidates.extend(
        project_root / name
        for name in (
            ".env.example",
            ".gitignore",
            "Dockerfile",
            "docker-compose.yml",
            "pyproject.toml",
            "README.md",
            "requirements.txt",
        )
        if (project_root / name).is_file()
    )
    matches: list[str] = []
    for path in candidates:
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(content) for pattern in _SECRET_PATTERNS):
            matches.append(path.relative_to(project_root).as_posix())
    return sorted(set(matches))


def run_project_completion_validation(
    project_root: Path,
    config: ProjectCompletionConfig,
    *,
    results_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the final release against data, quality, safety, and delivery contracts."""

    checks: list[dict[str, Any]] = []
    corpus_config = _load_json(project_root, config.corpus_config_path)
    hard_config = _load_json(project_root, config.hard_evaluation_config_path)
    knowledge = load_jsonl(resolve_project_path(project_root, config.knowledge_documents_path))
    support_cases = load_jsonl(resolve_project_path(project_root, config.support_cases_path))
    baseline_questions = load_jsonl(
        resolve_project_path(project_root, config.baseline_questions_path)
    )
    hard_questions = load_jsonl(resolve_project_path(project_root, config.hard_questions_path))
    guardrail_questions = load_jsonl(
        resolve_project_path(project_root, config.guardrail_questions_path)
    )
    chunks = load_jsonl(resolve_project_path(project_root, config.chunks_path))

    expected_types = corpus_config["document_counts"]
    expected_documents = sum(int(count) for count in expected_types.values())
    expected_support_cases = int(corpus_config["support_case_count"])
    expected_baseline_questions = int(corpus_config["evaluation_question_count"])
    counts_passed = (
        len(knowledge) == expected_documents
        and _unique_count(knowledge, "document_id") == expected_documents
        and len(support_cases) == expected_support_cases
        and _unique_count(support_cases, "case_id") == expected_support_cases
        and len(baseline_questions) == expected_baseline_questions
        and _unique_count(baseline_questions, "question_id") == expected_baseline_questions
    )
    checks.append(
        _check(
            "DATA-001",
            "synthetic corpus and baseline evaluation counts",
            counts_passed,
            ("FR-01", "FR-16", "NFR-01"),
            knowledge_documents=len(knowledge),
            support_cases=len(support_cases),
            baseline_questions=len(baseline_questions),
        )
    )

    actual_types = Counter(str(record["document_type"]) for record in knowledge)
    schema_passed = (
        actual_types == Counter({key: int(value) for key, value in expected_types.items()})
        and all(record.get("status") == "active" for record in knowledge)
        and all(
            "synthetic" in str(record.get("summary", "")).casefold() for record in support_cases
        )
    )
    checks.append(
        _check(
            "DATA-002",
            "document-family distribution and synthetic-data boundary",
            schema_passed,
            ("FR-01", "NFR-04", "NFR-07"),
            document_types=dict(sorted(actual_types.items())),
            all_documents_active=all(record.get("status") == "active" for record in knowledge),
            all_cases_synthetic=all(
                "synthetic" in str(record.get("summary", "")).casefold() for record in support_cases
            ),
        )
    )

    expected_hard_questions = sum(
        int(hard_config[name])
        for name in ("paraphrased_count", "multi_source_count", "unanswerable_count")
    )
    hard_passed = (
        len(hard_questions) == expected_hard_questions
        and _unique_count(hard_questions, "question_id") == expected_hard_questions
        and len(guardrail_questions) == config.expected_guardrail_cases
        and _unique_count(guardrail_questions, "case_id") == config.expected_guardrail_cases
    )
    checks.append(
        _check(
            "DATA-003",
            "hard-answerability and guardrail evaluation sets",
            hard_passed,
            ("FR-11", "FR-16", "NFR-05"),
            hard_questions=len(hard_questions),
            guardrail_cases=len(guardrail_questions),
        )
    )

    manifest = _load_json(project_root, config.ingestion_manifest_path)
    knowledge_path = resolve_project_path(project_root, config.knowledge_documents_path)
    chunks_path = resolve_project_path(project_root, config.chunks_path)
    manifest_passed = (
        manifest.get("source_document_count") == len(knowledge)
        and manifest.get("included_document_count") == len(knowledge)
        and manifest.get("chunk_count") == len(chunks)
        and _unique_count(chunks, "chunk_id") == len(chunks)
        and manifest.get("source_sha256") == _sha256(knowledge_path)
        and manifest.get("chunks_sha256") == _sha256(chunks_path)
        and {str(chunk["document_id"]) for chunk in chunks}
        == {str(document["document_id"]) for document in knowledge}
    )
    checks.append(
        _check(
            "ING-001",
            "reproducible ingestion manifest and chunk lineage",
            manifest_passed,
            ("FR-02", "NFR-01", "NFR-02"),
            chunks=len(chunks),
            source_hash_matches=manifest.get("source_sha256") == _sha256(knowledge_path),
            chunks_hash_matches=manifest.get("chunks_sha256") == _sha256(chunks_path),
        )
    )

    index_details: list[dict[str, Any]] = []
    indexes_passed = True
    for configured_path in config.vector_index_paths:
        index = _load_json(project_root, configured_path)
        metadata = index.get("metadata", {})
        records = index.get("records", [])
        passed = (
            index.get("schema_version") == 1
            and isinstance(records, list)
            and len(records) == len(chunks)
            and metadata.get("source_chunks_path") == config.chunks_path
            and metadata.get("source_chunks_sha256", manifest["chunks_sha256"])
            == manifest["chunks_sha256"]
        )
        indexes_passed = indexes_passed and passed
        index_details.append({"path": configured_path, "records": len(records), "passed": passed})
    checks.append(
        _check(
            "RET-001",
            "vector and keyword indexes map to every processed chunk",
            indexes_passed,
            ("FR-03", "FR-04", "FR-05", "NFR-02", "NFR-08"),
            indexes=index_details,
        )
    )

    metrics = {name: _load_json(project_root, path) for name, path in config.metric_paths.items()}
    threshold = config.minimum_quality_rate
    hybrid = metrics["hybrid_retrieval"]["summary"]
    hybrid_passed = (
        hybrid["evaluated_questions"] == len(hard_questions)
        and hybrid["evidence_recall_at_k"]["5"] >= threshold
        and hybrid["answerability_accuracy"] >= threshold
        and hybrid["unanswerable_abstention_accuracy"] >= threshold
    )
    checks.append(
        _check(
            "EVAL-001",
            "hybrid retrieval, reranking, and abstention quality",
            hybrid_passed,
            ("FR-05", "FR-06", "FR-11", "FR-16"),
            evaluated_questions=hybrid["evaluated_questions"],
            evidence_recall_at_5=hybrid["evidence_recall_at_k"]["5"],
            answerability_accuracy=hybrid["answerability_accuracy"],
        )
    )

    grounded = metrics["grounded_answers"]["summary"]
    grounded_passed = (
        grounded["evaluated_questions"] == len(hard_questions)
        and grounded["expected_evidence_coverage"] >= threshold
        and grounded["citation_verification_accuracy"] >= threshold
        and grounded["citation_source_precision"] >= threshold
        and grounded["unanswerable_refusal_accuracy"] >= threshold
    )
    checks.append(
        _check(
            "EVAL-002",
            "grounded answers, citations, and safe refusals",
            grounded_passed,
            ("FR-07", "FR-08", "FR-11", "NFR-02", "NFR-05"),
            evidence_coverage=grounded["expected_evidence_coverage"],
            citation_verification=grounded["citation_verification_accuracy"],
            refusal_accuracy=grounded["unanswerable_refusal_accuracy"],
        )
    )

    agent = metrics["agent_workflow"]["summary"]
    agent_passed = all(
        agent[name] >= threshold
        for name in (
            "adversarial_block_rate",
            "knowledge_evidence_coverage",
            "knowledge_status_accuracy",
            "safety_pass_rate",
        )
    )
    checks.append(
        _check(
            "EVAL-003",
            "agent routing, knowledge regression, and guardrail safety",
            agent_passed,
            ("FR-09", "FR-10", "FR-11", "NFR-05"),
            **agent,
        )
    )

    api = metrics["api_validation"]["summary"]
    api_passed = (
        api["passed"] == api["checks"]
        and api["pass_rate"] >= threshold
        and api["structured_log_events"] >= 1
        and api["raw_questions_logged"] is False
    )
    checks.append(
        _check(
            "API-001",
            "API, UI, schema, limits, metrics, and log privacy",
            api_passed,
            ("FR-12", "FR-13", "FR-14", "NFR-04", "NFR-06"),
            validation_summary=api,
        )
    )

    model = metrics["model_integration"]["summary"]
    model_passed = (
        model["passed"] == model["checks"]
        and model["pass_rate"] >= threshold
        and model["default_backend"] == "extractive"
        and model["fallback_backend"] == "extractive"
        and model["live_ollama_required"] is False
    )
    checks.append(
        _check(
            "MODEL-001",
            "optional model adapter and deterministic fallback",
            model_passed,
            ("FR-07", "FR-11", "NFR-05", "NFR-07", "NFR-08"),
            validation_summary=model,
        )
    )

    mlops = metrics["mlops_validation"]["summary"]
    mlops_passed = mlops["passed"] == mlops["checks"] and mlops["pass_rate"] >= threshold
    checks.append(
        _check(
            "DEL-001",
            "observability, experiment, container, and CI contracts",
            mlops_passed,
            ("FR-12", "FR-15", "FR-17", "FR-18", "NFR-01", "NFR-06"),
            validation_summary=mlops,
        )
    )

    missing_release_files = [
        path
        for path in config.required_release_files
        if not resolve_project_path(project_root, path).is_file()
        or resolve_project_path(project_root, path).stat().st_size == 0
    ]
    final_release_path = project_root / "docs" / "final_release.md"
    final_release_text = (
        final_release_path.read_text(encoding="utf-8") if final_release_path.is_file() else ""
    )
    missing_requirement_evidence = [
        requirement
        for requirement in (*_FR_IDS, *_NFR_IDS)
        if requirement not in final_release_text
    ]
    checks.append(
        _check(
            "DOC-001",
            "release documentation and requirement traceability matrix",
            not missing_release_files and not missing_requirement_evidence,
            ("NFR-01", "NFR-02", "NFR-08"),
            missing_files=missing_release_files,
            missing_requirement_ids=missing_requirement_evidence,
        )
    )

    api_config = _load_json(project_root, "configs/api_config.json")
    project_metadata = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    package_version = project_metadata["project"]["version"]
    version_passed = api_config["service_version"] == package_version == __version__ == "1.0.0"
    checks.append(
        _check(
            "REL-001",
            "consistent 1.0.0 release identity",
            version_passed,
            ("NFR-01", "NFR-03"),
            service_version=api_config["service_version"],
            package_version=package_version,
            module_version=__version__,
        )
    )

    embedded_secrets = _find_embedded_secrets(project_root)
    local_env_present = (project_root / ".env").exists()
    checks.append(
        _check(
            "SEC-001",
            "no embedded high-confidence secrets or committed local environment file",
            not embedded_secrets and not local_env_present,
            ("NFR-04",),
            secret_matches=embedded_secrets,
            local_env_present=local_env_present,
        )
    )

    passed = sum(bool(check["passed"]) for check in checks)
    covered_requirements = sorted(
        {requirement for check in checks for requirement in check["requirements"]}
    )
    results = {
        "summary": {
            "checks": len(checks),
            "passed": passed,
            "pass_rate": passed / len(checks),
            "functional_requirements_covered": sum(
                requirement in covered_requirements for requirement in _FR_IDS
            ),
            "functional_requirements_total": len(_FR_IDS),
            "nonfunctional_requirements_covered": sum(
                requirement in covered_requirements for requirement in _NFR_IDS
            ),
            "nonfunctional_requirements_total": len(_NFR_IDS),
            "release_version": package_version,
            "docker_local_status": (
                "available" if shutil.which("docker") else "not_available_ci_gate_configured"
            ),
            "live_model_status": "optional_not_required_mock_contract_validated",
            "cloud_status": "runbook_only_no_resources_provisioned",
        },
        "checks": checks,
    }
    output_path = results_path or resolve_project_path(project_root, config.results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return results

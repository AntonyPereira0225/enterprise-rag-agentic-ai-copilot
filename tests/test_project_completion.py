from pathlib import Path

import pytest

from enterprise_copilot.evaluation.project_completion import (
    ProjectCompletionConfig,
    run_project_completion_validation,
)


def test_completion_config_rejects_an_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        ProjectCompletionConfig(
            corpus_config_path="corpus.json",
            hard_evaluation_config_path="hard.json",
            knowledge_documents_path="knowledge.jsonl",
            support_cases_path="cases.jsonl",
            baseline_questions_path="baseline.jsonl",
            hard_questions_path="hard.jsonl",
            guardrail_questions_path="guardrails.jsonl",
            chunks_path="chunks.jsonl",
            ingestion_manifest_path="manifest.json",
            vector_index_paths=[],
            metric_paths={
                name: f"{name}.json"
                for name in (
                    "hybrid_retrieval",
                    "grounded_answers",
                    "agent_workflow",
                    "api_validation",
                    "mlops_validation",
                    "model_integration",
                )
            },
            expected_guardrail_cases=20,
            minimum_quality_rate=1.1,
            required_release_files=[],
            results_path="results.json",
        )


def test_final_project_acceptance_suite_passes(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = ProjectCompletionConfig.from_json(
        project_root / "configs" / "project_completion_config.json"
    )

    results = run_project_completion_validation(
        project_root,
        config,
        results_path=tmp_path / "completion.json",
    )

    assert results["summary"]["pass_rate"] == 1.0
    assert results["summary"]["functional_requirements_covered"] == 18
    assert results["summary"]["nonfunctional_requirements_covered"] == 8

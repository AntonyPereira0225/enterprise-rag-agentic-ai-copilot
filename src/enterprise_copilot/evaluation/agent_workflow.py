from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from enterprise_copilot.agents.graph import (
    AgentWorkflow,
    AgentWorkflowConfig,
    load_agent_workflow,
)
from enterprise_copilot.ingestion.loaders import load_jsonl


def evaluate_safety_cases(
    workflow: AgentWorkflow,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    if not cases:
        raise ValueError("At least one safety case is required")
    results = []
    for case in cases:
        result = workflow.ask(case["question"], conversation_id=case["case_id"])
        row = {
            "case_id": case["case_id"],
            "case_type": case["case_type"],
            "question": case["question"],
            "expected_route": case["expected_route"],
            "actual_route": result.route,
            "route_correct": result.route == case["expected_route"],
            "expected_status": case["expected_status"],
            "actual_status": result.status,
            "status_correct": result.status == case["expected_status"],
            "expected_input_category": case["expected_input_category"],
            "actual_input_category": result.input_guardrail.category,
            "input_category_correct": (
                result.input_guardrail.category == case["expected_input_category"]
            ),
            "output_safe": (result.output_guardrail is None or result.output_guardrail.allowed),
            "result": result.to_dict(),
        }
        row["passed"] = all(
            (
                row["route_correct"],
                row["status_correct"],
                row["input_category_correct"],
                row["output_safe"],
            )
        )
        results.append(row)

    adversarial = [row for row in results if row["case_type"] == "adversarial"]
    safe = [row for row in results if row["case_type"] == "safe_workflow"]
    return {
        "summary": {
            "cases": len(results),
            "safe_cases": len(safe),
            "adversarial_cases": len(adversarial),
            "overall_pass_rate": sum(row["passed"] for row in results) / len(results),
            "route_accuracy": sum(row["route_correct"] for row in results) / len(results),
            "status_accuracy": sum(row["status_correct"] for row in results) / len(results),
            "input_category_accuracy": (
                sum(row["input_category_correct"] for row in results) / len(results)
            ),
            "safe_workflow_pass_rate": sum(row["passed"] for row in safe) / len(safe),
            "adversarial_block_rate": (
                sum(row["actual_status"] == "blocked" for row in adversarial) / len(adversarial)
            ),
        },
        "cases": results,
    }


def evaluate_knowledge_regression(
    workflow: AgentWorkflow,
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    if not questions:
        raise ValueError("At least one knowledge question is required")
    rows = []
    for question in questions:
        result = workflow.ask(question["question"], conversation_id=question["question_id"])
        evidence_checks = [
            {
                **expected,
                "covered": any(
                    citation.source_id == expected["document_id"]
                    and expected["answer"].casefold() in citation.quote.casefold()
                    for citation in result.citations
                ),
            }
            for expected in question.get("expected_evidence", [])
        ]
        expected_status = "completed" if question.get("answerable", True) else "refused"
        rows.append(
            {
                "question_id": question["question_id"],
                "difficulty": question["difficulty"],
                "expected_status": expected_status,
                "actual_status": result.status,
                "status_correct": result.status == expected_status,
                "evidence_checks": evidence_checks,
                "all_expected_evidence_covered": bool(evidence_checks)
                and all(item["covered"] for item in evidence_checks),
                "route": result.route,
                "result": result.to_dict(),
            }
        )

    answerable = [row for row in rows if row["expected_status"] == "completed"]
    unanswerable = [row for row in rows if row["expected_status"] == "refused"]
    evidence = [item for row in answerable for item in row["evidence_checks"]]
    return {
        "summary": {
            "questions": len(rows),
            "status_accuracy": sum(row["status_correct"] for row in rows) / len(rows),
            "answerable_completion_rate": (
                sum(row["actual_status"] == "completed" for row in answerable) / len(answerable)
            ),
            "unanswerable_refusal_rate": (
                sum(row["actual_status"] == "refused" for row in unanswerable) / len(unanswerable)
            ),
            "expected_evidence_coverage": (
                sum(item["covered"] for item in evidence) / len(evidence)
            ),
            "all_expected_evidence_covered": (
                sum(row["all_expected_evidence_covered"] for row in answerable) / len(answerable)
            ),
            "blocked_safe_questions": sum(row["actual_status"] == "blocked" for row in rows),
            "failed_safe_outputs": sum(row["actual_status"] == "failed_safe" for row in rows),
        },
        "questions": rows,
    }


def run_agent_workflow_evaluation(
    project_root: Path,
    config: AgentWorkflowConfig,
) -> dict[str, Any]:
    workflow = load_agent_workflow(project_root, config)
    safety = evaluate_safety_cases(
        workflow,
        load_jsonl(project_root / config.safety_evaluation_path),
    )
    knowledge = evaluate_knowledge_regression(
        workflow,
        load_jsonl(project_root / config.hard_evaluation_path),
    )
    results = {
        "summary": {
            "safety_pass_rate": safety["summary"]["overall_pass_rate"],
            "adversarial_block_rate": safety["summary"]["adversarial_block_rate"],
            "knowledge_status_accuracy": knowledge["summary"]["status_accuracy"],
            "knowledge_evidence_coverage": knowledge["summary"]["expected_evidence_coverage"],
        },
        "safety": safety,
        "knowledge_regression": knowledge,
        "configuration": {
            "max_input_characters": config.max_input_characters,
            "grounded_answer_config_path": config.grounded_answer_config_path,
            "support_cases_path": config.support_cases_path,
        },
    }
    output_path = project_root / config.results_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return results

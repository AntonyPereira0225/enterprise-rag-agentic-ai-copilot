from __future__ import annotations

import importlib.util
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest

from enterprise_copilot.agents.state import (
    AgentCitation,
    WorkflowEvent,
    WorkflowResult,
)
from enterprise_copilot.api.fastapi_app import create_app
from enterprise_copilot.api.logging import JsonRequestLogger
from enterprise_copilot.api.metrics import ServiceMetrics
from enterprise_copilot.api.schemas import AskRequest, RequestValidationError
from enterprise_copilot.api.service import ApiConfig, CopilotService
from enterprise_copilot.evaluation.api_validation import run_api_validation
from enterprise_copilot.guardrails.schemas import GuardrailDecision


def _allowed(category: str = "safe_input") -> GuardrailDecision:
    return GuardrailDecision("allow", category, "Checks passed.", "ALLOW")


def _workflow_result() -> WorkflowResult:
    citation = AgentCitation(
        citation_id="C1",
        source_type="knowledge_chunk",
        source_id="DOC-1",
        title="Approved source",
        source_uri="nsg://knowledge/DOC-1",
        quote="The requirement is 20 minutes.",
    )
    return WorkflowResult(
        request_id="request-1",
        conversation_id="conversation-1",
        turn_number=1,
        route="support",
        status="completed",
        answer="The requirement is 20 minutes. [C1]",
        confidence=0.8,
        citations=(citation,),
        input_guardrail=_allowed(),
        output_guardrail=_allowed("verified_output"),
        trace=(WorkflowEvent("router", "support", "Support cue found."),),
    )


class _FakeWorkflow:
    def __init__(self, *, fails: bool = False) -> None:
        self.fails = fails
        self.calls: list[tuple[str, str]] = []

    def ask(self, question: str, *, conversation_id: str = "default") -> WorkflowResult:
        self.calls.append((question, conversation_id))
        if self.fails:
            raise RuntimeError("private failure details")
        return _workflow_result()


def _service(
    workflow: _FakeWorkflow | None = None,
    *,
    stream: io.StringIO | None = None,
) -> CopilotService:
    return CopilotService(
        workflow or _FakeWorkflow(),  # type: ignore[arg-type]
        name="test-copilot",
        version="0.8.0",
        logger=JsonRequestLogger(stream=stream or io.StringIO()),
    )


def test_request_schema_applies_a_safe_default_conversation() -> None:
    request = AskRequest.from_payload({"question": "  What is the policy?  "})

    assert request.question == "What is the policy?"
    assert request.conversation_id == "default"


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"question": " "},
        {"question": "Valid", "conversation_id": "contains spaces"},
        {"question": "Valid", "extra": True},
    ],
)
def test_request_schema_rejects_invalid_payloads(payload: object) -> None:
    with pytest.raises(RequestValidationError):
        AskRequest.from_payload(payload)


def test_service_returns_a_structured_answer_and_records_safe_logs() -> None:
    stream = io.StringIO()
    workflow = _FakeWorkflow()
    service = _service(workflow, stream=stream)
    question = "What response window should support staff use?"

    response = service.ask({"question": question, "conversation_id": "conversation-1"})
    log = json.loads(stream.getvalue())

    assert response.status_code == 200
    assert response.body["data"]["route"] == "support"
    assert workflow.calls == [(question, "conversation-1")]
    assert question not in stream.getvalue()
    assert "conversation-1" not in stream.getvalue()
    assert len(log["question_sha256"]) == 64
    assert len(log["conversation_sha256"]) == 64
    assert service.metric_snapshot().body["requests_total"] == 1


def test_service_returns_422_without_calling_workflow() -> None:
    workflow = _FakeWorkflow()
    service = _service(workflow)

    response = service.ask({"conversation_id": "missing-question"})

    assert response.status_code == 422
    assert response.body["error"]["code"] == "invalid_request"
    assert workflow.calls == []
    assert service.metric_snapshot().body["status_counts"] == {"validation_error": 1}


def test_service_hides_internal_exception_details() -> None:
    service = _service(_FakeWorkflow(fails=True))

    response = service.ask({"question": "A valid question"})

    assert response.status_code == 500
    assert "private failure details" not in json.dumps(response.body)
    assert service.metric_snapshot().body["errors_total"] == 1


def test_metrics_aggregate_routes_statuses_and_guardrails() -> None:
    metrics = ServiceMetrics()
    metrics.record(status="completed", route="support", latency_ms=10.0)
    metrics.record(
        status="blocked",
        route="blocked",
        latency_ms=20.0,
        guardrail_category="prompt_injection",
    )

    snapshot = metrics.snapshot()

    assert snapshot["requests_total"] == 2
    assert snapshot["average_latency_ms"] == 15.0
    assert snapshot["route_counts"] == {"blocked": 1, "support": 1}
    assert snapshot["guardrail_counts"] == {"prompt_injection": 1}


def test_optional_fastapi_adapter_has_a_clear_offline_fallback() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = ApiConfig.from_json(project_root / "configs" / "api_config.json")
    if importlib.util.find_spec("fastapi") is None:
        with pytest.raises(RuntimeError, match="FastAPI is not installed"):
            create_app(project_root, config, _service())
    else:
        from fastapi.testclient import TestClient

        service = _service()
        app = create_app(project_root, config, service)
        assert app.title == "Enterprise RAG & Agentic AI Copilot"
        client = TestClient(app)
        response = client.post("/ask", content="plain text")
        assert response.status_code == 415
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        response = client.post(
            "/ask",
            content="{invalid",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400
        response = client.post(
            "/ask",
            content=(chunk for chunk in (b"x" * 20_000, b"x" * 20_000)),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 413
        response = client.post(
            "/ask",
            json={"question": "What response window should support staff use?"},
        )
        assert response.status_code == 200
        assert service.metric_snapshot().body["requests_total"] == 4


def test_demo_page_is_local_accessible_and_connected_to_the_api() -> None:
    project_root = Path(__file__).resolve().parents[1]
    page = (project_root / "src" / "enterprise_copilot" / "ui" / "index.html").read_text(
        encoding="utf-8"
    )

    assert "Knowledge you can trace" in page
    assert "fetch('/ask'" in page
    assert "fetch('/health'" in page
    assert "https://" not in page
    assert 'aria-live="polite"' in page


def test_live_http_api_and_demo_validation(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = ApiConfig.from_json(project_root / "configs" / "api_config.json")
    config = replace(config, validation_results_path=str(tmp_path / "api_metrics.json"))

    results = run_api_validation(project_root, config)

    assert results["summary"]["checks"] == 11
    assert results["summary"]["pass_rate"] == 1.0
    assert results["summary"]["raw_questions_logged"] is False
    assert (tmp_path / "api_metrics.json").is_file()

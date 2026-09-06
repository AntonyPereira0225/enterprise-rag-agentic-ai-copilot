from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

from enterprise_copilot.evaluation.grounded_answers import evaluate_grounded_answers
from enterprise_copilot.ingestion.loaders import load_jsonl
from enterprise_copilot.llm.generation import ExtractiveAnswerGenerator
from enterprise_copilot.llm.ollama import OllamaClient, OllamaEvidenceSelectorGenerator
from enterprise_copilot.llm.pipeline import GroundedAnswerConfig, load_grounded_answer_pipeline
from enterprise_copilot.llm.schemas import ContextPackage, Evidence
from enterprise_copilot.llm.verification import verify_citations
from enterprise_copilot.monitoring.experiment import resolve_project_path


def _check(check_id: str, name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"check_id": check_id, "name": name, "passed": passed, **details}


def _context(*, answerable: bool = True) -> ContextPackage:
    if not answerable:
        return ContextPackage(
            question="What unsupported fact applies?",
            query_score=0.1,
            minimum_query_score=0.3,
            word_count=0,
            evidence=(),
            refusal_reason="The evidence threshold was not met.",
        )
    evidence = (
        Evidence(
            citation_id="C1",
            score=0.9,
            chunk_id="DOC-1::chunk-0000",
            document_id="DOC-1",
            document_type="support_procedure",
            title="Approved support procedure",
            region="Ireland",
            product="NSG Home",
            source_uri="nsg://knowledge/support_procedure/DOC-1",
            content="The documented requirement is 20 minutes. Escalate missing evidence.",
        ),
        Evidence(
            citation_id="C2",
            score=0.8,
            chunk_id="DOC-2::chunk-0000",
            document_id="DOC-2",
            document_type="faq",
            title="Approved FAQ",
            region="Ireland",
            product="NSG Home",
            source_uri="nsg://knowledge/faq/DOC-2",
            content="The documented requirement is 24 hours. Escalate missing evidence.",
        ),
    )
    return ContextPackage(
        question="What response window applies to NSG Home in Ireland?",
        query_score=0.9,
        minimum_query_score=0.3,
        word_count=sum(len(item.content.split()) for item in evidence),
        evidence=evidence,
    )


@contextmanager
def _stub_ollama(content: str) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    requests: list[dict[str, Any]] = []
    response_body = json.dumps({"message": {"content": content}}).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            requests.append(json.loads(body.decode("utf-8")))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _generator(base_url: str) -> OllamaEvidenceSelectorGenerator:
    return OllamaEvidenceSelectorGenerator(
        OllamaClient(
            base_url=base_url,
            model="validation-model:latest",
            timeout_seconds=2,
        ),
        max_citations=2,
        fallback=ExtractiveAnswerGenerator(max_citations=2),
    )


def run_model_integration_validation(
    project_root: Path,
    config: GroundedAnswerConfig,
    *,
    results_path: Path | None = None,
) -> dict[str, Any]:
    """Exercise the optional model boundary without requiring a live model installation."""

    checks: list[dict[str, Any]] = []
    context = _context()
    with _stub_ollama('{"citation_ids":["C1"]}') as (base_url, requests):
        response = _generator(base_url).generate(context)
        verification = verify_citations(response, context)
        request = requests[0] if requests else {}
        checks.append(
            _check(
                "MODEL-001",
                "validated Ollama evidence selection",
                response.generator == "ollama"
                and response.model == "validation-model:latest"
                and [item.citation_id for item in response.citations] == ["C1"]
                and response.citations[0].quote in response.answer
                and verification.valid
                and request.get("stream") is False
                and request.get("format") == "json",
                requests=len(requests),
                citation_verification=verification.valid,
            )
        )

    private_marker = "private-provider-stack-trace"
    with _stub_ollama(private_marker) as (base_url, requests):
        response = _generator(base_url).generate(context)
        serialized = json.dumps(response.to_dict(), ensure_ascii=False)
        checks.append(
            _check(
                "MODEL-002",
                "safe deterministic fallback for invalid model output",
                response.generator == "extractive_fallback"
                and response.fallback_reason == "invalid_provider_response"
                and private_marker not in serialized
                and verify_citations(response, context).valid,
                requests=len(requests),
                fallback_reason=response.fallback_reason,
            )
        )

    with _stub_ollama('{"citation_ids":["C1"]}') as (base_url, requests):
        response = _generator(base_url).generate(_context(answerable=False))
        checks.append(
            _check(
                "MODEL-003",
                "answerability gate avoids unnecessary model calls",
                not requests
                and response.status == "insufficient_evidence"
                and not response.citations,
                requests=len(requests),
            )
        )

    unsafe_http_rejected = False
    try:
        OllamaClient(base_url="http://example.com", model="safe-model")
    except ValueError:
        unsafe_http_rejected = True
    checks.append(
        _check(
            "MODEL-004",
            "provider endpoint and model settings validated",
            unsafe_http_rejected,
            remote_plain_http_rejected=unsafe_http_rejected,
        )
    )

    offline_pipeline = load_grounded_answer_pipeline(project_root, config)
    questions = load_jsonl(resolve_project_path(project_root, config.evaluation_path))
    offline_summary = evaluate_grounded_answers(offline_pipeline, questions)["summary"]
    offline_passed = (
        offline_pipeline.generator.backend_name == "extractive"
        and offline_summary["answerability_accuracy"] == 1.0
        and offline_summary["expected_evidence_coverage"] == 1.0
        and offline_summary["citation_verification_accuracy"] == 1.0
    )
    checks.append(
        _check(
            "MODEL-005",
            "offline grounded-answer regression remains deterministic",
            offline_passed,
            evaluated_questions=offline_summary["evaluated_questions"],
            answerability_accuracy=offline_summary["answerability_accuracy"],
            evidence_coverage=offline_summary["expected_evidence_coverage"],
            citation_verification=offline_summary["citation_verification_accuracy"],
        )
    )

    passed = sum(bool(check["passed"]) for check in checks)
    results = {
        "summary": {
            "checks": len(checks),
            "passed": passed,
            "pass_rate": passed / len(checks),
            "live_ollama_required": False,
            "default_backend": config.generator_backend,
            "fallback_backend": config.fallback_backend,
        },
        "checks": checks,
    }
    output_path = results_path or resolve_project_path(
        project_root,
        config.model_validation_results_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return results

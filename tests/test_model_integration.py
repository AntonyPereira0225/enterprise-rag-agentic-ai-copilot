from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.error import URLError

import pytest

from enterprise_copilot.evaluation.model_integration import (
    run_model_integration_validation,
)
from enterprise_copilot.llm.context import ContextBuilder, ContextBuilderConfig
from enterprise_copilot.llm.ollama import OllamaClient, OllamaEvidenceSelectorGenerator
from enterprise_copilot.llm.pipeline import GroundedAnswerConfig, GroundedAnswerPipeline
from enterprise_copilot.llm.schemas import Citation, ContextPackage, Evidence, GroundedAnswer
from enterprise_copilot.retrieval.vector_index import SearchResult


def _context(*, answerable: bool = True) -> ContextPackage:
    evidence = (
        Evidence(
            citation_id="C1",
            score=0.9,
            chunk_id="DOC-1::chunk-0000",
            document_id="DOC-1",
            document_type="support_procedure",
            title="Approved procedure",
            region="Ireland",
            product="NSG Home",
            source_uri="nsg://DOC-1",
            content="The documented requirement is 20 minutes.",
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
            source_uri="nsg://DOC-2",
            content="The documented requirement is 24 hours.",
        ),
    )
    return ContextPackage(
        question="What response window applies to NSG Home in Ireland?",
        query_score=0.9 if answerable else 0.1,
        minimum_query_score=0.3,
        word_count=14 if answerable else 0,
        evidence=evidence if answerable else (),
        refusal_reason=None if answerable else "Below threshold.",
    )


@contextmanager
def _server(response_body: bytes) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            requests.append(json.loads(body))
            self.send_response(200)
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


def _ollama_body(content: str) -> bytes:
    return json.dumps({"message": {"content": content}}).encode()


def _generator(base_url: str, *, max_response_bytes: int = 65_536):
    client = OllamaClient(
        base_url=base_url,
        model="test-model:1",
        timeout_seconds=1,
        max_response_bytes=max_response_bytes,
    )
    return OllamaEvidenceSelectorGenerator(client, max_citations=2)


def test_model_selects_ids_but_application_renders_exact_quotes() -> None:
    with _server(_ollama_body('{"citation_ids":["C2"]}')) as (base_url, requests):
        response = _generator(base_url).generate(_context())

    assert response.generator == "ollama"
    assert response.model == "test-model:1"
    assert response.answer == "The documented requirement is 24 hours. [C2]"
    assert requests[0]["stream"] is False
    assert requests[0]["options"] == {"temperature": 0, "seed": 7}


@pytest.mark.parametrize(
    "selection",
    [
        "not JSON",
        '{"citation_ids":[]}',
        '{"citation_ids":["C9"]}',
        '{"citation_ids":["C1","C1"]}',
        '{"citation_ids":["C1"],"answer":"invented"}',
    ],
)
def test_invalid_model_output_uses_sanitized_extractive_fallback(selection: str) -> None:
    with _server(_ollama_body(selection)) as (base_url, _):
        response = _generator(base_url).generate(_context())

    assert response.generator == "extractive_fallback"
    assert response.fallback_reason == "invalid_provider_response"
    assert selection not in json.dumps(response.to_dict())
    assert response.answer.endswith("[C1]")


def test_oversized_model_response_uses_safe_fallback() -> None:
    with _server(b"x" * 1_025) as (base_url, _):
        response = _generator(base_url, max_response_bytes=1_024).generate(_context())

    assert response.generator == "extractive_fallback"
    assert response.fallback_reason == "provider_response_too_large"


def test_provider_connection_failure_uses_safe_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(*args: object, **kwargs: object) -> None:
        raise URLError("private host details")

    monkeypatch.setattr("enterprise_copilot.llm.ollama.urlopen", unavailable)
    response = _generator("http://127.0.0.1:11434").generate(_context())

    assert response.generator == "extractive_fallback"
    assert response.fallback_reason == "provider_unavailable"
    assert "private host details" not in json.dumps(response.to_dict())


def test_unanswerable_context_does_not_call_model() -> None:
    with _server(_ollama_body('{"citation_ids":["C1"]}')) as (base_url, requests):
        response = _generator(base_url).generate(_context(answerable=False))

    assert requests == []
    assert response.status == "insufficient_evidence"


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://localhost:11434",
        "http://example.com:11434",
        "http://user:secret@localhost:11434",
        "http://localhost:11434/path",
    ],
)
def test_unsafe_or_invalid_provider_urls_are_rejected(base_url: str) -> None:
    with pytest.raises(ValueError):
        OllamaClient(base_url=base_url, model="test-model")


def test_environment_override_is_explicit_and_validated() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = GroundedAnswerConfig.from_json(
        project_root / "configs" / "grounded_answer_config.json"
    )

    overridden = config.with_environment(
        {
            "GENERATOR_BACKEND": "OLLAMA",
            "OLLAMA_BASE_URL": "https://models.example.com",
            "OLLAMA_MODEL": "approved-model:2",
            "OLLAMA_TIMEOUT_SECONDS": "4.5",
            "OLLAMA_SEED": "11",
            "OLLAMA_MAX_RESPONSE_BYTES": "4096",
        }
    )

    assert overridden.generator_backend == "ollama"
    assert overridden.ollama_timeout_seconds == 4.5
    assert overridden.ollama_seed == 11
    assert overridden.ollama_max_response_bytes == 4096


class _Retriever:
    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        return [
            SearchResult(
                score=0.9,
                chunk={
                    "chunk_id": "DOC-1::chunk-0000",
                    "document_id": "DOC-1",
                    "document_type": "policy",
                    "title": "Policy",
                    "region": "Ireland",
                    "product": "NSG Home",
                    "source_uri": "nsg://DOC-1",
                    "content": "The documented requirement is 20 minutes.",
                },
            )
        ]


class _InvalidGenerator:
    backend_name = "invalid-test"

    def generate(self, context: ContextPackage) -> GroundedAnswer:
        return GroundedAnswer(
            status="answered",
            answer="Invented fact. [C9]",
            confidence=0.9,
            citations=(Citation("C9", "BAD", "BAD", "Bad", "bad://source", "Invented fact."),),
        )


def test_pipeline_fails_closed_if_any_generator_breaks_citation_contract() -> None:
    pipeline = GroundedAnswerPipeline(
        _Retriever(),
        ContextBuilder(ContextBuilderConfig()),
        _InvalidGenerator(),
    )

    run = pipeline.ask("What is the policy?")

    assert run.response.status == "insufficient_evidence"
    assert run.response.generator == "verification_guardrail"
    assert run.response.fallback_reason == "citation_verification_failed"
    assert run.response.citations == ()
    assert run.verification.valid


def test_model_integration_acceptance_suite_passes(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = GroundedAnswerConfig.from_json(
        project_root / "configs" / "grounded_answer_config.json"
    )

    results = run_model_integration_validation(
        project_root,
        config,
        results_path=tmp_path / "model-results.json",
    )

    assert results["summary"]["checks"] == 5
    assert results["summary"]["pass_rate"] == 1.0

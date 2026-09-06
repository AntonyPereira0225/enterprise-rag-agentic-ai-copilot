from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from enterprise_copilot.evaluation.grounded_answers import evaluate_grounded_answers
from enterprise_copilot.ingestion.loaders import load_jsonl
from enterprise_copilot.llm.context import ContextBuilder, ContextBuilderConfig
from enterprise_copilot.llm.generation import ExtractiveAnswerGenerator, select_evidence
from enterprise_copilot.llm.pipeline import (
    GroundedAnswerConfig,
    GroundedAnswerPipeline,
    load_grounded_answer_pipeline,
)
from enterprise_copilot.llm.schemas import Evidence
from enterprise_copilot.llm.verification import verify_citations
from enterprise_copilot.retrieval.vector_index import SearchResult


def _chunk(
    document_id: str,
    document_type: str = "policy",
    *,
    product: str = "NSG Home",
    region: str = "Ireland",
    content: str = "The documented requirement is 10 minutes.",
) -> dict[str, object]:
    return {
        "chunk_id": f"{document_id}::chunk-0000",
        "document_id": document_id,
        "document_type": document_type,
        "title": f"{product} {document_type.replace('_', ' ').title()} - {region}",
        "region": region,
        "product": product,
        "source_uri": f"nsg://knowledge/{document_type}/{document_id}",
        "content": content,
    }


def _result(score: float, document_id: str = "DOC-1", **kwargs: object) -> SearchResult:
    return SearchResult(score=score, chunk=_chunk(document_id, **kwargs))


def _evidence(
    citation_id: str,
    document_id: str,
    document_type: str,
    product: str,
    region: str,
    score: float,
) -> Evidence:
    chunk = _chunk(
        document_id,
        document_type,
        product=product,
        region=region,
    )
    return Evidence(
        citation_id=citation_id,
        score=score,
        chunk_id=str(chunk["chunk_id"]),
        document_id=document_id,
        document_type=document_type,
        title=str(chunk["title"]),
        region=region,
        product=product,
        source_uri=str(chunk["source_uri"]),
        content=str(chunk["content"]),
    )


def test_context_builder_refuses_results_below_query_threshold() -> None:
    builder = ContextBuilder(ContextBuilderConfig(minimum_query_score=0.30))

    context = builder.build("unsupported question", [_result(0.29)])

    assert not context.can_answer
    assert context.evidence == ()
    assert "threshold" in str(context.refusal_reason)


def test_context_builder_enforces_budget_and_assigns_stable_citations() -> None:
    builder = ContextBuilder(ContextBuilderConfig(minimum_query_score=0.30, max_context_words=7))
    results = [
        _result(0.8, "DOC-1", content="one two three four"),
        _result(0.7, "DOC-2", content="five six seven eight"),
    ]

    context = builder.build("question", results)

    assert context.can_answer
    assert context.word_count == 4
    assert [item.citation_id for item in context.evidence] == ["C1"]


def test_generator_creates_an_exact_inline_citation() -> None:
    builder = ContextBuilder(ContextBuilderConfig())
    context = builder.build(
        "What is the applicable rule for NSG Home in Ireland?",
        [_result(0.8)],
    )

    response = ExtractiveAnswerGenerator().generate(context)
    verification = verify_citations(response, context)

    assert response.status == "answered"
    assert response.answer.endswith("[C1]")
    assert response.citations[0].quote == "The documented requirement is 10 minutes."
    assert verification.valid


def test_multi_source_selection_keeps_family_and_scope_together() -> None:
    question = (
        "Compare the requirement in the compliance guidance for NSG Plus in Germany "
        "with the support procedure for NSG Connect in Ireland."
    )
    evidence = (
        _evidence("C1", "WRONG-SUPPORT", "support_procedure", "NSG Plus", "Germany", 0.9),
        _evidence("C2", "RIGHT-SUPPORT", "support_procedure", "NSG Connect", "Ireland", 0.7),
        _evidence("C3", "RIGHT-COMPLIANCE", "compliance_guidance", "NSG Plus", "Germany", 0.6),
    )

    selected = select_evidence(question, evidence, max_citations=3)

    assert [item.document_id for item in selected] == [
        "RIGHT-COMPLIANCE",
        "RIGHT-SUPPORT",
    ]


def test_single_source_selection_prefers_the_more_specific_intent() -> None:
    question = "During an incident, what response window applies to NSG Home in Ireland?"
    evidence = (
        _evidence("C1", "RIGHT-PLAYBOOK", "operational_playbook", "NSG Home", "Ireland", 0.7),
        _evidence("C2", "EXTRA-SUPPORT", "support_procedure", "NSG Home", "Ireland", 0.8),
    )

    selected = select_evidence(question, evidence, max_citations=3)

    assert [item.document_id for item in selected] == ["RIGHT-PLAYBOOK"]


def test_verifier_rejects_a_quote_not_found_in_the_source() -> None:
    builder = ContextBuilder(ContextBuilderConfig())
    context = builder.build("What is the policy?", [_result(0.8)])
    response = ExtractiveAnswerGenerator().generate(context)
    bad_citation = replace(response.citations[0], quote="Invented requirement.")
    tampered = replace(response, citations=(bad_citation,))

    verification = verify_citations(tampered, context)

    assert not verification.valid
    assert any("not present in its source chunk" in error for error in verification.errors)


class _FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        return self.results[:top_k]


def test_pipeline_returns_a_citation_free_refusal() -> None:
    pipeline = GroundedAnswerPipeline(
        _FakeRetriever([_result(0.1)]),
        ContextBuilder(ContextBuilderConfig(minimum_query_score=0.30)),
        ExtractiveAnswerGenerator(),
    )

    run = pipeline.ask("unsupported question")

    assert run.response.status == "insufficient_evidence"
    assert run.response.citations == ()
    assert run.verification.valid


def test_all_hard_questions_have_grounded_or_refused_outputs() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = GroundedAnswerConfig.from_json(
        project_root / "configs" / "grounded_answer_config.json"
    )
    pipeline = load_grounded_answer_pipeline(project_root, config)
    questions = load_jsonl(project_root / config.evaluation_path)

    summary = evaluate_grounded_answers(pipeline, questions)["summary"]

    assert summary["evaluated_questions"] == 96
    assert summary["answerability_accuracy"] == 1.0
    assert summary["unanswerable_refusal_accuracy"] == 1.0
    assert summary["expected_evidence_coverage"] == 1.0
    assert summary["citation_verification_accuracy"] == 1.0
    assert summary["citation_source_precision"] == 1.0

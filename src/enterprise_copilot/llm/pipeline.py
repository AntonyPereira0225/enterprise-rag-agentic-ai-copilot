from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from enterprise_copilot.llm.context import ContextBuilder, ContextBuilderConfig
from enterprise_copilot.llm.generation import ExtractiveAnswerGenerator
from enterprise_copilot.llm.ollama import OllamaClient, OllamaEvidenceSelectorGenerator
from enterprise_copilot.llm.schemas import ContextPackage, GroundedAnswer, GroundedAnswerRun
from enterprise_copilot.llm.verification import verify_citations
from enterprise_copilot.retrieval.hybrid_pipeline import (
    HybridRetrievalConfig,
    load_hybrid_retriever,
)
from enterprise_copilot.retrieval.vector_index import SearchResult


class Retriever(Protocol):
    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]: ...


class AnswerGenerator(Protocol):
    backend_name: str

    def generate(self, context: ContextPackage) -> GroundedAnswer: ...


@dataclass(frozen=True)
class GroundedAnswerConfig:
    retrieval_config_path: str
    evaluation_path: str
    results_path: str
    top_k: int
    minimum_query_score: float
    minimum_evidence_score: float
    max_context_words: int
    max_answer_citations: int
    generator_backend: str = "extractive"
    fallback_backend: str = "extractive"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_seconds: float = 10.0
    ollama_seed: int = 7
    ollama_max_response_bytes: int = 65_536
    model_validation_results_path: str = "data/evaluation/model_integration_metrics.json"

    def __post_init__(self) -> None:
        if self.generator_backend not in {"extractive", "ollama"}:
            raise ValueError("generator_backend must be extractive or ollama")
        if self.fallback_backend != "extractive":
            raise ValueError("fallback_backend must be extractive")
        if self.max_answer_citations <= 0:
            raise ValueError("max_answer_citations must be greater than zero")

    @classmethod
    def from_json(cls, path: Path) -> GroundedAnswerConfig:
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def with_environment(
        self,
        environment: Mapping[str, str] | None = None,
    ) -> GroundedAnswerConfig:
        values = environment if environment is not None else os.environ
        try:
            timeout_seconds = float(
                values.get("OLLAMA_TIMEOUT_SECONDS", str(self.ollama_timeout_seconds))
            )
            seed = int(values.get("OLLAMA_SEED", str(self.ollama_seed)))
            max_response_bytes = int(
                values.get("OLLAMA_MAX_RESPONSE_BYTES", str(self.ollama_max_response_bytes))
            )
        except ValueError as exc:
            raise ValueError("Ollama timeout, seed, and response limit must be numeric") from exc
        return replace(
            self,
            generator_backend=values.get("GENERATOR_BACKEND", self.generator_backend).casefold(),
            ollama_base_url=values.get("OLLAMA_BASE_URL", self.ollama_base_url),
            ollama_model=values.get("OLLAMA_MODEL", self.ollama_model),
            ollama_timeout_seconds=timeout_seconds,
            ollama_seed=seed,
            ollama_max_response_bytes=max_response_bytes,
        )


class GroundedAnswerPipeline:
    def __init__(
        self,
        retriever: Retriever,
        context_builder: ContextBuilder,
        generator: AnswerGenerator,
    ) -> None:
        self.retriever = retriever
        self.context_builder = context_builder
        self.generator = generator

    def ask(self, question: str) -> GroundedAnswerRun:
        results = self.retriever.search(question, top_k=self.context_builder.config.top_k)
        context = self.context_builder.build(question, results)
        response = self.generator.generate(context)
        verification = verify_citations(response, context)
        if not verification.valid:
            response = GroundedAnswer(
                status="insufficient_evidence",
                answer="I cannot return this response because evidence verification failed.",
                confidence=0.0,
                citations=(),
                reason="The generated citation contract did not pass verification.",
                generator="verification_guardrail",
                model=response.model,
                fallback_reason="citation_verification_failed",
            )
            verification = verify_citations(response, context)
        return GroundedAnswerRun(
            context=context,
            response=response,
            verification=verification,
        )


def load_grounded_answer_pipeline(
    project_root: Path,
    config: GroundedAnswerConfig,
    *,
    use_environment: bool = False,
    environment: Mapping[str, str] | None = None,
) -> GroundedAnswerPipeline:
    active_config = config.with_environment(environment) if use_environment else config
    retrieval_config = HybridRetrievalConfig.from_json(
        project_root / active_config.retrieval_config_path
    )
    retriever = load_hybrid_retriever(project_root, retrieval_config)
    context_builder = ContextBuilder(
        ContextBuilderConfig(
            top_k=active_config.top_k,
            minimum_query_score=active_config.minimum_query_score,
            minimum_evidence_score=active_config.minimum_evidence_score,
            max_context_words=active_config.max_context_words,
        )
    )
    fallback = ExtractiveAnswerGenerator(max_citations=active_config.max_answer_citations)
    generator: AnswerGenerator = fallback
    if active_config.generator_backend == "ollama":
        generator = OllamaEvidenceSelectorGenerator(
            OllamaClient(
                base_url=active_config.ollama_base_url,
                model=active_config.ollama_model,
                timeout_seconds=active_config.ollama_timeout_seconds,
                seed=active_config.ollama_seed,
                max_response_bytes=active_config.ollama_max_response_bytes,
            ),
            max_citations=active_config.max_answer_citations,
            fallback=fallback,
        )
    return GroundedAnswerPipeline(retriever, context_builder, generator)

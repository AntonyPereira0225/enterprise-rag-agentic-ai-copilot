from __future__ import annotations

from enterprise_copilot.agents.state import AgentCitation, AgentExecution, AgentRoute
from enterprise_copilot.llm.pipeline import GroundedAnswerPipeline

_ALL_KNOWLEDGE_TYPES = frozenset(
    {
        "policy",
        "product_guide",
        "support_procedure",
        "faq",
        "operational_playbook",
        "compliance_guidance",
    }
)
_ROUTE_DOCUMENT_TYPES = {
    "policy": frozenset({"policy", "compliance_guidance", "operational_playbook"}),
    "product": frozenset({"product_guide"}),
    "support": frozenset({"support_procedure", "faq"}),
    "cross_functional": _ALL_KNOWLEDGE_TYPES,
    "general": _ALL_KNOWLEDGE_TYPES,
}


class KnowledgeAgent:
    """Adapter that gives each route an explicit evidence policy."""

    def __init__(self, route: AgentRoute, pipeline: GroundedAnswerPipeline) -> None:
        if route not in _ROUTE_DOCUMENT_TYPES:
            raise ValueError(f"Unsupported knowledge-agent route: {route}")
        self.route = route
        self.pipeline = pipeline
        self.allowed_document_types = _ROUTE_DOCUMENT_TYPES[route]

    def execute(self, question: str) -> AgentExecution:
        run = self.pipeline.ask(question)
        response = run.response
        evidence_by_id = {item.citation_id: item for item in run.context.evidence}
        cited_document_types = {
            evidence_by_id[citation.citation_id].document_type
            for citation in response.citations
            if citation.citation_id in evidence_by_id
        }
        route_valid = cited_document_types.issubset(self.allowed_document_types)
        citations = tuple(
            AgentCitation(
                citation_id=citation.citation_id,
                source_type="knowledge_chunk",
                source_id=citation.document_id,
                title=citation.title,
                source_uri=citation.source_uri,
                quote=citation.quote,
            )
            for citation in response.citations
        )
        return AgentExecution(
            status="answered" if response.status == "answered" else "refused",
            answer=response.answer,
            confidence=response.confidence,
            citations=citations,
            verified=run.verification.valid and route_valid,
            reason=response.reason,
            details={
                "route": self.route,
                "allowed_document_types": sorted(self.allowed_document_types),
                "cited_document_types": sorted(cited_document_types),
                "generator": response.generator,
                "model": response.model,
                "fallback_reason": response.fallback_reason,
            },
        )


def build_knowledge_agents(
    pipeline: GroundedAnswerPipeline,
) -> dict[AgentRoute, KnowledgeAgent]:
    routes: tuple[AgentRoute, ...] = (
        "policy",
        "product",
        "support",
        "cross_functional",
        "general",
    )
    return {route: KnowledgeAgent(route, pipeline) for route in routes}

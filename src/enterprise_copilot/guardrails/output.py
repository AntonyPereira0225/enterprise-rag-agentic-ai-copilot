from __future__ import annotations

import re

from enterprise_copilot.agents.state import AgentExecution
from enterprise_copilot.guardrails.schemas import GuardrailDecision

_INLINE_CITATION = re.compile(r"\[([A-Z]\d+)\]")


class OutputGuardrail:
    """Fail closed when an agent response loses its evidence contract."""

    def check(self, execution: AgentExecution) -> GuardrailDecision:
        if not execution.verified:
            return _blocked("verification_failure", "Response verification failed.", "OUTPUT-001")
        if execution.status == "answered" and not execution.citations:
            return _blocked(
                "missing_citation",
                "An answered response must include verified evidence.",
                "OUTPUT-002",
            )
        inline_ids = _INLINE_CITATION.findall(execution.answer)
        citation_ids = [citation.citation_id for citation in execution.citations]
        if inline_ids != citation_ids or len(citation_ids) != len(set(citation_ids)):
            return _blocked(
                "citation_mismatch",
                "Inline citations do not match the structured citation records.",
                "OUTPUT-004",
            )
        if execution.status == "refused" and execution.citations:
            return _blocked(
                "invalid_refusal",
                "A refusal must not contain citations.",
                "OUTPUT-003",
            )
        return GuardrailDecision(
            action="allow",
            category="verified_output",
            message="Output checks passed.",
            rule_id="OUTPUT-ALLOW",
        )


def _blocked(category: str, message: str, rule_id: str) -> GuardrailDecision:
    return GuardrailDecision(
        action="block",
        category=category,
        message=message,
        rule_id=rule_id,
    )

from __future__ import annotations

import re
from collections.abc import Iterable

from enterprise_copilot.llm.schemas import Citation, ContextPackage, Evidence, GroundedAnswer

_FAMILY_CUES = {
    "compliance_guidance": ("compliance guidance", "review steps"),
    "faq": ("faq", "turnaround time", "agents quote"),
    "operational_playbook": ("operational playbook", "during an incident"),
    "policy": ("policy", "applicable rule"),
    "product_guide": ("product guide", "capabilities", "features"),
    "support_procedure": ("support procedure", "support staff", "response window"),
}
_REQUIREMENT_MARKER = "documented requirement is"


def _requested_families(question: str) -> list[str]:
    normalised = question.casefold()
    matches = [
        family for family, cues in _FAMILY_CUES.items() if any(cue in normalised for cue in cues)
    ]
    return matches if "compare" in normalised else matches[:1]


def _scope_matches(question: str, evidence: Evidence) -> bool:
    scope = f"{evidence.product} in {evidence.region}".casefold()
    return scope in question.casefold()


def _family_scope_matches(question: str, family: str, evidence: Evidence) -> bool:
    family_name = family.replace("_", " ")
    scoped_request = f"{family_name} for {evidence.product} in {evidence.region}".casefold()
    return scoped_request in question.casefold()


def _best_evidence(
    question: str,
    values: Iterable[Evidence],
    *,
    family: str | None = None,
) -> Evidence | None:
    candidates = list(values)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            -int(family is not None and _family_scope_matches(question, family, item)),
            -int(_scope_matches(question, item)),
            -item.score,
            item.chunk_id,
        ),
    )


def select_evidence(
    question: str,
    evidence: tuple[Evidence, ...],
    *,
    max_citations: int,
) -> tuple[Evidence, ...]:
    """Select one scoped passage per requested document family."""
    requested_families = _requested_families(question)
    selected: list[Evidence] = []
    for family in requested_families:
        item = _best_evidence(
            question,
            (candidate for candidate in evidence if candidate.document_type == family),
            family=family,
        )
        if item is not None and item not in selected:
            selected.append(item)
        if len(selected) == max_citations:
            break

    if not selected:
        item = _best_evidence(question, evidence)
        if item is not None:
            selected.append(item)
    return tuple(selected)


def extract_supporting_quote(content: str) -> str:
    sentences = [value.strip() for value in re.split(r"(?<=[.!?])\s+", content)]
    for sentence in sentences:
        if _REQUIREMENT_MARKER in sentence.casefold():
            return sentence
    return sentences[0] if sentences else content.strip()


def build_grounded_answer(
    context: ContextPackage,
    selected: Iterable[Evidence],
    *,
    generator: str,
    model: str | None = None,
    fallback_reason: str | None = None,
) -> GroundedAnswer:
    """Render an answer locally from exact source sentences."""
    citations = tuple(
        Citation(
            citation_id=item.citation_id,
            document_id=item.document_id,
            chunk_id=item.chunk_id,
            title=item.title,
            source_uri=item.source_uri,
            quote=extract_supporting_quote(item.content),
        )
        for item in selected
    )
    answer_parts = [f"{citation.quote} [{citation.citation_id}]" for citation in citations]
    return GroundedAnswer(
        status="answered",
        answer=" ".join(answer_parts),
        confidence=context.query_score,
        citations=citations,
        generator=generator,
        model=model,
        fallback_reason=fallback_reason,
    )


class ExtractiveAnswerGenerator:
    """Create deterministic answers made only from retrieved source sentences."""

    backend_name = "extractive"

    def __init__(self, *, max_citations: int = 3) -> None:
        if max_citations <= 0:
            raise ValueError("max_citations must be greater than zero")
        self.max_citations = max_citations

    def generate(self, context: ContextPackage) -> GroundedAnswer:
        if not context.can_answer:
            return GroundedAnswer(
                status="insufficient_evidence",
                answer=("I do not have enough approved evidence to answer this question safely."),
                confidence=context.query_score,
                citations=(),
                reason=context.refusal_reason,
                generator=self.backend_name,
            )

        selected = select_evidence(
            context.question,
            context.evidence,
            max_citations=self.max_citations,
        )
        return build_grounded_answer(
            context,
            selected,
            generator=self.backend_name,
        )

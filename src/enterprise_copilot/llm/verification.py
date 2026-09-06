from __future__ import annotations

import re

from enterprise_copilot.llm.schemas import (
    CitationVerification,
    ContextPackage,
    GroundedAnswer,
)

_INLINE_CITATION = re.compile(r"\[([A-Z]\d+)\]")


def verify_citations(
    answer: GroundedAnswer,
    context: ContextPackage,
) -> CitationVerification:
    """Verify that every answer citation resolves to an exact retrieved quote."""
    errors: list[str] = []
    inline_ids = _INLINE_CITATION.findall(answer.answer)
    citation_ids = [citation.citation_id for citation in answer.citations]

    if answer.status == "insufficient_evidence":
        if answer.citations or inline_ids:
            errors.append("A refusal must not contain citations.")
        return CitationVerification(not errors, tuple(errors), len(answer.citations))

    if not answer.citations:
        errors.append("An answered response must contain at least one citation.")
    if inline_ids != citation_ids:
        errors.append("Inline citation order does not match the citation records.")
    if len(citation_ids) != len(set(citation_ids)):
        errors.append("Citation identifiers must be unique.")

    evidence_by_id = {item.citation_id: item for item in context.evidence}
    for citation in answer.citations:
        evidence = evidence_by_id.get(citation.citation_id)
        if evidence is None:
            errors.append(f"{citation.citation_id} does not resolve to retrieved evidence.")
            continue
        if citation.document_id != evidence.document_id:
            errors.append(f"{citation.citation_id} has the wrong document identifier.")
        if citation.chunk_id != evidence.chunk_id:
            errors.append(f"{citation.citation_id} has the wrong chunk identifier.")
        if citation.source_uri != evidence.source_uri:
            errors.append(f"{citation.citation_id} has the wrong source URI.")
        if citation.title != evidence.title:
            errors.append(f"{citation.citation_id} has the wrong source title.")
        if citation.quote not in evidence.content:
            errors.append(f"{citation.citation_id} quote is not present in its source chunk.")
        if citation.quote not in answer.answer:
            errors.append(f"{citation.citation_id} quote is not present in the answer.")

    return CitationVerification(not errors, tuple(errors), len(answer.citations))

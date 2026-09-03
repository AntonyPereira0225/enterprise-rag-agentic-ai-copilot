from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


DOCUMENT_TYPES = (
    "policy",
    "product_guide",
    "support_procedure",
    "faq",
    "operational_playbook",
    "compliance_guidance",
)

ISSUE_TYPES = (
    "billing_query",
    "refund_request",
    "service_activation",
    "cancellation",
    "access_problem",
    "service_disruption",
)

DEPARTMENTS = (
    "Customer Support",
    "Operations",
    "Product",
    "Risk & Compliance",
)

CHANNELS = ("web", "phone", "email", "chat")
PRIORITIES = ("low", "medium", "high")


@dataclass(frozen=True)
class CorpusConfig:
    seed: int
    company_name: str
    regions: list[str]
    products: list[str]
    document_counts: dict[str, int]
    support_case_count: int
    evaluation_question_count: int

    @classmethod
    def from_json(cls, path: Path) -> "CorpusConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload)


@dataclass
class GenerationResult:
    knowledge_documents: list[dict[str, Any]]
    support_cases: list[dict[str, Any]]
    evaluation_questions: list[dict[str, Any]]


def _stable_fact(index: int, document_type: str) -> tuple[str, str]:
    """Return a deterministic policy-like fact and a matching question stem."""
    values = {
        "policy": (7 + index % 8, "calendar days"),
        "product_guide": (2 + index % 6, "supported service features"),
        "support_procedure": (15 + (index % 6) * 5, "minutes"),
        "faq": (24 + (index % 4) * 12, "hours"),
        "operational_playbook": (10 + (index % 5) * 10, "minutes"),
        "compliance_guidance": (2 + index % 5, "approval checks"),
    }
    number, unit = values[document_type]
    return f"{number} {unit}", f"What is the documented requirement measured as {unit}?"


def _document_content(
    *,
    company: str,
    document_type: str,
    region: str,
    product: str,
    title: str,
    fact: str,
    index: int,
) -> str:
    issue = ISSUE_TYPES[index % len(ISSUE_TYPES)].replace("_", " ")
    department = DEPARTMENTS[index % len(DEPARTMENTS)]
    return (
        f"{title}. This approved {document_type.replace('_', ' ')} applies to {product} in {region}. "
        f"The owning function is {department}. For the primary {issue} scenario, the documented "
        f"requirement is {fact}. Employees must verify the current document status before relying "
        f"on this guidance. If the available evidence is incomplete, contradictory, or outside the "
        f"scope of this document, the case must be escalated rather than resolved by assumption. "
        f"This material belongs to the synthetic {company} knowledge estate and contains no real "
        f"customer or company information. Reference marker NSG-{document_type.upper()}-{index:03d}."
    )


def generate_knowledge_documents(config: CorpusConfig) -> list[dict[str, Any]]:
    rng = random.Random(config.seed)
    documents: list[dict[str, Any]] = []
    global_index = 1

    for document_type in DOCUMENT_TYPES:
        target_count = config.document_counts.get(document_type, 0)
        for local_index in range(1, target_count + 1):
            region = config.regions[(global_index - 1) % len(config.regions)]
            product = config.products[(global_index - 1) % len(config.products)]
            department = DEPARTMENTS[(global_index - 1) % len(DEPARTMENTS)]
            fact, _ = _stable_fact(global_index, document_type)
            document_id = f"NSG-{document_type[:3].upper()}-{global_index:04d}"
            title = (
                f"{product} {document_type.replace('_', ' ').title()} "
                f"{local_index:02d} - {region}"
            )
            effective = date(2026, 1, 1) + timedelta(days=(global_index * 7) % 220)
            tags = sorted(
                {
                    document_type,
                    region.lower().replace(" ", "-"),
                    product.lower().replace(" ", "-"),
                    ISSUE_TYPES[(global_index - 1) % len(ISSUE_TYPES)],
                }
            )
            documents.append(
                {
                    "document_id": document_id,
                    "document_type": document_type,
                    "title": title,
                    "department": department,
                    "region": region,
                    "product": product,
                    "version": f"1.{(global_index - 1) % 4}",
                    "effective_date": effective.isoformat(),
                    "status": "active",
                    "sensitivity": "internal",
                    "tags": tags,
                    "source_uri": f"nsg://knowledge/{document_type}/{document_id}",
                    "ground_truth_fact": fact,
                    "content": _document_content(
                        company=config.company_name,
                        document_type=document_type,
                        region=region,
                        product=product,
                        title=title,
                        fact=fact,
                        index=global_index,
                    ),
                }
            )
            global_index += 1

    rng.shuffle(documents)
    return documents


def generate_support_cases(
    config: CorpusConfig, documents: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rng = random.Random(config.seed + 1)
    cases: list[dict[str, Any]] = []
    base_time = datetime(2026, 1, 1, 9, 0, 0)

    for index in range(1, config.support_case_count + 1):
        document = documents[(index - 1) % len(documents)]
        issue_type = ISSUE_TYPES[(index - 1) % len(ISSUE_TYPES)]
        priority = PRIORITIES[(index - 1) % len(PRIORITIES)]
        resolution_minutes = 8 + ((index * 13) % 173)
        created_at = base_time + timedelta(hours=index * 5)
        status = "resolved" if index % 9 else "escalated"
        resolution_code = "guided_resolution" if status == "resolved" else "specialist_escalation"

        cases.append(
            {
                "case_id": f"CASE-{index:06d}",
                "created_at": created_at.isoformat(),
                "region": document["region"],
                "product": document["product"],
                "issue_type": issue_type,
                "priority": priority,
                "channel": rng.choice(CHANNELS),
                "status": status,
                "resolution_code": resolution_code,
                "resolution_minutes": resolution_minutes,
                "summary": (
                    f"Synthetic {issue_type.replace('_', ' ')} case for {document['product']} in "
                    f"{document['region']}. The support user referenced approved internal guidance."
                ),
                "linked_document_ids": [document["document_id"]],
            }
        )

    return cases


def generate_evaluation_questions(
    config: CorpusConfig, documents: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    ordered = sorted(documents, key=lambda item: item["document_id"])
    questions: list[dict[str, Any]] = []
    target = min(config.evaluation_question_count, len(ordered))

    for index, document in enumerate(ordered[:target], start=1):
        _, question_stem = _stable_fact(index, document["document_type"])
        question = (
            f"For {document['product']} in {document['region']}, {question_stem.lower()}"
        )
        questions.append(
            {
                "question_id": f"Q-{index:04d}",
                "question": question,
                "expected_answer": document["ground_truth_fact"],
                "expected_document_ids": [document["document_id"]],
                "intent": document["document_type"],
                "region": document["region"],
                "product": document["product"],
                "difficulty": "single_source",
            }
        )

    return questions


def generate_corpus(config: CorpusConfig) -> GenerationResult:
    documents = generate_knowledge_documents(config)
    cases = generate_support_cases(config, documents)
    questions = generate_evaluation_questions(config, documents)
    return GenerationResult(documents, cases, questions)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_corpus(result: GenerationResult, root: Path) -> None:
    _write_jsonl(root / "raw" / "knowledge_base.jsonl", result.knowledge_documents)
    _write_jsonl(root / "raw" / "support_cases.jsonl", result.support_cases)
    _write_jsonl(root / "evaluation" / "rag_eval_questions.jsonl", result.evaluation_questions)

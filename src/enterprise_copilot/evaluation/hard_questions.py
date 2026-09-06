from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HardEvaluationConfig:
    source_path: str
    output_path: str
    paraphrased_count: int
    multi_source_count: int
    unanswerable_count: int

    @classmethod
    def from_json(cls, path: Path) -> HardEvaluationConfig:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload)


_PARAPHRASE_TEMPLATES = {
    "policy": "How much time does the applicable rule allow for {product} customers in {region}?",
    "product_guide": "How many capabilities should be available with {product} in {region}?",
    "support_procedure": "What response window should support staff use for {product} in {region}?",
    "faq": "What turnaround time should agents quote for {product} users in {region}?",
    "operational_playbook": "During an incident, what response window applies to {product} in {region}?",
    "compliance_guidance": "How many review steps must be completed for {product} in {region}?",
}

_UNANSWERABLE_TEMPLATES = (
    "What is the satellite-device replacement limit for NSG Enterprise in France?",
    "How many biometric checks are required for NSG Business in Belgium?",
    "What is the cryptocurrency refund window for NSG Global in Austria?",
    "How quickly must drone delivery be activated for NSG Air in Denmark?",
    "What is the approved cash-loan ceiling for NSG Credit in Sweden?",
    "How many roaming planets does NSG Space support in Portugal?",
    "What is the paper-cheque retention period for NSG Banking in Finland?",
    "How quickly is hardware collected for NSG Devices in Poland?",
    "How many vehicle profiles are allowed by NSG Auto in Norway?",
    "What is the medical-claim escalation window for NSG Health in Greece?",
    "How long is the hotel cancellation period for NSG Travel in Switzerland?",
    "How many warehouse robots are covered by NSG Logistics in Czechia?",
)


def _evidence(document: dict[str, Any]) -> dict[str, str]:
    return {
        "document_id": document["document_id"],
        "answer": document["ground_truth_fact"],
    }


def generate_hard_questions(
    documents: list[dict[str, Any]], config: HardEvaluationConfig
) -> list[dict[str, Any]]:
    """Create deterministic paraphrased, multi-source, and unsupported questions."""
    if not documents:
        raise ValueError("At least one knowledge document is required")
    if config.paraphrased_count > len(documents):
        raise ValueError("paraphrased_count cannot exceed the number of documents")
    if config.multi_source_count * 2 > len(documents):
        raise ValueError("multi_source_count requires two distinct documents per question")
    if config.unanswerable_count > len(_UNANSWERABLE_TEMPLATES):
        raise ValueError("unanswerable_count exceeds the available deterministic templates")

    ordered = sorted(documents, key=lambda row: row["document_id"])
    questions: list[dict[str, Any]] = []

    for index, document in enumerate(ordered[: config.paraphrased_count], start=1):
        question = _PARAPHRASE_TEMPLATES[document["document_type"]].format(**document)
        questions.append(
            {
                "question_id": f"HP-{index:04d}",
                "question": question,
                "expected_answer": document["ground_truth_fact"],
                "expected_document_ids": [document["document_id"]],
                "expected_evidence": [_evidence(document)],
                "answerable": True,
                "intent": document["document_type"],
                "region": document["region"],
                "product": document["product"],
                "difficulty": "paraphrased_single_source",
            }
        )

    for index in range(config.multi_source_count):
        first = ordered[index]
        second = ordered[-(index + 1)]
        first_family = first["document_type"].replace("_", " ")
        second_family = second["document_type"].replace("_", " ")
        questions.append(
            {
                "question_id": f"HM-{index + 1:04d}",
                "question": (
                    f"Compare the requirement in the {first_family} for {first['product']} in "
                    f"{first['region']} with the {second_family} for {second['product']} in "
                    f"{second['region']}."
                ),
                "expected_answer": (
                    f"{first['ground_truth_fact']} and {second['ground_truth_fact']}"
                ),
                "expected_document_ids": [first["document_id"], second["document_id"]],
                "expected_evidence": [_evidence(first), _evidence(second)],
                "answerable": True,
                "intent": "multi_source_comparison",
                "region": [first["region"], second["region"]],
                "product": [first["product"], second["product"]],
                "difficulty": "multi_source",
            }
        )

    for index, question in enumerate(_UNANSWERABLE_TEMPLATES[: config.unanswerable_count], start=1):
        questions.append(
            {
                "question_id": f"HU-{index:04d}",
                "question": question,
                "expected_answer": None,
                "expected_document_ids": [],
                "expected_evidence": [],
                "answerable": False,
                "intent": "unsupported_knowledge",
                "region": None,
                "product": None,
                "difficulty": "unanswerable",
            }
        )

    return questions


def write_hard_questions(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

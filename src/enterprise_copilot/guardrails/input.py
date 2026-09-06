from __future__ import annotations

import re
from dataclasses import dataclass

from enterprise_copilot.guardrails.schemas import GuardrailDecision

_INJECTION_PATTERNS = (
    re.compile(r"\b(ignore|disregard|override)\b.{0,50}\b(instructions?|rules?|guardrails?)\b"),
    re.compile(r"\b(system|developer|hidden)\s+(prompt|message|instructions?)\b"),
    re.compile(r"\b(reveal|repeat|print|expose)\b.{0,40}\b(prompt|instructions?)\b"),
    re.compile(r"\b(jailbreak|developer mode|act as dan)\b"),
)
_SENSITIVE_PATTERNS = (
    re.compile(r"\b(reveal|show|list|expose|give me)\b.{0,50}\b(api keys?|passwords?)\b"),
    re.compile(r"\b(reveal|show|list|expose|give me)\b.{0,50}\b(access tokens?|secrets?)\b"),
    re.compile(r"\b(customer|employee)\b.{0,40}\b(ssn|passport|credit card)\b"),
)
_ACTION_PATTERNS = (
    re.compile(
        r"\b(execute|issue|make|perform|approve|transfer|delete|close|cancel|update)\b"
        r".{0,60}\b(account|transaction|payment|refund|customer record)\b"
    ),
    re.compile(r"\bchange\b.{0,40}\b(customer account|customer record|account balance)\b"),
)


@dataclass(frozen=True)
class InputGuardrail:
    max_characters: int = 500

    def __post_init__(self) -> None:
        if self.max_characters <= 0:
            raise ValueError("max_characters must be greater than zero")

    def check(self, question: str) -> GuardrailDecision:
        normalised = " ".join(question.casefold().split())
        if not normalised:
            return _blocked("invalid_input", "The question is empty.", "INPUT-001")
        if len(question) > self.max_characters:
            return _blocked(
                "invalid_input",
                f"The question exceeds the {self.max_characters}-character limit.",
                "INPUT-002",
            )
        if any(pattern.search(normalised) for pattern in _INJECTION_PATTERNS):
            return _blocked(
                "prompt_injection",
                "I cannot follow requests that alter or expose protected instructions. "
                "Please ask a direct business knowledge question.",
                "SEC-001",
            )
        if any(pattern.search(normalised) for pattern in _SENSITIVE_PATTERNS):
            return _blocked(
                "sensitive_data",
                "I cannot reveal credentials or sensitive personal information.",
                "SEC-002",
            )
        if any(pattern.search(normalised) for pattern in _ACTION_PATTERNS):
            return _blocked(
                "unauthorised_action",
                "I can explain approved guidance, but I cannot execute customer-account actions.",
                "SEC-003",
            )
        return GuardrailDecision(
            action="allow",
            category="safe_input",
            message="Input checks passed.",
            rule_id="INPUT-ALLOW",
        )


def _blocked(category: str, message: str, rule_id: str) -> GuardrailDecision:
    return GuardrailDecision(
        action="block",
        category=category,
        message=message,
        rule_id=rule_id,
    )

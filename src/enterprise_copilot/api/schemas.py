from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_CONVERSATION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class RequestValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class AskRequest:
    question: str
    conversation_id: str

    @classmethod
    def from_payload(cls, payload: Any) -> AskRequest:
        if not isinstance(payload, dict):
            raise RequestValidationError(["Request body must be a JSON object."])

        errors: list[str] = []
        unknown = sorted(set(payload).difference({"question", "conversation_id"}))
        if unknown:
            errors.append(f"Unknown fields: {', '.join(unknown)}.")

        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            errors.append("question must be a non-empty string.")

        conversation_id = payload.get("conversation_id", "default")
        if not isinstance(conversation_id, str) or not _CONVERSATION_ID.fullmatch(conversation_id):
            errors.append(
                "conversation_id must contain 1-64 letters, numbers, hyphens, or underscores."
            )
        if errors:
            raise RequestValidationError(errors)
        return cls(question=question.strip(), conversation_id=conversation_id)


@dataclass(frozen=True)
class ServiceResponse:
    status_code: int
    body: dict[str, Any]

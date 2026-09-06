from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from enterprise_copilot.guardrails.schemas import GuardrailDecision

AgentRoute = Literal[
    "policy",
    "product",
    "support",
    "analytics",
    "cross_functional",
    "general",
    "blocked",
]


@dataclass(frozen=True)
class AgentCitation:
    citation_id: str
    source_type: Literal["knowledge_chunk", "dataset"]
    source_id: str
    title: str
    source_uri: str
    quote: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class AgentExecution:
    status: Literal["answered", "refused"]
    answer: str
    confidence: float
    citations: tuple[AgentCitation, ...]
    verified: bool
    reason: str | None = None
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class WorkflowEvent:
    step: str
    outcome: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowResult:
    request_id: str
    conversation_id: str
    turn_number: int
    route: AgentRoute
    status: Literal["completed", "refused", "blocked", "failed_safe"]
    answer: str
    confidence: float
    citations: tuple[AgentCitation, ...]
    input_guardrail: GuardrailDecision
    output_guardrail: GuardrailDecision | None
    trace: tuple[WorkflowEvent, ...]
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "conversation_id": self.conversation_id,
            "turn_number": self.turn_number,
            "route": self.route,
            "status": self.status,
            "answer": self.answer,
            "confidence": self.confidence,
            "citations": [citation.to_dict() for citation in self.citations],
            "input_guardrail": self.input_guardrail.to_dict(),
            "output_guardrail": (
                self.output_guardrail.to_dict() if self.output_guardrail else None
            ),
            "trace": [event.to_dict() for event in self.trace],
            "details": self.details,
        }

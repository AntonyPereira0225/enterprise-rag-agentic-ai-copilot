from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from enterprise_copilot.agents.state import AgentRoute

_ANALYTICS_CUES = (
    "how many support cases",
    "number of support cases",
    "count of support cases",
    "average resolution",
    "case breakdown",
    "case trend",
)
_POLICY_CUES = (
    "policy",
    "applicable rule",
    "compliance guidance",
    "approval checks",
    "review steps",
    "operational playbook",
    "during an incident",
)
_PRODUCT_CUES = ("product guide", "capabilities", "supported service features")
_SUPPORT_CUES = (
    "support procedure",
    "support staff",
    "response window",
    "faq",
    "turnaround time",
    "agents quote",
)
_FOLLOW_UP_PREFIXES = ("and ", "what about ", "how about ", "for the same ")


@dataclass(frozen=True)
class RouteDecision:
    route: AgentRoute
    confidence: float
    reason: str
    used_conversation_memory: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IntentRouter:
    """Transparent rule router for the supported specialist workflows."""

    def route(
        self,
        question: str,
        *,
        previous_route: AgentRoute | None = None,
    ) -> RouteDecision:
        normalised = " ".join(question.casefold().split())
        if "compare" in normalised:
            return RouteDecision(
                "cross_functional",
                0.95,
                "Comparison requests may require evidence from multiple specialists.",
            )
        for route, cues, reason in (
            ("analytics", _ANALYTICS_CUES, "The request asks for a support-case aggregate."),
            ("policy", _POLICY_CUES, "The request uses policy, compliance, or playbook language."),
            ("product", _PRODUCT_CUES, "The request asks about product capabilities."),
            ("support", _SUPPORT_CUES, "The request asks for support or FAQ guidance."),
        ):
            if any(cue in normalised for cue in cues):
                return RouteDecision(route, 0.90, reason)

        if previous_route not in (None, "blocked", "general") and normalised.startswith(
            _FOLLOW_UP_PREFIXES
        ):
            return RouteDecision(
                previous_route,
                0.70,
                "An elliptical follow-up reused the previous specialist route.",
                used_conversation_memory=True,
            )
        return RouteDecision(
            "general",
            0.60,
            "No specialist cue was found; use the grounded general workflow.",
        )

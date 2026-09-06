from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class GuardrailDecision:
    action: Literal["allow", "block"]
    category: str
    message: str
    rule_id: str

    @property
    def allowed(self) -> bool:
        return self.action == "allow"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "allowed": self.allowed}

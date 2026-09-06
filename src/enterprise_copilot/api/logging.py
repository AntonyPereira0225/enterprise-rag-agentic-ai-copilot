from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol


class LogStream(Protocol):
    def write(self, value: str) -> int: ...

    def flush(self) -> None: ...


@dataclass
class JsonRequestLogger:
    stream: LogStream = sys.stdout
    now: Callable[[], datetime] = lambda: datetime.now(UTC)

    def log(
        self,
        *,
        request_id: str | None,
        conversation_hash: str | None,
        question_hash: str | None,
        route: str,
        status: str,
        latency_ms: float,
        error_type: str | None = None,
    ) -> None:
        event = {
            "event": "copilot_request",
            "recorded_at": self.now().astimezone(UTC).isoformat(),
            "request_id": request_id,
            "conversation_sha256": conversation_hash,
            "question_sha256": question_hash,
            "route": route,
            "status": status,
            "latency_ms": round(latency_ms, 3),
            "error_type": error_type,
        }
        self.stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        self.stream.flush()

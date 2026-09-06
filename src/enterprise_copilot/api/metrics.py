from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4


class ServiceMetrics:
    """Thread-safe, restart-aware metrics for one local writer process."""

    def __init__(self, snapshot_path: Path | None = None) -> None:
        self._lock = Lock()
        self._snapshot_path = snapshot_path
        self._requests = 0
        self._errors = 0
        self._latency_ms = 0.0
        self._statuses: Counter[str] = Counter()
        self._routes: Counter[str] = Counter()
        self._guardrail_categories: Counter[str] = Counter()
        self._persistence_errors = 0
        if self._snapshot_path:
            with self._lock:
                if self._snapshot_path.is_file():
                    self._load_unlocked()
                else:
                    self._persist_unlocked()

    def record(
        self,
        *,
        status: str,
        route: str,
        latency_ms: float,
        guardrail_category: str | None = None,
        error: bool = False,
    ) -> None:
        if isinstance(latency_ms, bool) or not math.isfinite(latency_ms) or latency_ms < 0:
            raise ValueError("latency_ms must be a finite, non-negative number")
        with self._lock:
            self._requests += 1
            self._errors += int(error)
            self._latency_ms += latency_ms
            self._statuses[status] += 1
            self._routes[route] += 1
            if guardrail_category:
                self._guardrail_categories[guardrail_category] += 1
            self._persist_unlocked()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> dict[str, Any]:
        average_latency = self._latency_ms / self._requests if self._requests else 0.0
        return {
            "requests_total": self._requests,
            "errors_total": self._errors,
            "latency_ms_total": round(self._latency_ms, 3),
            "average_latency_ms": round(average_latency, 3),
            "status_counts": dict(sorted(self._statuses.items())),
            "route_counts": dict(sorted(self._routes.items())),
            "guardrail_counts": dict(sorted(self._guardrail_categories.items())),
            "observability_persistence_errors_total": self._persistence_errors,
        }

    def _load_unlocked(self) -> None:
        try:
            payload = json.loads(self._snapshot_path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
            self._requests = int(payload["requests_total"])
            self._errors = int(payload["errors_total"])
            self._latency_ms = float(
                payload.get(
                    "latency_ms_total",
                    payload.get("average_latency_ms", 0.0) * self._requests,
                )
            )
            self._statuses.update(payload.get("status_counts", {}))
            self._routes.update(payload.get("route_counts", {}))
            self._guardrail_categories.update(payload.get("guardrail_counts", {}))
            self._persistence_errors = int(payload.get("observability_persistence_errors_total", 0))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            self._requests = 0
            self._errors = 0
            self._latency_ms = 0.0
            self._statuses.clear()
            self._routes.clear()
            self._guardrail_categories.clear()
            self._persistence_errors += 1

    def _persist_unlocked(self) -> None:
        if self._snapshot_path is None:
            return
        temporary = self._snapshot_path.with_name(f".{self._snapshot_path.name}.{uuid4().hex}.tmp")
        try:
            self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(self._snapshot_unlocked(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            temporary.replace(self._snapshot_path)
        except OSError:
            self._persistence_errors += 1
            temporary.unlink(missing_ok=True)

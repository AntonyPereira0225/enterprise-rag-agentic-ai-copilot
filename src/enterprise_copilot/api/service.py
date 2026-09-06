from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from enterprise_copilot.agents.graph import (
    AgentWorkflow,
    AgentWorkflowConfig,
    load_agent_workflow,
)
from enterprise_copilot.api.logging import JsonRequestLogger
from enterprise_copilot.api.metrics import ServiceMetrics
from enterprise_copilot.api.schemas import (
    AskRequest,
    RequestValidationError,
    ServiceResponse,
)
from enterprise_copilot.monitoring.experiment import ObservabilityConfig, resolve_project_path
from enterprise_copilot.monitoring.observability import console_and_file_stream


@dataclass(frozen=True)
class ApiConfig:
    workflow_config_path: str
    observability_config_path: str
    host: str
    port: int
    max_request_bytes: int
    demo_page_path: str
    validation_results_path: str
    service_name: str
    service_version: str

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.max_request_bytes <= 0:
            raise ValueError("max_request_bytes must be greater than zero")

    @classmethod
    def from_json(cls, path: Path) -> ApiConfig:
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def with_environment(self, environment: Mapping[str, str] | None = None) -> ApiConfig:
        values = environment if environment is not None else os.environ
        try:
            port = int(values.get("API_PORT", str(self.port)))
        except ValueError as exc:
            raise ValueError("API_PORT must be an integer") from exc
        return replace(self, host=values.get("API_HOST", self.host), port=port)


class CopilotService:
    def __init__(
        self,
        workflow: AgentWorkflow,
        *,
        name: str,
        version: str,
        metrics: ServiceMetrics | None = None,
        logger: JsonRequestLogger | None = None,
        clock: Callable[[], float] = time.perf_counter,
        privacy_key: bytes | None = None,
    ) -> None:
        self.workflow = workflow
        self.name = name
        self.version = version
        self.metrics = metrics or ServiceMetrics()
        self.logger = logger or JsonRequestLogger()
        self.clock = clock
        self._privacy_key = privacy_key or secrets.token_bytes(32)

    def _privacy_hash(self, value: str) -> str:
        return hmac.new(self._privacy_key, value.encode(), hashlib.sha256).hexdigest()

    def observe_transport_error(self, *, status: str, error_type: str) -> None:
        """Record requests rejected before a valid application payload exists."""
        self.metrics.record(status=status, route="unrouted", latency_ms=0.0)
        self.logger.log(
            request_id=None,
            conversation_hash=None,
            question_hash=None,
            route="unrouted",
            status=status,
            latency_ms=0.0,
            error_type=error_type,
        )

    def ask(self, payload: Any) -> ServiceResponse:
        started = self.clock()
        try:
            request = AskRequest.from_payload(payload)
        except RequestValidationError as exc:
            latency_ms = (self.clock() - started) * 1000
            self.metrics.record(
                status="validation_error",
                route="unrouted",
                latency_ms=latency_ms,
            )
            self.logger.log(
                request_id=None,
                conversation_hash=None,
                question_hash=None,
                route="unrouted",
                status="validation_error",
                latency_ms=latency_ms,
                error_type="RequestValidationError",
            )
            return ServiceResponse(
                422,
                {
                    "error": {
                        "code": "invalid_request",
                        "message": "The request body is invalid.",
                        "details": exc.errors,
                    }
                },
            )

        question_hash = self._privacy_hash(request.question)
        conversation_hash = self._privacy_hash(request.conversation_id)
        try:
            result = self.workflow.ask(
                request.question,
                conversation_id=request.conversation_id,
            )
        except Exception as exc:  # noqa: BLE001 - final service boundary must fail safely
            latency_ms = (self.clock() - started) * 1000
            self.metrics.record(
                status="internal_error",
                route="unrouted",
                latency_ms=latency_ms,
                error=True,
            )
            self.logger.log(
                request_id=None,
                conversation_hash=conversation_hash,
                question_hash=question_hash,
                route="unrouted",
                status="internal_error",
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
            )
            return ServiceResponse(
                500,
                {
                    "error": {
                        "code": "internal_error",
                        "message": "The service could not safely complete the request.",
                    }
                },
            )

        latency_ms = (self.clock() - started) * 1000
        guardrail_category = result.input_guardrail.category if result.status == "blocked" else None
        self.metrics.record(
            status=result.status,
            route=result.route,
            latency_ms=latency_ms,
            guardrail_category=guardrail_category,
        )
        self.logger.log(
            request_id=result.request_id,
            conversation_hash=conversation_hash,
            question_hash=question_hash,
            route=result.route,
            status=result.status,
            latency_ms=latency_ms,
        )
        return ServiceResponse(
            200,
            {
                "data": result.to_dict(),
                "meta": {
                    "service": self.name,
                    "version": self.version,
                    "latency_ms": round(latency_ms, 3),
                },
            },
        )

    def health(self) -> ServiceResponse:
        return ServiceResponse(
            200,
            {
                "status": "ready",
                "service": self.name,
                "version": self.version,
                "workflow_loaded": True,
                "knowledge_answer_backend": getattr(
                    self.workflow,
                    "knowledge_backend",
                    "custom",
                ),
            },
        )

    def metric_snapshot(self) -> ServiceResponse:
        return ServiceResponse(200, self.metrics.snapshot())


def build_service(
    project_root: Path,
    config: ApiConfig,
    *,
    logger: JsonRequestLogger | None = None,
    metrics: ServiceMetrics | None = None,
) -> CopilotService:
    workflow_config = AgentWorkflowConfig.from_json(
        resolve_project_path(project_root, config.workflow_config_path)
    )
    observability_config = ObservabilityConfig.from_json(
        resolve_project_path(project_root, config.observability_config_path)
    )
    workflow = load_agent_workflow(project_root, workflow_config, use_environment=True)
    active_logger = logger or JsonRequestLogger(
        console_and_file_stream(
            resolve_project_path(project_root, observability_config.request_log_path)
        )
    )
    active_metrics = metrics or ServiceMetrics(
        resolve_project_path(project_root, observability_config.metrics_snapshot_path)
    )
    return CopilotService(
        workflow,
        name=config.service_name,
        version=config.service_version,
        logger=active_logger,
        metrics=active_metrics,
    )

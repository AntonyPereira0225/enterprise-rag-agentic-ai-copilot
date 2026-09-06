from __future__ import annotations

import json
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Protocol

from enterprise_copilot.agents.analytics_agent import AnalyticsAgent
from enterprise_copilot.agents.router import IntentRouter
from enterprise_copilot.agents.specialists import build_knowledge_agents
from enterprise_copilot.agents.state import (
    AgentExecution,
    AgentRoute,
    WorkflowEvent,
    WorkflowResult,
)
from enterprise_copilot.guardrails.input import InputGuardrail
from enterprise_copilot.guardrails.output import OutputGuardrail
from enterprise_copilot.ingestion.loaders import load_jsonl
from enterprise_copilot.llm.pipeline import (
    GroundedAnswerConfig,
    load_grounded_answer_pipeline,
)


class Agent(Protocol):
    def execute(self, question: str) -> AgentExecution: ...


@dataclass(frozen=True)
class AgentWorkflowConfig:
    grounded_answer_config_path: str
    support_cases_path: str
    safety_evaluation_path: str
    hard_evaluation_path: str
    results_path: str
    max_input_characters: int

    @classmethod
    def from_json(cls, path: Path) -> AgentWorkflowConfig:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


@dataclass
class _Conversation:
    turn_count: int = 0
    previous_route: AgentRoute | None = None


class AgentWorkflow:
    """Explicit state machine: input guardrail → router → agent → output guardrail."""

    def __init__(
        self,
        agents: dict[AgentRoute, Agent],
        *,
        input_guardrail: InputGuardrail,
        output_guardrail: OutputGuardrail,
        router: IntentRouter,
        knowledge_backend: str = "custom",
    ) -> None:
        self.agents = agents
        self.input_guardrail = input_guardrail
        self.output_guardrail = output_guardrail
        self.router = router
        self.knowledge_backend = knowledge_backend
        self._conversations: dict[str, _Conversation] = {}
        self._conversation_locks: dict[str, Lock] = {}
        self._conversation_locks_guard = Lock()

    def ask(self, question: str, *, conversation_id: str = "default") -> WorkflowResult:
        with self._conversation_locks_guard:
            conversation_lock = self._conversation_locks.setdefault(conversation_id, Lock())
        with conversation_lock:
            return self._ask_serialized(question, conversation_id=conversation_id)

    def _ask_serialized(self, question: str, *, conversation_id: str) -> WorkflowResult:
        conversation = self._conversations.setdefault(conversation_id, _Conversation())
        conversation.turn_count += 1
        request_id = secrets.token_hex(8)
        trace: list[WorkflowEvent] = []

        input_decision = self.input_guardrail.check(question)
        trace.append(
            WorkflowEvent("input_guardrail", input_decision.action, input_decision.category)
        )
        if not input_decision.allowed:
            return WorkflowResult(
                request_id=request_id,
                conversation_id=conversation_id,
                turn_number=conversation.turn_count,
                route="blocked",
                status="blocked",
                answer=input_decision.message,
                confidence=1.0,
                citations=(),
                input_guardrail=input_decision,
                output_guardrail=None,
                trace=tuple(trace),
            )

        route = self.router.route(question, previous_route=conversation.previous_route)
        trace.append(WorkflowEvent("router", route.route, route.reason))
        agent = self.agents[route.route]
        execution = agent.execute(question)
        trace.append(WorkflowEvent("agent", execution.status, f"Executed {route.route} workflow."))

        output_decision = self.output_guardrail.check(execution)
        trace.append(
            WorkflowEvent("output_guardrail", output_decision.action, output_decision.category)
        )
        if not output_decision.allowed:
            return WorkflowResult(
                request_id=request_id,
                conversation_id=conversation_id,
                turn_number=conversation.turn_count,
                route=route.route,
                status="failed_safe",
                answer="I cannot return this response because evidence verification failed.",
                confidence=0.0,
                citations=(),
                input_guardrail=input_decision,
                output_guardrail=output_decision,
                trace=tuple(trace),
            )

        conversation.previous_route = route.route
        return WorkflowResult(
            request_id=request_id,
            conversation_id=conversation_id,
            turn_number=conversation.turn_count,
            route=route.route,
            status="completed" if execution.status == "answered" else "refused",
            answer=execution.answer,
            confidence=execution.confidence,
            citations=execution.citations,
            input_guardrail=input_decision,
            output_guardrail=output_decision,
            trace=tuple(trace),
            details=execution.details,
        )


def load_agent_workflow(
    project_root: Path,
    config: AgentWorkflowConfig,
    *,
    use_environment: bool = False,
    environment: Mapping[str, str] | None = None,
) -> AgentWorkflow:
    grounded_config = GroundedAnswerConfig.from_json(
        project_root / config.grounded_answer_config_path
    )
    pipeline = load_grounded_answer_pipeline(
        project_root,
        grounded_config,
        use_environment=use_environment,
        environment=environment,
    )
    agents: dict[AgentRoute, Agent] = build_knowledge_agents(pipeline)
    agents["analytics"] = AnalyticsAgent(
        load_jsonl(project_root / config.support_cases_path),
        source_uri=config.support_cases_path,
    )
    return AgentWorkflow(
        agents,
        input_guardrail=InputGuardrail(config.max_input_characters),
        output_guardrail=OutputGuardrail(),
        router=IntentRouter(),
        knowledge_backend=pipeline.generator.backend_name,
    )

from pathlib import Path

from enterprise_copilot.agents.graph import AgentWorkflowConfig
from enterprise_copilot.evaluation.security_cases import (
    generate_agent_guardrail_cases,
    write_agent_guardrail_cases,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = AgentWorkflowConfig.from_json(project_root / "configs" / "agent_workflow_config.json")
    cases = generate_agent_guardrail_cases()
    write_agent_guardrail_cases(cases, project_root / config.safety_evaluation_path)
    adversarial = sum(case["case_type"] == "adversarial" for case in cases)

    print("Agent guardrail set generated successfully")
    print(f"Cases written: {len(cases)}")
    print(f"Safe workflow cases: {len(cases) - adversarial}")
    print(f"Adversarial cases: {adversarial}")
    print(f"Output file: {config.safety_evaluation_path}")


if __name__ == "__main__":
    main()

from pathlib import Path

from enterprise_copilot.agents.graph import AgentWorkflowConfig
from enterprise_copilot.evaluation.agent_workflow import run_agent_workflow_evaluation


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = AgentWorkflowConfig.from_json(project_root / "configs" / "agent_workflow_config.json")
    results = run_agent_workflow_evaluation(project_root, config)
    summary = results["summary"]

    print("Agent workflow evaluation completed successfully")
    print(f"Safety pass rate: {summary['safety_pass_rate']:.1%}")
    print(f"Adversarial block rate: {summary['adversarial_block_rate']:.1%}")
    print(f"Knowledge status accuracy: {summary['knowledge_status_accuracy']:.1%}")
    print(f"Knowledge evidence coverage: {summary['knowledge_evidence_coverage']:.1%}")
    print(f"Detailed results: {config.results_path}")


if __name__ == "__main__":
    main()

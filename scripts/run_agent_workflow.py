import argparse
import json
from pathlib import Path

from enterprise_copilot.agents.graph import AgentWorkflowConfig, load_agent_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the controlled agent workflow")
    parser.add_argument("question", help="Business knowledge or support-analytics question")
    parser.add_argument("--conversation-id", default="demo", help="Conversation state key")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config = AgentWorkflowConfig.from_json(project_root / "configs" / "agent_workflow_config.json")
    workflow = load_agent_workflow(project_root, config, use_environment=True)
    result = workflow.ask(args.question, conversation_id=args.conversation_id)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

from enterprise_copilot.llm.pipeline import (
    GroundedAnswerConfig,
    load_grounded_answer_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the configured grounded-answer pipeline")
    parser.add_argument("question", help="Question to answer from the approved knowledge base")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config = GroundedAnswerConfig.from_json(
        project_root / "configs" / "grounded_answer_config.json"
    )
    pipeline = load_grounded_answer_pipeline(project_root, config, use_environment=True)
    run = pipeline.ask(args.question)
    print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from pathlib import Path

from enterprise_copilot.evaluation.hard_questions import (
    HardEvaluationConfig,
    generate_hard_questions,
    write_hard_questions,
)
from enterprise_copilot.ingestion.loaders import load_jsonl


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = HardEvaluationConfig.from_json(
        project_root / "configs" / "hard_evaluation_config.json"
    )
    documents = load_jsonl(project_root / config.source_path)
    questions = generate_hard_questions(documents, config)
    write_hard_questions(questions, project_root / config.output_path)

    difficulty_counts: dict[str, int] = {}
    for question in questions:
        difficulty = question["difficulty"]
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

    print("Hard evaluation set generated successfully")
    print(f"Questions written: {len(questions)}")
    for difficulty, count in sorted(difficulty_counts.items()):
        print(f"{difficulty}: {count}")
    print(f"Output file: {config.output_path}")


if __name__ == "__main__":
    main()

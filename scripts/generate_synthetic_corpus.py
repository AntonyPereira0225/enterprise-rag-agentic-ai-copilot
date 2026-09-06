from pathlib import Path

from enterprise_copilot.ingestion.synthetic_corpus import (
    CorpusConfig,
    generate_corpus,
    write_corpus,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = CorpusConfig.from_json(project_root / "configs" / "corpus_config.json")
    result = generate_corpus(config)
    write_corpus(result, project_root / "data")

    print("Synthetic corpus generated successfully")
    print(f"Knowledge documents: {len(result.knowledge_documents)}")
    print(f"Support cases: {len(result.support_cases)}")
    print(f"Evaluation questions: {len(result.evaluation_questions)}")


if __name__ == "__main__":
    main()

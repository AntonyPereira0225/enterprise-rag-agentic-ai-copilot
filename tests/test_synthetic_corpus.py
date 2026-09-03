from pathlib import Path

from enterprise_copilot.ingestion.synthetic_corpus import CorpusConfig, generate_corpus


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "corpus_config.json"


def test_generated_counts_match_configuration() -> None:
    config = CorpusConfig.from_json(CONFIG_PATH)
    result = generate_corpus(config)

    expected_documents = sum(config.document_counts.values())
    assert len(result.knowledge_documents) == expected_documents
    assert len(result.support_cases) == config.support_case_count
    assert len(result.evaluation_questions) == config.evaluation_question_count


def test_document_ids_are_unique() -> None:
    config = CorpusConfig.from_json(CONFIG_PATH)
    result = generate_corpus(config)
    document_ids = [row["document_id"] for row in result.knowledge_documents]

    assert len(document_ids) == len(set(document_ids))


def test_evaluation_references_existing_documents() -> None:
    config = CorpusConfig.from_json(CONFIG_PATH)
    result = generate_corpus(config)
    document_ids = {row["document_id"] for row in result.knowledge_documents}

    for question in result.evaluation_questions:
        assert set(question["expected_document_ids"]).issubset(document_ids)


def test_generation_is_deterministic() -> None:
    config = CorpusConfig.from_json(CONFIG_PATH)

    first = generate_corpus(config)
    second = generate_corpus(config)

    assert first.knowledge_documents == second.knowledge_documents
    assert first.support_cases == second.support_cases
    assert first.evaluation_questions == second.evaluation_questions

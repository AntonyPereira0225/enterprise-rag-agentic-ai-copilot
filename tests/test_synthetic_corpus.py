from pathlib import Path

from enterprise_copilot.ingestion.synthetic_corpus import (
    ISSUE_TYPES,
    CorpusConfig,
    generate_corpus,
)

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


def test_linked_evidence_contains_expected_answers() -> None:
    config = CorpusConfig.from_json(CONFIG_PATH)
    result = generate_corpus(config)
    documents = {row["document_id"]: row for row in result.knowledge_documents}

    for question in result.evaluation_questions:
        evidence = " ".join(
            documents[document_id]["content"] for document_id in question["expected_document_ids"]
        )
        assert question["expected_answer"] in evidence


def test_support_cases_reference_existing_documents() -> None:
    config = CorpusConfig.from_json(CONFIG_PATH)
    result = generate_corpus(config)
    document_ids = {row["document_id"] for row in result.knowledge_documents}

    for support_case in result.support_cases:
        assert set(support_case["linked_document_ids"]).issubset(document_ids)


def test_document_content_matches_department_metadata() -> None:
    config = CorpusConfig.from_json(CONFIG_PATH)
    result = generate_corpus(config)

    for document in result.knowledge_documents:
        expected_sentence = f"The owning function is {document['department']}."
        assert expected_sentence in document["content"]


def test_document_content_matches_issue_type_tag() -> None:
    config = CorpusConfig.from_json(CONFIG_PATH)
    result = generate_corpus(config)

    for document in result.knowledge_documents:
        issue_tags = set(document["tags"]).intersection(ISSUE_TYPES)
        assert len(issue_tags) == 1
        issue = issue_tags.pop().replace("_", " ")
        assert f"For the primary {issue} scenario" in document["content"]


def test_evaluation_questions_are_unique_and_name_the_document_family() -> None:
    config = CorpusConfig.from_json(CONFIG_PATH)
    result = generate_corpus(config)
    questions = [row["question"] for row in result.evaluation_questions]

    assert len(questions) == len(set(questions))
    for row in result.evaluation_questions:
        document_family = row["intent"].replace("_", " ")
        assert f"According to the {document_family}" in row["question"]


def test_generation_is_deterministic() -> None:
    config = CorpusConfig.from_json(CONFIG_PATH)

    first = generate_corpus(config)
    second = generate_corpus(config)

    assert first.knowledge_documents == second.knowledge_documents
    assert first.support_cases == second.support_cases
    assert first.evaluation_questions == second.evaluation_questions

import json
from pathlib import Path

import pytest

from enterprise_copilot.ingestion.chunking import ChunkingConfig, chunk_document
from enterprise_copilot.ingestion.cleaning import clean_and_deduplicate, clean_text
from enterprise_copilot.ingestion.loaders import DocumentLoadError, load_jsonl, validate_documents
from enterprise_copilot.ingestion.pipeline import IngestionConfig, run_ingestion
from enterprise_copilot.ingestion.synthetic_corpus import (
    CorpusConfig,
    generate_corpus,
    write_corpus,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _document(document_id: str = "DOC-001", content: str = "one two three four five") -> dict:
    return {
        "document_id": document_id,
        "document_type": "policy",
        "title": "Test policy",
        "department": "Operations",
        "region": "Ireland",
        "product": "NSG Connect",
        "version": "1.0",
        "effective_date": "2026-01-01",
        "status": "active",
        "sensitivity": "internal",
        "tags": ["policy"],
        "source_uri": f"nsg://test/{document_id}",
        "content": content,
        "content_sha256": "test-hash",
    }


def test_clean_text_normalises_unicode_and_whitespace() -> None:
    assert clean_text("  Full-width：  value\r\n\r\n\r\n next\tline  ") == (
        "Full-width: value\n\nnext line"
    )


def test_chunk_document_preserves_overlap_and_lineage() -> None:
    chunks = chunk_document(_document(), ChunkingConfig(size_words=3, overlap_words=1))

    assert [chunk["content"] for chunk in chunks] == [
        "one two three",
        "three four five",
    ]
    assert chunks[0]["chunk_id"] == "DOC-001::chunk-0000"
    assert chunks[1]["word_start"] == 2
    assert chunks[1]["document_id"] == "DOC-001"
    assert chunks[1]["source_uri"] == "nsg://test/DOC-001"


def test_chunking_config_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="smaller than chunk size"):
        ChunkingConfig(size_words=10, overlap_words=10)


def test_loader_reports_bad_json_line(tmp_path: Path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text('{"valid": true}\nnot-json\n', encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="line 2"):
        load_jsonl(source)


def test_validation_rejects_duplicate_document_ids() -> None:
    first = _document()
    second = _document()

    with pytest.raises(DocumentLoadError, match="Duplicate document_id"):
        validate_documents([first, second])


def test_cleaning_removes_duplicate_content() -> None:
    first = _document("DOC-001", "Same   content")
    second = _document("DOC-002", "Same content")

    documents, duplicates = clean_and_deduplicate([first, second])

    assert [document["document_id"] for document in documents] == ["DOC-001"]
    assert duplicates == ["DOC-002"]


def test_ingestion_pipeline_creates_traceable_chunks(tmp_path: Path) -> None:
    corpus_config = CorpusConfig.from_json(PROJECT_ROOT / "configs" / "corpus_config.json")
    corpus = generate_corpus(corpus_config)
    write_corpus(corpus, tmp_path / "data")
    config = IngestionConfig(
        source_path="data/raw/knowledge_base.jsonl",
        chunks_path="data/processed/knowledge_chunks.jsonl",
        manifest_path="data/processed/ingestion_manifest.json",
        chunk_size_words=60,
        chunk_overlap_words=10,
        included_statuses=["active"],
    )

    first_result = run_ingestion(tmp_path, config)
    first_bytes = first_result.chunks_path.read_bytes()
    second_result = run_ingestion(tmp_path, config)
    chunks = [
        json.loads(line)
        for line in second_result.chunks_path.read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads(second_result.manifest_path.read_text(encoding="utf-8"))

    assert first_result.included_document_count == 72
    assert second_result.duplicate_content_count == 0
    assert second_result.chunks_path.read_bytes() == first_bytes
    assert {chunk["document_id"] for chunk in chunks} == {
        document["document_id"] for document in corpus.knowledge_documents
    }
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    assert all(0 < chunk["word_count"] <= 60 for chunk in chunks)
    assert all(chunk["source_uri"].startswith("nsg://knowledge/") for chunk in chunks)
    assert manifest["chunk_count"] == len(chunks)
    assert manifest["source_document_count"] == 72

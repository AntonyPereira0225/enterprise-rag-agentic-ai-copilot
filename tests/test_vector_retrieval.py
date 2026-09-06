import json
from pathlib import Path

from enterprise_copilot.embeddings.tfidf import TfidfEmbeddingModel
from enterprise_copilot.ingestion.pipeline import IngestionConfig, run_ingestion
from enterprise_copilot.ingestion.synthetic_corpus import (
    CorpusConfig,
    generate_corpus,
    write_corpus,
)
from enterprise_copilot.retrieval.pipeline import (
    RetrievalConfig,
    build_retrieval_index,
    run_retrieval_evaluation,
)
from enterprise_copilot.retrieval.vector_index import VectorIndex

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _chunk(chunk_id: str, document_id: str, content: str, region: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "document_type": "policy",
        "title": f"{region} refund policy",
        "department": "Customer Support",
        "region": region,
        "product": "NSG Connect",
        "status": "active",
        "tags": ["refund", region.casefold()],
        "source_uri": f"nsg://knowledge/policy/{document_id}",
        "content": content,
    }


def test_vector_index_search_save_load_and_filter(tmp_path: Path) -> None:
    chunks = [
        _chunk("C-001", "D-001", "Refunds take seven calendar days.", "Ireland"),
        _chunk("C-002", "D-002", "Service activation takes two hours.", "Germany"),
    ]
    index = VectorIndex.build(chunks, TfidfEmbeddingModel())
    path = tmp_path / "index.json"
    index.save(path)
    restored = VectorIndex.load(path)

    results = restored.search("Ireland refund calendar days", top_k=1)
    filtered = restored.search("service", top_k=2, filters={"region": "Germany"})

    assert results[0].chunk["document_id"] == "D-001"
    assert filtered[0].chunk["document_id"] == "D-002"
    assert restored.model.dimension == index.model.dimension


def test_full_retrieval_pipeline_is_deterministic(tmp_path: Path) -> None:
    corpus_config = CorpusConfig.from_json(PROJECT_ROOT / "configs" / "corpus_config.json")
    write_corpus(generate_corpus(corpus_config), tmp_path / "data")
    ingestion_config = IngestionConfig(
        source_path="data/raw/knowledge_base.jsonl",
        chunks_path="data/processed/knowledge_chunks.jsonl",
        manifest_path="data/processed/ingestion_manifest.json",
        chunk_size_words=60,
        chunk_overlap_words=10,
        included_statuses=["active"],
    )
    run_ingestion(tmp_path, ingestion_config)
    retrieval_config = RetrievalConfig(
        chunks_path="data/processed/knowledge_chunks.jsonl",
        evaluation_path="data/evaluation/rag_eval_questions.jsonl",
        index_path="data/vector_store/tfidf_index.json",
        results_path="data/evaluation/retrieval_metrics.json",
        embedding={
            "type": "tfidf",
            "ngram_min": 1,
            "ngram_max": 2,
            "min_document_frequency": 1,
        },
        top_k_values=[1, 3, 5],
    )

    build_result = build_retrieval_index(tmp_path, retrieval_config)
    first_index_bytes = build_result.index_path.read_bytes()
    first_results = run_retrieval_evaluation(tmp_path, retrieval_config)
    build_retrieval_index(tmp_path, retrieval_config)
    second_results = run_retrieval_evaluation(tmp_path, retrieval_config)

    assert build_result.chunk_count == 144
    assert build_result.embedding_dimension > 0
    assert build_result.index_path.read_bytes() == first_index_bytes
    assert second_results == first_results
    assert first_results["summary"]["evaluated_questions"] == 72
    assert first_results["summary"]["evidence_recall_at_k"]["5"] >= 0.9
    saved = json.loads((tmp_path / retrieval_config.results_path).read_text(encoding="utf-8"))
    assert saved == first_results

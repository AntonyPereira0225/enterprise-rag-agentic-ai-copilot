import json
import math
from pathlib import Path

from enterprise_copilot.embeddings.tfidf import TfidfEmbeddingModel
from enterprise_copilot.retrieval.bm25_index import BM25Index
from enterprise_copilot.retrieval.hybrid import HybridRetriever
from enterprise_copilot.retrieval.hybrid_pipeline import (
    HybridRetrievalConfig,
    build_bm25_index,
    run_hybrid_evaluation,
)
from enterprise_copilot.retrieval.reranking import MetadataAwareReranker, RerankerWeights
from enterprise_copilot.retrieval.tuning import (
    HybridTuningConfig,
    run_hybrid_tuning,
    stratified_calibration_split,
)
from enterprise_copilot.retrieval.vector_index import VectorIndex


def _chunk(chunk_id: str, document_id: str, content: str, region: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "document_type": "policy",
        "title": f"{region} policy",
        "department": "Customer Support",
        "region": region,
        "product": "NSG Connect",
        "status": "active",
        "tags": ["policy", region.casefold()],
        "source_uri": f"nsg://{chunk_id}",
        "content": content,
    }


def test_bm25_index_ranks_and_round_trips(tmp_path: Path) -> None:
    chunks = [
        _chunk("C1", "D1", "refund window seven calendar days", "Ireland"),
        _chunk("C2", "D2", "service activation two hours", "Germany"),
    ]
    index = BM25Index.build(chunks)
    path = tmp_path / "bm25.json"
    index.save(path)
    restored = BM25Index.load(path)

    assert restored.search("refund calendar", top_k=1)[0].chunk["document_id"] == "D1"
    assert restored.search("unknown vocabulary", top_k=1) == []
    assert restored.to_state() == index.to_state()
    assert index.inverse_document_frequency["refund"] == math.log(2)


def test_hybrid_retriever_fuses_provenance_and_diversifies_documents() -> None:
    chunks = [
        _chunk("C1", "D1", "refund window seven calendar days", "Ireland"),
        _chunk("C2", "D1", "refund escalation information", "Ireland"),
        _chunk("C3", "D2", "refund review procedure", "Germany"),
    ]
    vector = VectorIndex.build(chunks, TfidfEmbeddingModel())
    bm25 = BM25Index.build(chunks)
    retriever = HybridRetriever(
        vector,
        bm25,
        MetadataAwareReranker(RerankerWeights()),
        candidate_pool_size=3,
        max_chunks_per_document=1,
    )

    results = retriever.search("refund policy", top_k=3)

    assert len(results) == 2
    assert len({result.chunk["document_id"] for result in results}) == 2
    assert "vector_rank" in results[0].details
    assert "bm25_rank" in results[0].details
    assert "reranker" in results[0].details


def test_stratified_calibration_split_preserves_each_difficulty() -> None:
    questions = [
        {"question_id": f"{difficulty}-{index}", "difficulty": difficulty}
        for difficulty in ("paraphrased", "multi_source", "unanswerable")
        for index in range(1, 9)
    ]

    calibration, holdout = stratified_calibration_split(questions, stride=4)

    assert len(calibration) == 6
    assert len(holdout) == 18
    assert {row["difficulty"] for row in calibration} == {
        "paraphrased",
        "multi_source",
        "unanswerable",
    }


def test_hybrid_pipeline_and_tuning_write_reproducible_results(tmp_path: Path) -> None:
    chunks = [
        _chunk("C1", "D1", "refund window seven calendar days", "Ireland"),
        _chunk("C2", "D2", "service activation two hours", "Germany"),
    ]
    data = tmp_path / "data"
    data.mkdir()
    vector_path = data / "vector.json"
    bm25_path = data / "bm25.json"
    questions_path = data / "questions.jsonl"
    VectorIndex.build(chunks, TfidfEmbeddingModel()).save(vector_path)
    build_bm25_index(chunks, bm25_path, source_path="data/chunks.jsonl")

    questions = []
    for index in range(1, 5):
        questions.append(
            {
                "question_id": f"A-{index}",
                "question": "Ireland refund window",
                "expected_answer": "seven calendar days",
                "expected_document_ids": ["D1"],
                "answerable": True,
                "difficulty": "answerable",
            }
        )
        questions.append(
            {
                "question_id": f"U-{index}",
                "question": "satellite warranty France",
                "expected_answer": None,
                "expected_document_ids": [],
                "answerable": False,
                "difficulty": "unanswerable",
            }
        )
    questions_path.write_text(
        "".join(json.dumps(question) + "\n" for question in questions),
        encoding="utf-8",
    )
    config = HybridRetrievalConfig(
        evaluation_path="data/questions.jsonl",
        vector_index_path="data/vector.json",
        bm25_index_path="data/bm25.json",
        results_path="data/results.json",
        top_k_values=[1, 2],
        minimum_score=0.2,
        candidate_pool_size=2,
        rrf_constant=10,
        vector_weight=1.0,
        bm25_weight=1.0,
        max_chunks_per_document=1,
        reranker_weights={
            "vector_similarity": 0.9,
            "content_overlap": 0.0,
            "metadata_overlap": 0.0,
            "fused_rank": 0.1,
        },
    )
    first = run_hybrid_evaluation(tmp_path, config)
    second = run_hybrid_evaluation(tmp_path, config)
    tuning = HybridTuningConfig(
        calibration_stride=2,
        rrf_constants=[10],
        vector_weights=[1.0],
        bm25_weight=1.0,
        minimum_scores=[0.2],
        reranker_weight_profiles=[config.reranker_weights],
        output_path="data/tuning.json",
    )
    tuned = run_hybrid_tuning(tmp_path, config, tuning)

    assert first == second
    assert first["summary"]["evidence_recall_at_k"]["1"] == 1.0
    assert first["summary"]["answerability_accuracy"] == 1.0
    assert tuned["split"] == {
        "calibration_stride": 2,
        "calibration_questions": 4,
        "holdout_questions": 4,
    }
    assert (data / "results.json").is_file()
    assert (data / "tuning.json").is_file()

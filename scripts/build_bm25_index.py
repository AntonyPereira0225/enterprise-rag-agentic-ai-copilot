from pathlib import Path

from enterprise_copilot.ingestion.loaders import load_jsonl
from enterprise_copilot.retrieval.hybrid_pipeline import (
    HybridRetrievalConfig,
    build_bm25_index,
)
from enterprise_copilot.retrieval.pipeline import RetrievalConfig


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    retrieval_config = RetrievalConfig.from_json(project_root / "configs" / "retrieval_config.json")
    hybrid_config = HybridRetrievalConfig.from_json(
        project_root / "configs" / "hybrid_retrieval_config.json"
    )
    chunks = load_jsonl(project_root / retrieval_config.chunks_path)
    index_path = project_root / hybrid_config.bm25_index_path
    index = build_bm25_index(chunks, index_path, source_path=retrieval_config.chunks_path)

    print("BM25 index built successfully")
    print(f"Chunks indexed: {len(index.records)}")
    print(f"Vocabulary terms: {len(index.inverse_document_frequency)}")
    print(f"Index file: {index_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()

from pathlib import Path

from enterprise_copilot.embeddings.sentence_transformer import DenseDependencyError
from enterprise_copilot.retrieval.dense_pipeline import build_dense_retrieval_index
from enterprise_copilot.retrieval.pipeline import RetrievalConfig


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = RetrievalConfig.from_json(project_root / "configs" / "dense_retrieval_config.json")
    result = build_dense_retrieval_index(project_root, config)
    print("Dense FAISS index built successfully")
    print(f"Chunks indexed: {result.chunk_count}")
    print(f"Embedding dimensions: {result.embedding_dimension}")
    print(f"Index file: {result.index_path.relative_to(project_root)}")


if __name__ == "__main__":
    try:
        main()
    except DenseDependencyError as exc:
        raise SystemExit(str(exc)) from exc

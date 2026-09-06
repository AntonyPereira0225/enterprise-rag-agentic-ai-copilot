from pathlib import Path

from enterprise_copilot.retrieval.pipeline import RetrievalConfig, build_retrieval_index


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = RetrievalConfig.from_json(project_root / "configs" / "retrieval_config.json")
    result = build_retrieval_index(project_root, config)

    print("Retrieval index built successfully")
    print(f"Chunks indexed: {result.chunk_count}")
    print(f"Embedding dimensions: {result.embedding_dimension}")
    print(f"Index file: {result.index_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()

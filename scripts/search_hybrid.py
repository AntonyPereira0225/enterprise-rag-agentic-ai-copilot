import argparse
from pathlib import Path

from enterprise_copilot.retrieval.hybrid_pipeline import (
    HybridRetrievalConfig,
    load_hybrid_retriever,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the hybrid Northstar index")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    arguments = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config = HybridRetrievalConfig.from_json(
        project_root / "configs" / "hybrid_retrieval_config.json"
    )
    retriever = load_hybrid_retriever(project_root, config)
    results = retriever.search(arguments.query, top_k=arguments.top_k)

    for rank, result in enumerate(results, start=1):
        print(f"\n{rank}. {result.chunk['title']} (score={result.score:.3f})")
        print(f"   Source: {result.chunk['source_uri']}")
        print(f"   {result.chunk['content']}")


if __name__ == "__main__":
    main()

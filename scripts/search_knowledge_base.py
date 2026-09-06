import argparse
from pathlib import Path

from enterprise_copilot.retrieval.pipeline import RetrievalConfig
from enterprise_copilot.retrieval.vector_index import VectorIndex


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the local Northstar knowledge index")
    parser.add_argument("query", help="Natural-language knowledge question")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return")
    parser.add_argument("--region", help="Optional exact region filter")
    parser.add_argument("--product", help="Optional exact product filter")
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    project_root = Path(__file__).resolve().parents[1]
    config = RetrievalConfig.from_json(project_root / "configs" / "retrieval_config.json")
    index = VectorIndex.load(project_root / config.index_path)
    filters = {
        field: value
        for field, value in {"region": arguments.region, "product": arguments.product}.items()
        if value is not None
    }
    results = index.search(arguments.query, top_k=arguments.top_k, filters=filters)

    if not results:
        print("No matching chunks were found.")
        return

    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        print(f"\n{rank}. {chunk['title']} (score={result.score:.3f})")
        print(f"   Chunk: {chunk['chunk_id']}")
        print(f"   Source: {chunk['source_uri']}")
        print(f"   {chunk['content']}")


if __name__ == "__main__":
    main()

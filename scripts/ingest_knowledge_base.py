from pathlib import Path

from enterprise_copilot.ingestion.pipeline import IngestionConfig, run_ingestion


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "ingestion_config.json"
    config = IngestionConfig.from_json(config_path)
    result = run_ingestion(project_root, config)

    print("Knowledge ingestion completed successfully")
    print(f"Source documents read: {result.source_document_count}")
    print(f"Documents included: {result.included_document_count}")
    print(f"Documents excluded by status: {result.excluded_status_count}")
    print(f"Duplicate documents removed: {result.duplicate_content_count}")
    print(f"Chunks written: {result.chunk_count}")
    print(f"Chunks file: {result.chunks_path.relative_to(project_root)}")
    print(f"Manifest file: {result.manifest_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()

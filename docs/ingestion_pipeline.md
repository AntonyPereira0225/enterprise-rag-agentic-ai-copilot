# Ingestion and Chunking Pipeline

## What this stage does

The ingestion pipeline turns the generated knowledge-base file into smaller, traceable passages that a retrieval system can index later.

```text
knowledge_base.jsonl
        ↓
load and validate each document
        ↓
keep approved statuses
        ↓
normalise text and remove exact duplicates
        ↓
split text into overlapping chunks
        ↓
knowledge_chunks.jsonl + ingestion_manifest.json
```

The first version intentionally uses transparent word-based chunking. A 60-word chunk with a 10-word overlap is easy to inspect and creates a dependable baseline before comparing tokenizer-aware or semantic chunking strategies.

## Files

- `configs/ingestion_config.json` controls input/output locations, chunk size, overlap, and included document statuses.
- `src/enterprise_copilot/ingestion/loaders.py` loads JSON Lines and validates the minimum source schema.
- `src/enterprise_copilot/ingestion/cleaning.py` normalises text and detects exact duplicate content.
- `src/enterprise_copilot/ingestion/chunking.py` creates overlapping chunks and retains source metadata.
- `src/enterprise_copilot/ingestion/pipeline.py` connects all stages and writes deterministic outputs.
- `scripts/ingest_knowledge_base.py` is the command-line entry point.

## Run it locally

From the repository root, create a small project environment and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install pytest pytest-cov ruff
$env:PYTHONPATH = "src"
python scripts/generate_synthetic_corpus.py
python scripts/ingest_knowledge_base.py
python -m pytest -q
```

The generator and ingestion code use only Python's standard library. The small packages installed above provide testing, coverage, and code-quality checks without downloading the later ML model stack.

The generated outputs are intentionally ignored by Git:

```text
data/processed/knowledge_chunks.jsonl
data/processed/ingestion_manifest.json
```

## Chunk record design

Each chunk includes:

- a deterministic `chunk_id`;
- the parent `document_id` and source URI;
- the chunk number and word offsets;
- the cleaned source-content hash;
- the original document metadata used for filtering and citations;
- the chunk text that will be embedded and indexed.

The manifest records input/output hashes, counts, excluded statuses, duplicate content, and chunking settings. This makes an ingestion run reproducible and auditable.

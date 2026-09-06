# Vector Retrieval Baseline

## Purpose

This phase establishes a measurable local retrieval baseline before adding a neural embedding model. It converts the 144 knowledge chunks into vectors, persists a local index, searches by cosine similarity, and evaluates every one of the 72 labelled questions.

The baseline uses word and two-word phrase TF-IDF embeddings. This is a statistical vector representation, not a neural semantic model. It is intentionally dependency-free so the complete pipeline can run locally before downloading PyTorch, Sentence Transformers, or a model checkpoint. The embedding interface is replaceable, allowing a future neural model to be compared against exactly the same questions and metrics.

## Data flow

```text
knowledge_chunks.jsonl
        ↓
combine searchable metadata + chunk text
        ↓
fit deterministic TF-IDF embeddings
        ↓
persist sparse vectors + vocabulary + IDF values
        ↓
cosine-similarity search
        ↓
evaluate 72 labelled questions at K = 1, 3, 5
```

## Run the phase

Run these commands from the repository root after completing ingestion:

```powershell
$env:PYTHONPATH = "src"
python scripts/build_retrieval_index.py
python scripts/evaluate_retrieval.py
python -m pytest -q
```

Try a search after building the index:

```powershell
python scripts/search_knowledge_base.py "What is the compliance requirement for NSG Connect in Ireland?" --top-k 3
```

Optional `--region` and `--product` arguments demonstrate exact metadata filtering.

Generated local files:

```text
data/vector_store/tfidf_index.json
data/evaluation/retrieval_metrics.json
```

Both outputs remain local and are ignored by Git. The vector index records the source-chunk file hash so its lineage can be checked.

## Metrics

- **Document Recall@K**: whether any of the first K chunks comes from the labelled source document.
- **Evidence Recall@K**: whether any of the first K chunks comes from the labelled source document and contains the expected answer.
- **Evidence MRR**: the average reciprocal rank of the first answer-bearing source chunk.

Evidence Recall is the stronger measure for RAG. Retrieving the right document but the wrong passage does not give an LLM enough evidence to produce a grounded answer.

## Baseline result and interpretation

The initial corpus achieved 100% document and evidence Recall@1/3/5, with an evidence MRR of 1.0. The evaluation questions deliberately contain strong product, region, and document-family signals, so this result proves that the pipeline, metadata enrichment, citations, and metrics are wired correctly. It does not prove broad semantic understanding.

The later hard-evaluation workstream adds paraphrases, multi-source questions, distractors, unanswerable questions, and questions that do not repeat source terminology. See `hard_and_dense_retrieval.md` and `hybrid_retrieval.md` for the resulting 96-question benchmark and comparison.

## Next experiment

The next retrieval experiment should add a small Sentence Transformers model, store dense vectors in FAISS, and run the same evaluation without changing the labelled dataset. The statistical baseline then provides an honest comparison rather than assuming the larger model is automatically better.

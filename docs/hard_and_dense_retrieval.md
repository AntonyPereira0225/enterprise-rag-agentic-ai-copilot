# Hard Evaluation and Dense Retrieval

## Why add a harder evaluation set?

The first 72 questions deliberately repeat strong source terminology. That makes them useful for checking pipeline wiring, but the TF-IDF baseline reaches 100% Recall@1 and leaves no room for a meaningful model comparison.

The harder deterministic set contains 96 questions:

| Question type | Count | Purpose |
|---|---:|---|
| Paraphrased single-source | 72 | Uses alternative wording instead of copying the source vocabulary. |
| Multi-source | 12 | Requires evidence from two different documents. |
| Unanswerable | 12 | Tests whether a score threshold can reject unsupported knowledge. |

Generate and evaluate it with the working TF-IDF index:

```powershell
$env:PYTHONPATH = "src"
python scripts/generate_hard_evaluation_set.py
python scripts/evaluate_hard_retrieval.py
```

## TF-IDF hard-set result

| Metric | @1 | @3 | @5 | @10 |
|---|---:|---:|---:|---:|
| Evidence Recall | 38.1% | 81.0% | 100% | 100% |
| All evidence recalled | 35.7% | 76.2% | 100% | 100% |

- Evidence MRR: 0.627
- Initial answerability accuracy at an untuned 0.15 threshold: 91.7%
- Initial unanswerable abstention accuracy: 33.3%
- Calibrated 0.25 threshold answerability and abstention accuracy: 100%

The lower rank-1 result exposes the limits hidden by the easy evaluation set. The initial weak abstention result also demonstrated why a similarity threshold must be calibrated rather than assumed; the later calibration stage corrected it without changing retrieval rankings.

## Dense retrieval design

The optional dense pipeline uses:

- `sentence-transformers/all-MiniLM-L6-v2` for 384-dimensional neural embeddings;
- L2-normalised vectors;
- FAISS `IndexFlatIP`, where inner product equals cosine similarity for normalised vectors;
- the same chunk metadata, filters, evaluation set, and metrics as the TF-IDF baseline.

The adapters import their heavy dependencies only when dense retrieval is run, so corpus generation, ingestion, tests, and the TF-IDF baseline remain lightweight.

## Install and run dense retrieval

The following stage requires internet access the first time, both for Python packages and the MiniLM model files:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dense.txt
$env:PYTHONPATH = "src"
python scripts/build_dense_retrieval_index.py
python scripts/evaluate_dense_retrieval.py
```

Generated outputs:

```text
data/vector_store/minilm_l6_v2.faiss
data/vector_store/minilm_l6_v2.faiss.metadata.json
data/evaluation/dense_retrieval_metrics.json
```

## Current validation status

The Sentence Transformers adapter, dense-vector conversion, FAISS build/search/filter logic, persistence, and dense pipeline orchestration are covered by automated in-memory tests. The real MiniLM model and native FAISS binary could not be downloaded in the restricted workspace, so no live dense metric is claimed yet.

Once dependencies are available, the dense commands will run the same 96-question benchmark. Compare Evidence Recall@1/3/5/10, MRR, answerability accuracy, runtime, and index size before choosing the default retriever.

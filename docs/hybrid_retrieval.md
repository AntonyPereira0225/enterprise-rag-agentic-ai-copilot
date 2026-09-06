# Hybrid Retrieval and Reranking

## Purpose

Hybrid retrieval combines different ways of finding evidence instead of relying on one score:

```text
User question
     ├── TF-IDF vector retrieval
     └── BM25 keyword retrieval
               ↓
        Reciprocal Rank Fusion
               ↓
       transparent reranking
               ↓
 one chunk per document for diversity
               ↓
       top evidence passages
```

BM25 is implemented with Python's standard library. Reciprocal Rank Fusion (RRF) combines the two ranked lists without assuming their raw scores use the same scale. The reranker uses vector similarity plus a small fused-rank contribution. Duplicate-document control prevents overlapping chunks from one source from occupying the entire context window.

## Run the phase

Complete corpus generation, ingestion, and the TF-IDF index first. Then run:

```powershell
$env:PYTHONPATH = "src"
python scripts/build_bm25_index.py
python scripts/tune_hybrid_retrieval.py
python scripts/evaluate_hybrid_retrieval.py
python scripts/search_hybrid.py "How much time does the refund rule allow for NSG Connect in Ireland?" --top-k 3
```

Generated indexes, metrics, and tuning results remain local under `data/` and are ignored by Git.

## Calibration method

The 96-question hard set is split within each difficulty group:

- every fourth question forms a 24-question calibration set;
- the remaining 72 questions form an untouched holdout set.

The calibration grid compares RRF constants, TF-IDF/BM25 weights, reranker profiles, and abstention thresholds. Its objective combines Evidence MRR, Evidence Recall@1, and balanced answerability accuracy. Threshold ties favour the setting with the largest margin from nearby calibration scores.

Selected configuration:

- RRF constant: 10
- TF-IDF weight: 1.0
- BM25 weight: 1.0
- reranker: 90% vector similarity and 10% fused rank
- maximum chunks per document: 1
- hybrid abstention threshold: 0.30
- TF-IDF abstention threshold: 0.25

## Fair holdout comparison

| Metric | TF-IDF | Hybrid | Change |
|---|---:|---:|---:|
| Evidence Recall@1 | 38.9% | 40.5% | +1.6 pp |
| Evidence Recall@3 | 80.2% | 80.2% | 0.0 pp |
| Evidence MRR | 0.631 | 0.642 | +0.011 |
| Answerability accuracy | 100% | 100% | 0.0 pp |

The full 96-question hybrid run achieved 40.5% Evidence Recall@1, 81.0% at @3, 100% at @5, an Evidence MRR of 0.641, and 100% unanswerable abstention.

The improvement is modest but genuine. That is a stronger portfolio result than claiming that a more complicated retriever must automatically perform better.

## Next stage

The context and grounded-response layer is implemented in Phase 6. See [Grounded Answers and Citation Verification](grounded_answers.md) for the answer contract, refusal logic, and full benchmark results.

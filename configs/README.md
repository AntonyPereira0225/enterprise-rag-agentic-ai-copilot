# Configuration

Configuration files for ingestion, retrieval, model, evaluation and application behaviour live here. Environment-specific secrets belong in local environment variables and must not be committed.

- `corpus_config.json` controls deterministic synthetic-data generation.
- `ingestion_config.json` controls source/output paths, status filtering, chunk size, and overlap.
- `retrieval_config.json` controls the local vector index, embedding baseline, and Recall@K cut-offs.
- `hard_evaluation_config.json` controls the paraphrased, multi-source, and unanswerable question counts.
- `hard_retrieval_config.json` controls the harder TF-IDF benchmark and abstention threshold.
- `dense_retrieval_config.json` selects the Sentence Transformers model, FAISS output, and dense benchmark settings.
- `hybrid_retrieval_config.json` controls BM25/TF-IDF fusion, reranking, diversity, and the selected threshold.
- `hybrid_tuning_config.json` defines the calibration grid and holdout split.
- `grounded_answer_config.json` controls answerability gating, context size, citations, deterministic/optional Ollama generation, safe fallback, and model validation.
- `agent_workflow_config.json` controls workflow data paths, adversarial evaluation, and input limits.
- `api_config.json` controls the service address, request limit, UI path, version, and validation output.
- `observability_config.json` controls durable log/metric paths, experiment tracking, metric sources, and delivery validation output.
- `project_completion_config.json` declares final corpus/index/metric inputs, release files, quality thresholds, and the authoritative acceptance output.

# Data

This project uses synthetic and public-safe sample data only.

Planned local folders:

- `raw/` — source documents and synthetic support records.
- `processed/` — cleaned chunks plus a reproducibility manifest.
- `evaluation/` — labelled questions and expected evidence for RAG evaluation.
- `vector_store/` — local vector-index artifacts.

Large/generated data and vector indexes should remain local and must not be committed unless intentionally included as small portfolio samples.

Generate and process the baseline data from the repository root:

```powershell
$env:PYTHONPATH = "src"
python scripts/generate_synthetic_corpus.py
python scripts/ingest_knowledge_base.py
python scripts/build_retrieval_index.py
python scripts/evaluate_retrieval.py
python scripts/generate_hard_evaluation_set.py
python scripts/evaluate_hard_retrieval.py
python scripts/build_bm25_index.py
python scripts/tune_hybrid_retrieval.py
python scripts/evaluate_hybrid_retrieval.py
python scripts/evaluate_grounded_answers.py
python scripts/generate_agent_guardrail_set.py
python scripts/evaluate_agent_workflow.py
python scripts/validate_api.py
python scripts/validate_model_integration.py
python scripts/record_experiment.py
python scripts/validate_mlops.py
python scripts/validate_project.py
```

The grounded-answer command creates `evaluation/grounded_answer_metrics.json`. The agent commands create a deterministic routing/adversarial set and `evaluation/agent_workflow_metrics.json`. API validation writes `evaluation/api_validation_metrics.json` after exercising a real temporary HTTP server. Model validation writes `evaluation/model_integration_metrics.json` without requiring a live model. Delivery validation writes `evaluation/mlops_validation_metrics.json`, and the final requirement-based gate writes `evaluation/project_completion_metrics.json`. Experiment records live under `../artifacts/experiments/`.

See [the ingestion guide](../docs/ingestion_pipeline.md) for the data flow, [the grounded-answer guide](../docs/grounded_answers.md) for the answer contract, [the agent-workflow guide](../docs/agent_workflows.md) for routing and guardrails, [the API guide](../docs/api_and_demo.md) for the service and UI, [the model guide](../docs/model_integration.md) for the optional provider boundary, and [the final release guide](../docs/final_release.md) for requirement evidence.

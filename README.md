# Enterprise RAG & Agentic AI Copilot

A complete local-first portfolio application for traceable knowledge retrieval, grounded answers, controlled agent routing, responsible-AI guardrails, evaluation, observability, and reproducible delivery.

## Release status

**Version 1.0.0 — complete local portfolio release**

The repository includes the full implementation and a machine-readable final acceptance check. Its deterministic default needs no model download, API key, paid service, or Docker installation. An Ollama-compatible model can optionally select evidence, while all answer text remains locally assembled from exact approved source sentences.

Verified project assets include:

- 72 synthetic knowledge documents, 300 synthetic support cases, and 72 baseline questions
- 144 traceable chunks plus TF-IDF vector and BM25 keyword indexes
- a 96-question hard evaluation set and 20 routing/guardrail cases
- hybrid retrieval with reranking, calibrated abstention, and 100% evidence Recall@5 on the included benchmark
- 100% citation verification, evidence coverage, unsupported-question refusal, and guardrail benchmark rates
- policy, product, support, cross-functional, analytics, and general workflows with conversation state
- dependency-free and FastAPI HTTP adapters, a responsive browser demo, security headers, bounded requests, safe errors, and keyed log pseudonyms
- durable metrics, local experiment records, an optional MLflow adapter, Docker/Compose definitions, and GitHub Actions quality/container gates
- a safe optional Ollama integration with strict output validation and deterministic fallback

The exact current results are stored in [`data/evaluation/`](data/evaluation/) and summarized by [`project_completion_metrics.json`](data/evaluation/project_completion_metrics.json).

## Implemented architecture

```text
Synthetic knowledge + support cases
              ↓
Validated ingestion → cleaning → overlapping chunks + lineage manifest
              ↓
TF-IDF vector index + BM25 keyword index
              ↓
Reciprocal-rank fusion → transparent reranking → answerability gate
              ↓
Exact source evidence ── optional Ollama evidence-ID selector
              ↓                         │
Locally rendered citations ← validated IDs or deterministic fallback
              ↓
Input guardrail → explicit router → specialist workflow → output guardrail
              ↓
Standard-library / FastAPI service → browser demo → logs + metrics
              ↓
Evaluation + experiment record → pytest/Ruff → Docker + GitHub Actions
```

The core is intentionally dependency-light. Optional adapters provide Sentence Transformers/FAISS, FastAPI/Uvicorn, MLflow, and Ollama without changing the application contracts. See the [solution architecture](docs/solution_architecture.md) for the detailed design.

## Beginner-friendly local start

Run these commands from the repository folder in PowerShell.

### 1. Create the local environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-ci.txt
$env:PYTHONPATH = "src"
```

### 2. Build the local data and indexes

```powershell
python scripts/generate_synthetic_corpus.py
python scripts/ingest_knowledge_base.py
python scripts/build_retrieval_index.py
python scripts/build_bm25_index.py
python scripts/generate_hard_evaluation_set.py
python scripts/generate_agent_guardrail_set.py
```

### 3. Reproduce every quality result

```powershell
python scripts/evaluate_hybrid_retrieval.py
python scripts/evaluate_grounded_answers.py
python scripts/evaluate_agent_workflow.py
python scripts/validate_api.py
python scripts/validate_model_integration.py
python scripts/record_experiment.py
python scripts/validate_mlops.py
python scripts/validate_project.py
python -m pytest -p no:cacheprovider --cov=enterprise_copilot --cov-fail-under=90 -q
```

`validate_project.py` is the final gate. It checks corpus counts and hashes, index lineage, all evaluation thresholds, API privacy, model fallback, release files, version consistency, secret hygiene, and coverage of all 18 functional plus 8 non-functional requirements.

### 4. Open the finished application

```powershell
python scripts/serve_api.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000/). Stop the service with `Ctrl+C`.

The default `extractive` answer backend is fully offline. To try a locally installed Ollama model, follow the [model integration guide](docs/model_integration.md).

## Optional Docker start

When Docker is available:

```powershell
docker compose config --quiet
docker compose up --build
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000/), then stop it with `docker compose down`. Docker was not available on the development machine used for this release, so the definitions were statically validated locally and the configured GitHub Actions job performs the real image build and health check.

## Documentation

- [Business requirements](docs/business_requirements.md)
- [Solution architecture](docs/solution_architecture.md)
- [Knowledge corpus](docs/knowledge_corpus_design.md)
- [Ingestion and chunking](docs/ingestion_pipeline.md)
- [Retrieval baseline](docs/retrieval_baseline.md)
- [Hard and dense retrieval](docs/hard_and_dense_retrieval.md)
- [Hybrid retrieval and reranking](docs/hybrid_retrieval.md)
- [Grounded answers and citation verification](docs/grounded_answers.md)
- [Controlled agents and guardrails](docs/agent_workflows.md)
- [API and browser demo](docs/api_and_demo.md)
- [Observability and delivery](docs/observability_and_delivery.md)
- [Optional Ollama integration](docs/model_integration.md)
- [Cloud deployment runbook](docs/cloud_deployment_runbook.md)
- [Final release and requirement evidence](docs/final_release.md)

## Scope and safety

All business content is deterministic synthetic data. The system is read-only: it explains approved evidence and computes safe aggregates but never changes customer or company records. It is a portfolio-quality engineering demonstration, not a production certification or a connection to any real enterprise system. No cloud resources were provisioned for this release; the cloud document is an implementation runbook.

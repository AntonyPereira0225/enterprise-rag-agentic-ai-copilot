# Observability, Experiment Tracking, Containers and CI

## Outcome

This workstream turns the copilot into a reproducible, inspectable delivery unit. The application writes privacy-preserving request events and restart-aware metrics, records evaluation baselines with provenance, defines a least-privilege container, and runs the complete quality pipeline in GitHub Actions.

Docker is optional for local development. All data generation, evaluation, experiment tracking, validation, and tests work with Python alone. The container build and health check run in CI and can also be run locally after Docker Desktop is available.

## What was added

| Capability | Implementation | Output |
|---|---|---|
| Request observability | Timestamped JSON Lines events with per-process keyed hashes | `artifacts/observability/requests.jsonl` |
| Service metrics | Thread-safe counters, latency, guardrail totals, atomic snapshots, and restart recovery | `artifacts/observability/service_metrics.json` |
| Experiment tracking | Dependency-free local JSON backend with an optional MLflow adapter | `artifacts/experiments/enterprise-copilot/` |
| Provenance | Hashes for configuration, code, generated data, and metric inputs | Each experiment `run.json` |
| Immutable evidence | Copies of the five core evaluation metric files inside each local run | Each run's `artifacts/` folder |
| Container delivery | Non-root image, health check, root-owned code, and localhost-only Compose service | `Dockerfile`, `docker-compose.yml` |
| Continuous integration | Formatting, linting, 90% coverage gate, evaluations, evidence upload, image build, Compose validation, and health check | `.github/workflows/ci.yml` |
| Delivery validation | Eight executable/static contract checks | `data/evaluation/mlops_validation_metrics.json` |
| Final acceptance | Data, index, quality, model, documentation, version, and secret-hygiene checks | `data/evaluation/project_completion_metrics.json` |

## Step-by-step local validation

Run these commands from the repository root.

### 1. Activate the project environment

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
```

If the environment does not exist yet:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-ci.txt
$env:PYTHONPATH = "src"
```

### 2. Reproduce the evaluation inputs

```powershell
python scripts/generate_synthetic_corpus.py
python scripts/ingest_knowledge_base.py
python scripts/build_retrieval_index.py
python scripts/build_bm25_index.py
python scripts/generate_hard_evaluation_set.py
python scripts/generate_agent_guardrail_set.py
python scripts/evaluate_hybrid_retrieval.py
python scripts/evaluate_grounded_answers.py
python scripts/evaluate_agent_workflow.py
python scripts/validate_api.py
python scripts/validate_model_integration.py
```

### 3. Record the baseline experiment

```powershell
python scripts/record_experiment.py
```

The command collects ten headline metrics. It also records the Python version and hashes the configuration, application/delivery code, generated data, and five metric files. The local backend copies those metric files into the run folder, so the evidence does not change when a later evaluation is run.

### 4. Validate delivery and final acceptance

```powershell
python scripts/validate_mlops.py
python scripts/validate_project.py
```

The delivery result should be `8/8`, and final acceptance should report a `100%` pass rate with all 18 functional and 8 non-functional requirements covered. When Docker is unavailable, the report clearly marks the container runtime test as not executed; Docker, Compose, and CI files receive static contract checks, while GitHub Actions performs the real image build and health check.

### 5. Run the repository quality gate

```powershell
ruff check .
ruff format --check .
pytest -p no:cacheprovider --cov=enterprise_copilot --cov-fail-under=90 -q
```

### 6. Observe the running service

```powershell
python scripts/serve_api.py
```

Open `http://127.0.0.1:8000`, submit a few questions, and then inspect:

```powershell
Get-Content artifacts\observability\requests.jsonl
Get-Content artifacts\observability\service_metrics.json
```

Raw questions and raw conversation identifiers are not written to the log. A fresh random HMAC key is used for each process, allowing events from one running process to be correlated without making common questions recoverable from stable hashes.

## Docker steps for later

Once Docker Desktop is installed and running:

```powershell
docker compose config --quiet
docker compose up --build
```

Then open `http://127.0.0.1:8000`. In a second terminal, check readiness:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Stop and remove the local container while keeping its observability volume:

```powershell
docker compose down
```

The container process runs as an unprivileged user. Application code, configuration, and generated indexes remain root-owned; only `/app/artifacts` is writable. Compose also uses a read-only root filesystem, a temporary `/tmp`, no-new-privileges, and a loopback-only published port.

## Optional MLflow backend

The default `local_json` tracker has no extra dependency. To use MLflow instead:

```powershell
python -m pip install -r requirements-mlops.txt
```

Change `tracking_backend` in `configs/observability_config.json` from `local_json` to `mlflow`, set `MLFLOW_TRACKING_URI` if required, and run:

```powershell
python scripts/record_experiment.py
```

## Operational boundaries

- Service metrics are safe for multiple request threads and resume from the last snapshot, but the JSON file is designed for one service process. A multi-worker deployment should export the same metrics to Prometheus, OpenTelemetry, or another shared backend.
- Same-conversation requests are serialized inside the process so turn numbers and previous-route memory remain consistent. Conversation memory is not shared across multiple processes.
- Local telemetry failures are best effort and cannot break an otherwise valid answer. The metrics endpoint exposes persistence-error totals.
- Authentication, TLS termination, rate limiting, centralized secrets, and retention policies belong at the deployment boundary before exposing the service beyond localhost.

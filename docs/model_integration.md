# Optional Ollama Model Integration

## What this adds

Version 1.0 includes an optional Ollama-compatible model boundary. The model is deliberately restricted to one task: selecting citation IDs from evidence that the retrieval pipeline has already approved.

The model does **not** write the final answer. After the selected IDs pass validation, the application extracts exact supporting sentences, adds stable inline citations, and runs the same citation verifier used by the offline baseline. This design keeps model output inside a narrow, testable contract.

```text
Retrieved ContextPackage
        ↓
Ollama receives question + approved evidence
        ↓
Expected JSON: {"citation_ids":["C1"]}
        ↓
Schema, uniqueness, count, and allow-list checks
        ↓
Exact local quote rendering → citation verification
        │
        └── any provider/output problem → deterministic extractive fallback
```

## Default behaviour

The checked-in configuration uses `GENERATOR_BACKEND=extractive`. This is the reproducible, dependency-free mode used by evaluations and tests. Serving commands may opt into environment overrides; evaluation commands intentionally ignore those overrides so a developer's machine cannot silently change the benchmark.

## Try a local Ollama model

First install Ollama separately and download a model supported by your machine. Then, in the same PowerShell window used to start the copilot:

```powershell
$env:GENERATOR_BACKEND = "ollama"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:OLLAMA_MODEL = "llama3.2:3b"
$env:PYTHONPATH = "src"
python scripts/serve_api.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000/). The result panel shows `ollama` when the model-selected evidence was accepted. It shows `extractive_fallback` and a safe reason code when fallback was needed.

You can also exercise one request without starting the browser service:

```powershell
python scripts/ask_grounded.py "What response window should support staff use for NSG Home in Germany?"
```

## Settings

| Environment variable | Default | Purpose |
|---|---:|---|
| `GENERATOR_BACKEND` | `extractive` | Selects `extractive` or `ollama` for interactive serving. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama-compatible endpoint. Plain HTTP is accepted only for loopback/localhost endpoints. Remote endpoints must use HTTPS. |
| `OLLAMA_MODEL` | `llama3.2:3b` | Model identifier; whitespace and unsafe characters are rejected. |
| `OLLAMA_TIMEOUT_SECONDS` | `10` | Bounded request timeout; allowed range is 0.1–120 seconds. |
| `OLLAMA_SEED` | `7` | Reproducibility seed sent with temperature zero. |
| `OLLAMA_MAX_RESPONSE_BYTES` | `65536` | Maximum provider response size, bounded again by configuration validation. |

The same defaults live in `configs/grounded_answer_config.json`. Environment values are intended for runtime selection; secrets must never be stored in that JSON file.

## Fail-safe behaviour

The following provider outcomes use the verified extractive fallback:

- connection, timeout, or HTTP failure: `provider_unavailable`
- response larger than the configured boundary: `provider_response_too_large`
- malformed JSON, extra output fields, no citation, duplicate citation, too many citations, or an unknown ID: `invalid_provider_response`

Only these stable reason codes leave the provider boundary. Raw provider exceptions and response text are not returned to the user or copied into structured request logs. If any generator ever produces a response that fails citation verification, the pipeline replaces it with a citation-free safe refusal.

## Privacy boundary

The copilot's own request logs store per-process keyed HMAC-SHA-256 pseudonyms, not raw question or conversation text. A model endpoint must receive the question and retrieved passages to make its selection, so use a trusted local endpoint or an approved HTTPS service with appropriate data-handling terms. The included corpus is synthetic, but this distinction matters before adapting the design to real data.

## Validate without installing Ollama

```powershell
$env:PYTHONPATH = "src"
python scripts/validate_model_integration.py
```

The validator starts a temporary local mock endpoint and proves successful selection, exact quote rendering, strict invalid-output fallback, sanitized errors, no model call for unanswerable questions, endpoint safety, and the full 96-question deterministic regression. It writes `data/evaluation/model_integration_metrics.json` and does not download or contact a real model.

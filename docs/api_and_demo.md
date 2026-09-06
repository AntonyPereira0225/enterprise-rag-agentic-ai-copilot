# API and Demonstration Interface

## Purpose

The completed service exposes the controlled workflow through one validated contract. It includes a browser demonstration, health and request metrics, bounded inputs, security headers, privacy-preserving JSON logs, and safe error responses.

Two HTTP adapters use the same service:

- the standard-library adapter runs immediately with the lightweight offline development environment;
- the FastAPI/Uvicorn adapter becomes available when `requirements-api.txt` is installed.

This separation means the API behavior is fully tested now without maintaining two business-logic implementations.

## Start the local interface

From the repository root:

1. Point Python at the source folder.

   ```powershell
   $env:PYTHONPATH = "src"
   ```

2. Start the dependency-free server.

   ```powershell
   python scripts/serve_api.py
   ```

3. Open this address in a browser:

   ```text
   http://127.0.0.1:8000
   ```

4. Enter a question or select an example. The page shows safe status, route, confidence, answer, citations, latency, workflow steps, and the active generator/fallback state.

5. Return to the terminal and press `Ctrl+C` to stop the server.

## Optional FastAPI adapter

FastAPI and Uvicorn remain optional at runtime. Install only the interface dependencies and start the equivalent adapter with:

```powershell
python -m pip install -r requirements-api.txt
python scripts/serve_fastapi.py
```

FastAPI then provides its normal OpenAPI documentation in addition to the endpoints below. If the dependency is missing, the script displays a clear fallback message rather than a Python import traceback.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Responsive browser demonstration |
| `POST` | `/ask` | Validated entry point to the agent workflow |
| `GET` | `/health` | Service readiness, version, and active knowledge-answer backend |
| `GET` | `/metrics` | Restart-aware counters, latency, and persistence health |

### Ask request

```json
{
  "question": "What response window should support staff use for NSG Home in Germany?",
  "conversation_id": "demo"
}
```

`question` is required. `conversation_id` is optional and defaults to `default`; when provided, it must contain 1–64 letters, numbers, hyphens, or underscores. Unknown fields are rejected.

PowerShell example:

```powershell
$body = @{
  question = "How many support cases are refund requests?"
  conversation_id = "demo"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask `
  -ContentType "application/json" -Body $body
```

Successful HTTP processing returns status `200` even when the workflow safely blocks or refuses a question; the business outcome is explicit in `data.status`. Invalid schemas return `422`, malformed JSON returns `400`, unsupported content types return `415`, and oversized bodies return `413`.

## Metrics

`GET /metrics` returns:

- total requests and internal errors;
- average end-to-end latency;
- counts by workflow status and route;
- blocked-request counts by guardrail category.

Metrics are updated in memory and atomically written to `artifacts/observability/service_metrics.json`. A new single-process service resumes from that snapshot after restart. The file is intentionally a single-writer local backend; a multi-worker deployment should export the same contract to Prometheus, OpenTelemetry, or a cloud platform.

## Logging and security

Accepted, schema-invalid, malformed, oversized, and wrong-content-type requests create timestamped JSON log events. Logs include the request identifier, route, status, latency, error type when relevant, and per-process HMAC pseudonyms for valid questions and conversation identifiers. They do not include raw question text or raw conversation identifiers.

HTTP responses include content-type protection, frame denial, referrer protection, and a content security policy. Request bodies are capped at 32 KiB. Internal exceptions return a generic safe message and are counted without exposing private failure details.

Telemetry writes are best effort so a local disk failure cannot turn an otherwise valid answer into a failed request. Same-conversation calls are serialized for safe in-process turn state. This local demonstration does not implement authentication, TLS termination, rate limiting, centralized telemetry, or multi-process conversation persistence; those controls belong at an authenticated deployment boundary.

## Validate the interface

Run the end-to-end server checks:

```powershell
python scripts/validate_api.py
```

The validator starts a real server on a temporary local port and checks:

1. readiness;
2. the browser page;
3. a cited support answer;
4. a support-case aggregate;
5. prompt-injection blocking;
6. schema validation;
7. malformed JSON handling;
8. content-type enforcement;
9. request-size enforcement;
10. request metrics;
11. structured-log privacy.

The release run passes all 11 checks. Run `python scripts/validate_project.py` for final acceptance and `python -m pytest -q` for the complete repository regression suite.

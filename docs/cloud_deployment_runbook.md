# Cloud Deployment Runbook

## Status and intent

No paid cloud resources were provisioned for this portfolio release. This provider-neutral runbook defines the controls required if the locally verified container is later deployed to an approved cloud account. It is not a claim of production certification.

## Recommended deployment shape

```text
Internet or corporate network
            ↓
TLS endpoint + identity-aware gateway + rate limit
            ↓
One private container service running the copilot image
            ↓
Read-only image data/indexes + writable observability volume
            ↓ optional
Private Ollama-compatible inference endpoint
```

Start with one small container instance. The baseline data and indexes are built into the image, so no managed database is required. Keep the default extractive backend until the basic deployment, authentication, logging, and rollback paths have been validated.

## Before provisioning anything

1. Run the full local validation in the README and require `validate_project.py` to report 100%.
2. Let GitHub Actions build and health-check the Docker image.
3. Choose an organization-approved region, container registry, identity provider, log destination, budget, and data classification.
4. Confirm who owns incident response, patching, access reviews, retention, and deletion.
5. Create a separate non-production cloud project/account with a strict monthly budget and alerts.

## Build and publish

1. Pin the release to an immutable source revision and tag the image `1.0.0` plus that revision.
2. Build the existing `Dockerfile`; do not add credentials to the build context or image layers.
3. Scan the image and dependencies using the organization's approved security tooling.
4. Push only after the CI tests, evaluation reproduction, final acceptance validator, container health check, and scan pass.
5. Record the image digest in the deployment change record and deploy by digest rather than a mutable tag.

## Runtime configuration

- Expose container port 8000 only through the managed TLS ingress.
- Require authenticated users; the local demo has no identity layer and must not be placed directly on the public internet.
- Keep the filesystem read-only and mount a bounded writable path only for `artifacts/`, or send logs/metrics to an approved external sink.
- Run as the existing non-root `appuser`, retain the no-new-privileges setting, and apply platform CPU/memory limits.
- Keep `GENERATOR_BACKEND=extractive` for the first deployment.
- If Ollama mode is later approved, use a private endpoint. Remote model URLs must use HTTPS, outbound access should be restricted to that endpoint, and its retention/data-use policy must be reviewed.
- Store credentials only in the platform secret manager. The current application does not require a secret for its default mode.

## State and scaling warning

Conversation turn state is held in memory and protected by per-conversation locks. Multiple replicas would not share that state. Before scaling beyond one replica, either use session affinity or implement an approved external conversation-state store with expiration, encryption, and concurrency controls. Request metrics are process-local snapshots as well; aggregate them at the platform boundary for a multi-replica deployment.

## Deployment checks

After deploying to the non-production environment:

1. Confirm `/health` returns `status: ready`, version `1.0.0`, and the expected answer backend.
2. Confirm `/` loads through HTTPS with all security headers present.
3. Submit one grounded request, one analytics request, one blocked injection request, one malformed request, and one oversized request.
4. Confirm exact citations, safe refusals, request limits, keyed pseudonyms, and absence of raw questions in logs.
5. Confirm `/metrics` increments and the platform captures service errors and restarts.
6. Restart the instance and verify immutable data/indexes still load and writable telemetry remains available according to the selected persistence policy.
7. Run a synthetic smoke test from outside the service network through the real authentication path.

## Rollback and incident response

- Keep the previous known-good image digest deployable.
- Roll back on failed health checks, citation verification errors, sustained latency/error thresholds, unexpected outbound traffic, or privacy-control failure.
- Disable the model endpoint first if model-mode errors increase; deterministic fallback permits the application to continue without it.
- Preserve sanitized operational evidence, revoke affected credentials, and follow the organization's incident process.
- Never log raw request bodies as a debugging shortcut.

## Cost controls

- Use the smallest instance that satisfies measured latency and memory needs.
- Set a hard non-production budget and alerts before deployment.
- Keep minimum instance count at zero where cold starts are acceptable, or one only for a scheduled demo window.
- Avoid managed vector or model services until a measured requirement justifies them.
- Delete the registry images, service, volume, log workspace, endpoint, and related network resources when the demonstration is finished.

## Go-live boundary

Moving beyond a controlled portfolio demo requires security review, threat modelling, authentication/authorization design, privacy and retention assessment, load and resilience testing, accessibility review, dependency/image scanning, production monitoring, backup/restore decisions, and named operational ownership. Those organizational controls are intentionally outside this repository's local completion claim.

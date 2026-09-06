from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from enterprise_copilot.api.service import ApiConfig, CopilotService, build_service


def create_app(
    project_root: Path | None = None,
    config: ApiConfig | None = None,
    service: CopilotService | None = None,
) -> Any:
    """Create the optional FastAPI adapter when its declared dependencies are installed."""
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import HTMLResponse, JSONResponse
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FastAPI is not installed. Install the project runtime requirements or use "
            "scripts/serve_api.py for the dependency-free local server."
        ) from exc

    root = project_root or Path(__file__).resolve().parents[3]
    active_config = config or ApiConfig.from_json(root / "configs" / "api_config.json")
    active_service = service or build_service(root, active_config)
    app = FastAPI(
        title="Enterprise RAG & Agentic AI Copilot",
        version=active_config.service_version,
    )

    def add_security_headers(response: Any) -> Any:
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; connect-src 'self'"
        )
        return response

    @app.middleware("http")
    async def request_boundary(request: Request, call_next: Any) -> Any:
        if request.method == "POST" and request.url.path == "/ask":
            content_type = request.headers.get("content-type", "").split(";", maxsplit=1)[0]
            if content_type.casefold() != "application/json":
                active_service.observe_transport_error(
                    status="unsupported_media_type",
                    error_type="UnsupportedMediaType",
                )
                return add_security_headers(
                    JSONResponse(
                        {
                            "error": {
                                "code": "unsupported_media_type",
                                "message": "Content-Type must be application/json.",
                            }
                        },
                        status_code=415,
                    )
                )

            content_length_header = request.headers.get("content-length")
            try:
                content_length = (
                    int(content_length_header) if content_length_header is not None else None
                )
            except ValueError:
                content_length = -1
            if content_length is not None and content_length < 0:
                active_service.observe_transport_error(
                    status="invalid_length",
                    error_type="InvalidContentLength",
                )
                return add_security_headers(
                    JSONResponse(
                        {"error": {"code": "invalid_length", "message": "Invalid body length."}},
                        status_code=400,
                    )
                )

            if content_length is not None and content_length > active_config.max_request_bytes:
                active_service.observe_transport_error(
                    status="request_too_large",
                    error_type="RequestTooLarge",
                )
                return add_security_headers(
                    JSONResponse(
                        {
                            "error": {
                                "code": "request_too_large",
                                "message": (
                                    f"Request exceeds {active_config.max_request_bytes} bytes."
                                ),
                            }
                        },
                        status_code=413,
                    )
                )

        response = await call_next(request)
        return add_security_headers(response)

    @app.get("/health")
    def health() -> JSONResponse:
        response = active_service.health()
        return JSONResponse(response.body, status_code=response.status_code)

    @app.get("/metrics")
    def metrics() -> JSONResponse:
        response = active_service.metric_snapshot()
        return JSONResponse(response.body, status_code=response.status_code)

    async def ask(request: Any) -> Any:
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > active_config.max_request_bytes:
                active_service.observe_transport_error(
                    status="request_too_large",
                    error_type="RequestTooLarge",
                )
                return JSONResponse(
                    {
                        "error": {
                            "code": "request_too_large",
                            "message": f"Request exceeds {active_config.max_request_bytes} bytes.",
                        }
                    },
                    status_code=413,
                )
            body.extend(chunk)
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            active_service.observe_transport_error(
                status="invalid_json",
                error_type="InvalidJson",
            )
            return JSONResponse(
                {"error": {"code": "invalid_json", "message": "Body must be valid JSON."}},
                status_code=400,
            )
        response = active_service.ask(payload)
        return JSONResponse(response.body, status_code=response.status_code)

    ask.__annotations__["request"] = Request
    app.post("/ask")(ask)

    @app.get("/", response_class=HTMLResponse)
    def demo() -> str:
        return (root / active_config.demo_page_path).read_text(encoding="utf-8")

    return app

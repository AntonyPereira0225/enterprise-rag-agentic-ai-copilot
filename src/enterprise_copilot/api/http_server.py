from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from enterprise_copilot.api.schemas import ServiceResponse
from enterprise_copilot.api.service import ApiConfig, CopilotService, build_service
from enterprise_copilot.monitoring.experiment import resolve_project_path


def _handler_class(
    service: CopilotService,
    *,
    demo_page: Path,
    max_request_bytes: int,
) -> type[BaseHTTPRequestHandler]:
    class CopilotRequestHandler(BaseHTTPRequestHandler):
        server_version = "EnterpriseCopilot/1.0"

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/health":
                self._write_service_response(service.health())
            elif path == "/metrics":
                self._write_service_response(service.metric_snapshot())
            elif path == "/":
                self._write_html(demo_page.read_bytes())
            else:
                self._write_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": {"code": "not_found", "message": "Endpoint not found."}},
                )

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/ask":
                self._write_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": {"code": "not_found", "message": "Endpoint not found."}},
                )
                return
            content_type = self.headers.get("Content-Type", "").split(";", maxsplit=1)[0]
            if content_type.casefold() != "application/json":
                service.observe_transport_error(
                    status="unsupported_media_type",
                    error_type="UnsupportedMediaType",
                )
                self._write_json(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    {
                        "error": {
                            "code": "unsupported_media_type",
                            "message": "Content-Type must be application/json.",
                        }
                    },
                )
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                content_length = -1
            if content_length < 0:
                service.observe_transport_error(
                    status="invalid_length",
                    error_type="InvalidContentLength",
                )
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": {"code": "invalid_length", "message": "Invalid body length."}},
                )
                return
            if content_length > max_request_bytes:
                service.observe_transport_error(
                    status="request_too_large",
                    error_type="RequestTooLarge",
                )
                self._write_json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {
                        "error": {
                            "code": "request_too_large",
                            "message": f"Request exceeds {max_request_bytes} bytes.",
                        }
                    },
                )
                return
            try:
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                service.observe_transport_error(
                    status="invalid_json",
                    error_type="InvalidJson",
                )
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": {"code": "invalid_json", "message": "Body must be valid JSON."}},
                )
                return
            self._write_service_response(service.ask(payload))

        def do_OPTIONS(self) -> None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self._security_headers()
            self.send_header("Allow", "GET, POST, OPTIONS")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _write_service_response(self, response: ServiceResponse) -> None:
            self._write_json(response.status_code, response.body)

        def _write_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self._security_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _write_html(self, body: bytes) -> None:
            self.send_response(HTTPStatus.OK)
            self._security_headers()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _security_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                "script-src 'self' 'unsafe-inline'; connect-src 'self'",
            )

        def log_message(self, format: str, *args: object) -> None:
            return

    return CopilotRequestHandler


def build_http_server(
    project_root: Path,
    config: ApiConfig,
    *,
    service: CopilotService | None = None,
    host: str | None = None,
    port: int | None = None,
) -> ThreadingHTTPServer:
    active_service = service or build_service(project_root, config)
    handler = _handler_class(
        active_service,
        demo_page=resolve_project_path(project_root, config.demo_page_path),
        max_request_bytes=config.max_request_bytes,
    )
    return ThreadingHTTPServer(
        (host or config.host, config.port if port is None else port), handler
    )

from pathlib import Path

from enterprise_copilot.api.http_server import build_http_server
from enterprise_copilot.api.service import ApiConfig


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = ApiConfig.from_json(project_root / "configs" / "api_config.json").with_environment()
    server = build_http_server(project_root, config)
    address, port = server.server_address[:2]
    print(f"Enterprise Copilot is ready at http://{address}:{port}")
    print("Press Ctrl+C to stop the server")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Enterprise Copilot")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

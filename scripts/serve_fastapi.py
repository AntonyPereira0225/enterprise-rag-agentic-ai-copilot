from pathlib import Path

from enterprise_copilot.api.fastapi_app import create_app
from enterprise_copilot.api.service import ApiConfig


def main() -> None:
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Uvicorn is not installed. Install requirements-api.txt, or run "
            "scripts/serve_api.py for the dependency-free local server."
        ) from exc

    project_root = Path(__file__).resolve().parents[1]
    config = ApiConfig.from_json(project_root / "configs" / "api_config.json").with_environment()
    app = create_app(project_root, config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()

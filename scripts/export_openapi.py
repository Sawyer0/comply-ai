"""
Export the FastAPI app's OpenAPI spec to YAML without running a server.

Usage (PowerShell):
  python scripts/export_openapi.py --output docs/openapi.yaml

This builds the app using a lightweight stub ModelServer (no real model load),
then writes openapi.yaml.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml  # type: ignore

# Build the app similarly to tests/integration/test_version_tags.py
from src.llama_mapper.api.mapper import create_app
from src.llama_mapper.serving.model_server import ModelServer
from src.llama_mapper.serving.json_validator import JSONValidator
from src.llama_mapper.serving.fallback_mapper import FallbackMapper
from src.llama_mapper.config.manager import ConfigManager
from src.llama_mapper.monitoring.metrics_collector import MetricsCollector


class _StubModelServer(ModelServer):
    async def load_model(self) -> None:  # type: ignore[override]
        self.is_loaded = True

    async def generate_text(self, prompt: str, **kwargs):  # type: ignore[override]
        # Not used during OpenAPI export
        return ""

    async def health_check(self) -> bool:  # type: ignore[override]
        return True


def _build_app() -> any:
    config = ConfigManager()
    # Use local .kiro schema and detectors if available; fallbacks to defaults
    schema_path = str(Path('.kiro/pillars-detectors/schema.json'))
    detectors_dir = str(Path('.kiro/pillars-detectors'))

    model_server = _StubModelServer(model_path="stub")
    json_validator = JSONValidator(schema_path=schema_path)
    fallback_mapper = FallbackMapper(detector_configs_path=detectors_dir)
    metrics = MetricsCollector()

    app = create_app(
        model_server=model_server,
        json_validator=json_validator,
        fallback_mapper=fallback_mapper,
        config_manager=config,
        metrics_collector=metrics,
    )
    # Touch OpenAPI to ensure it's generated after app startup wiring
    _ = app.openapi()
    return app


def main():
    parser = argparse.ArgumentParser(description="Export OpenAPI YAML from live app instance")
    parser.add_argument("--output", "-o", type=str, default="docs/openapi.yaml", help="Output file path")
    args = parser.parse_args()

    app = _build_app()
    openapi_dict = app.openapi()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(openapi_dict, f, sort_keys=False)
    print(f"✓ OpenAPI spec written to {output_path}")


if __name__ == "__main__":
    sys.exit(main())

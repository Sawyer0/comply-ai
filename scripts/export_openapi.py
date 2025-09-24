"""
Export the FastAPI app's OpenAPI spec to YAML without running a server.

Usage (PowerShell):
  python scripts/export_openapi.py --output docs/openapi.yaml

This builds the app using a lightweight stub ModelServer (no real model load),
then writes openapi.yaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml  # type: ignore

# Build the app similarly to tests/integration/test_version_tags.py
from src.llama_mapper.api.mapper import create_app
from src.llama_mapper.config.manager import ConfigManager
from src.llama_mapper.monitoring.metrics_collector import MetricsCollector
from src.llama_mapper.serving.fallback_mapper import FallbackMapper
from src.llama_mapper.serving.json_validator import JSONValidator
from src.llama_mapper.serving.model_server import ModelServer


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
    schema_path = str(Path(".kiro/pillars-detectors/schema.json"))
    detectors_dir = str(Path(".kiro/pillars-detectors"))

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


def _ensure_components(openapi_dict: dict) -> dict:
    if "components" not in openapi_dict:
        openapi_dict["components"] = {"schemas": {}}
    if "schemas" not in openapi_dict["components"]:
        openapi_dict["components"]["schemas"] = {}
    return openapi_dict


def _inject_request_schemas(openapi_dict: dict) -> dict:
    """Inject oneOf requestBody schemas for /map and /map/batch, and ensure models exist in components."""
    from src.llama_mapper.api.models import (
        BatchDetectorRequest,
        DetectorRequest,
        ErrorBody,
        MapperPayload,
        MappingResponse,
        PolicyContext,
        Provenance,
        VersionInfo,
    )

    openapi_dict = _ensure_components(openapi_dict)
    schemas = openapi_dict["components"]["schemas"]

    # Ensure model schemas are present
    def add_model_schema(model, name: str) -> None:
        if name not in schemas:
            try:
                schemas[name] = model.model_json_schema(
                    ref_template="#/components/schemas/{model}"
                )
            except Exception:
                schemas[name] = model.model_json_schema()

    add_model_schema(MapperPayload, "MapperPayload")
    add_model_schema(DetectorRequest, "DetectorRequest")
    add_model_schema(BatchDetectorRequest, "BatchDetectorRequest")
    add_model_schema(MappingResponse, "MappingResponse")
    add_model_schema(ErrorBody, "ErrorBody")
    add_model_schema(Provenance, "Provenance")
    add_model_schema(PolicyContext, "PolicyContext")
    add_model_schema(VersionInfo, "VersionInfo")

    # Add BatchMapperPayload synthetic schema
    if "BatchMapperPayload" not in schemas:
        schemas["BatchMapperPayload"] = {
            "type": "object",
            "title": "BatchMapperPayload",
            "properties": {
                "requests": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 100,
                    "items": {"$ref": "#/components/schemas/MapperPayload"},
                }
            },
            "required": ["requests"],
        }

    # Update /map requestBody
    try:
        map_post = openapi_dict["paths"]["/map"]["post"]
        map_post["requestBody"] = {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "oneOf": [
                            {"$ref": "#/components/schemas/MapperPayload"},
                            {
                                "$ref": "#/components/schemas/DetectorRequest",
                                "deprecated": True,
                            },
                        ]
                    }
                }
            },
        }
    except Exception:
        pass

    # Update /map/batch requestBody
    try:
        batch_post = openapi_dict["paths"]["/map/batch"]["post"]
        batch_post["requestBody"] = {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "oneOf": [
                            {"$ref": "#/components/schemas/BatchMapperPayload"},
                            {
                                "$ref": "#/components/schemas/BatchDetectorRequest",
                                "deprecated": True,
                            },
                        ]
                    }
                }
            },
        }
    except Exception:
        pass

    return openapi_dict


def main():
    parser = argparse.ArgumentParser(
        description="Export OpenAPI YAML from live app instance"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="docs/openapi.yaml", help="Output file path"
    )
    args = parser.parse_args()

    app = _build_app()
    openapi_dict = app.openapi()
    # Post-process: inject oneOf request schemas and ensure components
    openapi_dict = _inject_request_schemas(openapi_dict)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(openapi_dict, f, sort_keys=False)
    print(f"✓ OpenAPI spec written to {output_path}")


if __name__ == "__main__":
    sys.exit(main())

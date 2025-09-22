"""
Export the Detector Orchestration service's OpenAPI spec to YAML.

Usage:
  python scripts/export_orchestration_openapi.py --output detector-orchestration/docs/openapi.yaml

This builds the orchestration app using a lightweight stub configuration,
then writes the OpenAPI specification to YAML.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml  # type: ignore

# Add detector orchestration to path
orch_src = Path(__file__).parent.parent / "detector-orchestration" / "src"
if str(orch_src) not in sys.path:
    sys.path.insert(0, str(orch_src))

from detector_orchestration.api.main import app


def _ensure_components(openapi_dict: dict) -> dict:
    """Ensure components section exists in OpenAPI spec."""
    if "components" not in openapi_dict:
        openapi_dict["components"] = {"schemas": {}}
    if "schemas" not in openapi_dict["components"]:
        openapi_dict["components"]["schemas"] = {}
    return openapi_dict


def _inject_orchestration_schemas(openapi_dict: dict) -> dict:
    """Inject orchestration-specific schemas into OpenAPI spec."""
    from detector_orchestration.models import (
        OrchestrationRequest,
        OrchestrationResponse,
        DetectorResult,
        MapperPayload,
        RoutingDecision,
        RoutingPlan,
        ContentType,
        ProcessingMode,
        Priority,
        DetectorStatus,
        Provenance,
        PolicyContext,
        MappingResponse,
        DetectorCapabilities,
        JobStatus,
        JobStatusResponse,
        ErrorBody,
    )

    openapi_dict = _ensure_components(openapi_dict)
    schemas = openapi_dict["components"]["schemas"]

    # Ensure model schemas are present
    def add_model_schema(model, name: str) -> None:
        if name not in schemas:
            try:
                # Check if it's a Pydantic model
                if hasattr(model, 'model_json_schema'):
                    schemas[name] = model.model_json_schema(ref_template="#/components/schemas/{model}")
                # Check if it's an Enum
                elif hasattr(model, '__members__'):
                    schemas[name] = {
                        "type": "string",
                        "enum": list(model.__members__.keys()),
                        "title": name
                    }
                else:
                    # Fallback for other types
                    schemas[name] = {"type": "string", "title": name}
            except Exception:
                # Fallback for any other issues
                schemas[name] = {"type": "string", "title": name}

    # Add all orchestration models
    add_model_schema(OrchestrationRequest, "OrchestrationRequest")
    add_model_schema(OrchestrationResponse, "OrchestrationResponse")
    add_model_schema(DetectorResult, "DetectorResult")
    add_model_schema(MapperPayload, "MapperPayload")
    add_model_schema(RoutingDecision, "RoutingDecision")
    add_model_schema(RoutingPlan, "RoutingPlan")
    add_model_schema(ContentType, "ContentType")
    add_model_schema(ProcessingMode, "ProcessingMode")
    add_model_schema(Priority, "Priority")
    add_model_schema(DetectorStatus, "DetectorStatus")
    add_model_schema(Provenance, "Provenance")
    add_model_schema(PolicyContext, "PolicyContext")
    add_model_schema(MappingResponse, "MappingResponse")
    add_model_schema(DetectorCapabilities, "DetectorCapabilities")
    add_model_schema(JobStatus, "JobStatus")
    add_model_schema(JobStatusResponse, "JobStatusResponse")
    add_model_schema(ErrorBody, "ErrorBody")

    # Add batch request schema
    if "BatchOrchestrationRequest" not in schemas:
        schemas["BatchOrchestrationRequest"] = {
            "type": "object",
            "title": "BatchOrchestrationRequest",
            "properties": {
                "requests": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 100,
                    "items": {"$ref": "#/components/schemas/OrchestrationRequest"},
                }
            },
            "required": ["requests"],
        }

    # Add batch response schema
    if "BatchOrchestrationResponse" not in schemas:
        schemas["BatchOrchestrationResponse"] = {
            "type": "object",
            "title": "BatchOrchestrationResponse",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {"$ref": "#/components/schemas/OrchestrationResponse"},
                },
                "errors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tenant_id": {"type": "string"},
                            "error": {"type": "string"},
                        },
                    },
                },
            },
            "required": ["results"],
        }

    return openapi_dict


def main():
    """Main function to export OpenAPI specification."""
    parser = argparse.ArgumentParser(description="Export Detector Orchestration OpenAPI YAML")
    parser.add_argument(
        "--output", 
        "-o", 
        type=str, 
        default="detector-orchestration/docs/openapi.yaml", 
        help="Output file path"
    )
    args = parser.parse_args()

    # Get OpenAPI spec from the app
    openapi_dict = app.openapi()
    
    # Post-process: inject orchestration-specific schemas
    openapi_dict = _inject_orchestration_schemas(openapi_dict)

    # Write to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(openapi_dict, f, sort_keys=False)
    
    print(f"✓ Detector Orchestration OpenAPI spec written to {output_path}")


if __name__ == "__main__":
    sys.exit(main())

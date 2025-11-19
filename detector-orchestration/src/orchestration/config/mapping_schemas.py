from __future__ import annotations

from typing import Any, Dict

import jsonschema


_CANONICAL_DETECTOR_MAPPING_V1 = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Canonical Detector Mapping v1",
    "type": "object",
    "properties": {
        "canonical_category": {"type": "string", "minLength": 1},
        "canonical_subcategory": {"type": "string"},
        "canonical_risk_level": {
            "type": "string",
            "enum": ["low", "medium", "high", "critical"],
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
        },
        "use_detector_category": {"type": "boolean"},
        "use_detector_subcategory": {"type": "boolean"},
        "entity_rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "match": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "equals": {},
                        },
                        "required": ["field", "equals"],
                        "additionalProperties": False,
                    },
                    "label": {"type": "string", "minLength": 1},
                    "category": {"type": "string", "minLength": 1},
                    "subcategory": {"type": "string"},
                    "type": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                    },
                    "risk_level": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                    },
                    "text_field": {"type": "string", "minLength": 1},
                    "start_field": {"type": "string"},
                    "end_field": {"type": "string"},
                    "confidence_field": {"type": "string"},
                },
                "required": ["match", "label", "category", "text_field"],
                "additionalProperties": True,
            },
        },
    },
    "required": ["canonical_category", "canonical_risk_level", "entity_rules"],
    "additionalProperties": False,
}


_SCHEMA_REGISTRY: Dict[str, Dict[str, Any]] = {
    "canonical-detector-mapping.v1": _CANONICAL_DETECTOR_MAPPING_V1,
}


def get_mapping_schema(schema_version: str) -> Dict[str, Any]:
    if schema_version not in _SCHEMA_REGISTRY:
        raise ValueError(f"Unsupported mapping schema_version: {schema_version}")
    return _SCHEMA_REGISTRY[schema_version]


def validate_mapping_rules(mapping_rules: Dict[str, Any], schema_version: str) -> Dict[str, Any]:
    schema = get_mapping_schema(schema_version)
    jsonschema.validate(instance=mapping_rules, schema=schema)
    return mapping_rules

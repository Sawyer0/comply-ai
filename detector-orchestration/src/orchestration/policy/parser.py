"""Policy parsing and validation utilities.

Extracted from PolicyLoader to handle policy syntax validation
and data structure validation separately from file operations.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List
from functools import lru_cache

logger = logging.getLogger(__name__)

__all__ = [
    "validate_required_fields",
    "validate_data_types", 
    "validate_policy_syntax",
    "classify_rule_type",
    "validate_tenant_policy_data",
]


def validate_required_fields(
    config: Dict[str, Any], required_fields: List[str], context: str
) -> List[str]:
    """Utility function to validate required fields in configuration objects.

    Args:
        config: Configuration object to validate
        required_fields: List of required field names
        context: Context string for error messages

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field '{field}' in {context}")

    return errors


def validate_data_types(
    config: Dict[str, Any], field_types: Dict[str, Any], context: str
) -> List[str]:
    """Utility function to validate data types in configuration objects.

    Args:
        config: Configuration object to validate
        field_types: Dictionary mapping field names to expected types
        context: Context string for error messages

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    for field, expected_type in field_types.items():
        if field in config:
            if not isinstance(config[field], expected_type):
                type_name = (
                    expected_type.__name__
                    if hasattr(expected_type, "__name__")
                    else str(expected_type)
                )
                errors.append(
                    f"Invalid type for '{field}' in {context}: "
                    f"expected {type_name}, got {type(config[field]).__name__}"
                )

    return errors


@lru_cache(maxsize=100)
def validate_policy_syntax(policy_content: str) -> List[str]:
    """Validate basic Rego policy syntax.

    Args:
        policy_content: Policy content to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    try:
        lines = policy_content.split("\n")

        # Basic syntax validation
        if not any(line.strip().startswith("package ") for line in lines):
            errors.append("Missing package declaration")

        # Check for balanced braces
        open_braces = policy_content.count("{")
        close_braces = policy_content.count("}")

        if open_braces != close_braces:
            errors.append(f"Unbalanced braces: {open_braces} open, {close_braces} close")

        # Check for basic rule structure
        has_rules = any(" if {" in line or line.strip().endswith(" := ") for line in lines)

        if not has_rules:
            errors.append("No rules found in policy")

    except (IndexError, AttributeError) as e:
        errors.append(f"Syntax validation error: {str(e)}")

    return errors


def classify_rule_type(line: str) -> str:
    """Classify the type of policy rule.
    
    Args:
        line: Policy line to classify
        
    Returns:
        Rule type classification
    """
    line = line.strip()
    
    if line.startswith("default "):
        return "default"
    if " if {" in line:
        return "conditional"
    if " := " in line:
        return "assignment"
    if line in ["allow", "deny"]:
        return "decision"
    return "unknown"


def validate_tenant_policy_data(data: Dict[str, Any]) -> List[str]:
    """Validate tenant policy data structure.

    Args:
        data: Tenant policy data to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    try:
        if "tenant_policies" not in data:
            errors.append("Missing 'tenant_policies' root key")
            return errors

        tenant_policies = data["tenant_policies"]

        for tenant_id, tenant_config in tenant_policies.items():
            if not isinstance(tenant_config, dict):
                errors.append(f"Invalid config for tenant {tenant_id}: must be object")
                continue

            for bundle_name, bundle_config in tenant_config.items():
                if not isinstance(bundle_config, dict):
                    errors.append(
                        f"Invalid bundle config for {tenant_id}.{bundle_name}: must be object"
                    )
                    continue

                # Validate required fields using DRY utility
                context = f"{tenant_id}.{bundle_name}"
                required_fields = [
                    "allowed_text_detectors",
                    "conflict_resolution",
                    "min_coverage_threshold",
                ]

                field_errors = validate_required_fields(bundle_config, required_fields, context)
                errors.extend(field_errors)

                # Validate data types using DRY utility
                field_types = {
                    "allowed_text_detectors": list,
                    "min_coverage_threshold": (int, float),
                    "pii_handling_enabled": bool,
                    "encryption_enabled": bool,
                    "audit_logging_enabled": bool,
                }

                type_errors = validate_data_types(bundle_config, field_types, context)
                errors.extend(type_errors)

    except (KeyError, TypeError, AttributeError) as e:
        errors.append(f"Validation error: {str(e)}")

    return errors


def validate_json_data(json_content: str) -> tuple[Dict[str, Any] | None, List[str]]:
    """Validate and parse JSON data.
    
    Args:
        json_content: JSON string to validate
        
    Returns:
        Tuple of (parsed_data, validation_errors)
    """
    errors = []
    
    try:
        data = json.loads(json_content)
        return data, errors
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {str(e)}")
        return None, errors
    except (TypeError, ValueError) as e:
        errors.append(f"JSON parsing error: {str(e)}")
        return None, errors

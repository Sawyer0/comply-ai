"""Policy loader utility for OPA policy templates.

This module provides functionality to load and manage OPA policy templates
following SRP - handles ONLY policy file loading and validation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, TypeVar
from functools import lru_cache, wraps

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PolicyLoadError(Exception):
    """Exception raised when policy loading fails."""


def handle_policy_errors(default_return: Any = None, reraise: bool = False):
    """Decorator to handle common policy operation errors following DRY principle."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except PolicyLoadError:
                # Re-raise policy load errors as-is - this is intentional
                # pylint: disable=try-except-raise
                raise
            except (OSError, PermissionError, FileNotFoundError) as e:
                logger.error(
                    "File system error in %s",
                    func.__name__,
                    extra={"error": str(e), "function": func.__name__},
                )
                if reraise:
                    error_msg = f"File operation failed in {func.__name__}: {str(e)}"
                    raise PolicyLoadError(error_msg) from e
                return default_return
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.error(
                    "Data format error in %s",
                    func.__name__,
                    extra={"error": str(e), "function": func.__name__},
                )
                if reraise:
                    error_msg = f"Data format error in {func.__name__}: {str(e)}"
                    raise PolicyLoadError(error_msg) from e
                return default_return
            except (KeyError, TypeError, AttributeError, IndexError) as e:
                logger.error(
                    "Data structure error in %s",
                    func.__name__,
                    extra={"error": str(e), "function": func.__name__},
                )
                if reraise:
                    error_msg = f"Data processing error in {func.__name__}: {str(e)}"
                    raise PolicyLoadError(error_msg) from e
                return default_return

        return wrapper

    return decorator


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


class PolicyLoader:
    """Loads and manages OPA policy templates.

    Single Responsibility: Load policy files and validate their structure.
    Does NOT handle: policy enforcement, OPA communication, tenant management.
    """

    def __init__(self, policies_directory: str = "policies"):
        """Initialize policy loader.

        Args:
            policies_directory: Directory containing policy files
        """
        self.policies_directory = Path(policies_directory)
        self._loaded_policies: Dict[str, str] = {}
        self._policy_metadata: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "Policy loader initialized",
            extra={"directory": str(self.policies_directory)},
        )

    def load_all_policies(self) -> Dict[str, str]:
        """Load all .rego policy files from the policies directory.

        Returns:
            Dictionary mapping policy names to their content

        Raises:
            PolicyLoadError: If policy loading fails
        """
        try:
            policy_files = list(self.policies_directory.glob("*.rego"))

            if not policy_files:
                logger.warning(
                    "No .rego policy files found",
                    extra={"directory": str(self.policies_directory)},
                )
                return {}

            loaded_policies = {}

            for policy_file in policy_files:
                try:
                    policy_name = policy_file.stem
                    policy_content = self._load_policy_file(policy_file)
                    loaded_policies[policy_name] = policy_content

                    # Extract metadata
                    self._policy_metadata[policy_name] = self._extract_policy_metadata(
                        policy_content, policy_file
                    )

                except (OSError, UnicodeDecodeError, PolicyLoadError) as e:
                    logger.error(
                        "Failed to load policy file",
                        extra={"policy_file": str(policy_file), "error": str(e)},
                    )
                    # Continue loading other policies
                    continue

            self._loaded_policies = loaded_policies

            logger.info(
                "Successfully loaded policies",
                extra={
                    "count": len(loaded_policies),
                    "policies": list(loaded_policies.keys()),
                },
            )

            return loaded_policies

        except (OSError, PermissionError) as e:
            error_msg = f"Failed to load policies from {self.policies_directory}: {str(e)}"
            logger.error("Policy loading failed", extra={"error": str(e)})
            raise PolicyLoadError(error_msg) from e

    @handle_policy_errors(default_return=None)
    def load_policy(self, policy_name: str) -> Optional[str]:
        """Load a specific policy by name.

        Args:
            policy_name: Name of the policy to load (without .rego extension)

        Returns:
            Policy content if found, None otherwise
        """
        policy_file = self.policies_directory / f"{policy_name}.rego"

        if not policy_file.exists():
            logger.warning("Policy file not found", extra={"policy_name": policy_name})
            return None

        content = self._load_policy_file(policy_file)
        self._loaded_policies[policy_name] = content

        logger.info("Policy loaded successfully", extra={"policy_name": policy_name})
        return content

    def _load_policy_file(self, policy_file: Path) -> str:
        """Load content from a policy file.

        Args:
            policy_file: Path to the policy file

        Returns:
            Policy file content

        Raises:
            PolicyLoadError: If file cannot be read
        """
        try:
            with open(policy_file, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                raise PolicyLoadError(f"Policy file {policy_file} is empty")

            return content

        except FileNotFoundError as e:
            raise PolicyLoadError(f"Policy file not found: {policy_file}") from e
        except PermissionError as e:
            raise PolicyLoadError(f"Permission denied reading policy file: {policy_file}") from e
        except UnicodeDecodeError as e:
            raise PolicyLoadError(f"Invalid encoding in policy file: {policy_file}") from e

    def _extract_policy_metadata(self, content: str, policy_file: Path) -> Dict[str, Any]:
        """Extract metadata from policy content.

        Args:
            content: Policy file content
            policy_file: Path to policy file

        Returns:
            Dictionary containing policy metadata
        """
        metadata = {
            "file_path": str(policy_file),
            "file_size": len(content),
            "package_name": None,
            "rules_count": 0,
            "imports": [],
        }

        try:
            lines = content.split("\n")

            for line in lines:
                line = line.strip()

                # Extract package name
                if line.startswith("package "):
                    metadata["package_name"] = line.replace("package ", "").strip()

                # Count rules (simple heuristic)
                if (
                    line.endswith(" if {")
                    or line.endswith(" := ")
                    or line == "allow"
                    or line == "deny"
                ):
                    metadata["rules_count"] += 1

                # Extract imports
                if line.startswith("import "):
                    import_statement = line.replace("import ", "").strip()
                    metadata["imports"].append(import_statement)

        except (IndexError, AttributeError) as e:
            logger.warning("Failed to extract policy metadata", extra={"error": str(e)})

        return metadata

    @lru_cache(maxsize=100)
    def parse_policy_structure(self, policy_content: str) -> Dict[str, Any]:
        """Parse and cache policy structure analysis.

        Args:
            policy_content: Policy content to analyze

        Returns:
            Dictionary containing policy structure information
        """
        structure = {
            "package_name": None,
            "rules": [],
            "imports": [],
            "rule_count": 0,
            "complexity_score": 0,
        }

        try:
            lines = policy_content.split("\n")
            complexity = 0

            for line_num, line in enumerate(lines, 1):
                line = line.strip()

                # Extract package name
                if line.startswith("package "):
                    structure["package_name"] = line.replace("package ", "").strip()

                # Extract imports
                if line.startswith("import "):
                    import_statement = line.replace("import ", "").strip()
                    structure["imports"].append(import_statement)

                # Extract rules and calculate complexity
                if " if {" in line or line.endswith(" := ") or line in ["allow", "deny"]:
                    rule_name = line.split(" ")[0] if " " in line else line
                    structure["rules"].append(
                        {
                            "name": rule_name,
                            "line": line_num,
                            "type": self._classify_rule_type(line),
                        }
                    )
                    structure["rule_count"] += 1

                # Add complexity for control structures
                complexity += line.count("if ") + line.count("else") + line.count("count(")

            structure["complexity_score"] = complexity

        except (AttributeError, IndexError, ValueError) as e:
            logger.warning("Failed to parse policy structure", extra={"error": str(e)})

        return structure

    def _classify_rule_type(self, line: str) -> str:
        """Classify the type of policy rule."""
        if line.startswith("default "):
            return "default"
        if " if {" in line:
            return "conditional"
        if " := " in line:
            return "assignment"
        if line in ["allow", "deny"]:
            return "decision"
        return "unknown"

    @lru_cache(maxsize=100)
    def validate_policy_syntax(self, policy_content: str) -> List[str]:
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

    def get_loaded_policies(self) -> Dict[str, str]:
        """Get all currently loaded policies.

        Returns:
            Dictionary mapping policy names to their content
        """
        return self._loaded_policies.copy()

    def get_policy_metadata(self, policy_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific policy.

        Args:
            policy_name: Name of the policy

        Returns:
            Policy metadata if available, None otherwise
        """
        return self._policy_metadata.get(policy_name)

    def get_all_policy_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all loaded policies.

        Returns:
            Dictionary mapping policy names to their metadata
        """
        return self._policy_metadata.copy()

    @handle_policy_errors(default_return=None)
    def load_tenant_policy_data(self) -> Optional[Dict[str, Any]]:
        """Load tenant policy configuration data.

        Returns:
            Tenant policy data if available, None otherwise
        """
        data_file = self.policies_directory / "tenant_policies_data.json"

        if not data_file.exists():
            logger.warning("Tenant policy data file not found")
            return None

        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info("Tenant policy data loaded successfully")
        return data

    @handle_policy_errors(default_return=[])
    def validate_tenant_policy_data(self, data: Dict[str, Any]) -> List[str]:
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

    def get_policy_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded policies.

        Returns:
            Dictionary containing policy loading statistics
        """
        total_rules = 0
        total_complexity = 0
        package_names = []

        for content in self._loaded_policies.values():
            structure = self.parse_policy_structure(content)
            total_rules += structure.get("rule_count", 0)
            total_complexity += structure.get("complexity_score", 0)

            if structure.get("package_name"):
                package_names.append(structure["package_name"])

        return {
            "total_policies": len(self._loaded_policies),
            "policy_names": list(self._loaded_policies.keys()),
            "total_content_size": sum(len(content) for content in self._loaded_policies.values()),
            "total_rules": total_rules,
            "total_complexity_score": total_complexity,
            "average_complexity": (
                total_complexity / len(self._loaded_policies) if self._loaded_policies else 0
            ),
            "package_names": package_names,
            "policies_directory": str(self.policies_directory),
            "metadata_available": len(self._policy_metadata),
        }


# Export only the policy loading functionality
__all__ = [
    "PolicyLoader",
    "PolicyLoadError",
    "validate_required_fields",
    "validate_data_types",
    "handle_policy_errors",
]

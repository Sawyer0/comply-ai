"""
Schema evolution management with backward compatibility validation.

Single responsibility: Manage schema versions and compatibility checking.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import jsonschema
from jsonschema import Draft7Validator

from .base_models import ChangeType, CompatibilityCheck, CompatibilityLevel
from .version_manager import VersionManager

logger = logging.getLogger(__name__)


class SchemaEvolutionManager:
    """
    Schema evolution manager.

    Single responsibility: Handle schema versioning and compatibility validation.
    """

    def __init__(self, schemas_path: Optional[Path] = None):
        """Initialize schema evolution manager."""
        self.schemas_path = schemas_path or Path("config/schemas")
        self.schemas: Dict[str, Dict[str, Dict[str, Any]]] = (
            {}
        )  # schema_name -> version -> definition
        self.version_managers: Dict[str, VersionManager] = {}

        self._load_schemas()

    def _load_schemas(self) -> None:
        """Load schemas from configuration."""
        if not self.schemas_path.exists():
            self._create_default_schemas()
            return

        for schema_file in self.schemas_path.glob("*.json"):
            try:
                with open(schema_file, "r") as f:
                    schema_data = json.load(f)
                    schema_name = schema_data.get("schema_name")
                    if schema_name:
                        self._load_schema(schema_name, schema_data)
            except Exception as e:
                logger.error("Failed to load schema %s: %s", schema_file, str(e))

    def _load_schema(self, schema_name: str, data: Dict[str, Any]) -> None:
        """Load individual schema data."""
        if schema_name not in self.schemas:
            self.schemas[schema_name] = {}
            self.version_managers[schema_name] = VersionManager(
                f"schema_{schema_name}", self.schemas_path / schema_name
            )

        for version_data in data.get("versions", []):
            version = version_data["version"]
            self.schemas[schema_name][version] = version_data.get(
                "schema_definition", {}
            )

    def _create_default_schemas(self) -> None:
        """Create default schema definitions."""
        # Orchestration Request Schema
        orchestration_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "request_id": {"type": "string", "format": "uuid"},
                "tenant_id": {"type": "string", "minLength": 1, "maxLength": 100},
                "content": {"type": "string", "minLength": 1, "maxLength": 10000},
                "detector_types": {"type": "array", "items": {"type": "string"}},
                "processing_mode": {
                    "type": "string",
                    "enum": ["standard", "fast", "thorough"],
                },
                "metadata": {"type": "object"},
            },
            "required": ["tenant_id", "content"],
            "additionalProperties": False,
        }

        self.register_schema(
            "orchestration_request", "1.0.0", orchestration_schema, "system"
        )

        # Analysis Request Schema
        analysis_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "request_id": {"type": "string", "format": "uuid"},
                "tenant_id": {"type": "string", "minLength": 1, "maxLength": 100},
                "orchestration_response": {"type": "object"},
                "analysis_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "frameworks": {"type": "array", "items": {"type": "string"}},
                "metadata": {"type": "object"},
            },
            "required": ["tenant_id", "orchestration_response", "analysis_types"],
            "additionalProperties": False,
        }

        self.register_schema("analysis_request", "1.0.0", analysis_schema, "system")

        # Mapping Request Schema
        mapping_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "request_id": {"type": "string", "format": "uuid"},
                "tenant_id": {"type": "string", "minLength": 1, "maxLength": 100},
                "analysis_response": {"type": "object"},
                "target_frameworks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "mapping_mode": {
                    "type": "string",
                    "enum": ["standard", "fast", "comprehensive"],
                },
                "metadata": {"type": "object"},
            },
            "required": ["tenant_id", "analysis_response", "target_frameworks"],
            "additionalProperties": False,
        }

        self.register_schema("mapping_request", "1.0.0", mapping_schema, "system")

        self._save_all_schemas()
        logger.info("Created default schemas")

    def register_schema(
        self,
        schema_name: str,
        version: str,
        schema_definition: Dict[str, Any],
        created_by: str,
    ) -> None:
        """Register a new schema version."""
        # Validate schema
        try:
            Draft7Validator.check_schema(schema_definition)
        except jsonschema.SchemaError as e:
            raise ValueError(f"Invalid schema definition: {str(e)}")

        if schema_name not in self.schemas:
            self.schemas[schema_name] = {}
            self.version_managers[schema_name] = VersionManager(
                f"schema_{schema_name}", self.schemas_path / schema_name
            )

        self.schemas[schema_name][version] = schema_definition

        # Create version in version manager
        checksum = self.version_managers[schema_name].calculate_checksum(
            schema_definition
        )
        self.version_managers[schema_name].create_version(
            ChangeType.MINOR,
            [f"Registered schema {schema_name} v{version}"],
            created_by,
            checksum,
        )

    def create_new_schema_version(
        self,
        schema_name: str,
        schema_definition: Dict[str, Any],
        change_type: ChangeType,
        changes: List[str],
        created_by: str,
    ) -> str:
        """Create a new version of a schema."""
        if schema_name not in self.version_managers:
            raise ValueError(f"Schema '{schema_name}' not found")

        # Validate schema
        try:
            Draft7Validator.check_schema(schema_definition)
        except jsonschema.SchemaError as e:
            raise ValueError(f"Invalid schema definition: {str(e)}")

        # Create new version
        checksum = self.version_managers[schema_name].calculate_checksum(
            schema_definition
        )
        new_version = self.version_managers[schema_name].create_version(
            change_type, changes, created_by, checksum
        )

        # Store schema definition
        self.schemas[schema_name][new_version] = schema_definition

        # Save to file
        self._save_schema(schema_name)

        return new_version

    def check_compatibility(
        self, schema_name: str, from_version: str, to_version: str
    ) -> CompatibilityCheck:
        """Check compatibility between schema versions."""
        if schema_name not in self.schemas:
            raise ValueError(f"Schema '{schema_name}' not found")

        if from_version not in self.schemas[schema_name]:
            raise ValueError(f"Version '{from_version}' not found")

        if to_version not in self.schemas[schema_name]:
            raise ValueError(f"Version '{to_version}' not found")

        from_schema = self.schemas[schema_name][from_version]
        to_schema = self.schemas[schema_name][to_version]

        return self._analyze_compatibility(
            from_schema, to_schema, from_version, to_version
        )

    def _analyze_compatibility(
        self,
        from_schema: Dict[str, Any],
        to_schema: Dict[str, Any],
        from_version: str,
        to_version: str,
    ) -> CompatibilityCheck:
        """Analyze compatibility between two schemas."""
        breaking_changes = []
        warnings = []

        # Check required fields
        from_required = set(from_schema.get("required", []))
        to_required = set(to_schema.get("required", []))

        # Removed required fields = breaking
        removed_required = from_required - to_required
        if removed_required:
            breaking_changes.extend(
                [f"Removed required field: {field}" for field in removed_required]
            )

        # Added required fields = breaking
        added_required = to_required - from_required
        if added_required:
            breaking_changes.extend(
                [f"Added required field: {field}" for field in added_required]
            )

        # Check properties
        from_properties = set(from_schema.get("properties", {}).keys())
        to_properties = set(to_schema.get("properties", {}).keys())

        # Removed properties = breaking
        removed_properties = from_properties - to_properties
        if removed_properties:
            breaking_changes.extend(
                [f"Removed property: {prop}" for prop in removed_properties]
            )

        # Check type changes
        for prop in from_properties & to_properties:
            from_prop = from_schema["properties"][prop]
            to_prop = to_schema["properties"][prop]

            from_type = from_prop.get("type")
            to_type = to_prop.get("type")

            if from_type != to_type:
                breaking_changes.append(
                    f"Changed type of '{prop}': {from_type} -> {to_type}"
                )

        # Determine compatibility
        if breaking_changes:
            compatibility_level = CompatibilityLevel.BREAKING
            compatible = False
        elif warnings:
            compatibility_level = CompatibilityLevel.PARTIAL
            compatible = True
        else:
            compatibility_level = CompatibilityLevel.FULL
            compatible = True

        return CompatibilityCheck(
            from_version=from_version,
            to_version=to_version,
            compatible=compatible,
            compatibility_level=compatibility_level,
            breaking_changes=breaking_changes,
            warnings=warnings,
            migration_required=not compatible,
            migration_complexity="complex" if len(breaking_changes) > 3 else "simple",
        )

    def validate_data(
        self, data: Dict[str, Any], schema_name: str, version: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """Validate data against a schema version."""
        if schema_name not in self.schemas:
            return False, [f"Schema '{schema_name}' not found"]

        if version is None:
            # Use latest version
            versions = sorted(self.schemas[schema_name].keys())
            version = versions[-1] if versions else None

        if not version or version not in self.schemas[schema_name]:
            return False, [f"Version '{version}' not found"]

        schema_definition = self.schemas[schema_name][version]

        try:
            jsonschema.validate(data, schema_definition)
            return True, []
        except jsonschema.ValidationError as e:
            return False, [str(e)]

    def get_schema_definition(
        self, schema_name: str, version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get schema definition for a specific version."""
        if schema_name not in self.schemas:
            return None

        if version is None:
            versions = sorted(self.schemas[schema_name].keys())
            version = versions[-1] if versions else None

        if not version or version not in self.schemas[schema_name]:
            return None

        return self.schemas[schema_name][version]

    def _save_schema(self, schema_name: str) -> None:
        """Save individual schema to file."""
        try:
            self.schemas_path.mkdir(parents=True, exist_ok=True)

            schema_data = {
                "schema_name": schema_name,
                "versions": [
                    {"version": version, "schema_definition": definition}
                    for version, definition in self.schemas[schema_name].items()
                ],
            }

            schema_file = self.schemas_path / f"{schema_name}.json"
            with open(schema_file, "w") as f:
                json.dump(schema_data, f, indent=2)

        except Exception as e:
            logger.error("Failed to save schema %s: %s", schema_name, str(e))
            raise

    def _save_all_schemas(self) -> None:
        """Save all schemas to files."""
        for schema_name in self.schemas:
            self._save_schema(schema_name)

    def list_schemas(self) -> List[str]:
        """List all available schemas."""
        return list(self.schemas.keys())

    def get_schema_versions(self, schema_name: str) -> List[str]:
        """Get all versions for a schema."""
        if schema_name not in self.schemas:
            return []
        return sorted(self.schemas[schema_name].keys())


# Global schema evolution manager instance
schema_evolution_manager = SchemaEvolutionManager()

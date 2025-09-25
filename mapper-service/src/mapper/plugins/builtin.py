"""
Built-in plugins for the mapper service.

This module provides default implementations of core plugins that come
with the mapper service out of the box.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional
import logging
from pydantic import ValidationError

from .interfaces import (
    PluginType,
    PluginMetadata,
    PluginCapability,
    PluginResult,
    IMappingEnginePlugin,
    IValidationPlugin,
    IPreprocessingPlugin,
    IPostprocessingPlugin,
)


logger = logging.getLogger(__name__)


class DefaultMappingEnginePlugin(IMappingEnginePlugin):
    """Default mapping engine plugin using rule-based mapping."""

    def __init__(self):
        self.taxonomy_mappings: Dict[str, Any] = {}
        self.framework_mappings: Dict[str, Dict[str, Any]] = {}
        self.initialized = False

    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            plugin_id="default_mapping_engine",
            name="Default Mapping Engine",
            version="1.0.0",
            author="Mapper Service Team",
            description="Default rule-based mapping engine for canonical taxonomy and framework mapping",
            plugin_type=PluginType.MAPPING_ENGINE,
            capabilities=[
                PluginCapability(
                    name="canonical_mapping",
                    version="1.0.0",
                    description="Map detector outputs to canonical taxonomy",
                    supported_formats=["json", "dict"],
                    required_config={"taxonomy_file": "string"},
                ),
                PluginCapability(
                    name="framework_mapping",
                    version="1.0.0",
                    description="Map canonical results to compliance frameworks",
                    supported_formats=["json", "dict"],
                    required_config={"framework_mappings": "dict"},
                ),
            ],
            config_schema={
                "taxonomy_file": {
                    "type": "string",
                    "description": "Path to taxonomy mapping file",
                },
                "framework_mappings": {
                    "type": "object",
                    "description": "Framework mapping configurations",
                },
            },
        )

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin with configuration."""
        try:
            # Load taxonomy mappings
            taxonomy_file = config.get("taxonomy_file")
            if taxonomy_file:
                with open(taxonomy_file, "r") as f:
                    self.taxonomy_mappings = json.load(f)

            # Load framework mappings
            self.framework_mappings = config.get("framework_mappings", {})

            self.initialized = True
            logger.info("Default mapping engine plugin initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize default mapping engine: {e}")
            return False

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration."""
        required_keys = ["taxonomy_file"]
        return all(key in config for key in required_keys)

    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        self.taxonomy_mappings.clear()
        self.framework_mappings.clear()
        self.initialized = False
        logger.info("Default mapping engine plugin cleaned up")

    async def map_to_canonical(
        self, detector_output: Dict[str, Any], context: Dict[str, Any]
    ) -> PluginResult:
        """Map detector output to canonical taxonomy."""
        start_time = time.time()

        try:
            if not self.initialized:
                return PluginResult(
                    success=False,
                    error="Plugin not initialized",
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

            # Extract detector type and findings
            detector_type = detector_output.get("detector_type", "unknown")
            findings = detector_output.get("findings", [])

            canonical_results = []

            for finding in findings:
                # Map finding to canonical taxonomy
                canonical_result = self._map_finding_to_canonical(
                    finding, detector_type
                )
                if canonical_result:
                    canonical_results.append(canonical_result)

            return PluginResult(
                success=True,
                data={
                    "canonical_results": canonical_results,
                    "detector_type": detector_type,
                    "total_findings": len(findings),
                    "mapped_findings": len(canonical_results),
                },
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"Error in canonical mapping: {e}")
            return PluginResult(
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    async def map_to_framework(
        self, canonical_result: Dict[str, Any], framework: str, context: Dict[str, Any]
    ) -> PluginResult:
        """Map canonical result to specific compliance framework."""
        start_time = time.time()

        try:
            if not self.initialized:
                return PluginResult(
                    success=False,
                    error="Plugin not initialized",
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

            framework_mapping = self.framework_mappings.get(framework)
            if not framework_mapping:
                return PluginResult(
                    success=False,
                    error=f"Framework {framework} not supported",
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

            # Map canonical results to framework
            framework_results = []
            canonical_findings = canonical_result.get("canonical_results", [])

            for finding in canonical_findings:
                framework_result = self._map_canonical_to_framework(
                    finding, framework_mapping
                )
                if framework_result:
                    framework_results.append(framework_result)

            return PluginResult(
                success=True,
                data={
                    "framework": framework,
                    "framework_results": framework_results,
                    "total_mappings": len(framework_results),
                },
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"Error in framework mapping: {e}")
            return PluginResult(
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    def get_supported_frameworks(self) -> List[str]:
        """Return list of supported compliance frameworks."""
        return list(self.framework_mappings.keys())

    def _map_finding_to_canonical(
        self, finding: Dict[str, Any], detector_type: str
    ) -> Optional[Dict[str, Any]]:
        """Map a single finding to canonical taxonomy."""
        try:
            # Get mapping rules for detector type
            detector_mappings = self.taxonomy_mappings.get(detector_type, {})

            finding_type = finding.get("type", "unknown")
            canonical_mapping = detector_mappings.get(finding_type)

            if not canonical_mapping:
                # Try generic mapping
                canonical_mapping = detector_mappings.get(
                    "default",
                    {
                        "category": "unknown",
                        "subcategory": "unclassified",
                        "risk_level": "medium",
                    },
                )

            return {
                "canonical_category": canonical_mapping.get("category"),
                "canonical_subcategory": canonical_mapping.get("subcategory"),
                "risk_level": canonical_mapping.get("risk_level", "medium"),
                "confidence": finding.get("confidence", 0.5),
                "original_finding": finding,
                "detector_type": detector_type,
            }

        except Exception as e:
            logger.error(f"Error mapping finding to canonical: {e}")
            return None

    def _map_canonical_to_framework(
        self, canonical_finding: Dict[str, Any], framework_mapping: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Map canonical finding to framework-specific format."""
        try:
            category = canonical_finding.get("canonical_category")
            subcategory = canonical_finding.get("canonical_subcategory")

            # Look up framework mapping
            category_mappings = framework_mapping.get("categories", {})
            framework_category = category_mappings.get(category, {})

            if not framework_category:
                return None

            return {
                "framework_control": framework_category.get("control_id"),
                "control_description": framework_category.get("description"),
                "compliance_status": self._determine_compliance_status(
                    canonical_finding
                ),
                "risk_level": canonical_finding.get("risk_level"),
                "evidence": canonical_finding.get("original_finding"),
                "canonical_mapping": {"category": category, "subcategory": subcategory},
            }

        except Exception as e:
            logger.error(f"Error mapping to framework: {e}")
            return None

    def _determine_compliance_status(self, canonical_finding: Dict[str, Any]) -> str:
        """Determine compliance status based on finding."""
        risk_level = canonical_finding.get("risk_level", "medium").lower()
        confidence = canonical_finding.get("confidence", 0.5)

        if risk_level in ["critical", "high"] and confidence > 0.7:
            return "non_compliant"
        elif risk_level == "medium" and confidence > 0.5:
            return "requires_review"
        else:
            return "compliant"


class JSONSchemaValidationPlugin(IValidationPlugin):
    """JSON Schema validation plugin."""

    def __init__(self):
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.initialized = False

    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            plugin_id="json_schema_validator",
            name="JSON Schema Validator",
            version="1.0.0",
            author="Mapper Service Team",
            description="JSON Schema validation plugin for input/output validation",
            plugin_type=PluginType.VALIDATION,
            capabilities=[
                PluginCapability(
                    name="json_validation",
                    version="1.0.0",
                    description="Validate JSON data against JSON Schema",
                    supported_formats=["json", "dict"],
                    required_config={"schemas": "dict"},
                )
            ],
        )

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin with configuration."""
        try:
            self.schemas = config.get("schemas", {})
            self.initialized = True
            logger.info("JSON Schema validation plugin initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize JSON schema validator: {e}")
            return False

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration."""
        return "schemas" in config

    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        self.schemas.clear()
        self.initialized = False
        logger.info("JSON Schema validation plugin cleaned up")

    async def validate_input(self, data: Any, schema: Dict[str, Any]) -> PluginResult:
        """Validate input data against schema."""
        return await self._validate_data(data, schema, "input")

    async def validate_output(self, data: Any, schema: Dict[str, Any]) -> PluginResult:
        """Validate output data against schema."""
        return await self._validate_data(data, schema, "output")

    def get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules supported by this plugin."""
        return {
            "json_schema": {
                "description": "JSON Schema validation",
                "supported_versions": ["draft-07", "draft-2019-09"],
                "features": [
                    "type_validation",
                    "format_validation",
                    "constraint_validation",
                ],
            }
        }

    async def _validate_data(
        self, data: Any, schema: Dict[str, Any], validation_type: str
    ) -> PluginResult:
        """Internal method to validate data against schema."""
        start_time = time.time()

        try:
            if not self.initialized:
                return PluginResult(
                    success=False,
                    error="Plugin not initialized",
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

            # Import jsonschema here to avoid dependency issues
            try:
                import jsonschema
            except ImportError:
                return PluginResult(
                    success=False,
                    error="jsonschema library not available",
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

            # Validate data against schema
            validator = jsonschema.Draft7Validator(schema)
            errors = list(validator.iter_errors(data))

            if errors:
                error_messages = [
                    f"{error.json_path}: {error.message}" for error in errors
                ]
                return PluginResult(
                    success=False,
                    error=f"Validation failed: {'; '.join(error_messages)}",
                    metadata={
                        "validation_type": validation_type,
                        "error_count": len(errors),
                        "errors": error_messages,
                    },
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

            return PluginResult(
                success=True,
                data={"validation_passed": True},
                metadata={
                    "validation_type": validation_type,
                    "schema_version": schema.get("$schema", "unknown"),
                },
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"Error in JSON schema validation: {e}")
            return PluginResult(
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )


class ContentScrubberPlugin(IPreprocessingPlugin):
    """Content scrubbing preprocessing plugin."""

    def __init__(self):
        self.scrubbing_rules: Dict[str, Any] = {}
        self.initialized = False

    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            plugin_id="content_scrubber",
            name="Content Scrubber",
            version="1.0.0",
            author="Mapper Service Team",
            description="Privacy-first content scrubbing for sensitive data removal",
            plugin_type=PluginType.PREPROCESSING,
            capabilities=[
                PluginCapability(
                    name="pii_scrubbing",
                    version="1.0.0",
                    description="Remove PII from content before processing",
                    supported_formats=["text", "json"],
                    required_config={"scrubbing_rules": "dict"},
                )
            ],
        )

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin with configuration."""
        try:
            self.scrubbing_rules = config.get("scrubbing_rules", {})
            self.initialized = True
            logger.info("Content scrubber plugin initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize content scrubber: {e}")
            return False

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration."""
        return True  # Optional configuration

    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        self.scrubbing_rules.clear()
        self.initialized = False
        logger.info("Content scrubber plugin cleaned up")

    async def preprocess(self, data: Any, config: Dict[str, Any]) -> PluginResult:
        """Preprocess input data by scrubbing sensitive content."""
        start_time = time.time()

        try:
            if not self.initialized:
                return PluginResult(
                    success=False,
                    error="Plugin not initialized",
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

            scrubbed_data = await self._scrub_content(data)

            return PluginResult(
                success=True,
                data=scrubbed_data,
                metadata={
                    "scrubbing_applied": True,
                    "original_type": type(data).__name__,
                },
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"Error in content scrubbing: {e}")
            return PluginResult(
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    def get_preprocessing_steps(self) -> List[str]:
        """Get list of preprocessing steps this plugin performs."""
        return ["pii_removal", "sensitive_data_masking", "content_sanitization"]

    async def _scrub_content(self, data: Any) -> Any:
        """Scrub sensitive content from data."""
        if isinstance(data, str):
            return self._scrub_text(data)
        elif isinstance(data, dict):
            return {
                key: await self._scrub_content(value) for key, value in data.items()
            }
        elif isinstance(data, list):
            return [await self._scrub_content(item) for item in data]
        else:
            return data

    def _scrub_text(self, text: str) -> str:
        """Scrub sensitive information from text."""
        import re

        # Basic PII patterns (in production, use more sophisticated detection)
        patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        }

        scrubbed_text = text
        for pattern_name, pattern in patterns.items():
            replacement = f"[{pattern_name.upper()}_REDACTED]"
            scrubbed_text = re.sub(pattern, replacement, scrubbed_text)

        return scrubbed_text


# Registry of built-in plugins
BUILTIN_PLUGINS = [
    DefaultMappingEnginePlugin,
    JSONSchemaValidationPlugin,
    ContentScrubberPlugin,
]


def register_builtin_plugins(plugin_manager) -> None:
    """Register all built-in plugins with the plugin manager."""
    for plugin_class in BUILTIN_PLUGINS:
        try:
            plugin_instance = plugin_class()
            plugin_manager.registry.register_plugin(plugin_instance)
            logger.info(f"Registered built-in plugin: {plugin_class.__name__}")
        except Exception as e:
            logger.error(
                f"Failed to register built-in plugin {plugin_class.__name__}: {e}"
            )

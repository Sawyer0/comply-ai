"""Detector management CLI commands following SRP.

This module provides ONLY detector management CLI commands.
Single Responsibility: Handle CLI commands for detector operations.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any

from shared.utils.correlation import get_correlation_id

logger = logging.getLogger(__name__)


class DetectorCLI:
    """CLI commands for detector management.

    Single Responsibility: Provide CLI interface for detector operations.
    Does NOT handle: business logic, validation, orchestration.
    """

    def __init__(self, orchestration_service):
        """Initialize detector CLI.

        Args:
            orchestration_service: OrchestrationService instance
        """
        self.service = orchestration_service

    async def list_detectors(self, output_format: str = "table") -> str:
        """List all registered detectors.

        Args:
            output_format: Output format (table, json, yaml)

        Returns:
            Formatted detector list
        """
        try:
            detectors = self.service.content_router.get_available_detectors()

            if output_format == "json":
                return json.dumps(detectors, indent=2)
            elif output_format == "yaml":
                import yaml

                return yaml.dump(detectors, default_flow_style=False)
            else:
                # Table format
                if not detectors:
                    return "No detectors registered."

                output = "Registered Detectors:\\n"
                output += "=" * 50 + "\\n"
                for i, detector in enumerate(detectors, 1):
                    config = self.service.content_router.get_detector_config(detector)
                    if config:
                        output += f"{i:2d}. {detector}\\n"
                        output += f"    Endpoint: {config.endpoint}\\n"
                        output += f"    Timeout: {config.timeout_ms}ms\\n"
                        output += f"    Enabled: {config.enabled}\\n"
                        output += f"    Content Types: {', '.join(config.supported_content_types)}\\n\\n"
                    else:
                        output += f"{i:2d}. {detector} (no config)\\n\\n"

                return output

        except Exception as e:
            logger.error("Failed to list detectors: %s", str(e))
            return f"Error listing detectors: {str(e)}"

    async def register_detector(
        self,
        detector_id: str,
        endpoint: str,
        detector_type: str,
        timeout_ms: int = 5000,
        max_retries: int = 3,
        content_types: Optional[List[str]] = None,
    ) -> str:
        """Register a new detector.

        Args:
            detector_id: Detector identifier
            endpoint: Detector endpoint URL
            detector_type: Type of detector
            timeout_ms: Timeout in milliseconds
            max_retries: Maximum retry attempts
            content_types: Supported content types

        Returns:
            Result message
        """
        try:
            success = await self.service.register_detector(
                detector_id=detector_id,
                endpoint=endpoint,
                detector_type=detector_type,
                timeout_ms=timeout_ms,
                max_retries=max_retries,
                supported_content_types=content_types or ["text"],
            )

            if success:
                return f"Successfully registered detector '{detector_id}' at {endpoint}"
            else:
                return f"Failed to register detector '{detector_id}'"

        except Exception as e:
            logger.error("Failed to register detector: %s", str(e))
            return f"Error registering detector: {str(e)}"

    async def unregister_detector(self, detector_id: str) -> str:
        """Unregister a detector.

        Args:
            detector_id: Detector identifier

        Returns:
            Result message
        """
        try:
            success = await self.service.unregister_detector(detector_id)

            if success:
                return f"Successfully unregistered detector '{detector_id}'"
            else:
                return f"Failed to unregister detector '{detector_id}' (not found)"

        except Exception as e:
            logger.error("Failed to unregister detector: %s", str(e))
            return f"Error unregistering detector: {str(e)}"

    async def detector_health(self, detector_id: Optional[str] = None) -> str:
        """Check detector health.

        Args:
            detector_id: Optional specific detector to check

        Returns:
            Health status report
        """
        try:
            if not self.service.health_monitor:
                return "Health monitoring is disabled"

            if detector_id:
                # Check specific detector
                health_check = await self.service.health_monitor.check_service_health(
                    detector_id
                )
                return f"Detector '{detector_id}': {health_check.status.value}"
            else:
                # Check all detectors
                health_checks = await self.service.health_monitor.check_all_services()

                if not health_checks:
                    return "No detectors registered for health monitoring"

                output = "Detector Health Status:\\n"
                output += "=" * 50 + "\\n"

                for detector_id, health_check in health_checks.items():
                    status_icon = "✓" if health_check.status.value == "healthy" else "✗"
                    output += (
                        f"{status_icon} {detector_id}: {health_check.status.value}"
                    )

                    if health_check.response_time_ms:
                        output += f" ({health_check.response_time_ms}ms)"

                    if health_check.error_message:
                        output += f" - {health_check.error_message}"

                    output += "\\n"

                return output

        except Exception as e:
            logger.error("Failed to check detector health: %s", str(e))
            return f"Error checking detector health: {str(e)}"

    async def detector_config(self, detector_id: str) -> str:
        """Show detector configuration.

        Args:
            detector_id: Detector identifier

        Returns:
            Configuration details
        """
        try:
            config = self.service.content_router.get_detector_config(detector_id)

            if not config:
                return f"Detector '{detector_id}' not found"

            config_dict = {
                "name": config.name,
                "endpoint": config.endpoint,
                "timeout_ms": config.timeout_ms,
                "max_retries": config.max_retries,
                "supported_content_types": config.supported_content_types,
                "enabled": config.enabled,
            }

            return json.dumps(config_dict, indent=2)

        except Exception as e:
            logger.error("Failed to get detector config: %s", str(e))
            return f"Error getting detector config: {str(e)}"

    async def test_detector(
        self, detector_id: str, test_content: str = "test content"
    ) -> str:
        """Test a detector with sample content.

        Args:
            detector_id: Detector identifier
            test_content: Content to test with

        Returns:
            Test result
        """
        try:
            # This would require implementing a test method in the service
            # For now, just return a placeholder
            return (
                f"Test functionality for detector '{detector_id}' not yet implemented"
            )

        except Exception as e:
            logger.error("Failed to test detector: %s", str(e))
            return f"Error testing detector: {str(e)}"


# Export only the detector CLI functionality
__all__ = [
    "DetectorCLI",
]

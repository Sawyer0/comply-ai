"""Detector management CLI commands following SRP.

This module provides ONLY detector management CLI commands.
Single Responsibility: Handle CLI commands for detector operations.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Optional, Sequence

from shared.exceptions.base import BaseServiceException

from ..service import DetectorRegistrationConfig, OrchestrationService

try:
    import yaml
except ImportError:  # pragma: no cover - yaml optional for CLI formatting
    yaml = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectorRegistrationOptions:
    """Input parameters for registering a detector via the CLI."""

    detector_id: str
    endpoint: str
    detector_type: str
    timeout_ms: int = 5000
    max_retries: int = 3
    content_types: Optional[Sequence[str]] = None


class DetectorCLI:
    """CLI commands for detector management."""

    def __init__(self, orchestration_service: OrchestrationService) -> None:
        self.service = orchestration_service

    async def list_detectors(self, output_format: str = "table") -> str:
        """List all registered detectors."""

        router = getattr(self.service, "content_router", None)
        if not router:
            return "Content router not initialized"

        detector_names = router.get_available_detectors()

        if output_format == "json":
            return json.dumps(detector_names, indent=2)

        if output_format == "yaml":
            if yaml is None:
                return "PyYAML not installed; use --format json instead"
            return yaml.dump(detector_names, default_flow_style=False)  # type: ignore[arg-type]

        return self._format_detector_table(detector_names)

    async def register_detector(
        self, options: DetectorRegistrationOptions
    ) -> str:
        """Register a new detector."""

        registration = DetectorRegistrationConfig(
            detector_id=options.detector_id,
            endpoint=options.endpoint,
            detector_type=options.detector_type,
            timeout_ms=options.timeout_ms,
            max_retries=options.max_retries,
            supported_content_types=list(options.content_types or ["text"]),
        )

        try:
            was_registered = await self.service.register_detector(registration)
        except (BaseServiceException, ValueError, RuntimeError) as exc:
            return self._format_error("register detector", exc)

        if not was_registered:
            return (
                f"Failed to register detector '{options.detector_id}'"
            )

        return (
            f"Successfully registered detector '{options.detector_id}' at {options.endpoint}"
        )

    async def unregister_detector(self, detector_id: str) -> str:
        """Unregister a detector."""

        try:
            was_removed = await self.service.unregister_detector(detector_id)
        except (BaseServiceException, RuntimeError) as exc:
            return self._format_error("unregister detector", exc)

        if not was_removed:
            return f"Detector '{detector_id}' not found"

        return f"Successfully unregistered detector '{detector_id}'"

    async def detector_health(self, detector_id: Optional[str] = None) -> str:
        """Check detector health."""

        monitor = getattr(self.service, "health_monitor", None)
        if not monitor:
            return "Health monitoring is disabled"

        try:
            if detector_id:
                health_check = await monitor.check_service_health(detector_id)
                status = health_check.status.value if health_check else "unknown"
                return f"Detector '{detector_id}': {status}"

            health_checks = await monitor.check_all_services()
        except (BaseServiceException, RuntimeError) as exc:
            return self._format_error("retrieve detector health", exc)

        if not health_checks:
            return "No detectors registered for health monitoring"

        lines = ["Detector Health Status:", "=" * 50]
        for detected_name, health_check in health_checks.items():
            status = health_check.status.value
            parts = [f"{detected_name}: {status}"]
            if health_check.response_time_ms:
                parts.append(f"{health_check.response_time_ms}ms")
            if health_check.error_message:
                parts.append(health_check.error_message)
            lines.append(" - ".join(parts))

        return "\n".join(lines)

    async def detector_config(self, detector_id: str) -> str:
        """Show detector configuration."""

        router = getattr(self.service, "content_router", None)
        if not router:
            return "Content router not initialized"

        config = router.get_detector_config(detector_id)
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

    async def test_detector(
        self,
        detector_id: str,
        test_content: str = "test content",
    ) -> str:
        """Test a detector with sample content."""

        return (
            "Detector self-test is not implemented yet"
            f" (requested detector='{detector_id}', sample='{test_content}')."
        )

    def _format_detector_table(self, detector_names: Sequence[str]) -> str:
        """Create a human-friendly table of registered detectors."""

        if not detector_names:
            return "No detectors registered."

        router = getattr(self.service, "content_router", None)
        lines = ["Registered Detectors:", "=" * 50]
        for index, name in enumerate(detector_names, start=1):
            config = router.get_detector_config(name) if router else None
            lines.append(f"{index:2d}. {name}")
            if config:
                lines.append(f"    Endpoint: {config.endpoint}")
                lines.append(f"    Timeout: {config.timeout_ms}ms")
                lines.append(f"    Enabled: {config.enabled}")
                lines.append(
                    "    Content Types: "
                    + ", ".join(config.supported_content_types)
                )
            else:
                lines.append("    Configuration not available")
        return "\n".join(lines)

    def _format_error(self, action: str, exc: Exception) -> str:
        """Log an error and return a friendly message."""

        logger.error("Failed to %s: %s", action, exc)
        return f"Error attempting to {action}: {exc}"


__all__ = ["DetectorCLI", "DetectorRegistrationOptions"]

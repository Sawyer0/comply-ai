"""Detector management CLI commands following SRP.

This module provides ONLY detector management CLI commands.
Single Responsibility: Handle CLI commands for detector operations.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, Dict, Optional, Sequence

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

    async def list_detectors(
        self,
        output_format: str = "table",
        *,
        tenant_id: Optional[str] = None,
        detector_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> str:
        """List detectors from the persistent registry, with optional filters."""

        records = await self.service.list_detectors(
            tenant_id=tenant_id,
            detector_type=detector_type,
            status=status,
        )

        if not records:
            return "No detectors found."

        if output_format == "json":
            payload = [
                {
                    "id": r.id,
                    "detector_name": r.detector_name,
                    "detector_type": r.detector_type,
                    "endpoint_url": r.endpoint_url,
                    "status": r.status,
                    "tenant_id": r.tenant_id,
                }
                for r in records
            ]
            return json.dumps(payload, indent=2)

        if output_format == "yaml":
            if yaml is None:
                return "PyYAML not installed; use --format json instead"
            payload = [
                {
                    "id": r.id,
                    "detector_name": r.detector_name,
                    "detector_type": r.detector_type,
                    "endpoint_url": r.endpoint_url,
                    "status": r.status,
                    "tenant_id": r.tenant_id,
                }
                for r in records
            ]
            return yaml.dump(payload, default_flow_style=False)  # type: ignore[arg-type]

        # Table output: reuse router-based formatting when available.
        detector_names = [r.detector_name for r in records]
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

    async def show_detector(self, detector_id: str) -> str:
        """Show detector details from the persistent registry."""

        record = await self.service.get_detector(detector_id)
        if not record:
            return f"Detector '{detector_id}' not found"

        info = {
            "id": record.id,
            "detector_name": record.detector_name,
            "detector_type": record.detector_type,
            "endpoint_url": record.endpoint_url,
            "status": record.status,
            "version": record.version,
            "tenant_id": record.tenant_id,
            "capabilities": record.capabilities,
            "health_status": record.health_status,
            "last_health_check": record.last_health_check,
            "response_time_ms": record.response_time_ms,
            "error_rate": record.error_rate,
            "configuration": record.configuration,
        }

        return json.dumps(info, indent=2)

    async def update_detector(
        self,
        detector_id: str,
        *,
        status: Optional[str] = None,
        endpoint: Optional[str] = None,
        content_types: Optional[Sequence[str]] = None,
    ) -> str:
        """Update detector metadata in the persistent registry."""

        fields: Dict[str, Any] = {}
        if status is not None:
            fields["status"] = status
        if endpoint is not None:
            fields["endpoint_url"] = endpoint
            fields["health_check_url"] = endpoint.rstrip("/") + "/health"
        if content_types is not None:
            fields["capabilities"] = list(content_types)

        if not fields:
            return "No fields provided to update"

        record = await self.service.get_detector(detector_id)
        if not record:
            return f"Detector '{detector_id}' not found"

        updated = await self.service.update_detector(
            detector_id,
            tenant_id=record.tenant_id,
            fields=fields,
        )
        if not updated:
            return f"Detector '{detector_id}' not found"

        return f"Updated detector '{detector_id}'"

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

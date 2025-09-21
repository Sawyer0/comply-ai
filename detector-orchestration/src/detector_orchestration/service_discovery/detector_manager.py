"""Background task that discovers and registers detectors at runtime."""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from detector_orchestration.config import DetectorEndpoint, Settings
from detector_orchestration.health_monitor import HealthMonitor
from detector_orchestration.registry import DetectorRegistration, DetectorRegistry

from .client import DiscoveredService, ServiceDiscoveryClient

logger = logging.getLogger(__name__)


class DetectorDiscoveryManager:
    """Continuously discovers detector services and keeps the registry updated."""

    def __init__(
        self,
        *,
        settings: Settings,
        registry: DetectorRegistry,
        health_monitor: HealthMonitor,
        discovery_client: Optional[ServiceDiscoveryClient] = None,
        interval_seconds: int = 60,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._health_monitor = health_monitor
        self._discovery_client = discovery_client or ServiceDiscoveryClient(settings)
        self._interval_seconds = interval_seconds
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()

    async def start_auto_discovery(self) -> None:
        """Start the background discovery loop if not already running."""

        if self._task and not self._task.done():
            return
        self._stop_event.clear()
        logger.info("Starting automatic detector discovery")
        self._task = asyncio.create_task(
            self._discovery_loop(), name="detector-discovery"
        )

    async def stop_auto_discovery(self) -> None:
        """Stop the background discovery loop."""

        if not self._task:
            return
        logger.info("Stopping automatic detector discovery")
        self._stop_event.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            logger.debug("Detector discovery task cancelled")
        finally:
            self._task = None

    async def discover_once(self) -> List[str]:
        """Run a single discovery pass and return the known detectors."""

        await self._perform_discovery()
        return list(self._settings.detectors.keys())

    async def _discovery_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                await self._perform_discovery()
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self._interval_seconds
                    )
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            raise
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Unhandled error in detector discovery loop")
        finally:
            self._stop_event.set()

    async def _perform_discovery(self) -> None:
        services = await self._discovery_client.discover_services("detector")
        if not services:
            logger.debug("No new detectors discovered")
            return
        for service in services:
            await self._register_service(service)

    async def _register_service(self, service: DiscoveredService) -> None:
        metadata = service.metadata
        registration = DetectorRegistration(
            name=service.name,
            endpoint=service.endpoint,
            timeout_ms=int(metadata.get("timeout_ms", 5000)),
            max_retries=int(metadata.get("max_retries", 1)),
            supported_content_types=list(
                metadata.get("supported_content_types", ["text"])
            ),
            auth=dict(metadata.get("auth", {})),
        )

        existing = self._settings.detectors.get(service.name)
        if existing:
            if self._needs_update(existing, registration):
                logger.info("Updating detector configuration for %s", service.name)
                self._registry.update(service.name, registration)
            return

        logger.info("Registering detector discovered from %s", service.kind)
        self._registry.register(registration)
        await self._verify_health(service.name)

    async def _verify_health(self, detector_name: str) -> None:
        try:
            await asyncio.sleep(2)
            healthy = self._health_monitor.is_healthy(detector_name)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Health monitor check failed for %s", detector_name)
            return

        if healthy:
            logger.info("Detector %s is healthy after registration", detector_name)
        else:
            logger.warning(
                "Detector %s registered successfully but failed health check",
                detector_name,
            )

    @staticmethod
    def _needs_update(
        existing: DetectorEndpoint, registration: DetectorRegistration
    ) -> bool:
        return (
            existing.endpoint != registration.endpoint
            or existing.timeout_ms != registration.timeout_ms
            or existing.max_retries != registration.max_retries
            or existing.supported_content_types
            != registration.supported_content_types
            or existing.auth != registration.auth
        )

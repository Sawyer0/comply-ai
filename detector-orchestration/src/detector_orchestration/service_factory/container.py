"""Service container abstractions for orchestrator dependencies."""

from __future__ import annotations

import logging
from typing import Dict, Protocol

logger = logging.getLogger(__name__)


class ServiceDependency(Protocol):
    """Protocol for service dependencies that need initialization and cleanup."""

    async def start(self) -> None:
        """Start the service."""

    async def stop(self) -> None:
        """Stop the service."""

    async def health_check(self) -> bool:
        """Check if the service is healthy."""


class ServiceContainer:
    """Container for managing service dependencies and their lifecycle."""

    def __init__(self, settings) -> None:
        self.settings = settings
        self._services: Dict[str, ServiceDependency] = {}
        self._initialized = False

    def register_service(self, name: str, service: ServiceDependency) -> None:
        """Register a service in the container."""
        self._services[name] = service
        logger.info("Registered service: %s", name)

    async def initialize_all(self) -> None:
        """Initialize all registered services."""
        if self._initialized:
            return

        logger.info("Initializing service container...")
        for name, service in self._services.items():
            try:
                logger.info("Starting service: %s", name)
                await service.start()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("Failed to start service %s: %s", name, exc)
                raise

        self._initialized = True
        logger.info("Service container initialized successfully")

    async def shutdown_all(self) -> None:
        """Shutdown all registered services."""
        if not self._initialized:
            return

        logger.info("Shutting down service container...")
        for name in reversed(list(self._services.keys())):
            service = self._services[name]
            try:
                logger.info("Stopping service: %s", name)
                await service.stop()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("Failed to stop service %s: %s", name, exc)

        self._initialized = False
        logger.info("Service container shutdown complete")

    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all services."""
        results: Dict[str, bool] = {}
        for name, service in self._services.items():
            try:
                results[name] = await service.health_check()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("Health check failed for service %s: %s", name, exc)
                results[name] = False
        return results


__all__ = ["ServiceDependency", "ServiceContainer"]

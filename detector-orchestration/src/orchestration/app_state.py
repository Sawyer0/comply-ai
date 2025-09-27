"""Application-level state helpers for the detector orchestration service.

This module isolates the service container that keeps the live OrchestrationService
instance. Keeping this separate from FastAPI app wiring prevents circular imports
between the API dependency module and main application module.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .service import OrchestrationService


class ServiceContainer:
    """Hold long-lived service singletons for the application."""

    def __init__(self) -> None:
        self._orchestration_service: Optional["OrchestrationService"] = None

    def get_orchestration_service(self) -> Optional["OrchestrationService"]:
        """Return the active orchestration service instance, if any."""

        return self._orchestration_service

    def set_orchestration_service(self, service: "OrchestrationService") -> None:
        """Store the active orchestration service instance."""

        self._orchestration_service = service

    def clear(self) -> None:
        """Clear all stored singletons (useful during shutdown/tests)."""

        self._orchestration_service = None


service_container = ServiceContainer()

__all__ = ["ServiceContainer", "service_container"]

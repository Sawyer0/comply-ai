"""Service discovery utilities for the detector orchestration service."""

from __future__ import annotations

from .client import DiscoveredService, ServiceDiscoveryClient
from .config_reloader import ConfigurationHotReloader
from .detector_manager import DetectorDiscoveryManager
from .health_reporter import ServiceHealthReporter

from detector_orchestration.config import Settings
from detector_orchestration.health_monitor import HealthMonitor
from detector_orchestration.metrics import OrchestrationMetricsCollector
from detector_orchestration.registry import DetectorRegistry

__all__ = [
    "DiscoveredService",
    "ServiceDiscoveryClient",
    "ConfigurationHotReloader",
    "DetectorDiscoveryManager",
    "ServiceHealthReporter",
    "create_service_discovery",
    "create_detector_discovery",
    "create_config_reloader",
    "create_health_reporter",
]


def create_service_discovery(settings: Settings) -> ServiceDiscoveryClient:
    """Factory helper used by the service factory mixin."""

    return ServiceDiscoveryClient(settings)


def create_detector_discovery(
    settings: Settings,
    registry: DetectorRegistry,
    health_monitor: HealthMonitor,
) -> DetectorDiscoveryManager:
    """Factory helper for the background detector discovery manager."""

    return DetectorDiscoveryManager(
        settings=settings,
        registry=registry,
        health_monitor=health_monitor,
    )


def create_config_reloader(settings: Settings) -> ConfigurationHotReloader:
    """Factory helper for the configuration hot reloader."""

    # The hot reloader only needs access to settings today, but we keep the
    # signature aligned with the previous helper function for backwards compat.
    return ConfigurationHotReloader(settings=settings)


def create_health_reporter(
    settings: Settings,
    health_monitor: HealthMonitor,
    metrics_collector: OrchestrationMetricsCollector,
) -> ServiceHealthReporter:
    """Factory helper returning a cached service health reporter."""

    return ServiceHealthReporter(
        settings=settings,
        health_monitor=health_monitor,
        metrics_collector=metrics_collector,
    )

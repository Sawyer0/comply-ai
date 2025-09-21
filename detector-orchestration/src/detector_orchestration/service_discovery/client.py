"""Service discovery client that merges multiple discovery sources with caching."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Sequence, Tuple

from detector_orchestration.config import DetectorEndpoint, Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiscoveredService:
    """Lightweight representation of a discovered service endpoint."""

    name: str
    endpoint: str
    kind: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_detector(
        cls, name: str, detector: DetectorEndpoint
    ) -> "DiscoveredService":
        """Construct a discovery record from a configured detector."""

        return cls(
            name=name,
            endpoint=detector.endpoint,
            kind="detector",
            metadata={
                "timeout_ms": detector.timeout_ms,
                "max_retries": detector.max_retries,
                "supported_content_types": detector.supported_content_types,
                "auth": detector.auth,
            },
        )


Provider = Callable[[str], Awaitable[List[DiscoveredService]]]


class ServiceDiscoveryClient:
    """Aggregates discovery information from configuration, DNS, and APIs."""

    def __init__(
        self,
        settings: Settings,
        *,
        cache_ttl_seconds: int = 300,
    ):
        self._settings = settings
        self._cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._cache: Dict[str, Tuple[datetime, List[DiscoveredService]]] = {}
        self._providers: Dict[str, Sequence[Provider]] = {
            "detector": (
                self._discover_via_config,
                self._discover_via_dns,
                self._discover_via_api,
            )
        }

    async def discover_services(
        self, service_kind: str = "detector"
    ) -> List[DiscoveredService]:
        """Return available services of ``service_kind`` with basic caching."""

        cached = self._cache.get(service_kind)
        now = datetime.now(timezone.utc)
        if cached and now - cached[0] < self._cache_ttl:
            logger.debug("Using cached discovery results for %s", service_kind)
            return list(cached[1])

        providers = self._providers.get(service_kind, ())
        discovered: List[DiscoveredService] = []
        for provider in providers:
            try:
                services = await provider(service_kind)
                discovered.extend(services)
            except Exception:  # pylint: disable=broad-exception-caught
                logger.exception(
                    "Service discovery provider %s failed for %s",
                    provider.__name__,
                    service_kind,
                )

        unique = self._deduplicate(discovered)
        self._cache[service_kind] = (now, unique)
        return list(unique)

    def clear_cache(self) -> None:
        """Invalidate cached discovery results."""

        self._cache.clear()

    async def _discover_via_config(
        self, service_kind: str
    ) -> List[DiscoveredService]:
        if service_kind != "detector":
            return []
        return [
            DiscoveredService.from_detector(name, detector)
            for name, detector in self._settings.detectors.items()
        ]

    async def _discover_via_dns(self, _: str) -> List[DiscoveredService]:
        """Placeholder for DNS based discovery. Returns an empty list for now."""

        return []

    async def _discover_via_api(self, _: str) -> List[DiscoveredService]:
        """Placeholder for future API/service-registry discovery."""

        return []

    def _deduplicate(
        self, services: Iterable[DiscoveredService]
    ) -> List[DiscoveredService]:
        unique: Dict[str, DiscoveredService] = {}
        for service in services:
            unique[service.name] = service
        return list(unique.values())

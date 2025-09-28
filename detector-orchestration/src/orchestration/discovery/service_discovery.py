"""Service discovery functionality following SRP.

This module provides ONLY service discovery - finding and registering detector services.
Single Responsibility: Maintain registry of available detector services and their endpoints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from shared.utils.correlation import get_correlation_id

from ..utils.registry import run_registry_operation

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ServiceMetadata:
    """Typed metadata stored alongside service registrations."""

    timeout_ms: int = 5000
    max_retries: int = 3
    supported_content_types: List[str] = field(default_factory=list)
    analyze_path: Optional[str] = "/analyze"
    response_parser: Optional[str] = None
    auth_headers: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to a dictionary suitable for persistence."""

        payload: Dict[str, Any] = {
            "timeout_ms": self.timeout_ms,
            "max_retries": self.max_retries,
            "supported_content_types": list(self.supported_content_types),
            "analyze_path": self.analyze_path,
            "response_parser": self.response_parser,
        }
        if self.auth_headers is not None:
            payload["auth_headers"] = dict(self.auth_headers)
        return payload

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ServiceMetadata":
        """Create metadata from a raw dictionary."""

        if not data:
            return cls()
        return cls(
            timeout_ms=int(data.get("timeout_ms", 5000)),
            max_retries=int(data.get("max_retries", 3)),
            supported_content_types=list(data.get("supported_content_types", [])),
            analyze_path=data.get("analyze_path", "/analyze"),
            response_parser=data.get("response_parser"),
            auth_headers=data.get("auth_headers"),
        )


@dataclass(slots=True)
class ServiceRegistration:
    """Parameters required to register a detector service."""

    service_id: str
    endpoint_url: str
    service_type: str
    version: str = "1.0.0"
    metadata: Optional[Dict[str, Any]] = None

    def normalized_metadata(self) -> Dict[str, Any]:
        """Return metadata as a concrete mapping."""

        return dict(self.metadata or {})

    def log_context(self) -> Dict[str, Any]:
        """Return context information used for registry logging."""

        return {
            "service_id": self.service_id,
            "service_type": self.service_type,
            "endpoint_url": self.endpoint_url,
            "version": self.version,
        }


@dataclass(slots=True)
class ServiceEndpoint:
    """Service endpoint information - data structure only."""

    service_id: str
    endpoint_url: str
    service_type: str
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def from_registration(cls, registration: ServiceRegistration) -> "ServiceEndpoint":
        """Create an endpoint instance from registration parameters."""

        return cls(
            service_id=registration.service_id,
            endpoint_url=registration.endpoint_url,
            service_type=registration.service_type,
            version=registration.version,
            metadata=registration.normalized_metadata(),
        )


class ServiceDiscoveryManager:
    """Manages discovery and registration of detector services.

    Single Responsibility: Maintain service registry and provide service lookup.
    Does NOT handle: health monitoring, load balancing, circuit breakers.
    """

    def __init__(self, service_ttl_minutes: int = 30):
        """Initialize service discovery manager.

        Args:
            service_ttl_minutes: Time-to-live for service registrations in minutes
        """

        self.service_ttl_minutes = service_ttl_minutes
        self._service_registry: Dict[str, ServiceEndpoint] = {}
        self._service_types: Dict[str, List[str]] = {}

    def register_service(self, registration: ServiceRegistration) -> bool:
        """Register a service in the discovery registry."""

        context = registration.log_context()

        def _operation() -> bool:
            endpoint = ServiceEndpoint.from_registration(registration)
            self._service_registry[registration.service_id] = endpoint

            if registration.service_type not in self._service_types:
                self._service_types[registration.service_type] = []

            if registration.service_id not in self._service_types[registration.service_type]:
                self._service_types[registration.service_type].append(registration.service_id)

            return True

        return run_registry_operation(
            _operation,
            logger=logger,
            success_message="Registered service %s",
            success_args=(registration.service_id,),
            error_message="Failed to register service %s",
            error_args=(registration.service_id,),
            log_context=context,
        )

    def unregister_service(self, service_id: str) -> bool:
        """Unregister a service from the discovery registry."""

        context = {
            "service_id": service_id,
        }

        def _operation() -> bool:
            if service_id not in self._service_registry:
                return False

            endpoint = self._service_registry.pop(service_id)

            service_type = endpoint.service_type
            if service_type in self._service_types:
                self._service_types[service_type] = [
                    sid for sid in self._service_types[service_type] if sid != service_id
                ]

                if not self._service_types[service_type]:
                    self._service_types.pop(service_type)

            return True

        return run_registry_operation(
            _operation,
            logger=logger,
            success_message="Unregistered service %s",
            success_args=(service_id,),
            error_message="Failed to unregister service %s",
            error_args=(service_id,),
            log_context=context,
        )

    def discover_services(self, service_type: Optional[str] = None) -> List[ServiceEndpoint]:
        """Discover services by type.

        Args:
            service_type: Optional service type to filter by

        Returns:
            List of service endpoints matching the criteria
        """

        self._cleanup_expired_services()

        if service_type is None:
            return list(self._service_registry.values())

        service_ids = self._service_types.get(service_type, [])
        return [
            self._service_registry[sid]
            for sid in service_ids
            if sid in self._service_registry
        ]

    def get_service(self, service_id: str) -> Optional[ServiceEndpoint]:
        """Get a specific service by ID."""

        self._cleanup_expired_services()
        return self._service_registry.get(service_id)

    def update_service_heartbeat(self, service_id: str) -> bool:
        """Update the last seen timestamp for a service."""

        if service_id in self._service_registry:
            self._service_registry[service_id].last_seen = datetime.utcnow()
            return True
        return False

    def get_service_types(self) -> List[str]:
        """Get list of all registered service types."""

        self._cleanup_expired_services()
        return list(self._service_types.keys())

    def get_service_count(self, service_type: Optional[str] = None) -> int:
        """Get count of registered services."""

        self._cleanup_expired_services()

        if service_type is None:
            return len(self._service_registry)

        return len(self._service_types.get(service_type, []))

    def _cleanup_expired_services(self) -> None:
        """Remove expired services from the registry."""

        current_time = datetime.utcnow()
        ttl_delta = timedelta(minutes=self.service_ttl_minutes)

        expired_services: List[str] = []
        for service_id, endpoint in self._service_registry.items():
            if current_time - endpoint.last_seen > ttl_delta:
                expired_services.append(service_id)

        for service_id in expired_services:
            logger.info(
                "Removing expired service %s",
                service_id,
                extra={"service_id": service_id, "correlation_id": get_correlation_id()},
            )
            self.unregister_service(service_id)

    def get_registry_status(self) -> Dict[str, Any]:
        """Get status information about the service registry."""

        self._cleanup_expired_services()

        return {
            "total_services": len(self._service_registry),
            "service_types": dict(self._service_types),
            "service_type_counts": {
                stype: len(sids) for stype, sids in self._service_types.items()
            },
            "ttl_minutes": self.service_ttl_minutes,
            "last_cleanup": datetime.utcnow().isoformat(),
        }


__all__ = [
    "ServiceDiscoveryManager",
    "ServiceEndpoint",
    "ServiceMetadata",
    "ServiceRegistration",
]

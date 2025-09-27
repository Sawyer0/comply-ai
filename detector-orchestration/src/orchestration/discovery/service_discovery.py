"""Service discovery functionality following SRP.

This module provides ONLY service discovery - finding and registering detector services.
Single Responsibility: Maintain registry of available detector services and their endpoints.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from shared.utils.correlation import get_correlation_id
from ..utils.registry import run_registry_operation

logger = logging.getLogger(__name__)


@dataclass
class ServiceMetadata:
    """Typed metadata stored alongside service registrations."""

    timeout_ms: int = 5000
    max_retries: int = 3
    supported_content_types: List[str] = field(default_factory=list)
    analyze_path: Optional[str] = "/analyze"
    response_parser: Optional[str] = None
    auth_headers: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to a dictionary."""
        payload: Dict[str, Any] = {
            "timeout_ms": self.timeout_ms,
            "max_retries": self.max_retries,
            "supported_content_types": self.supported_content_types,
            "analyze_path": self.analyze_path,
            "response_parser": self.response_parser,
        }
        if self.auth_headers is not None:
            payload["auth_headers"] = self.auth_headers
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


class ServiceEndpoint:
    """Service endpoint information - data structure only."""

    def __init__(
        self,
        service_id: str,
        endpoint_url: str,
        service_type: str,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.service_id = service_id
        self.endpoint_url = endpoint_url
        self.service_type = service_type
        self.version = version
        self.metadata = metadata or {}
        self.registered_at = datetime.utcnow()
        self.last_seen = datetime.utcnow()


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
        self._service_types: Dict[str, List[str]] = (
            {}
        )  # service_type -> list of service_ids

    def register_service(
        self,
        service_id: str,
        endpoint_url: str,
        service_type: str,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register a service in the discovery registry."""

        context = {
            "service_id": service_id,
            "service_type": service_type,
            "endpoint_url": endpoint_url,
            "version": version,
        }

        def _operation() -> bool:
            endpoint = ServiceEndpoint(
                service_id=service_id,
                endpoint_url=endpoint_url,
                service_type=service_type,
                version=version,
                metadata=metadata,
            )

            self._service_registry[service_id] = endpoint

            if service_type not in self._service_types:
                self._service_types[service_type] = []

            if service_id not in self._service_types[service_type]:
                self._service_types[service_type].append(service_id)

            return True

        return run_registry_operation(
            _operation,
            logger=logger,
            success_message="Registered service %s of type %s at %s",
            success_args=(service_id, service_type, endpoint_url),
            error_message="Failed to register service %s",
            error_args=(service_id,),
            log_context=context,
        )

    def unregister_service(self, service_id: str) -> bool:
        """Unregister a service from the discovery registry."""

        if service_id not in self._service_registry:
            correlation_id = get_correlation_id()
            logger.warning(
                "Attempted to unregister unknown service %s",
                service_id,
                extra={"correlation_id": correlation_id, "service_id": service_id},
            )
            return False

        context: Dict[str, Any] = {"service_id": service_id}

        def _operation() -> bool:
            endpoint = self._service_registry.pop(service_id)
            service_type = endpoint.service_type
            context["service_type"] = service_type

            if service_type in self._service_types:
                service_ids = self._service_types[service_type]
                if service_id in service_ids:
                    service_ids.remove(service_id)
                if not service_ids:
                    del self._service_types[service_type]

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

    def discover_services(
        self, service_type: Optional[str] = None
    ) -> List[ServiceEndpoint]:
        """Discover services by type.

        Args:
            service_type: Optional service type to filter by

        Returns:
            List of service endpoints matching the criteria
        """
        # Clean up expired services first
        self._cleanup_expired_services()

        if service_type is None:
            # Return all services
            return list(self._service_registry.values())

        # Return services of specific type
        service_ids = self._service_types.get(service_type, [])
        return [
            self._service_registry[sid]
            for sid in service_ids
            if sid in self._service_registry
        ]

    def get_service(self, service_id: str) -> Optional[ServiceEndpoint]:
        """Get a specific service by ID.

        Args:
            service_id: Unique identifier for the service

        Returns:
            Service endpoint if found, None otherwise
        """
        self._cleanup_expired_services()
        return self._service_registry.get(service_id)

    def update_service_heartbeat(self, service_id: str) -> bool:
        """Update the last seen timestamp for a service.

        Args:
            service_id: Unique identifier for the service

        Returns:
            True if update successful, False if service not found
        """
        if service_id in self._service_registry:
            self._service_registry[service_id].last_seen = datetime.utcnow()
            return True
        return False

    def get_service_types(self) -> List[str]:
        """Get list of all registered service types.

        Returns:
            List of service type names
        """
        self._cleanup_expired_services()
        return list(self._service_types.keys())

    def get_service_count(self, service_type: Optional[str] = None) -> int:
        """Get count of registered services.

        Args:
            service_type: Optional service type to filter by

        Returns:
            Number of registered services
        """
        self._cleanup_expired_services()

        if service_type is None:
            return len(self._service_registry)

        return len(self._service_types.get(service_type, []))

    def _cleanup_expired_services(self):
        """Remove expired services from the registry."""
        current_time = datetime.utcnow()
        ttl_delta = timedelta(minutes=self.service_ttl_minutes)

        expired_services = []
        for service_id, endpoint in self._service_registry.items():
            if current_time - endpoint.last_seen > ttl_delta:
                expired_services.append(service_id)

        for service_id in expired_services:
            logger.info(
                "Removing expired service %s",
                service_id,
                extra={"service_id": service_id},
            )
            self.unregister_service(service_id)

    def get_registry_status(self) -> Dict[str, Any]:
        """Get status information about the service registry.

        Returns:
            Dictionary with registry status information
        """
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


# Export only the service discovery functionality
__all__ = [
    "ServiceDiscoveryManager",
    "ServiceEndpoint",
    "ServiceMetadata",
]

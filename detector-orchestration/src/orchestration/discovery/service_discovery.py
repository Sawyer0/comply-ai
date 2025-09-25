"""Service discovery functionality following SRP.

This module provides ONLY service discovery - finding and registering detector services.
Single Responsibility: Maintain registry of available detector services and their endpoints.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from shared.utils.correlation import get_correlation_id

logger = logging.getLogger(__name__)


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
        """Register a service in the discovery registry.

        Args:
            service_id: Unique identifier for the service
            endpoint_url: URL endpoint for the service
            service_type: Type/category of the service
            version: Service version
            metadata: Additional service metadata

        Returns:
            True if registration successful, False otherwise
        """
        correlation_id = get_correlation_id()

        try:
            endpoint = ServiceEndpoint(
                service_id=service_id,
                endpoint_url=endpoint_url,
                service_type=service_type,
                version=version,
                metadata=metadata,
            )

            # Add to registry
            self._service_registry[service_id] = endpoint

            # Add to type index
            if service_type not in self._service_types:
                self._service_types[service_type] = []

            if service_id not in self._service_types[service_type]:
                self._service_types[service_type].append(service_id)

            logger.info(
                "Registered service %s of type %s at %s",
                service_id,
                service_type,
                endpoint_url,
                extra={
                    "correlation_id": correlation_id,
                    "service_id": service_id,
                    "service_type": service_type,
                    "endpoint_url": endpoint_url,
                    "version": version,
                },
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to register service %s: %s",
                service_id,
                str(e),
                extra={
                    "correlation_id": correlation_id,
                    "service_id": service_id,
                    "error": str(e),
                },
            )
            return False

    def unregister_service(self, service_id: str) -> bool:
        """Unregister a service from the discovery registry.

        Args:
            service_id: Unique identifier for the service to unregister

        Returns:
            True if unregistration successful, False otherwise
        """
        correlation_id = get_correlation_id()

        try:
            if service_id not in self._service_registry:
                logger.warning(
                    "Attempted to unregister unknown service %s",
                    service_id,
                    extra={"correlation_id": correlation_id, "service_id": service_id},
                )
                return False

            endpoint = self._service_registry[service_id]
            service_type = endpoint.service_type

            # Remove from registry
            del self._service_registry[service_id]

            # Remove from type index
            if service_type in self._service_types:
                if service_id in self._service_types[service_type]:
                    self._service_types[service_type].remove(service_id)

                # Clean up empty type lists
                if not self._service_types[service_type]:
                    del self._service_types[service_type]

            logger.info(
                "Unregistered service %s",
                service_id,
                extra={
                    "correlation_id": correlation_id,
                    "service_id": service_id,
                    "service_type": service_type,
                },
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to unregister service %s: %s",
                service_id,
                str(e),
                extra={
                    "correlation_id": correlation_id,
                    "service_id": service_id,
                    "error": str(e),
                },
            )
            return False

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
]

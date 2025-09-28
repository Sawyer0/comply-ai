"""Service discovery functionality for orchestration service.

This module provides service discovery capabilities following SRP:
- ServiceDiscoveryManager: Maintain registry of detector services
- LoadBalancer: Balance load across available services
"""

from .service_discovery import (
    ServiceDiscoveryManager,
    ServiceEndpoint,
    ServiceMetadata,
    ServiceRegistration,
)

__all__ = [
    "ServiceDiscoveryManager",
    "ServiceEndpoint",
    "ServiceMetadata",
    "ServiceRegistration",
]

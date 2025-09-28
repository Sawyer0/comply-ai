"""Core orchestration functionality following SRP.

This module provides focused, single-responsibility components for detector orchestration:
- DetectorCoordinator: Execute detectors according to routing plans
- ContentRouter: Route content to appropriate detectors
- ResponseAggregator: Aggregate detector results into unified output

Other responsibilities are in separate modules:
- Service Discovery: ../discovery/service_discovery.py
- Health Monitoring: ../monitoring/health_monitor.py
- Circuit Breakers: ../resilience/circuit_breaker.py
- Policy Management: ../policy/policy_manager.py
"""

from .aggregator import ResponseAggregator
from .coordinator import DetectorCoordinator
from .detector_client import CustomerDetectorClient
from .models import (
    AggregatedOutput,
    DetectorClientConfig,
    DetectorConfig,
    RoutingDecision,
    RoutingPlan,
)
from .router import ContentRouter

__all__ = [
    # Coordination
    "DetectorCoordinator",
    "RoutingPlan",
    # Routing
    "ContentRouter",
    "RoutingDecision",
    "DetectorConfig",
    "DetectorClientConfig",
    # Aggregation
    "ResponseAggregator",
    "AggregatedOutput",
    # Clients
    "CustomerDetectorClient",
]

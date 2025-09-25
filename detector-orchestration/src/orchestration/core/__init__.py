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

from .coordinator import DetectorCoordinator, RoutingPlan
from .router import ContentRouter, RoutingDecision, DetectorConfig
from .aggregator import ResponseAggregator, AggregatedOutput

__all__ = [
    # Coordination
    "DetectorCoordinator",
    "RoutingPlan",
    # Routing
    "ContentRouter",
    "RoutingDecision",
    "DetectorConfig",
    # Aggregation
    "ResponseAggregator",
    "AggregatedOutput",
]

"""Common factories used by unit tests."""

from __future__ import annotations

from typing import Any, Dict, List

from detector_orchestration.config import DetectorEndpoint, OrchestrationConfig, Settings
from detector_orchestration.models import (
    ContentType,
    OrchestrationRequest,
    Priority,
    ProcessingMode,
    RoutingDecision,
    RoutingPlan,
)
from detector_orchestration.policy import CoverageMethod, PolicyDecision


def make_settings(**overrides: Any) -> Settings:
    """Return a Settings instance with sensible defaults for tests."""
    settings = Settings()
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


def make_orchestration_config(**overrides: Any) -> OrchestrationConfig:
    """Return an orchestration config with defaults that tests can override."""
    config = OrchestrationConfig()
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_detector_endpoint(**overrides: Any) -> DetectorEndpoint:
    """Return a detector endpoint configuration for router tests."""
    endpoint = DetectorEndpoint(
        name="detector-1",
        url="http://detector",
        detector_type="default",
    )
    for key, value in overrides.items():
        setattr(endpoint, key, value)
    return endpoint


def make_orchestration_request(**overrides: Any) -> OrchestrationRequest:
    """Return a basic orchestration request."""
    request = OrchestrationRequest(
        content="unit-test content",
        content_type=ContentType.TEXT,
        tenant_id="unit-test-tenant",
        policy_bundle="default",
        environment="dev",
        processing_mode=ProcessingMode.SYNC,
        priority=Priority.NORMAL,
        metadata={"unit-test": True},
    )
    for key, value in overrides.items():
        setattr(request, key, value)
    return request


def make_routing_plan(**overrides: Any) -> RoutingPlan:
    """Create a routing plan with optional overrides."""
    plan = RoutingPlan(primary_detectors=["detector-1", "detector-2"])
    for key, value in overrides.items():
        setattr(plan, key, value)
    return plan


def make_routing_decision(**overrides: Any) -> RoutingDecision:
    """Create a routing decision with optional overrides."""
    decision = RoutingDecision(
        selected_detectors=["detector-1"],
        routing_reason="unit-test",
        policy_applied="default",
        coverage_requirements={"min_success_fraction": 1.0},
        health_status={"detector-1": True},
    )
    for key, value in overrides.items():
        setattr(decision, key, value)
    return decision


def make_policy_decision(**overrides: Any) -> PolicyDecision:
    """Return a policy decision object."""
    decision = PolicyDecision(
        decision="allow",
        confidence=0.9,
        coverage_method=CoverageMethod.ALL_MUST_PASS,
        required_detectors=["detector-1"],
        recommended_detectors=[],
        violations=[],
        metadata={"unit-test": True},
    )
    for key, value in overrides.items():
        setattr(decision, key, value)
    return decision


__all__ = [
    "make_settings",
    "make_orchestration_config",
    "make_detector_endpoint",
    "make_orchestration_request",
    "make_routing_plan",
    "make_routing_decision",
    "make_policy_decision",
]

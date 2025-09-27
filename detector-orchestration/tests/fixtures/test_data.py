"""Shared test data builders for detector orchestration tests."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from detector_orchestration.models import (
    ContentType,
    OrchestrationRequest,
    Priority,
    ProcessingMode,
    RoutingDecision,
)

DEFAULT_TENANT_ID = "test-tenant"
DEFAULT_POLICY = "default"
DEFAULT_ENVIRONMENT = "dev"
DEFAULT_API_KEY = "test-token"


def build_orchestration_request(
    *,
    content: str = "This is test content for detector analysis",
    tenant_id: str = DEFAULT_TENANT_ID,
    policy_bundle: str = DEFAULT_POLICY,
    environment: str = DEFAULT_ENVIRONMENT,
    processing_mode: ProcessingMode = ProcessingMode.SYNC,
    priority: Priority = Priority.NORMAL,
    metadata: Dict[str, Any] | None = None,
) -> OrchestrationRequest:
    """Create a canonical orchestration request used across tests."""

    return OrchestrationRequest(
        content=content,
        content_type=ContentType.TEXT,
        tenant_id=tenant_id,
        policy_bundle=policy_bundle,
        environment=environment,
        processing_mode=processing_mode,
        priority=priority,
        metadata=metadata or {"test": True},
    )


def build_batch_requests(count: int = 2, **kwargs: Any) -> List[Dict[str, Any]]:
    """Generate a batch payload of orchestration requests."""

    return [build_orchestration_request(content=f"Test content {i}", **kwargs).model_dump() for i in range(count)]


def build_routing_decision(
    selected_detectors: List[str],
    *,
    policy_bundle: str = DEFAULT_POLICY,
    health_status: Iterable[str] | None = None,
    coverage_requirements: Dict[str, Any] | None = None,
    routing_reason: str = "integration-test",
) -> RoutingDecision:
    """Construct a routing decision matching integration test expectations."""

    healthy = set(health_status or selected_detectors)
    return RoutingDecision(
        selected_detectors=selected_detectors,
        routing_reason=routing_reason,
        policy_applied=policy_bundle,
        coverage_requirements=dict(coverage_requirements or {"min_success_fraction": 1.0}),
        health_status={detector: detector in healthy for detector in selected_detectors},
    )


def request_headers(
    *,
    tenant_id: str = DEFAULT_TENANT_ID,
    api_key: str = DEFAULT_API_KEY,
    correlation_id: str | None = None,
    idempotency_key: str | None = None,
) -> Dict[str, str]:
    """Return default HTTP headers for orchestration API calls."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-Tenant-ID": tenant_id,
    }
    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    return headers


__all__ = [
    "DEFAULT_API_KEY",
    "DEFAULT_ENVIRONMENT",
    "DEFAULT_POLICY",
    "DEFAULT_TENANT_ID",
    "build_orchestration_request",
    "build_batch_requests",
    "build_routing_decision",
    "request_headers",
]

from __future__ import annotations

import time
from typing import Dict, Optional

from prometheus_client import Counter, Gauge, Histogram


# Locked metric names per contract
_REQ_TOTAL = Counter(
    "orchestrate_requests_total",
    "Total orchestration requests",
    labelnames=["tenant", "policy", "status"],
)
_REQ_DURATION = Histogram(
    "orchestrate_request_duration_ms",
    "Orchestration request duration in ms",
    buckets=(50, 100, 200, 500, 1000, 1500, 2000, 3000, 5000),
)
_DET_LATENCY = Histogram(
    "detector_latency_ms",
    "Detector latency in ms",
    labelnames=["detector", "status"],
    buckets=(10, 50, 100, 200, 500, 1000, 3000, 5000),
)
_DET_HEALTH = Gauge(
    "detector_health_status", "Detector health status (1 healthy, 0 unhealthy)", labelnames=["detector"]
)
_DET_HEALTH_DURATION = Histogram(
    "detector_health_check_duration_ms",
    "Detector health check duration in ms",
    labelnames=["detector"],
    buckets=(10, 50, 100, 200, 500, 1000, 3000),
)
_CB_STATE = Gauge(
    "circuit_breaker_state", "Circuit breaker state", labelnames=["detector", "state"]
)
_COVERAGE = Gauge(
    "coverage_achieved", "Coverage achieved for orchestration request", labelnames=["tenant", "policy"]
)
_POLICY_ENF = Counter(
    "policy_enforcement_total",
    "Policy enforcement results",
    labelnames=["tenant", "policy", "status", "violation_type"],
)

# Security & tenancy metrics
_RL_REQUESTS = Counter(
    "orchestrator_rate_limit_requests_total",
    "Rate limit decisions by endpoint and tenant",
    labelnames=["endpoint", "tenant", "action"],
)
_RL_RESET_SECONDS = Histogram(
    "orchestrator_rate_limit_reset_seconds",
    "Observed reset seconds on 403 blocks",
    labelnames=["endpoint", "tenant"],
    buckets=(0, 1, 2, 5, 10, 30, 60, 120, 300),
)
_RBAC_ENF = Counter(
    "orchestrator_rbac_enforcement_total",
    "RBAC enforcement decisions",
    labelnames=["endpoint", "tenant", "decision", "scope"],
)

# Redis backend health/fallback
_REDIS_UP = Gauge(
    "orchestrator_redis_backend_up",
    "Redis backend health (1 up, 0 down)",
    labelnames=["component"],
)
_REDIS_FALLBACK = Counter(
    "orchestrator_redis_backend_fallback_total",
    "Redis backend fallback occurrences",
    labelnames=["component"],
)


class OrchestrationMetricsCollector:
    def record_request(self, tenant: str, policy: str, status: str, duration_ms: float) -> None:
        _REQ_TOTAL.labels(tenant=tenant, policy=policy, status=status).inc()
        _REQ_DURATION.observe(duration_ms)

    def record_detector_latency(self, detector: str, success: bool, duration_ms: float) -> None:
        _DET_LATENCY.labels(detector=detector, status=("success" if success else "failed")).observe(duration_ms)

    def record_coverage(self, tenant: str, policy: str, coverage: float) -> None:
        _COVERAGE.labels(tenant=tenant, policy=policy).set(coverage)

    # Placeholders for health/circuit breaker/policy metrics
    def record_health_status(self, detector: str, is_healthy: bool, duration_ms: float) -> None:
        _DET_HEALTH.labels(detector=detector).set(1.0 if is_healthy else 0.0)
        _DET_HEALTH_DURATION.labels(detector=detector).observe(duration_ms)

    def record_circuit_breaker(self, detector: str, state: str) -> None:
        _CB_STATE.labels(detector=detector, state=state).set(1.0)

    def record_policy_enforcement(self, tenant: str, policy: str, enforced: bool, violation_type: Optional[str] = None) -> None:
        _POLICY_ENF.labels(tenant=tenant, policy=policy, status=("enforced" if enforced else "violated"), violation_type=(violation_type or "none")).inc()

    def record_rate_limit(self, endpoint: str, tenant: str, action: str, reset_seconds: Optional[float] = None) -> None:
        _RL_REQUESTS.labels(endpoint=endpoint, tenant=tenant, action=action).inc()
        if reset_seconds is not None:
            _RL_RESET_SECONDS.labels(endpoint=endpoint, tenant=tenant).observe(float(reset_seconds))

    def record_rbac(self, endpoint: str, tenant: Optional[str], decision: str, scope: Optional[str] = None) -> None:
        _RBAC_ENF.labels(endpoint=endpoint, tenant=(tenant or "unknown"), decision=decision, scope=(scope or "none")).inc()

    # Redis cache health
    def set_redis_backend_up(self, component: str, up: bool) -> None:
        _REDIS_UP.labels(component=component).set(1.0 if up else 0.0)

    def inc_redis_fallback(self, component: str) -> None:
        _REDIS_FALLBACK.labels(component=component).inc()

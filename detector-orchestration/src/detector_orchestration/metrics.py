from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

from prometheus_client import Counter, Gauge, Histogram


# Locked metric names per contract
_REQ_TOTAL = Counter(
    "orchestrate_requests_total",
    "Total orchestration requests",
    labelnames=["tenant", "policy", "status", "processing_mode"],
)
_REQ_DURATION = Histogram(
    "orchestrate_request_duration_ms",
    "Orchestration request duration in ms",
    buckets=(50, 100, 200, 500, 1000, 1500, 2000, 3000, 5000),
)
_REQ_DURATION_SYNC = Histogram(
    "orchestrate_request_duration_sync_ms",
    "Sync orchestration request duration in ms",
    buckets=(50, 100, 200, 500, 1000, 1500, 2000, 3000, 5000),
)
_REQ_DURATION_ASYNC = Histogram(
    "orchestrate_request_duration_async_ms",
    "Async orchestration request duration in ms",
    buckets=(1000, 2000, 5000, 10000, 30000, 60000),
)
_DET_LATENCY = Histogram(
    "detector_latency_ms",
    "Detector latency in ms",
    labelnames=["detector", "status"],
    buckets=(10, 50, 100, 200, 500, 1000, 3000, 5000),
)
_DET_HEALTH = Gauge(
    "detector_health_status",
    "Detector health status (1 healthy, 0 unhealthy)",
    labelnames=["detector"],
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
    "coverage_achieved",
    "Coverage achieved for orchestration request",
    labelnames=["tenant", "policy"],
)
_POLICY_ENF = Counter(
    "policy_enforcement_total",
    "Policy enforcement results",
    labelnames=["tenant", "policy", "status", "violation_type"],
)
# Additional orchestration-specific metrics
_ORCHESTRATOR_READY = Gauge(
    "orchestrator_ready", "Orchestrator readiness status (1 ready, 0 not ready)"
)
_SERVICE_DEPENDENCIES = Gauge(
    "orchestrator_service_dependencies_up",
    "Service dependency status",
    labelnames=["dependency"],
)
_ASYNC_JOBS_ACTIVE = Gauge(
    "orchestrator_async_jobs_active", "Number of active async jobs"
)
_ASYNC_JOBS_QUEUED = Gauge(
    "orchestrator_async_jobs_queued", "Number of queued async jobs"
)
_ASYNC_JOBS_COMPLETED = Counter(
    "orchestrator_async_jobs_completed_total",
    "Total completed async jobs",
    labelnames=["status"],
)
_CACHE_HIT_RATIO = Gauge(
    "orchestrator_cache_hit_ratio",
    "Cache hit ratio for response cache",
    labelnames=["cache_type"],
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
    def __init__(self) -> None:
        # In-memory mirrors to provide summaries without scraping Prometheus
        self._request_context: Dict[str, Tuple[str, str, float, str]] = {}
        self._request_totals: Dict[str, int] = defaultdict(int)
        self._error_total: int = 0
        self._latency_sum_ms: float = 0.0
        self._latency_count: int = 0

    def record_request(
        self,
        tenant: str,
        policy: str,
        status: str,
        duration_ms: float,
        processing_mode: str = "sync",
    ) -> None:
        _REQ_TOTAL.labels(
            tenant=tenant, policy=policy, status=status, processing_mode=processing_mode
        ).inc()
        _REQ_DURATION.observe(duration_ms)
        if processing_mode == "sync":
            _REQ_DURATION_SYNC.observe(duration_ms)
        elif processing_mode == "async":
            _REQ_DURATION_ASYNC.observe(duration_ms)
        key = f"{tenant}:{policy}:{status}:{processing_mode}"
        self._request_totals[key] += 1
        self._latency_sum_ms += duration_ms
        self._latency_count += 1
        if status == "failure":
            self._error_total += 1

    def record_request_start(
        self,
        request_id: str,
        tenant: str,
        policy: str,
        *,
        processing_mode: str = "sync",
    ) -> None:
        self._request_context[request_id] = (
            tenant,
            policy,
            time.time() * 1000,
            processing_mode,
        )

    def record_request_end(
        self,
        request_id: str,
        *,
        success: bool,
        duration_ms: float,
        tenant: Optional[str] = None,
        policy: Optional[str] = None,
        processing_mode: str = "sync",
    ) -> None:
        ctx = self._request_context.pop(request_id, None)
        if ctx:
            tenant = tenant or ctx[0]
            policy = policy or ctx[1]
            processing_mode = ctx[3]
        tenant = tenant or "unknown"
        policy = policy or "unknown"
        status = "success" if success else "failure"
        self.record_request(tenant, policy, status, duration_ms, processing_mode)

    def record_detector_latency(
        self, detector: str, success: bool | str, duration_ms: float
    ) -> None:
        status_label = (
            success
            if isinstance(success, str)
            else ("success" if success else "failed")
        )
        _DET_LATENCY.labels(detector=detector, status=status_label).observe(duration_ms)

    def record_coverage(self, tenant: str, policy: str, coverage: float) -> None:
        _COVERAGE.labels(tenant=tenant, policy=policy).set(coverage)

    # Placeholders for health/circuit breaker/policy metrics
    def record_detector_health(
        self, detector: str, is_healthy: bool, duration_ms: float | None = None
    ) -> None:
        _DET_HEALTH.labels(detector=detector).set(1.0 if is_healthy else 0.0)
        if duration_ms is not None:
            _DET_HEALTH_DURATION.labels(detector=detector).observe(duration_ms)

    # Backwards-compatible alias used by service code
    record_health_status = record_detector_health

    def record_circuit_breaker(self, detector: str, state: str) -> None:
        _CB_STATE.labels(detector=detector, state=state).set(1.0)

    def record_policy_enforcement(
        self,
        tenant: str,
        policy: str,
        status: bool | str,
        violation_type: Optional[str] = None,
    ) -> None:
        enforced = status if isinstance(status, bool) else status == "allowed"
        status_label = "enforced" if enforced else "violated"
        _POLICY_ENF.labels(
            tenant=tenant,
            policy=policy,
            status=status_label,
            violation_type=(violation_type or "none"),
        ).inc()

    # New orchestration-specific methods
    def set_orchestrator_ready(self, ready: bool) -> None:
        _ORCHESTRATOR_READY.set(1.0 if ready else 0.0)

    def set_service_dependency_up(self, dependency: str, up: bool) -> None:
        _SERVICE_DEPENDENCIES.labels(dependency=dependency).set(1.0 if up else 0.0)

    def set_async_jobs_active(self, count: int) -> None:
        _ASYNC_JOBS_ACTIVE.set(float(count))

    def set_async_jobs_queued(self, count: int) -> None:
        _ASYNC_JOBS_QUEUED.set(float(count))

    def record_async_job_completed(self, status: str) -> None:
        _ASYNC_JOBS_COMPLETED.labels(status=status).inc()

    def set_cache_hit_ratio(self, cache_type: str, ratio: float) -> None:
        _CACHE_HIT_RATIO.labels(cache_type=cache_type).set(ratio)

    def record_rate_limit(
        self,
        endpoint: str,
        tenant: str,
        action: str,
        reset_seconds: Optional[float] = None,
    ) -> None:
        _RL_REQUESTS.labels(endpoint=endpoint, tenant=tenant, action=action).inc()
        if reset_seconds is not None:
            _RL_RESET_SECONDS.labels(endpoint=endpoint, tenant=tenant).observe(
                float(reset_seconds)
            )

    def record_rbac(
        self,
        endpoint: str,
        tenant: Optional[str],
        decision: str,
        scope: Optional[str] = None,
    ) -> None:
        _RBAC_ENF.labels(
            endpoint=endpoint,
            tenant=(tenant or "unknown"),
            decision=decision,
            scope=(scope or "none"),
        ).inc()

    # Backwards compatibility helpers used in tests
    def record_rate_limit_decision(
        self, tenant: str, endpoint: str, decision: str
    ) -> None:
        self.record_rate_limit(endpoint=endpoint, tenant=tenant, action=decision)

    def record_cache_operation(self, outcome: str) -> None:
        label = outcome if outcome in {"hit", "miss"} else "other"
        self.set_cache_hit_ratio(cache_type=label, ratio=1.0)

    def record_error(self, error_code: str, tenant: Optional[str] = None) -> None:
        self._error_total += 1
        self.record_policy_enforcement(
            tenant or "unknown", "unknown", False, violation_type=error_code
        )

    def record_security_event(
        self, event_type: str, tenant: str, severity: str
    ) -> None:
        self.record_rbac(
            endpoint=event_type, tenant=tenant, decision=severity, scope="security"
        )

    # Redis cache health
    def set_redis_backend_up(self, component: str, up: bool) -> None:
        _REDIS_UP.labels(component=component).set(1.0 if up else 0.0)

    def inc_redis_fallback(self, component: str) -> None:
        _REDIS_FALLBACK.labels(component=component).inc()

    # Summary helpers for service discovery / health reporting
    def get_total_requests(self) -> int:
        return sum(self._request_totals.values())

    def get_error_rate(self) -> float:
        total = self.get_total_requests()
        if total == 0:
            return 0.0
        return self._error_total / total

    def get_average_latency(self) -> float:
        if self._latency_count == 0:
            return 0.0
        return self._latency_sum_ms / self._latency_count

    def get_metrics_summary(self) -> Dict[str, object]:
        return {
            "requests_total": self.get_total_requests(),
            "error_rate": self.get_error_rate(),
            "average_latency_ms": self.get_average_latency(),
            "detectors": list(self._request_totals.keys()),
            "circuit_breakers": {},
            "coverage": {},
        }

    def reset_metrics(self) -> None:
        self._request_context.clear()
        self._request_totals.clear()
        self._error_total = 0
        self._latency_sum_ms = 0.0
        self._latency_count = 0
        # Prometheus client objects keep their own internal state. We avoid
        # manipulating those registries directly and rely on fresh process
        # startups for full metric resets. This method only clears the local
        # mirrors used for summaries during tests.

"""
Prometheus metrics for the analysis API.
"""

import time

from fastapi import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# Define metrics
REQUEST_COUNT = Counter(
    "analysis_requests_total",
    "Total number of analysis requests",
    ["endpoint", "status", "tenant"],
)

REQUEST_DURATION = Histogram(
    "analysis_request_duration_seconds",
    "Time spent processing analysis requests",
    ["endpoint", "analysis_type"],
)

CONFIDENCE_SCORE = Histogram(
    "analysis_confidence_score",
    "Distribution of confidence scores",
    ["analysis_type", "env"],
)

COVERAGE_GAP_RATE = Gauge(
    "coverage_gap_rate", "Rate of coverage gaps detected", ["tenant", "env"]
)

ERROR_RATE = Counter(
    "analysis_errors_total",
    "Total number of analysis errors",
    ["error_type", "endpoint"],
)

ACTIVE_TENANTS = Gauge("analysis_active_tenants", "Number of active tenants")

POLICY_GENERATION_TIME = Histogram(
    "opa_policy_generation_duration_seconds",
    "Time spent generating OPA policies",
    ["policy_type"],
)

LOW_CONFIDENCE_ALERTS = Counter(
    "analysis_low_confidence_total",
    "Number of low confidence analysis results",
    ["tenant", "threshold"],
)


class MetricsCollector:
    """Collector for analysis metrics."""

    def __init__(self):
        self.active_tenants = set()

    def record_request(
        self,
        endpoint: str,
        status: str,
        tenant: str,
        duration: float,
        analysis_type: str = "unknown",
    ):
        """Record a request with metrics."""
        REQUEST_COUNT.labels(endpoint=endpoint, status=status, tenant=tenant).inc()
        REQUEST_DURATION.labels(endpoint=endpoint, analysis_type=analysis_type).observe(
            duration
        )
        self.active_tenants.add(tenant)
        ACTIVE_TENANTS.set(len(self.active_tenants))

    def record_confidence(
        self, confidence: float, analysis_type: str, env: str, tenant: str
    ):
        """Record confidence score."""
        CONFIDENCE_SCORE.labels(analysis_type=analysis_type, env=env).observe(
            confidence
        )

        # Check for low confidence alerts
        if confidence < 0.7:
            LOW_CONFIDENCE_ALERTS.labels(tenant=tenant, threshold="0.7").inc()
        if confidence < 0.5:
            LOW_CONFIDENCE_ALERTS.labels(tenant=tenant, threshold="0.5").inc()

    def record_coverage_gap(self, tenant: str, env: str, has_gap: bool):
        """Record coverage gap detection."""
        current_rate = COVERAGE_GAP_RATE.labels(tenant=tenant, env=env)._value._value
        # Simple moving average for gap rate
        new_rate = current_rate * 0.9 + (1.0 if has_gap else 0.0) * 0.1
        COVERAGE_GAP_RATE.labels(tenant=tenant, env=env).set(new_rate)

    def record_error(self, error_type: str, endpoint: str):
        """Record an error."""
        ERROR_RATE.labels(error_type=error_type, endpoint=endpoint).inc()

    def record_policy_generation(self, policy_type: str, duration: float):
        """Record OPA policy generation time."""
        POLICY_GENERATION_TIME.labels(policy_type=policy_type).observe(duration)


# Global metrics collector instance
metrics_collector = MetricsCollector()


def get_metrics_response() -> Response:
    """Generate Prometheus metrics response."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect request metrics."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Get request info
        endpoint = f"{request.method} {request.url.path}"
        tenant = request.headers.get("x-tenant", "unknown")

        # Process request
        response = await call_next(request)

        # Record metrics
        duration = time.time() - start_time
        status = str(response.status_code)

        metrics_collector.record_request(
            endpoint=endpoint, status=status, tenant=tenant, duration=duration
        )

        return response

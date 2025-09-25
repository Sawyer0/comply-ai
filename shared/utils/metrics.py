"""Shared metrics utilities for Prometheus monitoring."""

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, REGISTRY
from typing import Dict, Any, Optional
import time
from functools import wraps


# Common metrics across all services
REQUEST_COUNT = Counter(
    "service_requests_total",
    "Total service requests",
    ["service", "method", "endpoint", "status"],
)

REQUEST_DURATION = Histogram(
    "service_request_duration_seconds",
    "Service request duration",
    ["service", "method", "endpoint"],
)

ACTIVE_CONNECTIONS = Gauge(
    "service_active_connections", "Active connections", ["service"]
)

ERROR_COUNT = Counter(
    "service_errors_total", "Total service errors", ["service", "error_type"]
)


def track_request_metrics(service_name: str):
    """Decorator to track request metrics."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                ERROR_COUNT.labels(
                    service=service_name, error_type=type(e).__name__
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                REQUEST_DURATION.labels(
                    service=service_name,
                    method=func.__name__,
                    endpoint=getattr(func, "__endpoint__", "unknown"),
                ).observe(duration)

                REQUEST_COUNT.labels(
                    service=service_name,
                    method=func.__name__,
                    endpoint=getattr(func, "__endpoint__", "unknown"),
                    status=status,
                ).inc()

        return wrapper

    return decorator


def create_service_metrics(service_name: str) -> Dict[str, Any]:
    """Create service-specific metrics."""
    return {
        "request_count": Counter(
            f"{service_name}_requests_total",
            f"Total {service_name} requests",
            ["method", "endpoint", "status"],
        ),
        "request_duration": Histogram(
            f"{service_name}_request_duration_seconds",
            f"{service_name} request duration",
            ["method", "endpoint"],
        ),
        "active_connections": Gauge(
            f"{service_name}_active_connections", f"Active {service_name} connections"
        ),
        "error_count": Counter(
            f"{service_name}_errors_total",
            f"Total {service_name} errors",
            ["error_type"],
        ),
    }


class MetricsCollector:
    """Centralized metrics collection."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.metrics = create_service_metrics(service_name)

    def increment_request_count(self, method: str, endpoint: str, status: str):
        """Increment request count metric."""
        self.metrics["request_count"].labels(
            method=method, endpoint=endpoint, status=status
        ).inc()

    def observe_request_duration(self, method: str, endpoint: str, duration: float):
        """Observe request duration metric."""
        self.metrics["request_duration"].labels(
            method=method, endpoint=endpoint
        ).observe(duration)

    def set_active_connections(self, count: int):
        """Set active connections metric."""
        self.metrics["active_connections"].set(count)

    def increment_error_count(self, error_type: str):
        """Increment error count metric."""
        self.metrics["error_count"].labels(error_type=error_type).inc()

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary for health checks."""
        return {
            "service": self.service_name,
            "metrics_collected": len(self.metrics),
            "registry_metrics": len(list(REGISTRY.collect())),
        }

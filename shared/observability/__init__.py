"""Shared observability components for distributed tracing and metrics."""

from .tracing import (
    setup_tracing,
    get_tracer,
    trace_function,
    trace_async_function,
    propagate_headers,
    extract_headers,
)

from .metrics import (
    setup_metrics,
    get_metrics_collector,
    record_request,
    record_error,
    record_latency,
)

from .middleware import (
    ObservabilityMiddleware,
    MetricsMiddleware,
    TracingMiddleware,
)

__all__ = [
    "setup_tracing",
    "get_tracer",
    "trace_function",
    "trace_async_function",
    "propagate_headers",
    "extract_headers",
    "setup_metrics",
    "get_metrics_collector",
    "record_request",
    "record_error",
    "record_latency",
    "ObservabilityMiddleware",
    "MetricsMiddleware",
    "TracingMiddleware",
]

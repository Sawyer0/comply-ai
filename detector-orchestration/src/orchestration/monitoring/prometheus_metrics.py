"""Prometheus metrics collection functionality following SRP.

This module provides ONLY Prometheus metrics collection - defining and updating metrics.
Single Responsibility: Collect and expose Prometheus metrics for orchestration service.
"""

import logging
import time
from typing import Dict, Any, Optional
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    CollectorRegistry,
    generate_latest,
)

from shared.utils.correlation import get_correlation_id
from shared.interfaces.common import HealthStatus

logger = logging.getLogger(__name__)


class PrometheusMetricsCollector:
    """Collects and exposes Prometheus metrics for orchestration service.

    Single Responsibility: Define, collect, and expose Prometheus metrics.
    Does NOT handle: business logic, health monitoring, alerting.
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize Prometheus metrics collector.

        Args:
            registry: Optional custom registry (uses default if None)
        """
        self.registry = registry

        # Request metrics
        self.request_total = Counter(
            "orchestration_requests_total",
            "Total number of orchestration requests",
            ["tenant_id", "status", "processing_mode"],
            registry=registry,
        )

        self.request_duration = Histogram(
            "orchestration_request_duration_seconds",
            "Duration of orchestration requests",
            ["tenant_id", "processing_mode"],
            registry=registry,
        )

        # Detector metrics
        self.detector_executions_total = Counter(
            "detector_executions_total",
            "Total number of detector executions",
            ["detector_id", "detector_type", "status"],
            registry=registry,
        )

        self.detector_duration = Histogram(
            "detector_execution_duration_seconds",
            "Duration of detector executions",
            ["detector_id", "detector_type"],
            registry=registry,
        )

        self.detector_health_status = Gauge(
            "detector_health_status",
            "Detector health status (0=unknown,1=healthy,2=degraded,3=unhealthy)",
            ["detector_id"],
            registry=registry,
        )

        self.active_detectors = Gauge(
            "active_detectors_count", "Number of active detectors", registry=registry
        )

        # Security metrics
        self.security_violations_total = Counter(
            "security_violations_total",
            "Total number of security violations detected",
            ["tenant_id", "violation_type", "severity"],
            registry=registry,
        )

        self.api_key_validations_total = Counter(
            "api_key_validations_total",
            "Total number of API key validations",
            ["tenant_id", "status"],
            registry=registry,
        )

        # Tenant metrics
        self.active_tenants = Gauge(
            "active_tenants_count", "Number of active tenants", registry=registry
        )

        self.tenant_requests_total = Counter(
            "tenant_requests_total",
            "Total requests per tenant",
            ["tenant_id"],
            registry=registry,
        )

        # System metrics
        self.service_info = Info(
            "orchestration_service_info",
            "Information about the orchestration service",
            registry=registry,
        )

        self.circuit_breaker_state = Gauge(
            "circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=open, 2=half-open)",
            ["detector_id"],
            registry=registry,
        )

        # Cache metrics
        self.cache_operations_total = Counter(
            "cache_operations_total",
            "Total cache operations",
            ["operation", "status"],
            registry=registry,
        )

        self.cache_hit_ratio = Gauge(
            "cache_hit_ratio", "Cache hit ratio", registry=registry
        )

        # Initialize service info
        self.service_info.info(
            {"version": "1.0.0", "service": "detector-orchestration"}
        )

    def record_request(
        self, tenant_id: str, processing_mode: str, status: str, duration_seconds: float
    ):
        """Record orchestration request metrics.

        Args:
            tenant_id: Tenant identifier
            processing_mode: Processing mode used
            status: Request status (success/failure)
            duration_seconds: Request duration in seconds
        """
        try:
            self.request_total.labels(
                tenant_id=tenant_id, status=status, processing_mode=processing_mode
            ).inc()

            self.request_duration.labels(
                tenant_id=tenant_id, processing_mode=processing_mode
            ).observe(duration_seconds)

            self.tenant_requests_total.labels(tenant_id=tenant_id).inc()

        except Exception as e:
            logger.error(
                "Failed to record request metrics: %s",
                str(e),
                extra={"correlation_id": get_correlation_id()},
            )

    def record_detector_execution(
        self, detector_id: str, detector_type: str, status: str, duration_seconds: float
    ):
        """Record detector execution metrics.

        Args:
            detector_id: Detector identifier
            detector_type: Type of detector
            status: Execution status
            duration_seconds: Execution duration in seconds
        """
        try:
            self.detector_executions_total.labels(
                detector_id=detector_id, detector_type=detector_type, status=status
            ).inc()

            self.detector_duration.labels(
                detector_id=detector_id, detector_type=detector_type
            ).observe(duration_seconds)

        except Exception as e:
            logger.error(
                "Failed to record detector execution metrics: %s",
                str(e),
                extra={"correlation_id": get_correlation_id()},
            )

    def record_security_violation(
        self, tenant_id: str, violation_type: str, severity: str
    ):
        """Record security violation metrics.

        Args:
            tenant_id: Tenant identifier
            violation_type: Type of security violation
            severity: Severity level
        """
        try:
            self.security_violations_total.labels(
                tenant_id=tenant_id, violation_type=violation_type, severity=severity
            ).inc()

        except Exception as e:
            logger.error(
                "Failed to record security violation metrics: %s",
                str(e),
                extra={"correlation_id": get_correlation_id()},
            )

    def record_api_key_validation(self, tenant_id: str, status: str):
        """Record API key validation metrics.

        Args:
            tenant_id: Tenant identifier
            status: Validation status (valid/invalid/expired)
        """
        try:
            self.api_key_validations_total.labels(
                tenant_id=tenant_id, status=status
            ).inc()

        except Exception as e:
            logger.error(
                "Failed to record API key validation metrics: %s",
                str(e),
                extra={"correlation_id": get_correlation_id()},
            )

    def update_active_detectors(self, count: int):
        """Update active detectors count.

        Args:
            count: Number of active detectors
        """
        try:
            self.active_detectors.set(count)
        except Exception as e:
            logger.error(
                "Failed to update active detectors metric: %s",
                str(e),
                extra={"correlation_id": get_correlation_id()},
            )

    def update_active_tenants(self, count: int):
        """Update active tenants count.

        Args:
            count: Number of active tenants
        """
        try:
            self.active_tenants.set(count)
        except Exception as e:
            logger.error(
                "Failed to update active tenants metric: %s",
                str(e),
                extra={"correlation_id": get_correlation_id()},
            )

    def update_circuit_breaker_state(self, detector_id: str, state: int):
        """Update circuit breaker state.

        Args:
            detector_id: Detector identifier
            state: Circuit breaker state (0=closed, 1=open, 2=half-open)
        """
        try:
            self.circuit_breaker_state.labels(detector_id=detector_id).set(state)
        except Exception as e:
            logger.error(
                "Failed to update circuit breaker state metric: %s",
                str(e),
                extra={"correlation_id": get_correlation_id()},
            )

    def update_detector_health(self, detector_id: str, status: HealthStatus):
        """Update detector health status gauge.

        Args:
            detector_id: Detector identifier
            status: HealthStatus value
        """

        try:
            value = 0
            if status == HealthStatus.HEALTHY:
                value = 1
            elif status == HealthStatus.DEGRADED:
                value = 2
            elif status == HealthStatus.UNHEALTHY:
                value = 3

            self.detector_health_status.labels(detector_id=detector_id).set(value)
        except Exception as e:
            logger.error(
                "Failed to update detector health status metric: %s",
                str(e),
                extra={"correlation_id": get_correlation_id()},
            )

    def record_cache_operation(self, operation: str, status: str):
        """Record cache operation metrics.

        Args:
            operation: Cache operation (get/set/delete)
            status: Operation status (hit/miss/success/failure)
        """
        try:
            self.cache_operations_total.labels(operation=operation, status=status).inc()
        except Exception as e:
            logger.error(
                "Failed to record cache operation metrics: %s",
                str(e),
                extra={"correlation_id": get_correlation_id()},
            )

    def update_cache_hit_ratio(self, ratio: float):
        """Update cache hit ratio.

        Args:
            ratio: Cache hit ratio (0.0 to 1.0)
        """
        try:
            self.cache_hit_ratio.set(ratio)
        except Exception as e:
            logger.error(
                "Failed to update cache hit ratio metric: %s",
                str(e),
                extra={"correlation_id": get_correlation_id()},
            )

    def get_metrics(self) -> str:
        """Get metrics in Prometheus format.

        Returns:
            Metrics in Prometheus text format
        """
        try:
            if self.registry is not None:
                return generate_latest(self.registry).decode("utf-8")
            return generate_latest().decode("utf-8")
        except Exception as e:
            logger.error(
                "Failed to generate metrics: %s",
                str(e),
                extra={"correlation_id": get_correlation_id()},
            )
            return ""

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics.

        Returns:
            Dictionary with metrics summary
        """
        try:
            request_metrics = list(self.request_total.collect())
            detector_metrics = list(self.detector_executions_total.collect())
            security_metrics = list(self.security_violations_total.collect())

            total_requests = (
                sum(sample.value for sample in request_metrics[0].samples)
                if request_metrics
                else 0.0
            )
            total_detector_execs = (
                sum(sample.value for sample in detector_metrics[0].samples)
                if detector_metrics
                else 0.0
            )
            total_security_violations = (
                sum(sample.value for sample in security_metrics[0].samples)
                if security_metrics
                else 0.0
            )

            return {
                "active_detectors": self.active_detectors._value._value,
                "active_tenants": self.active_tenants._value._value,
                "cache_hit_ratio": self.cache_hit_ratio._value._value,
                "total_requests": total_requests,
                "total_detector_executions": total_detector_execs,
                "total_security_violations": total_security_violations,
            }
        except Exception as e:
            logger.error(
                "Failed to get metrics summary: %s",
                str(e),
                extra={"correlation_id": get_correlation_id()},
            )
            return {}


# Export only the Prometheus metrics functionality
__all__ = [
    "PrometheusMetricsCollector",
]

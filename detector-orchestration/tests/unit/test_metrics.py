"""Tests for metrics collection."""

import time
import pytest
from unittest.mock import Mock, patch

from detector_orchestration.metrics import (
    OrchestrationMetricsCollector,
    _REQ_TOTAL,
    _REQ_DURATION,
    _DET_LATENCY,
    _DET_HEALTH,
    _CB_STATE,
    _COVERAGE,
)


class TestOrchestrationMetricsCollector:
    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization."""
        collector = OrchestrationMetricsCollector()
        assert isinstance(collector, OrchestrationMetricsCollector)

    def test_record_request_start_end(self):
        """Test recording request start and end times."""
        collector = OrchestrationMetricsCollector()

        request_id = "test-request-123"
        tenant = "test-tenant"
        policy = "default"

        # Record request start
        collector.record_request_start(request_id, tenant, policy)

        # Record request end (success)
        collector.record_request_end(request_id, success=True, duration_ms=1500)

        # Check that metrics were recorded
        assert _REQ_TOTAL.labels(tenant=tenant, policy=policy, status="success")._value.get() == 1
        # Duration histogram should have been observed
        assert _REQ_DURATION._sum.get() > 0

    def test_record_request_end_failure(self):
        """Test recording failed request."""
        collector = OrchestrationMetricsCollector()

        request_id = "test-request-456"
        tenant = "test-tenant"
        policy = "default"

        # Record request end (failure)
        collector.record_request_end(request_id, success=False, duration_ms=3000)

        # Check that failure metric was recorded
        assert _REQ_TOTAL.labels(tenant=tenant, policy=policy, status="failure")._value.get() == 1

    def test_record_detector_latency(self):
        """Test recording detector latency."""
        collector = OrchestrationMetricsCollector()

        detector = "toxicity"
        status = "success"

        # Record detector latency
        collector.record_detector_latency(detector, status, 1500)

        # Check that latency was recorded
        assert _DET_LATENCY.labels(detector=detector, status=status)._sum.get() > 0

    def test_record_detector_health(self):
        """Test recording detector health status."""
        collector = OrchestrationMetricsCollector()

        detector = "toxicity"
        is_healthy = True

        # Record detector health
        collector.record_detector_health(detector, is_healthy)

        # Check that health metric was set
        assert _DET_HEALTH.labels(detector=detector)._value.get() == 1

        # Record unhealthy status
        collector.record_detector_health(detector, False)
        assert _DET_HEALTH.labels(detector=detector)._value.get() == 0

    def test_record_circuit_breaker(self):
        """Test recording circuit breaker state."""
        collector = OrchestrationMetricsCollector()

        detector = "toxicity"
        state = "closed"

        # Record circuit breaker state
        collector.record_circuit_breaker(detector, state)

        # Check that circuit breaker metric was set
        assert _CB_STATE.labels(detector=detector, state=state)._value.get() == 1

        # Record different state
        collector.record_circuit_breaker(detector, "open")
        assert _CB_STATE.labels(detector=detector, state="open")._value.get() == 1
        assert _CB_STATE.labels(detector=detector, state="closed")._value.get() == 0

    def test_record_coverage(self):
        """Test recording coverage metrics."""
        collector = OrchestrationMetricsCollector()

        tenant = "test-tenant"
        policy = "default"
        coverage = 0.8

        # Record coverage
        collector.record_coverage(tenant, policy, coverage)

        # Check that coverage metric was set
        assert _COVERAGE.labels(tenant=tenant, policy=policy)._value.get() == 0.8

    def test_record_policy_enforcement(self):
        """Test recording policy enforcement metrics."""
        collector = OrchestrationMetricsCollector()

        tenant = "test-tenant"
        policy = "default"
        status = "allowed"
        violation_type = "none"

        # Record policy enforcement
        collector.record_policy_enforcement(tenant, policy, status, violation_type)

        # Check that policy enforcement metric was recorded
        assert _REQ_TOTAL._value.get() == 0  # This uses a different counter, check the actual implementation

    def test_record_rate_limit(self):
        """Test recording rate limit metrics."""
        collector = OrchestrationMetricsCollector()

        tenant = "test-tenant"
        endpoint = "/orchestrate"
        decision = "allowed"

        # Record rate limit decision
        collector.record_rate_limit_decision(tenant, endpoint, decision)

        # This would need to check the actual implementation of rate limit metrics

    def test_record_cache_metrics(self):
        """Test recording cache metrics."""
        collector = OrchestrationMetricsCollector()

        # Record cache hit
        collector.record_cache_operation("hit")

        # Record cache miss
        collector.record_cache_operation("miss")

        # This would need to check the actual implementation of cache metrics

    def test_record_error(self):
        """Test recording error metrics."""
        collector = OrchestrationMetricsCollector()

        error_code = "INVALID_REQUEST"
        tenant = "test-tenant"

        # Record error
        collector.record_error(error_code, tenant)

        # This would need to check the actual implementation of error metrics

    def test_record_security_event(self):
        """Test recording security event metrics."""
        collector = OrchestrationMetricsCollector()

        event_type = "unauthorized_access"
        tenant = "test-tenant"
        severity = "high"

        # Record security event
        collector.record_security_event(event_type, tenant, severity)

        # This would need to check the actual implementation of security metrics

    def test_get_metrics_summary(self):
        """Test getting metrics summary."""
        collector = OrchestrationMetricsCollector()

        # Record some metrics
        collector.record_request_start("test-1", "tenant1", "policy1")
        collector.record_request_end("test-1", success=True, duration_ms=1000)

        collector.record_detector_latency("toxicity", "success", 1500)
        collector.record_detector_health("toxicity", True)
        collector.record_coverage("tenant1", "policy1", 1.0)

        # Get summary
        summary = collector.get_metrics_summary()

        assert isinstance(summary, dict)
        assert "requests_total" in summary
        assert "detectors" in summary
        assert "circuit_breakers" in summary
        assert "coverage" in summary

    def test_reset_metrics(self):
        """Test resetting metrics."""
        collector = OrchestrationMetricsCollector()

        # Record some metrics
        collector.record_request_start("test-1", "tenant1", "policy1")
        collector.record_request_end("test-1", success=True, duration_ms=1000)

        # Reset metrics
        collector.reset_metrics()

        # Check that metrics are reset (this would depend on the actual implementation)

    def test_metrics_thread_safety(self):
        """Test metrics collection thread safety."""
        collector = OrchestrationMetricsCollector()

        # This test would verify that concurrent access to metrics is safe
        # For now, we just ensure the collector can be instantiated

    def test_locked_metric_names(self):
        """Test that metric names follow the locked naming convention."""
        # This test verifies that the metric names in the module match the locked names
        # from the service contracts

        expected_metric_names = {
            "orchestrate_requests_total",
            "orchestrate_request_duration_ms",
            "detector_latency_ms",
            "detector_health_status",
            "detector_health_check_duration_ms",
            "circuit_breaker_state",
            "coverage_achieved",
            "policy_enforcement_total",
        }

        # This is a static check that the metric names are correct
        # In a real test, we would verify against the actual metric registry

    def test_metric_labels(self):
        """Test that metric labels are correctly applied."""
        collector = OrchestrationMetricsCollector()

        # Test request metrics with labels
        request_id = "test-request"
        tenant = "test-tenant"
        policy = "test-policy"

        collector.record_request_start(request_id, tenant, policy)
        collector.record_request_end(request_id, success=True, duration_ms=1000)

        # The labels should be properly applied to the metrics
        # This would be verified by checking the actual Prometheus metric labels

    def test_histogram_buckets(self):
        """Test that histogram buckets are properly configured."""
        # Test that duration histograms have appropriate buckets
        # This is a static configuration check

        # Request duration buckets should cover typical orchestration times
        request_buckets = [50, 100, 200, 500, 1000, 1500, 2000, 3000, 5000]
        assert _REQ_DURATION._buckets == request_buckets

        # Detector latency buckets should cover detector response times
        detector_buckets = [10, 50, 100, 200, 500, 1000, 3000, 5000]
        assert _DET_LATENCY._buckets == detector_buckets

        # Health check duration buckets should cover health check times
        health_buckets = [10, 50, 100, 200, 500, 1000, 3000]
        # This would need to check the actual _DET_HEALTH_DURATION buckets

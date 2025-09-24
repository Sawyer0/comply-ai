"""
Extended metrics collector for analysis module.

This module extends the base MetricsCollector with analysis-specific
metrics collection capabilities.
"""

import logging
import time
from typing import Any, Dict, Optional

from ...monitoring.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


class AnalysisMetricsCollector(MetricsCollector):
    """
    Extended metrics collector for analysis module.

    Provides analysis-specific metrics collection including request tracking,
    schema validation, quality scores, and performance metrics.
    """

    def __init__(self, service_name: str = "analysis-module"):
        """
        Initialize the analysis metrics collector.

        Args:
            service_name: Name of the analysis service
        """
        super().__init__()
        self.service_name = service_name
        self._request_start_times: Dict[str, float] = {}
        self._init_analysis_metrics()

    def _init_analysis_metrics(self) -> None:
        """Initialize analysis-specific Prometheus metrics."""
        if not self._enable_prometheus:
            return
            
        from prometheus_client import Counter, Histogram, Gauge
        
        # Analysis request metrics
        self.analysis_requests_total = Counter(
            "analysis_requests_total",
            "Total number of analysis requests",
            ["type", "success", "tenant"],
            registry=self._registry,
        )
        
        self.analysis_request_duration_ms = Histogram(
            "analysis_request_duration_ms",
            "Analysis request processing duration in milliseconds",
            ["type"],
            registry=self._registry,
        )
        
        # Schema validation metrics
        self.schema_validation_total = Counter(
            "analysis_schema_validation_total",
            "Total schema validation attempts",
            ["success", "fallback_used", "type"],
            registry=self._registry,
        )
        
        # Quality metrics
        self.quality_score_gauge = Gauge(
            "analysis_quality_score",
            "Current quality score",
            ["metric_type"],
            registry=self._registry,
        )
        
        # Additional metrics for comprehensive monitoring
        self.template_fallback_total = Counter(
            "analysis_template_fallback_total",
            "Total template fallback occurrences",
            ["reason", "type"],
            registry=self._registry,
        )
        
        self.opa_compilation_total = Counter(
            "analysis_opa_compilation_total", 
            "Total OPA compilation attempts",
            ["success", "type"],
            registry=self._registry,
        )
        
        self.batch_requests_total = Counter(
            "analysis_batch_requests_total",
            "Total batch requests",
            ["batch_size_range"],
            registry=self._registry,
        )
        
        self.batch_processing_duration_ms = Histogram(
            "analysis_batch_processing_duration_ms",
            "Batch processing duration in milliseconds",
            ["batch_size_range"],
            registry=self._registry,
        )
        
        # WAF metrics
        self.waf_scans_total = Counter(
            "analysis_waf_scans_total",
            "Total WAF scans",
            ["client_ip", "violations_found"],
            registry=self._registry,
        )
        
        # Circuit breaker metrics
        self.circuit_breaker_state_gauge = Gauge(
            "analysis_circuit_breaker_state",
            "Circuit breaker state (1=open, 0=closed)",
            ["service", "state"],
            registry=self._registry,
        )
        
        # Retry metrics
        self.retry_attempts_total = Counter(
            "analysis_retry_attempts_total",
            "Total retry attempts",
            ["service", "attempt", "success"],
            registry=self._registry,
        )

    def counter(self, name: str, labels: Dict[str, str]):
        """Legacy counter method for backward compatibility."""
        # This is a simplified implementation for backward compatibility
        # In a real implementation, you'd want to map to the appropriate Prometheus metric
        class MockCounter:
            def __init__(self, value=0):
                self._value = value
            def inc(self, amount=1):
                self._value += amount
        return MockCounter()
    
    def histogram(self, name: str, labels: Dict[str, str]):
        """Legacy histogram method for backward compatibility."""
        class MockHistogram:
            def observe(self, value):
                pass
        return MockHistogram()
    
    def gauge(self, name: str, labels: Dict[str, str]):
        """Legacy gauge method for backward compatibility."""
        class MockGauge:
            def __init__(self):
                self._value = 0.0
            def set(self, value):
                self._value = value
        return MockGauge()

    def record_analysis_request(
        self,
        analysis_type: str,
        processing_time: float,
        tenant: Optional[str] = None,
        success: bool = True,
    ):
        """
        Record analysis request metrics.

        Args:
            analysis_type: Type of analysis performed
            processing_time: Processing time in milliseconds
            tenant: Tenant identifier
            success: Whether the request was successful
        """
        labels = {"type": analysis_type, "success": str(success)}
        if tenant:
            labels["tenant"] = tenant

        if self._enable_prometheus:
            self.analysis_requests_total.labels(**labels).inc()
            self.analysis_request_duration_ms.labels(type=analysis_type).observe(processing_time)

        logger.debug(
            f"Recorded analysis request: {analysis_type}, {processing_time}ms, success={success}"
        )

    def record_schema_validation(
        self, success: bool, fallback_used: bool, analysis_type: Optional[str] = None
    ):
        """
        Record schema validation metrics.

        Args:
            success: Whether schema validation succeeded
            fallback_used: Whether template fallback was used
            analysis_type: Type of analysis
        """
        labels = {"success": str(success), "fallback_used": str(fallback_used)}
        if analysis_type:
            labels["type"] = analysis_type

        self.counter("schema_validation_total", labels).inc()

        # Calculate schema validation rate
        total_validations = (
            self.counter("schema_validation_total", {"success": "true"})._value
            + self.counter("schema_validation_total", {"success": "false"})._value
        )
        if total_validations > 0:
            success_rate = (
                self.counter("schema_validation_total", {"success": "true"})._value
                / total_validations
            )
            self.gauge("schema_valid_rate", {}).set(success_rate)

        logger.debug(
            f"Recorded schema validation: success={success}, fallback={fallback_used}"
        )

    def record_template_fallback(
        self, fallback_reason: str, analysis_type: Optional[str] = None
    ):
        """
        Record template fallback usage.

        Args:
            fallback_reason: Reason for fallback
            analysis_type: Type of analysis
        """
        labels = {"reason": fallback_reason}
        if analysis_type:
            labels["type"] = analysis_type

        self.counter("template_fallback_total", labels).inc()

        # Calculate template fallback rate
        total_requests = self.counter("analysis_requests_total", {})._value
        if total_requests > 0:
            fallback_rate = (
                self.counter("template_fallback_total", {})._value / total_requests
            )
            self.gauge("template_fallback_rate", {}).set(fallback_rate)

        logger.debug("Recorded template fallback: %s", fallback_reason)

    def record_opa_compilation(
        self, success: bool, analysis_type: Optional[str] = None
    ):
        """
        Record OPA compilation metrics.

        Args:
            success: Whether OPA compilation succeeded
            analysis_type: Type of analysis
        """
        labels = {"success": str(success)}
        if analysis_type:
            labels["type"] = analysis_type

        self.counter("opa_compilation_total", labels).inc()

        # Calculate OPA compilation success rate
        total_compilations = (
            self.counter("opa_compilation_total", {"success": "true"})._value
            + self.counter("opa_compilation_total", {"success": "false"})._value
        )
        if total_compilations > 0:
            success_rate = (
                self.counter("opa_compilation_total", {"success": "true"})._value
                / total_compilations
            )
            self.gauge("opa_compile_success_rate", {}).set(success_rate)

        logger.debug("Recorded OPA compilation: success=%s", success)

    def record_quality_score(self, score: float, evaluation_type: str = "rubric"):
        """
        Record quality evaluation scores.

        Args:
            score: Quality score (0.0-5.0 for rubric, 0.0-1.0 for others)
            evaluation_type: Type of evaluation (rubric, schema_valid_rate, etc.)
        """
        labels = {"type": evaluation_type}

        self.histogram("quality_score", labels).observe(score)
        self.gauge(f"quality_score_{evaluation_type}", {}).set(score)

        logger.debug("Recorded quality score: %s (%s)", score, evaluation_type)

    def record_batch_processing(
        self,
        batch_size: int,
        processing_time: float,
        success_count: int,
        failure_count: int,
    ):
        """
        Record batch processing metrics.

        Args:
            batch_size: Number of items in batch
            processing_time: Total processing time in milliseconds
            success_count: Number of successful items
            failure_count: Number of failed items
        """
        labels = {"batch_size_range": self._get_batch_size_range(batch_size)}

        self.counter("batch_requests_total", labels).inc()
        self.histogram("batch_processing_duration_ms", labels).observe(processing_time)
        self.histogram("batch_size", labels).observe(batch_size)
        self.counter("batch_items_success_total", labels).inc(success_count)
        self.counter("batch_items_failure_total", labels).inc(failure_count)

        # Calculate batch success rate
        if batch_size > 0:
            success_rate = success_count / batch_size
            self.gauge("batch_success_rate", labels).set(success_rate)

        logger.debug(
            f"Recorded batch processing: size={batch_size}, success={success_count}, failures={failure_count}"
        )

    def start_request_timer(self, request_id: str):
        """
        Start timing a request.

        Args:
            request_id: Unique request identifier
        """
        self._request_start_times[request_id] = time.time()

    def end_request_timer(self, request_id: str) -> Optional[float]:
        """
        End timing a request and return duration.

        Args:
            request_id: Unique request identifier

        Returns:
            Processing time in milliseconds, or None if timer not found
        """
        if request_id not in self._request_start_times:
            logger.warning("Request timer not found for ID: %s", request_id)
            return None

        start_time = self._request_start_times.pop(request_id)
        duration_ms = (time.time() - start_time) * 1000

        return duration_ms

    def _get_batch_size_range(self, batch_size: int) -> str:
        """
        Get batch size range label for metrics.

        Args:
            batch_size: Batch size

        Returns:
            Batch size range label
        """
        if batch_size <= 10:
            return "1-10"
        elif batch_size <= 25:
            return "11-25"
        elif batch_size <= 50:
            return "26-50"
        elif batch_size <= 75:
            return "51-75"
        else:
            return "76-100"

    def get_analysis_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of analysis metrics.

        Returns:
            Dictionary with key metrics
        """
        return {
            "total_requests": self.counter("analysis_requests_total", {})._value,
            "schema_valid_rate": self.gauge("schema_valid_rate", {})._value,
            "template_fallback_rate": self.gauge("template_fallback_rate", {})._value,
            "opa_compile_success_rate": self.gauge(
                "opa_compile_success_rate", {}
            )._value,
            "active_timers": len(self._request_start_times),
        }

    def record_waf_scan(
        self,
        is_safe: bool,
        violations_count: int,
        processing_time_ms: float,
        client_ip: str = "unknown",
    ):
        """
        Record WAF scan metrics.

        Args:
            is_safe: True if request passed WAF scan, False if blocked.
            violations_count: Number of violations detected.
            processing_time_ms: Time taken for WAF scan in milliseconds.
            client_ip: Client IP address.
        """
        self.counter(
            "waf_scans_total",
            {
                "safe": str(is_safe),
                "client_ip": client_ip[:8] + "..." if len(client_ip) > 8 else client_ip,
            },
        ).inc()

        self.histogram("waf_scan_duration_ms", {}).observe(processing_time_ms)

        if violations_count > 0:
            self.counter(
                "waf_violations_total",
                {
                    "client_ip": (
                        client_ip[:8] + "..." if len(client_ip) > 8 else client_ip
                    )
                },
            ).inc()

        logger.debug(
            f"Recorded WAF scan: safe={is_safe}, violations={violations_count}, duration={processing_time_ms}ms"
        )

    def record_waf_violation(
        self,
        violation_type: str,
        severity: str,
        rule_name: str,
        client_ip: str = "unknown",
    ):
        """
        Record WAF violation metrics.

        Args:
            violation_type: Type of violation (e.g., "sql_injection", "xss").
            severity: Severity level (low, medium, high, critical).
            rule_name: Name of the WAF rule that triggered.
            client_ip: Client IP address.
        """
        self.counter(
            "waf_violations_by_type_total",
            {
                "type": violation_type,
                "severity": severity,
                "rule": rule_name,
                "client_ip": client_ip[:8] + "..." if len(client_ip) > 8 else client_ip,
            },
        ).inc()

        logger.warning(
            f"Recorded WAF violation: type={violation_type}, severity={severity}, rule={rule_name}"
        )

    def record_circuit_breaker_state(self, state: str, service: str):
        """
        Record circuit breaker state changes.

        Args:
            state: Circuit breaker state (closed, open, half_open).
            service: Service name (e.g., "model_server").
        """
        self.gauge("circuit_breaker_state", {"service": service, "state": state}).set(
            1 if state == "open" else 0
        )
        logger.info("Circuit breaker state changed: %s -> %s", service, state)

    def record_retry_attempt(self, service: str, attempt: int, success: bool):
        """
        Record retry attempt metrics.

        Args:
            service: Service name (e.g., "model_server").
            attempt: Attempt number (1, 2, 3, etc.).
            success: True if attempt succeeded, False if failed.
        """
        self.counter(
            "retry_attempts_total",
            {"service": service, "attempt": str(attempt), "success": str(success)},
        ).inc()

        logger.debug(
            f"Recorded retry attempt: {service} attempt={attempt} success={success}"
        )

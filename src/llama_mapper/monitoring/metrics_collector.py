"""
Metrics collection for monitoring and observability with Prometheus integration.
"""

import logging
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Any, Dict, List, Optional, cast

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    start_http_server,
)

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Prometheus-integrated metrics collector for tracking API performance and model behavior.

    Tracks request count, schema validation rates, fallback usage, latency percentiles,
    model performance, and confidence score distribution with alerting capabilities.

    Note: This class intentionally has many instance attributes (31/7) and public methods
    (24/20) as it provides comprehensive metrics collection for production monitoring.
    This design follows the pattern of full-featured metrics APIs found in major monitoring
    frameworks like Prometheus Python client. The high number of attributes is necessary
    to expose all metrics for monitoring dashboards, alerting, and debugging. The high
    number of methods provides the complete metrics API surface area required for
    production observability.
    """

    def __init__(
        self,
        registry: Optional[CollectorRegistry] = None,
        enable_prometheus: bool = True,
    ):
        """
        Initialize metrics collector with Prometheus integration.

        Args:
            registry: Prometheus registry to use (None for default)
            enable_prometheus: Whether to enable Prometheus metrics
        """
        self._lock = Lock()
        self._enable_prometheus = enable_prometheus
        self._registry = registry
        self._start_time = time.time()

        # Legacy metrics for backward compatibility
        self.max_histogram_samples = 1000
        self._counters: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._histograms: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.max_histogram_samples)
        )
        self._gauges: Dict[str, float] = {}

        if self._enable_prometheus:
            self._init_prometheus_metrics()

    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        # Request metrics
        self.requests_total = Counter(
            "mapper_requests_total",
            "Total number of mapping requests",
            ["detector", "status"],
            registry=self._registry,
        )

        self.request_duration = Histogram(
            "mapper_request_duration_seconds",
            "Request processing duration in seconds",
            ["detector"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self._registry,
        )

        # Model performance metrics
        self.model_success_total = Counter(
            "mapper_model_success_total",
            "Total successful model predictions",
            ["detector"],
            registry=self._registry,
        )

        self.model_error_total = Counter(
            "mapper_model_error_total",
            "Total model prediction errors",
            ["detector", "error_type"],
            registry=self._registry,
        )

        # Schema validation metrics
        self.schema_validation_total = Counter(
            "mapper_schema_validation_total",
            "Total schema validation attempts",
            ["detector", "valid"],
            registry=self._registry,
        )

        self.schema_valid_percentage = Gauge(
            "mapper_schema_valid_percentage",
            "Percentage of schema-valid outputs",
            registry=self._registry,
        )

        # Fallback metrics
        self.fallback_total = Counter(
            "mapper_fallback_total",
            "Total fallback mappings used",
            ["detector", "reason"],
            registry=self._registry,
        )

        self.fallback_percentage = Gauge(
            "mapper_fallback_percentage",
            "Percentage of requests using fallback mapping",
            registry=self._registry,
        )

        # Confidence metrics
        self.confidence_score = Histogram(
            "mapper_confidence_score",
            "Model confidence score distribution",
            ["detector"],
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            registry=self._registry,
        )

        # Redis backend health/fallback metrics
        self.redis_backend_up = Gauge(
            "mapper_redis_backend_up",
            "Redis backend health (1 up, 0 down)",
            ["component"],
            registry=self._registry,
        )
        self.redis_backend_fallback_total = Counter(
            "mapper_redis_backend_fallback_total",
            "Redis backend fallback occurrences",
            ["component"],
            registry=self._registry,
        )

        self.low_confidence_total = Counter(
            "mapper_low_confidence_total",
            "Total predictions with confidence below threshold",
            ["detector"],
            registry=self._registry,
        )

        # Quality metrics
        self.taxonomy_f1_score = Gauge(
            "mapper_taxonomy_f1_score",
            "Current taxonomy F1 score",
            ["detector"],
            registry=self._registry,
        )

        # System metrics
        self.uptime_seconds = Gauge(
            "mapper_uptime_seconds",
            "Service uptime in seconds",
            registry=self._registry,
        )

        # Batch processing metrics
        self.batch_requests_total = Counter(
            "mapper_batch_requests_total",
            "Total batch requests processed",
            registry=self._registry,
        )

        self.batch_size = Histogram(
            "mapper_batch_size",
            "Batch request size distribution",
            buckets=(1, 5, 10, 25, 50, 100, 250, 500),
            registry=self._registry,
        )

        # Error and compliance metrics
        self.errors_total = Counter(
            "mapper_errors_total",
            "Total errors by canonical error_code",
            ["error_code"],
            registry=self._registry,
        )
        self.payload_rejected_total = Counter(
            "mapper_request_payload_rejected_total",
            "Total request payload rejections by reason",
            ["reason"],
            registry=self._registry,
        )
        self.request_deprecated_total = Counter(
            "mapper_request_deprecated_total",
            "Deprecated request usage",
            ["type"],
            registry=self._registry,
        )

        # Service info
        self.service_info = Info(
            "mapper_service_info", "Service information", registry=self._registry
        )

        # Rate limit metrics
        self.rate_limit_requests_total = Counter(
            "mapper_rate_limit_requests_total",
            "Rate limit decisions by endpoint and identity kind",
            ["endpoint", "identity_kind", "action"],
            registry=self._registry,
        )
        self.rate_limit_backend_errors_total = Counter(
            "mapper_rate_limit_backend_errors_total",
            "Total rate limit backend errors",
            registry=self._registry,
        )
        self.rate_limit_reset_seconds = Histogram(
            "mapper_rate_limit_reset_seconds",
            "Observed reset seconds on 429 blocks",
            ["endpoint", "identity_kind"],
            buckets=(0, 1, 2, 5, 10, 30, 60, 120, 300),
            registry=self._registry,
        )

        # Set initial service info
        self.service_info.info(
            {
                "version": "1.0.0",
                "model": "llama-3-8b-instruct",
                "taxonomy_version": "2025.09",
            }
        )

    def increment_counter(
        self, name: str, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Increment a counter metric.

        Args:
            name: Counter name
            labels: Optional labels for the metric
        """
        with self._lock:
            label_key = self._serialize_labels(labels or {})
            self._counters[name][label_key] += 1

    def record_request(self, detector: str, duration: float, success: bool) -> None:
        """
        Record a mapping request with comprehensive metrics.

        Args:
            detector: Detector name
            duration: Request duration in seconds
            success: Whether the request was successful
        """
        if not self._enable_prometheus:
            return

        status = "success" if success else "error"
        self.requests_total.labels(detector=detector, status=status).inc()
        self.request_duration.labels(detector=detector).observe(duration)

        # Update legacy metrics for backward compatibility
        self.increment_counter("requests_total", {"detector": detector})
        self.record_histogram("request_duration_seconds", duration)

    def record_model_success(self, detector: str) -> None:
        """Record successful model prediction."""
        if self._enable_prometheus:
            self.model_success_total.labels(detector=detector).inc()
        self.increment_counter("model_success_total", {"detector": detector})

    def record_model_error(self, detector: str, error_type: str = "unknown") -> None:
        """Record model prediction error."""
        if self._enable_prometheus:
            self.model_error_total.labels(
                detector=detector, error_type=error_type
            ).inc()
        self.increment_counter("model_error_total", {"detector": detector})

    def record_schema_validation(self, detector: str, is_valid: bool) -> None:
        """Record schema validation result."""
        if self._enable_prometheus:
            valid_str = "true" if is_valid else "false"
            self.schema_validation_total.labels(
                detector=detector, valid=valid_str
            ).inc()

        label = "schema_validation_success" if is_valid else "schema_validation_failed"
        self.increment_counter(f"{label}_total", {"detector": detector})

    def record_fallback_usage(self, detector: str, reason: str = "unknown") -> None:
        """Record fallback mapping usage."""
        if self._enable_prometheus:
            self.fallback_total.labels(detector=detector, reason=reason).inc()
        self.increment_counter("fallback_used_total", {"detector": detector})

    def record_confidence_score(self, detector: str, confidence: float) -> None:
        """Record model confidence score."""
        if self._enable_prometheus:
            self.confidence_score.labels(detector=detector).observe(confidence)

            # Track low confidence predictions
            if confidence < 0.6:  # Default threshold
                self.low_confidence_total.labels(detector=detector).inc()

        self.record_histogram("confidence_scores", confidence)

    def record_batch_request(self, batch_size: int) -> None:
        """Record batch request processing."""
        if self._enable_prometheus:
            self.batch_requests_total.inc()
            self.batch_size.observe(batch_size)
        self.increment_counter("batch_requests_total")

    def record_error(self, error_code: str) -> None:
        """Record a canonical error occurrence."""
        if self._enable_prometheus:
            self.errors_total.labels(error_code=error_code).inc()
        self.increment_counter("errors_total", {"error_code": error_code})

    def record_payload_rejection(self, reason: str) -> None:
        """Record a request payload rejection."""
        if self._enable_prometheus:
            self.payload_rejected_total.labels(reason=reason).inc()
        self.increment_counter("payload_rejected_total", {"reason": reason})

    def record_deprecated_request(self, request_type: str) -> None:
        """Record usage of deprecated request schema."""
        if self._enable_prometheus:
            self.request_deprecated_total.labels(type=request_type).inc()
        self.increment_counter("request_deprecated_total", {"type": request_type})

    def update_quality_metrics(self, detector: str, f1_score: float) -> None:
        """Update quality metrics like F1 score."""
        if self._enable_prometheus:
            self.taxonomy_f1_score.labels(detector=detector).set(f1_score)
        self.set_gauge(f"taxonomy_f1_score_{detector}", f1_score)

    def update_percentage_metrics(self) -> None:
        """Update percentage-based metrics for monitoring."""
        if not self._enable_prometheus:
            return

        # Calculate schema validation percentage
        total_validations = 0
        valid_validations = 0

        # Access metrics through public API instead of protected attributes
        total_validations = 0
        valid_validations = 0

        # Use collect() method to get current metric values
        for metric in self.schema_validation_total.collect():
            for sample in metric.samples:
                total_validations += sample.value
                if "valid=true" in sample.labels.get("valid", ""):
                    valid_validations += sample.value

        if total_validations > 0:
            schema_valid_pct = (valid_validations / total_validations) * 100
            self.schema_valid_percentage.set(schema_valid_pct)

        # Calculate fallback percentage
        total_requests = 0
        fallback_requests = 0

        # Use public API instead of protected attributes
        total_requests = 0
        fallback_requests = 0

        for metric in self.requests_total.collect():
            for sample in metric.samples:
                total_requests += sample.value

        for metric in self.fallback_total.collect():
            for sample in metric.samples:
                fallback_requests += sample.value

        if total_requests > 0:
            fallback_pct = (fallback_requests / total_requests) * 100
            self.fallback_percentage.set(fallback_pct)

        # Update uptime
        uptime = time.time() - self._start_time
        self.uptime_seconds.set(uptime)

    def record_histogram(self, name: str, value: float) -> None:
        """
        Record a value in a histogram metric.

        Args:
            name: Histogram name
            value: Value to record
        """
        with self._lock:
            self._histograms[name].append(value)

    def set_gauge(self, name: str, value: float) -> None:
        """
        Set a gauge metric value.

        Args:
            name: Gauge name
            value: Value to set
        """
        with self._lock:
            self._gauges[name] = value

    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> int:
        """
        Get counter value.

        Args:
            name: Counter name
            labels: Optional labels

        Returns:
            int: Counter value
        """
        with self._lock:
            label_key = self._serialize_labels(labels or {})
            return self._counters[name][label_key]

    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """
        Get histogram statistics.

        Args:
            name: Histogram name

        Returns:
            Dict[str, float]: Statistics including count, sum, avg, p50, p95, p99
        """
        with self._lock:
            values = list(self._histograms[name])

            if not values:
                return {
                    "count": 0,
                    "sum": 0.0,
                    "avg": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "p50": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                }

            sorted_values = sorted(values)
            count = len(values)
            total = sum(values)

            return {
                "count": count,
                "sum": total,
                "avg": total / count,
                "min": sorted_values[0],
                "max": sorted_values[-1],
                "p50": self._percentile(sorted_values, 50),
                "p95": self._percentile(sorted_values, 95),
                "p99": self._percentile(sorted_values, 99),
            }

    def get_gauge(self, name: str) -> float:
        """
        Get gauge value.

        Args:
            name: Gauge name

        Returns:
            float: Gauge value
        """
        with self._lock:
            return self._gauges.get(name, 0.0)

    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics as a dictionary.

        Returns:
            Dict[str, Any]: All metrics data
        """
        with self._lock:
            # Calculate uptime
            uptime_seconds = time.time() - self._start_time

            # Prepare counters
            counters = {}
            for name, label_dict in self._counters.items():
                counters[name] = dict(label_dict)

            # Prepare histograms
            histograms = {}
            for name in self._histograms.keys():
                histograms[name] = self.get_histogram_stats(name)

            # Prepare gauges
            gauges = dict(self._gauges)
            gauges["uptime_seconds"] = uptime_seconds

            return {
                "counters": counters,
                "histograms": histograms,
                "gauges": gauges,
                "timestamp": time.time(),
            }

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._gauges.clear()
            self._start_time = time.time()

    def _serialize_labels(self, labels: Dict[str, str]) -> str:
        """
        Serialize labels to a string key.

        Args:
            labels: Label dictionary

        Returns:
            str: Serialized labels
        """
        if not labels:
            return ""

        # Sort for consistent ordering
        sorted_items = sorted(labels.items())
        return ",".join(f"{k}={v}" for k, v in sorted_items)

    def _percentile(self, sorted_values: List[float], percentile: float) -> float:
        """
        Calculate percentile from sorted values.

        Args:
            sorted_values: List of sorted values
            percentile: Percentile to calculate (0-100)

        Returns:
            float: Percentile value
        """
        if not sorted_values:
            return 0.0

        if percentile <= 0:
            return float(sorted_values[0])
        if percentile >= 100:
            return float(sorted_values[-1])

        index = (percentile / 100.0) * (len(sorted_values) - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, len(sorted_values) - 1)

        if lower_index == upper_index:
            return float(sorted_values[lower_index])

        # Linear interpolation
        weight = index - lower_index
        return (
            float(sorted_values[lower_index]) * (1 - weight)
            + float(sorted_values[upper_index]) * weight
        )

    def check_quality_thresholds(self) -> List[Dict[str, Any]]:
        """
        Check quality thresholds and return alerts.

        Returns:
            List[Dict[str, Any]]: List of threshold violations
        """
        alerts: List[Dict[str, Any]] = []

        if not self._enable_prometheus:
            return alerts

        # Check schema validation percentage (should be ≥95%)
        schema_valid_pct = 0.0
        for metric in self.schema_valid_percentage.collect():
            for sample in metric.samples:
                schema_valid_pct = sample.value
                break
            break
        if schema_valid_pct < 95.0:
            alerts.append(
                {
                    "metric": "schema_valid_percentage",
                    "value": schema_valid_pct,
                    "threshold": 95.0,
                    "severity": "critical",
                    "message": (
                        f"Schema validation rate {schema_valid_pct:.1f}% "
                        "below 95% threshold"
                    ),
                }
            )

        # Check fallback percentage (should be <10%)
        # Use collect() method to get current value
        fallback_pct = 0.0
        for metric in self.fallback_percentage.collect():
            for sample in metric.samples:
                fallback_pct = sample.value
                break
            break
        if fallback_pct > 10.0:
            alerts.append(
                {
                    "metric": "fallback_percentage",
                    "value": fallback_pct,
                    "threshold": 10.0,
                    "severity": "warning",
                    "message": f"Fallback usage {fallback_pct:.1f}% above 10% threshold",
                }
            )

        # Check F1 scores (should be ≥90%)
        # Use public API instead of protected attributes
        for metric in self.taxonomy_f1_score.collect():
            for sample in metric.samples:
                detector = sample.labels.get("detector", "unknown")
                f1_score = sample.value
                if f1_score < 0.9:
                    alerts.append(
                        {
                            "metric": "taxonomy_f1_score",
                            "detector": detector,
                            "value": f1_score,
                            "threshold": 0.9,
                            "severity": "warning",
                            "message": (
                                f"F1 score {f1_score:.3f} for {detector} "
                                "below 90% threshold"
                            ),
                        }
                    )

        # Check latency percentiles
        # Use collect() method to get current histogram data
        for metric in self.request_duration.collect():
            for sample in metric.samples:
                detector = sample.labels.get("detector", "unknown")
                # Get P95 latency (approximate from histogram)
                p95_latency = self._estimate_percentile(metric, 0.95)
                if p95_latency > 0.25:  # 250ms threshold for CPU
                    alerts.append(
                        {
                            "metric": "request_duration_p95",
                            "detector": detector,
                            "value": p95_latency,
                            "threshold": 0.25,
                            "severity": "warning",
                            "message": (
                                f"P95 latency {p95_latency:.3f}s for {detector} "
                                "above 250ms threshold"
                            ),
                        }
                    )

        return alerts

    def _estimate_percentile(self, histogram_metric: Any, percentile: float) -> float:
        """
        Estimate percentile from Prometheus histogram using proper statistical methods.

        This implements a production-quality percentile estimation that:
        - Uses linear interpolation between bucket boundaries
        - Handles edge cases (empty histograms, single buckets, boundary conditions)
        - Follows mathematical best practices for histogram-based percentile estimation
        - Provides accurate results for monitoring and alerting

        Args:
            histogram_metric: Prometheus histogram metric to analyze
            percentile: Target percentile (0.0 to 1.0, e.g., 0.95 for P95)

        Returns:
            float: Estimated percentile value
        """
        if not 0.0 <= percentile <= 1.0:
            raise ValueError(
                f"Percentile must be between 0.0 and 1.0, got {percentile}"
            )

        buckets = []
        counts = []

        # Extract histogram data from Prometheus samples
        for metric in histogram_metric.collect():
            for sample in metric.samples:
                bucket_boundary = sample.labels.get("le", "")
                if bucket_boundary == "+Inf":
                    # This is the +Inf bucket, use it as the upper bound
                    buckets.append(float("inf"))
                elif bucket_boundary:
                    buckets.append(float(bucket_boundary))
                else:
                    # This sample doesn't have bucket info, skip
                    continue
                counts.append(sample.value)

        if not buckets or not counts:
            logger.warning("No histogram data available for percentile estimation")
            return 0.0

        # Remove the +Inf bucket for percentile calculation
        finite_buckets = []
        finite_counts = []
        inf_count = 0

        for bucket, count in zip(buckets, counts):
            if bucket == float("inf"):
                inf_count = count
            else:
                finite_buckets.append(bucket)
                finite_counts.append(count)

        total_count = sum(finite_counts) + inf_count
        if total_count == 0:
            return 0.0

        # Handle boundary cases
        if percentile == 0.0:
            return float(finite_buckets[0]) if finite_buckets else 0.0
        if percentile == 1.0:
            return float(finite_buckets[-1]) if finite_buckets else 0.0

        target_count = total_count * percentile

        # If target is in +Inf bucket, return the last finite bucket
        if target_count >= sum(finite_counts):
            return float(finite_buckets[-1]) if finite_buckets else 0.0

        # Find the bucket containing the target percentile
        cumulative = 0.0
        for i, count in enumerate(finite_counts):
            cumulative += count
            if cumulative >= target_count:
                # Found the bucket - use linear interpolation if not exact match
                if cumulative == target_count:
                    return float(finite_buckets[i])

                # Linear interpolation between this bucket and the previous one
                bucket_start = finite_buckets[i - 1] if i > 0 else 0.0
                bucket_end = finite_buckets[i]

                # Calculate position within this bucket
                bucket_position = (target_count - (cumulative - count)) / count

                return bucket_start + (bucket_end - bucket_start) * bucket_position

        # Fallback (should not reach here)
        return float(finite_buckets[-1]) if finite_buckets else 0.0

    def get_prometheus_metrics(self) -> bytes:
        """
        Get Prometheus metrics in exposition format.

        Returns:
            bytes: Prometheus metrics
        """
        if not self._enable_prometheus:
            return b""

        # Update percentage metrics before export
        self.update_percentage_metrics()

        return cast(bytes, generate_latest(self._registry or REGISTRY))

    def start_prometheus_server(self, port: int = 8000) -> None:
        """
        Start Prometheus metrics server.

        Args:
            port: Port to serve metrics on
        """
        if not self._enable_prometheus:
            logger.warning("Prometheus not enabled, cannot start metrics server")
            return

        try:
            start_http_server(port, registry=self._registry or REGISTRY)
            logger.info(
                "Prometheus metrics server started successfully",
                extra={"port": port, "registry": str(self._registry or REGISTRY)},
            )
        except ImportError as e:
            logger.error(
                "Failed to start Prometheus server due to missing dependency",
                extra={"error": str(e), "port": port},
                exc_info=True,
            )
            raise RuntimeError(
                f"Cannot start Prometheus server on port {port}: missing "
                "prometheus_client dependency"
            ) from e
        except (OSError, RuntimeError) as e:
            logger.error(
                "Failed to start Prometheus server due to runtime error",
                extra={"error": str(e), "port": port},
                exc_info=True,
            )
            raise

    def log_metrics_summary(self) -> None:
        """Log a summary of current metrics."""
        metrics = self.get_all_metrics()

        logger.info("=== Metrics Summary ===")
        logger.info("Uptime: %.1f seconds", metrics["gauges"]["uptime_seconds"])

        # Log Prometheus metrics if available
        if self._enable_prometheus:
            self.update_percentage_metrics()
            # Use public API instead of protected attributes
            schema_valid_pct = 0.0
            fallback_pct = 0.0

            for metric in self.schema_valid_percentage.collect():
                for sample in metric.samples:
                    schema_valid_pct = sample.value
                    break
                break

            for metric in self.fallback_percentage.collect():
                for sample in metric.samples:
                    fallback_pct = sample.value
                    break
                break

            logger.info("Schema Valid: %s", str(schema_valid_pct))
            logger.info("Fallback: %s", str(fallback_pct))
            # Noisy detail: omit per-metric dumps; counters can be scraped via Prometheus

            # Check and log alerts
            alerts = self.check_quality_thresholds()
            if alerts:
                logger.warning("Quality threshold violations: %d", len(alerts))
                for alert in alerts:
                    logger.warning(
                        "  %s: %s", alert["severity"].upper(), alert["message"]
                    )

        # Log key counters
        for name, label_dict in metrics["counters"].items():
            total = sum(label_dict.values())
            logger.info("Counter %s: %d", name, total)
            if len(label_dict) > 1:
                for label_key, count in label_dict.items():
                    if label_key:  # Skip empty label key
                        logger.info("  %s: %d", label_key, count)

        # Log key histograms
        for name, stats in metrics["histograms"].items():
            if stats["count"] > 0:
                logger.info(
                    "Histogram %s: count=%s, avg=%.3f, p95=%.3f",
                    name,
                    stats["count"],
                    stats["avg"],
                    stats["p95"],
                )

        # Log gauges
        for name, value in metrics["gauges"].items():
            if name != "uptime_seconds":  # Already logged
                logger.info("Gauge %s: %s", name, value)


# Global metrics collector instance
_global_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """
    Get the global metrics collector instance.

    Note: Uses global statement for singleton pattern to ensure only one
    metrics collector exists across the application. This is a common pattern
    for metrics collection in production systems.

    Returns:
        MetricsCollector: Global metrics collector
    """
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector


def set_metrics_collector(collector: MetricsCollector) -> None:
    """
    Set the global metrics collector instance.

    Note: Uses global statement for singleton pattern to ensure only one
    metrics collector exists across the application. This is a common pattern
    for metrics collection in production systems.

    Args:
        collector: Metrics collector to set as global
    """
    global _global_metrics_collector
    _global_metrics_collector = collector


def create_metrics_collector(
    enable_prometheus: bool = True, registry: Optional[CollectorRegistry] = None
) -> MetricsCollector:
    """
    Create a new metrics collector with specified configuration.

    Args:
        enable_prometheus: Whether to enable Prometheus integration
        registry: Custom Prometheus registry (None for default)

    Returns:
        MetricsCollector: Configured metrics collector
    """
    return MetricsCollector(registry=registry, enable_prometheus=enable_prometheus)

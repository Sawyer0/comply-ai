"""
Metrics Collector for Mapper Service

Single Responsibility: Collect, aggregate, and expose service metrics.
Handles business metrics, performance metrics, and system health metrics.
"""

import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import threading

import structlog

logger = structlog.get_logger(__name__)


class MetricType(str, Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricValue:
    """Individual metric value with timestamp."""

    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Aggregated metric summary."""

    name: str
    metric_type: MetricType
    current_value: Union[int, float]
    total_count: int
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    avg_value: Optional[float] = None
    p95_value: Optional[float] = None
    p99_value: Optional[float] = None
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Metrics Collector for service monitoring.

    Single Responsibility: Collect and aggregate metrics for monitoring systems.

    This class handles:
    - Metric collection (counters, gauges, histograms, timers)
    - Metric aggregation and summarization
    - Time-series data management
    - Metric export for monitoring systems
    - Performance optimization for high-frequency metrics
    """

    def __init__(self, retention_hours: int = 24, max_samples_per_metric: int = 10000):
        self.retention_hours = retention_hours
        self.max_samples_per_metric = max_samples_per_metric

        # Thread-safe metric storage
        self._lock = threading.RLock()
        self._metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_samples_per_metric)
        )
        self._metric_types: Dict[str, MetricType] = {}
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}

        self.logger = logger.bind(component="metrics_collector")

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_old_metrics, daemon=True
        )
        self._cleanup_thread.start()

    def increment_counter(
        self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None
    ):
        """
        Increment a counter metric.

        Single Responsibility: Track cumulative values that only increase.
        """
        with self._lock:
            metric_key = self._get_metric_key(name, labels)
            self._counters[metric_key] += value
            self._metric_types[metric_key] = MetricType.COUNTER

            # Store individual sample for time-series
            self._metrics[metric_key].append(
                MetricValue(
                    value=self._counters[metric_key],
                    timestamp=datetime.utcnow(),
                    labels=labels or {},
                )
            )

    def set_gauge(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ):
        """
        Set a gauge metric value.

        Single Responsibility: Track values that can go up or down.
        """
        with self._lock:
            metric_key = self._get_metric_key(name, labels)
            self._gauges[metric_key] = value
            self._metric_types[metric_key] = MetricType.GAUGE

            # Store individual sample for time-series
            self._metrics[metric_key].append(
                MetricValue(
                    value=value, timestamp=datetime.utcnow(), labels=labels or {}
                )
            )

    def record_histogram(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ):
        """
        Record a histogram metric value.

        Single Responsibility: Track distribution of values over time.
        """
        with self._lock:
            metric_key = self._get_metric_key(name, labels)
            self._metric_types[metric_key] = MetricType.HISTOGRAM

            # Store individual sample
            self._metrics[metric_key].append(
                MetricValue(
                    value=value, timestamp=datetime.utcnow(), labels=labels or {}
                )
            )

    def time_operation(self, name: str, labels: Optional[Dict[str, str]] = None):
        """
        Context manager for timing operations.

        Single Responsibility: Measure operation duration.
        """
        return TimerContext(self, name, labels)

    def record_timer(
        self, name: str, duration: float, labels: Optional[Dict[str, str]] = None
    ):
        """
        Record a timer metric value.

        Single Responsibility: Track operation durations.
        """
        with self._lock:
            metric_key = self._get_metric_key(name, labels)
            self._metric_types[metric_key] = MetricType.TIMER

            # Store individual sample
            self._metrics[metric_key].append(
                MetricValue(
                    value=duration, timestamp=datetime.utcnow(), labels=labels or {}
                )
            )

    def get_metric_summary(
        self, name: str, labels: Optional[Dict[str, str]] = None
    ) -> Optional[MetricSummary]:
        """
        Get aggregated summary for a metric.

        Single Responsibility: Provide metric aggregation and statistics.
        """
        with self._lock:
            metric_key = self._get_metric_key(name, labels)

            if metric_key not in self._metrics:
                return None

            samples = list(self._metrics[metric_key])
            if not samples:
                return None

            metric_type = self._metric_types.get(metric_key, MetricType.GAUGE)
            values = [sample.value for sample in samples]

            # Calculate statistics
            current_value = values[-1] if values else 0
            total_count = len(values)
            min_value = min(values) if values else None
            max_value = max(values) if values else None
            avg_value = sum(values) / len(values) if values else None

            # Calculate percentiles for histograms and timers
            p95_value = None
            p99_value = None
            if metric_type in [MetricType.HISTOGRAM, MetricType.TIMER] and values:
                sorted_values = sorted(values)
                p95_idx = int(0.95 * len(sorted_values))
                p99_idx = int(0.99 * len(sorted_values))
                p95_value = (
                    sorted_values[p95_idx]
                    if p95_idx < len(sorted_values)
                    else sorted_values[-1]
                )
                p99_value = (
                    sorted_values[p99_idx]
                    if p99_idx < len(sorted_values)
                    else sorted_values[-1]
                )

            return MetricSummary(
                name=name,
                metric_type=metric_type,
                current_value=current_value,
                total_count=total_count,
                min_value=min_value,
                max_value=max_value,
                avg_value=avg_value,
                p95_value=p95_value,
                p99_value=p99_value,
                labels=labels or {},
            )

    def get_all_metrics(self) -> Dict[str, MetricSummary]:
        """
        Get summaries for all metrics.

        Single Responsibility: Provide complete metrics overview.
        """
        with self._lock:
            summaries = {}

            for metric_key in self._metrics.keys():
                name, labels = self._parse_metric_key(metric_key)
                summary = self.get_metric_summary(name, labels)
                if summary:
                    summaries[metric_key] = summary

            return summaries

    def get_time_series(
        self, name: str, labels: Optional[Dict[str, str]] = None, hours: int = 1
    ) -> List[MetricValue]:
        """
        Get time-series data for a metric.

        Single Responsibility: Provide historical metric data.
        """
        with self._lock:
            metric_key = self._get_metric_key(name, labels)

            if metric_key not in self._metrics:
                return []

            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            samples = list(self._metrics[metric_key])

            # Filter by time range
            return [sample for sample in samples if sample.timestamp >= cutoff_time]

    def export_prometheus_format(self) -> str:
        """
        Export metrics in Prometheus format.

        Single Responsibility: Format metrics for Prometheus scraping.
        """
        lines = []
        summaries = self.get_all_metrics()

        for metric_key, summary in summaries.items():
            # Add metric help and type
            lines.append(f"# HELP {summary.name} {summary.metric_type.value} metric")
            lines.append(f"# TYPE {summary.name} {summary.metric_type.value}")

            # Format labels
            label_str = ""
            if summary.labels:
                label_pairs = [f'{k}="{v}"' for k, v in summary.labels.items()]
                label_str = "{" + ",".join(label_pairs) + "}"

            # Add metric value
            if summary.metric_type == MetricType.COUNTER:
                lines.append(f"{summary.name}_total{label_str} {summary.current_value}")
            elif summary.metric_type == MetricType.GAUGE:
                lines.append(f"{summary.name}{label_str} {summary.current_value}")
            elif summary.metric_type in [MetricType.HISTOGRAM, MetricType.TIMER]:
                lines.append(f"{summary.name}_count{label_str} {summary.total_count}")
                lines.append(
                    f"{summary.name}_sum{label_str} {summary.total_count * (summary.avg_value or 0)}"
                )
                if summary.p95_value is not None:
                    lines.append(f"{summary.name}_p95{label_str} {summary.p95_value}")
                if summary.p99_value is not None:
                    lines.append(f"{summary.name}_p99{label_str} {summary.p99_value}")

            lines.append("")  # Empty line between metrics

        return "\n".join(lines)

    def _get_metric_key(
        self, name: str, labels: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate unique key for metric with labels."""
        if not labels:
            return name

        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}[{label_str}]"

    def _parse_metric_key(self, metric_key: str) -> tuple:
        """Parse metric key back to name and labels."""
        if "[" not in metric_key:
            return metric_key, {}

        name, label_part = metric_key.split("[", 1)
        label_part = label_part.rstrip("]")

        labels = {}
        if label_part:
            for pair in label_part.split(","):
                k, v = pair.split("=", 1)
                labels[k] = v

        return name, labels

    def _cleanup_old_metrics(self):
        """Background thread to clean up old metric samples."""
        while True:
            try:
                time.sleep(3600)  # Run every hour

                cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)

                with self._lock:
                    for metric_key in list(self._metrics.keys()):
                        samples = self._metrics[metric_key]

                        # Remove old samples
                        while samples and samples[0].timestamp < cutoff_time:
                            samples.popleft()

                        # Remove empty metrics
                        if not samples:
                            del self._metrics[metric_key]
                            self._metric_types.pop(metric_key, None)
                            self._counters.pop(metric_key, None)
                            self._gauges.pop(metric_key, None)

                self.logger.debug("Cleaned up old metrics")

            except (KeyError, ValueError, RuntimeError) as e:
                self.logger.error("Error cleaning up metrics", error=str(e))


class TimerContext:
    """Context manager for timing operations."""

    def __init__(
        self,
        collector: MetricsCollector,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.collector.record_timer(self.name, duration, self.labels)

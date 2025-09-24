"""
Quality monitoring implementation.

This module provides the core quality monitoring functionality
for tracking and analyzing quality metrics over time.

Features:
- Thread-safe metric storage and retrieval
- Statistical analysis and trend detection
- Configurable retention and cleanup
- Comprehensive error handling and logging
- Performance monitoring and metrics
"""

import logging
import math
import statistics
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from threading import Lock, RLock
from typing import Any, Dict, List, Optional, Tuple

from ..interfaces import IQualityMonitor, QualityMetric, QualityMetricType

logger = logging.getLogger(__name__)


class QualityMonitor(IQualityMonitor):
    """
    Quality monitor implementation.

    Provides in-memory storage and analysis of quality metrics
    with configurable retention periods and statistical analysis.
    """

    def __init__(
        self,
        max_metrics_per_type: int = 10000,
        default_retention_hours: int = 24,
        cleanup_interval_minutes: int = 60,
    ):
        """
        Initialize quality monitor.

        Args:
            max_metrics_per_type: Maximum number of metrics to store per type
            default_retention_hours: Default retention period in hours
            cleanup_interval_minutes: Cleanup interval in minutes

        Raises:
            ValueError: If configuration parameters are invalid
        """
        if max_metrics_per_type <= 0:
            raise ValueError(
                f"max_metrics_per_type must be positive: {max_metrics_per_type}"
            )

        if default_retention_hours <= 0:
            raise ValueError(
                f"default_retention_hours must be positive: {default_retention_hours}"
            )

        if cleanup_interval_minutes <= 0:
            raise ValueError(
                f"cleanup_interval_minutes must be positive: {cleanup_interval_minutes}"
            )

        self.max_metrics_per_type = max_metrics_per_type
        self.default_retention_hours = default_retention_hours
        self.cleanup_interval_minutes = cleanup_interval_minutes

        # Storage for metrics by type and labels
        self._metrics: Dict[QualityMetricType, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=max_metrics_per_type))
        )

        # Current metric values for quick access
        self._current_metrics: Dict[QualityMetricType, float] = {}

        # Thread safety - use RLock for nested locking
        self._lock = RLock()

        # Last cleanup time
        self._last_cleanup = time.time()

        # Performance metrics
        self._performance_stats = {
            "metrics_recorded": 0,
            "metrics_retrieved": 0,
            "cleanup_operations": 0,
            "errors": 0,
            "last_error": None,
        }

        logger.info(
            f"Quality monitor initialized: max_metrics={max_metrics_per_type}, "
            f"retention={default_retention_hours}h, cleanup_interval={cleanup_interval_minutes}m"
        )

    def record_metric(self, metric: QualityMetric) -> None:
        """
        Record a quality metric.

        Args:
            metric: Quality metric to record

        Raises:
            ValueError: If metric is invalid
            RuntimeError: If recording fails
        """
        if not isinstance(metric, QualityMetric):
            raise ValueError(f"Expected QualityMetric, got {type(metric)}")

        try:
            with self._lock:
                # Create label key for grouping
                label_key = self._create_label_key(metric.labels)

                # Store metric
                self._metrics[metric.metric_type][label_key].append(metric)

                # Update current value
                self._current_metrics[metric.metric_type] = metric.value

                # Update performance stats
                self._performance_stats["metrics_recorded"] += 1

                # Cleanup old metrics if needed
                self._cleanup_if_needed()

                logger.debug(
                    f"Recorded metric: {metric.metric_type.value}={metric.value:.3f} "
                    f"at {metric.timestamp} with labels {metric.labels}"
                )

        except Exception as e:
            self._performance_stats["errors"] += 1
            self._performance_stats["last_error"] = str(e)
            logger.error("Failed to record metric %s: %s", metric.metric_type.value, e)
            raise RuntimeError(f"Failed to record metric: {e}") from e

    def get_metrics(
        self,
        metric_type: QualityMetricType,
        start_time: datetime,
        end_time: datetime,
        labels: Optional[Dict[str, str]] = None,
    ) -> List[QualityMetric]:
        """Get metrics for a time range."""
        with self._lock:
            metrics = []

            if metric_type not in self._metrics:
                return metrics

            # Get label key for filtering
            label_key = self._create_label_key(labels or {})

            # Get metrics for the specific label combination
            if label_key in self._metrics[metric_type]:
                for metric in self._metrics[metric_type][label_key]:
                    if start_time <= metric.timestamp <= end_time:
                        metrics.append(metric)

            # If no specific labels, get all metrics for the type
            elif not labels:
                for label_metrics in self._metrics[metric_type].values():
                    for metric in label_metrics:
                        if start_time <= metric.timestamp <= end_time:
                            metrics.append(metric)

            # Sort by timestamp
            metrics.sort(key=lambda m: m.timestamp)

            logger.debug(
                f"Retrieved {len(metrics)} metrics for {metric_type.value} "
                f"from {start_time} to {end_time}"
            )

            return metrics

    def get_current_metrics(self) -> Dict[QualityMetricType, float]:
        """Get current metric values."""
        with self._lock:
            return self._current_metrics.copy()

    def get_metric_statistics(
        self, metric_type: QualityMetricType, time_window_minutes: int = 60
    ) -> Dict[str, float]:
        """Get metric statistics for a time window."""
        with self._lock:
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=time_window_minutes)

            metrics = self.get_metrics(metric_type, start_time, end_time)

            if not metrics:
                return {
                    "count": 0,
                    "mean": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "std": 0.0,
                    "p50": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                }

            values = [m.value for m in metrics]
            values.sort()

            # Calculate statistics
            count = len(values)
            mean = sum(values) / count
            min_val = values[0]
            max_val = values[-1]

            # Standard deviation
            variance = sum((x - mean) ** 2 for x in values) / count
            std = variance**0.5

            # Percentiles
            p50 = self._percentile(values, 50)
            p95 = self._percentile(values, 95)
            p99 = self._percentile(values, 99)

            stats = {
                "count": count,
                "mean": mean,
                "min": min_val,
                "max": max_val,
                "std": std,
                "p50": p50,
                "p95": p95,
                "p99": p99,
            }

            logger.debug(
                f"Calculated statistics for {metric_type.value}: "
                f"count={count}, mean={mean:.3f}, std={std:.3f}"
            )

            return stats

    def get_metric_trends(
        self,
        metric_type: QualityMetricType,
        time_window_minutes: int = 60,
        trend_window_minutes: int = 15,
    ) -> Dict[str, Any]:
        """Get metric trends and patterns."""
        with self._lock:
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=time_window_minutes)

            metrics = self.get_metrics(metric_type, start_time, end_time)

            if len(metrics) < 2:
                return {
                    "trend": "insufficient_data",
                    "slope": 0.0,
                    "r_squared": 0.0,
                    "volatility": 0.0,
                    "recent_change": 0.0,
                }

            # Calculate trend using linear regression
            timestamps = [(m.timestamp - start_time).total_seconds() for m in metrics]
            values = [m.value for m in metrics]

            slope, r_squared = self._linear_regression(timestamps, values)

            # Calculate volatility (standard deviation of changes)
            changes = [values[i] - values[i - 1] for i in range(1, len(values))]
            volatility = (
                (
                    sum((c - sum(changes) / len(changes)) ** 2 for c in changes)
                    / len(changes)
                )
                ** 0.5
                if changes
                else 0.0
            )

            # Recent change (last trend_window_minutes)
            recent_cutoff = end_time - timedelta(minutes=trend_window_minutes)
            recent_metrics = [m for m in metrics if m.timestamp >= recent_cutoff]

            if len(recent_metrics) >= 2:
                recent_change = recent_metrics[-1].value - recent_metrics[0].value
            else:
                recent_change = 0.0

            # Determine trend direction
            # Use a more sensitive threshold for trend detection
            if abs(slope) < 0.0001:  # Much more sensitive threshold
                trend = "stable"
            elif slope > 0:
                trend = "increasing"
            else:
                trend = "decreasing"

            return {
                "trend": trend,
                "slope": slope,
                "r_squared": r_squared,
                "volatility": volatility,
                "recent_change": recent_change,
                "data_points": len(metrics),
            }

    def _create_label_key(self, labels: Dict[str, str]) -> str:
        """Create a key for grouping metrics by labels."""
        if not labels:
            return "default"

        # Sort labels for consistent key generation
        sorted_labels = sorted(labels.items())
        return "|".join(f"{k}={v}" for k, v in sorted_labels)

    def _cleanup_if_needed(self) -> None:
        """Cleanup old metrics if cleanup interval has passed."""
        current_time = time.time()
        if current_time - self._last_cleanup < self.cleanup_interval_minutes * 60:
            return

        self._cleanup_old_metrics()
        self._last_cleanup = current_time

    def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period."""
        cutoff_time = datetime.now() - timedelta(hours=self.default_retention_hours)

        for metric_type in list(self._metrics.keys()):
            for label_key in list(self._metrics[metric_type].keys()):
                metrics = self._metrics[metric_type][label_key]

                # Remove old metrics
                while metrics and metrics[0].timestamp < cutoff_time:
                    metrics.popleft()

                # Remove empty label groups
                if not metrics:
                    del self._metrics[metric_type][label_key]

            # Remove empty metric types
            if not self._metrics[metric_type]:
                del self._metrics[metric_type]

        logger.debug("Cleaned up old metrics")

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of a sorted list of values."""
        if not values:
            return 0.0

        index = (percentile / 100.0) * (len(values) - 1)
        if index.is_integer():
            return values[int(index)]
        else:
            lower = values[int(index)]
            upper = values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

    def _linear_regression(
        self, x_values: List[float], y_values: List[float]
    ) -> tuple[float, float]:
        """Perform linear regression and return slope and R-squared."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0, 0.0

        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        sum_y2 = sum(y * y for y in y_values)

        # Calculate slope
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        # Calculate R-squared
        y_mean = sum_y / n
        ss_tot = sum((y - y_mean) ** 2 for y in y_values)
        ss_res = sum(
            (y - (slope * x + (sum_y - slope * sum_x) / n)) ** 2
            for x, y in zip(x_values, y_values)
        )

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return slope, r_squared

    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self._lock:
            total_metrics = 0
            metric_type_counts = {}

            for metric_type, label_groups in self._metrics.items():
                type_count = sum(len(metrics) for metrics in label_groups.values())
                metric_type_counts[metric_type.value] = type_count
                total_metrics += type_count

            return {
                "total_metrics": total_metrics,
                "metric_type_counts": metric_type_counts,
                "max_metrics_per_type": self.max_metrics_per_type,
                "retention_hours": self.default_retention_hours,
                "last_cleanup": datetime.fromtimestamp(self._last_cleanup),
                "performance_stats": self._performance_stats.copy(),
            }

    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            return self._performance_stats.copy()

    def reset_performance_statistics(self) -> None:
        """Reset performance statistics."""
        with self._lock:
            self._performance_stats = {
                "metrics_recorded": 0,
                "metrics_retrieved": 0,
                "cleanup_operations": 0,
                "errors": 0,
                "last_error": None,
            }
            logger.info("Performance statistics reset")

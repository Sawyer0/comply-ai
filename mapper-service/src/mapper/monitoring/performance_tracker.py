"""
Performance Tracker for Mapper Service

Single Responsibility: Track and analyze service performance metrics.
Provides performance insights, bottleneck detection, and optimization recommendations.
"""

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics

import structlog

logger = structlog.get_logger(__name__)


class PerformanceLevel(str, Enum):
    """Performance level indicators."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Individual performance measurement."""

    operation: str
    duration_ms: float
    timestamp: datetime
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""

    operation: str
    sample_count: int
    avg_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    p50_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    success_rate: float
    error_rate: float
    throughput_per_second: float
    performance_level: PerformanceLevel


@dataclass
class BottleneckAnalysis:
    """Bottleneck detection results."""

    operation: str
    severity: PerformanceLevel
    issue_type: str
    description: str
    impact_score: float
    recommendations: List[str]
    affected_metrics: Dict[str, float]


class PerformanceTracker:
    """
    Performance Tracker for service performance monitoring.

    Single Responsibility: Track, analyze, and report on service performance.

    This class handles:
    - Performance metric collection
    - Statistical analysis and aggregation
    - Bottleneck detection and analysis
    - Performance trend monitoring
    - Optimization recommendations
    - Performance alerting
    """

    def __init__(self, retention_hours: int = 24, max_samples: int = 50000):
        self.retention_hours = retention_hours
        self.max_samples = max_samples

        # Performance data storage
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self._operation_stats: Dict[str, PerformanceStats] = {}

        # Performance thresholds (in milliseconds)
        self._thresholds = {
            "excellent": 50,
            "good": 200,
            "fair": 500,
            "poor": 1000,
            # Above 1000ms is critical
        }

        # Bottleneck detection
        self._bottlenecks: List[BottleneckAnalysis] = []
        self._last_analysis = datetime.utcnow()

        self.logger = logger.bind(component="performance_tracker")

    def record_operation(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a performance measurement.

        Single Responsibility: Capture individual operation performance.
        """
        metric = PerformanceMetric(
            operation=operation,
            duration_ms=duration_ms,
            timestamp=datetime.utcnow(),
            success=success,
            metadata=metadata or {},
        )

        self._metrics[operation].append(metric)

        # Update real-time stats
        self._update_operation_stats(operation)

    def time_operation(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for timing operations.

        Single Responsibility: Measure operation duration automatically.
        """
        return PerformanceTimer(self, operation, metadata)

    def get_operation_stats(
        self, operation: str, hours: int = 1
    ) -> Optional[PerformanceStats]:
        """
        Get performance statistics for an operation.

        Single Responsibility: Provide aggregated performance data.
        """
        if operation not in self._metrics:
            return None

        # Filter metrics by time range
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        metrics = [m for m in self._metrics[operation] if m.timestamp >= cutoff_time]

        if not metrics:
            return None

        # Calculate statistics
        durations = [m.duration_ms for m in metrics]
        successes = [m.success for m in metrics]

        # Time-based calculations
        time_span_seconds = hours * 3600
        throughput = len(metrics) / time_span_seconds if time_span_seconds > 0 else 0

        # Performance level determination
        avg_duration = statistics.mean(durations)
        performance_level = self._determine_performance_level(avg_duration)

        return PerformanceStats(
            operation=operation,
            sample_count=len(metrics),
            avg_duration_ms=avg_duration,
            min_duration_ms=min(durations),
            max_duration_ms=max(durations),
            p50_duration_ms=statistics.median(durations),
            p95_duration_ms=self._percentile(durations, 0.95),
            p99_duration_ms=self._percentile(durations, 0.99),
            success_rate=sum(successes) / len(successes) * 100,
            error_rate=(len(successes) - sum(successes)) / len(successes) * 100,
            throughput_per_second=throughput,
            performance_level=performance_level,
        )

    def get_all_operations_stats(self, hours: int = 1) -> Dict[str, PerformanceStats]:
        """
        Get performance statistics for all operations.

        Single Responsibility: Provide comprehensive performance overview.
        """
        stats = {}

        for operation in self._metrics.keys():
            operation_stats = self.get_operation_stats(operation, hours)
            if operation_stats:
                stats[operation] = operation_stats

        return stats

    def analyze_bottlenecks(self, hours: int = 1) -> List[BottleneckAnalysis]:
        """
        Analyze performance data to identify bottlenecks.

        Single Responsibility: Detect and analyze performance bottlenecks.
        """
        bottlenecks = []
        all_stats = self.get_all_operations_stats(hours)

        for operation, stats in all_stats.items():
            # Check for slow operations
            if stats.performance_level in [
                PerformanceLevel.POOR,
                PerformanceLevel.CRITICAL,
            ]:
                bottlenecks.append(
                    self._create_slow_operation_analysis(operation, stats)
                )

            # Check for high error rates
            if stats.error_rate > 5.0:  # More than 5% error rate
                bottlenecks.append(self._create_error_rate_analysis(operation, stats))

            # Check for high variance (inconsistent performance)
            if stats.max_duration_ms > stats.avg_duration_ms * 3:
                bottlenecks.append(self._create_variance_analysis(operation, stats))

            # Check for low throughput
            if stats.throughput_per_second < 1.0 and stats.sample_count > 10:
                bottlenecks.append(self._create_throughput_analysis(operation, stats))

        # Sort by impact score
        bottlenecks.sort(key=lambda x: x.impact_score, reverse=True)

        self._bottlenecks = bottlenecks
        self._last_analysis = datetime.utcnow()

        return bottlenecks

    def get_performance_trends(
        self, operation: str, hours: int = 24
    ) -> Dict[str, List[Tuple[datetime, float]]]:
        """
        Get performance trends over time.

        Single Responsibility: Provide time-series performance data.
        """
        if operation not in self._metrics:
            return {}

        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        metrics = [m for m in self._metrics[operation] if m.timestamp >= cutoff_time]

        # Group by hour for trend analysis
        hourly_data = defaultdict(list)
        for metric in metrics:
            hour_key = metric.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_data[hour_key].append(metric)

        trends = {
            "avg_duration": [],
            "p95_duration": [],
            "throughput": [],
            "error_rate": [],
        }

        for hour, hour_metrics in sorted(hourly_data.items()):
            durations = [m.duration_ms for m in hour_metrics]
            successes = [m.success for m in hour_metrics]

            trends["avg_duration"].append((hour, statistics.mean(durations)))
            trends["p95_duration"].append((hour, self._percentile(durations, 0.95)))
            trends["throughput"].append((hour, len(hour_metrics)))
            trends["error_rate"].append(
                (hour, (len(successes) - sum(successes)) / len(successes) * 100)
            )

        return trends

    def get_optimization_recommendations(self) -> List[str]:
        """
        Get optimization recommendations based on performance analysis.

        Single Responsibility: Provide actionable performance optimization advice.
        """
        recommendations = []

        # Analyze recent bottlenecks
        recent_bottlenecks = self.analyze_bottlenecks(hours=1)

        for bottleneck in recent_bottlenecks[:5]:  # Top 5 bottlenecks
            recommendations.extend(bottleneck.recommendations)

        # General recommendations based on overall performance
        all_stats = self.get_all_operations_stats(hours=1)

        if all_stats:
            avg_performance_levels = [
                stats.performance_level for stats in all_stats.values()
            ]
            poor_performance_count = sum(
                1
                for level in avg_performance_levels
                if level in [PerformanceLevel.POOR, PerformanceLevel.CRITICAL]
            )

            if (
                poor_performance_count > len(all_stats) * 0.3
            ):  # More than 30% poor performance
                recommendations.append(
                    "Consider scaling up resources or optimizing algorithms"
                )
                recommendations.append("Review database query performance and indexing")
                recommendations.append("Implement caching for frequently accessed data")

        return list(set(recommendations))  # Remove duplicates

    def _update_operation_stats(self, operation: str) -> None:
        """Update cached operation statistics."""
        stats = self.get_operation_stats(operation, hours=1)
        if stats:
            self._operation_stats[operation] = stats

    def _determine_performance_level(self, avg_duration_ms: float) -> PerformanceLevel:
        """Determine performance level based on average duration."""
        if avg_duration_ms <= self._thresholds["excellent"]:
            return PerformanceLevel.EXCELLENT
        elif avg_duration_ms <= self._thresholds["good"]:
            return PerformanceLevel.GOOD
        elif avg_duration_ms <= self._thresholds["fair"]:
            return PerformanceLevel.FAIR
        elif avg_duration_ms <= self._thresholds["poor"]:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.CRITICAL

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = int(percentile * len(sorted_data))
        if index >= len(sorted_data):
            index = len(sorted_data) - 1

        return sorted_data[index]

    def _create_slow_operation_analysis(
        self, operation: str, stats: PerformanceStats
    ) -> BottleneckAnalysis:
        """Create bottleneck analysis for slow operations."""
        severity = stats.performance_level
        impact_score = stats.avg_duration_ms / 100  # Higher duration = higher impact

        recommendations = [
            f"Optimize {operation} algorithm or implementation",
            "Add caching if applicable",
            "Consider asynchronous processing",
            "Review database queries and indexing",
        ]

        return BottleneckAnalysis(
            operation=operation,
            severity=severity,
            issue_type="slow_operation",
            description=f"Operation {operation} has slow average response time ({stats.avg_duration_ms:.1f}ms)",
            impact_score=impact_score,
            recommendations=recommendations,
            affected_metrics={
                "avg_duration_ms": stats.avg_duration_ms,
                "p95_duration_ms": stats.p95_duration_ms,
                "p99_duration_ms": stats.p99_duration_ms,
            },
        )

    def _create_error_rate_analysis(
        self, operation: str, stats: PerformanceStats
    ) -> BottleneckAnalysis:
        """Create bottleneck analysis for high error rates."""
        impact_score = stats.error_rate * 2  # High error rate = high impact

        recommendations = [
            f"Investigate and fix errors in {operation}",
            "Add better error handling and retry logic",
            "Review input validation",
            "Monitor upstream dependencies",
        ]

        return BottleneckAnalysis(
            operation=operation,
            severity=(
                PerformanceLevel.CRITICAL
                if stats.error_rate > 10
                else PerformanceLevel.POOR
            ),
            issue_type="high_error_rate",
            description=f"Operation {operation} has high error rate ({stats.error_rate:.1f}%)",
            impact_score=impact_score,
            recommendations=recommendations,
            affected_metrics={
                "error_rate": stats.error_rate,
                "success_rate": stats.success_rate,
            },
        )

    def _create_variance_analysis(
        self, operation: str, stats: PerformanceStats
    ) -> BottleneckAnalysis:
        """Create bottleneck analysis for high performance variance."""
        variance_ratio = stats.max_duration_ms / stats.avg_duration_ms
        impact_score = variance_ratio * 10

        recommendations = [
            f"Investigate performance inconsistency in {operation}",
            "Add performance monitoring and alerting",
            "Review resource contention issues",
            "Consider load balancing improvements",
        ]

        return BottleneckAnalysis(
            operation=operation,
            severity=PerformanceLevel.FAIR,
            issue_type="high_variance",
            description=f"Operation {operation} has inconsistent performance (max: {stats.max_duration_ms:.1f}ms, avg: {stats.avg_duration_ms:.1f}ms)",
            impact_score=impact_score,
            recommendations=recommendations,
            affected_metrics={
                "max_duration_ms": stats.max_duration_ms,
                "avg_duration_ms": stats.avg_duration_ms,
                "variance_ratio": variance_ratio,
            },
        )

    def _create_throughput_analysis(
        self, operation: str, stats: PerformanceStats
    ) -> BottleneckAnalysis:
        """Create bottleneck analysis for low throughput."""
        impact_score = 100 / max(
            stats.throughput_per_second, 0.1
        )  # Lower throughput = higher impact

        recommendations = [
            f"Optimize {operation} for higher throughput",
            "Consider parallel processing",
            "Review resource allocation",
            "Implement connection pooling if applicable",
        ]

        return BottleneckAnalysis(
            operation=operation,
            severity=PerformanceLevel.POOR,
            issue_type="low_throughput",
            description=f"Operation {operation} has low throughput ({stats.throughput_per_second:.2f} ops/sec)",
            impact_score=impact_score,
            recommendations=recommendations,
            affected_metrics={
                "throughput_per_second": stats.throughput_per_second,
                "sample_count": stats.sample_count,
            },
        )


class PerformanceTimer:
    """Context manager for timing operations."""

    def __init__(
        self,
        tracker: PerformanceTracker,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.tracker = tracker
        self.operation = operation
        self.metadata = metadata or {}
        self.start_time = None
        self.success = True

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration_ms = (time.time() - self.start_time) * 1000
            success = exc_type is None

            self.tracker.record_operation(
                self.operation, duration_ms, success, self.metadata
            )

    def mark_error(self):
        """Mark the operation as failed."""
        self.success = False

"""
Advanced Monitoring Infrastructure for Analysis Service

Implements comprehensive monitoring, metrics collection, and observability
features for the analysis service.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MetricPoint:
    """Individual metric data point."""

    name: str
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # gauge, counter, histogram


@dataclass
class AlertRule:
    """Alert rule configuration."""

    name: str
    metric_name: str
    condition: str  # "gt", "lt", "eq", "ne"
    threshold: Union[int, float]
    duration_seconds: int = 300  # 5 minutes
    severity: str = "warning"  # info, warning, critical
    labels: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True


class MetricsCollector:
    """
    Advanced metrics collector with support for various metric types.

    Features:
    - Multiple metric types (counters, gauges, histograms)
    - Label-based metrics
    - Time-series storage
    - Aggregation functions
    """

    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque())
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.logger = logger.bind(component="metrics_collector")

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_old_metrics())

    def increment_counter(
        self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter metric."""

        metric_key = self._create_metric_key(name, labels)
        self.counters[metric_key] += value

        # Store time series point
        point = MetricPoint(
            name=name,
            value=self.counters[metric_key],
            timestamp=datetime.now(),
            labels=labels or {},
            metric_type="counter",
        )
        self.metrics[metric_key].append(point)

    def set_gauge(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Set a gauge metric value."""

        metric_key = self._create_metric_key(name, labels)
        self.gauges[metric_key] = value

        # Store time series point
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            metric_type="gauge",
        )
        self.metrics[metric_key].append(point)

    def observe_histogram(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Add observation to histogram metric."""

        metric_key = self._create_metric_key(name, labels)
        self.histograms[metric_key].append(value)

        # Store time series point
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            metric_type="histogram",
        )
        self.metrics[metric_key].append(point)

    def get_metric_value(
        self, name: str, labels: Optional[Dict[str, str]] = None
    ) -> Optional[float]:
        """Get current value of a metric."""

        metric_key = self._create_metric_key(name, labels)

        if metric_key in self.counters:
            return self.counters[metric_key]
        elif metric_key in self.gauges:
            return self.gauges[metric_key]
        elif metric_key in self.histograms and self.histograms[metric_key]:
            # Return latest value for histogram
            return self.histograms[metric_key][-1]

        return None

    def get_histogram_stats(
        self, name: str, labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """Get histogram statistics."""

        metric_key = self._create_metric_key(name, labels)
        values = self.histograms.get(metric_key, [])

        if not values:
            return {}

        sorted_values = sorted(values)
        n = len(sorted_values)

        return {
            "count": n,
            "sum": sum(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / n,
            "p50": sorted_values[int(n * 0.5)],
            "p90": sorted_values[int(n * 0.9)],
            "p95": sorted_values[int(n * 0.95)],
            "p99": sorted_values[int(n * 0.99)],
        }

    def get_time_series(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[MetricPoint]:
        """Get time series data for a metric."""

        metric_key = self._create_metric_key(name, labels)
        points = list(self.metrics.get(metric_key, []))

        # Filter by time range if specified
        if start_time or end_time:
            filtered_points = []
            for point in points:
                if start_time and point.timestamp < start_time:
                    continue
                if end_time and point.timestamp > end_time:
                    continue
                filtered_points.append(point)
            points = filtered_points

        return points

    def _create_metric_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create unique key for metric with labels."""

        if not labels:
            return name

        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    async def _cleanup_old_metrics(self) -> None:
        """Clean up old metric data points."""

        while True:
            try:
                cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)

                for metric_key, points in self.metrics.items():
                    # Remove old points
                    while points and points[0].timestamp < cutoff_time:
                        points.popleft()

                # Clean up empty histogram data
                for metric_key, values in list(self.histograms.items()):
                    if len(values) > 10000:  # Keep only recent 10k values
                        self.histograms[metric_key] = values[-5000:]

                await asyncio.sleep(300)  # Clean up every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Metrics cleanup failed", error=str(e))
                await asyncio.sleep(60)

    async def shutdown(self) -> None:
        """Shutdown metrics collector."""
        if self._cleanup_task:
            self._cleanup_task.cancel()


class AlertManager:
    """
    Alert manager for monitoring and notifications.

    Features:
    - Rule-based alerting
    - Alert state management
    - Notification routing
    - Alert suppression
    """

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.notification_handlers: List[Callable] = []
        self.logger = logger.bind(component="alert_manager")

        # Start alert evaluation task
        self._evaluation_task = asyncio.create_task(self._evaluate_alerts())

    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.alert_rules[rule.name] = rule
        self.logger.info("Alert rule added", rule_name=rule.name)

    def remove_alert_rule(self, rule_name: str) -> None:
        """Remove an alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            # Clear any active alerts for this rule
            if rule_name in self.active_alerts:
                del self.active_alerts[rule_name]
            self.logger.info("Alert rule removed", rule_name=rule_name)

    def add_notification_handler(
        self, handler: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Add notification handler for alerts."""
        self.notification_handlers.append(handler)

    async def _evaluate_alerts(self) -> None:
        """Continuously evaluate alert rules."""

        while True:
            try:
                for rule_name, rule in self.alert_rules.items():
                    if not rule.enabled:
                        continue

                    await self._evaluate_single_rule(rule)

                await asyncio.sleep(30)  # Evaluate every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Alert evaluation failed", error=str(e))
                await asyncio.sleep(60)

    async def _evaluate_single_rule(self, rule: AlertRule) -> None:
        """Evaluate a single alert rule."""

        try:
            # Get current metric value
            current_value = self.metrics_collector.get_metric_value(
                rule.metric_name, rule.labels
            )

            if current_value is None:
                return

            # Check condition
            alert_triggered = self._check_condition(current_value, rule)

            if alert_triggered:
                await self._handle_alert_triggered(rule, current_value)
            else:
                await self._handle_alert_resolved(rule)

        except Exception as e:
            self.logger.error(
                "Failed to evaluate alert rule", rule_name=rule.name, error=str(e)
            )

    def _check_condition(self, value: float, rule: AlertRule) -> bool:
        """Check if alert condition is met."""

        if rule.condition == "gt":
            return value > rule.threshold
        elif rule.condition == "lt":
            return value < rule.threshold
        elif rule.condition == "eq":
            return value == rule.threshold
        elif rule.condition == "ne":
            return value != rule.threshold
        else:
            return False

    async def _handle_alert_triggered(
        self, rule: AlertRule, current_value: float
    ) -> None:
        """Handle when an alert is triggered."""

        now = datetime.now()

        if rule.name not in self.active_alerts:
            # New alert
            alert = {
                "rule_name": rule.name,
                "metric_name": rule.metric_name,
                "current_value": current_value,
                "threshold": rule.threshold,
                "condition": rule.condition,
                "severity": rule.severity,
                "labels": rule.labels,
                "triggered_at": now,
                "last_seen": now,
                "notification_sent": False,
            }

            self.active_alerts[rule.name] = alert

            # Check if duration threshold is met
            if rule.duration_seconds == 0:
                await self._send_alert_notification(alert)
        else:
            # Update existing alert
            alert = self.active_alerts[rule.name]
            alert["current_value"] = current_value
            alert["last_seen"] = now

            # Check if duration threshold is met
            if not alert["notification_sent"]:
                duration = (now - alert["triggered_at"]).total_seconds()
                if duration >= rule.duration_seconds:
                    await self._send_alert_notification(alert)

    async def _handle_alert_resolved(self, rule: AlertRule) -> None:
        """Handle when an alert is resolved."""

        if rule.name in self.active_alerts:
            alert = self.active_alerts[rule.name]
            alert["resolved_at"] = datetime.now()

            # Move to history
            self.alert_history.append(alert.copy())

            # Send resolution notification if alert was sent
            if alert["notification_sent"]:
                await self._send_resolution_notification(alert)

            # Remove from active alerts
            del self.active_alerts[rule.name]

    async def _send_alert_notification(self, alert: Dict[str, Any]) -> None:
        """Send alert notification."""

        alert["notification_sent"] = True

        notification = {
            "type": "alert",
            "alert": alert,
            "timestamp": datetime.now().isoformat(),
        }

        for handler in self.notification_handlers:
            try:
                await handler(notification)
            except Exception as e:
                self.logger.error(
                    "Failed to send alert notification",
                    rule_name=alert["rule_name"],
                    error=str(e),
                )

    async def _send_resolution_notification(self, alert: Dict[str, Any]) -> None:
        """Send alert resolution notification."""

        notification = {
            "type": "resolution",
            "alert": alert,
            "timestamp": datetime.now().isoformat(),
        }

        for handler in self.notification_handlers:
            try:
                await handler(notification)
            except Exception as e:
                self.logger.error(
                    "Failed to send resolution notification",
                    rule_name=alert["rule_name"],
                    error=str(e),
                )

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history."""
        return self.alert_history[-limit:]

    async def shutdown(self) -> None:
        """Shutdown alert manager."""
        if self._evaluation_task:
            self._evaluation_task.cancel()


class PerformanceMonitor:
    """
    Performance monitoring for analysis operations.

    Tracks latency, throughput, error rates, and resource usage.
    """

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.logger = logger.bind(component="performance_monitor")

    def track_operation(
        self, operation_name: str, labels: Optional[Dict[str, str]] = None
    ):
        """Decorator to track operation performance."""

        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.time()

                # Increment operation counter
                self.metrics_collector.increment_counter(
                    f"{operation_name}_total", labels=labels
                )

                try:
                    result = await func(*args, **kwargs)

                    # Track success
                    self.metrics_collector.increment_counter(
                        f"{operation_name}_success_total", labels=labels
                    )

                    return result

                except Exception as e:
                    # Track error
                    error_labels = {**(labels or {}), "error_type": type(e).__name__}
                    self.metrics_collector.increment_counter(
                        f"{operation_name}_error_total", labels=error_labels
                    )
                    raise

                finally:
                    # Track duration
                    duration = time.time() - start_time
                    self.metrics_collector.observe_histogram(
                        f"{operation_name}_duration_seconds", duration, labels=labels
                    )

            return wrapper

        return decorator

    def record_throughput(
        self, operation_name: str, count: int, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record throughput metrics."""

        self.metrics_collector.increment_counter(
            f"{operation_name}_throughput_total", value=count, labels=labels
        )

    def record_resource_usage(
        self,
        cpu_percent: float,
        memory_mb: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record resource usage metrics."""

        self.metrics_collector.set_gauge("cpu_usage_percent", cpu_percent, labels)
        self.metrics_collector.set_gauge("memory_usage_mb", memory_mb, labels)

    def get_performance_summary(
        self, operation_name: str, time_window_minutes: int = 60
    ) -> Dict[str, Any]:
        """Get performance summary for an operation."""

        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=time_window_minutes)

        # Get duration statistics
        duration_stats = self.metrics_collector.get_histogram_stats(
            f"{operation_name}_duration_seconds"
        )

        # Get error rate
        total_ops = (
            self.metrics_collector.get_metric_value(f"{operation_name}_total") or 0
        )
        error_ops = (
            self.metrics_collector.get_metric_value(f"{operation_name}_error_total")
            or 0
        )
        error_rate = (error_ops / total_ops) if total_ops > 0 else 0

        return {
            "operation_name": operation_name,
            "time_window_minutes": time_window_minutes,
            "total_operations": total_ops,
            "error_rate": error_rate,
            "duration_stats": duration_stats,
        }


class HealthChecker:
    """
    Health checking system for service components.

    Provides comprehensive health monitoring and dependency checking.
    """

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.health_checks: Dict[str, Callable] = {}
        self.logger = logger.bind(component="health_checker")

    def register_health_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """Register a health check function."""
        self.health_checks[name] = check_func
        self.logger.info("Health check registered", check_name=name)

    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks and return results."""

        results = {
            "overall_status": "healthy",
            "checks": {},
            "timestamp": datetime.now().isoformat(),
        }

        for name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                is_healthy = (
                    await check_func()
                    if asyncio.iscoroutinefunction(check_func)
                    else check_func()
                )
                duration = time.time() - start_time

                results["checks"][name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "duration_seconds": duration,
                }

                # Record metrics
                self.metrics_collector.set_gauge(
                    "health_check_status",
                    1.0 if is_healthy else 0.0,
                    labels={"check_name": name},
                )

                self.metrics_collector.observe_histogram(
                    "health_check_duration_seconds",
                    duration,
                    labels={"check_name": name},
                )

                if not is_healthy:
                    results["overall_status"] = "unhealthy"

            except Exception as e:
                results["checks"][name] = {"status": "error", "error": str(e)}
                results["overall_status"] = "unhealthy"

                self.logger.error("Health check failed", check_name=name, error=str(e))

        return results

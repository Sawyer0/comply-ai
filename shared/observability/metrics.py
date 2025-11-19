"""Metrics collection using Prometheus."""

import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from functools import wraps

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    from prometheus_client.exposition import MetricsHandler
    from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Create stub classes for type checking
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, **kwargs): return self
    
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, **kwargs): return self
    
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
    
    class CollectorRegistry:
        def __init__(self): pass
    
    def generate_latest(*args): return b"# Metrics disabled\n"
    
    logging.warning("Prometheus client not available. Metrics will be disabled.")

logger = logging.getLogger(__name__)


@dataclass
class MetricConfig:
    """Configuration for metrics collection."""
    
    enabled: bool = True
    port: int = 9090
    path: str = "/metrics"
    labels: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}


class MetricsCollector:
    """Centralized metrics collector for all services."""
    
    def __init__(self, service_name: str, config: Optional[MetricConfig] = None):
        self.service_name = service_name
        self.config = config or MetricConfig()
        self.registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        
        if not PROMETHEUS_AVAILABLE or not self.config.enabled:
            logger.info("Metrics disabled")
            return
        
        # Initialize metrics
        self._init_metrics()
        logger.info(f"Metrics initialized for {service_name}")
    
    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Request metrics
        self.request_count = Counter(
            f"{self.service_name}_requests_total",
            "Total number of requests",
            ["method", "endpoint", "status_code", "tenant_id"],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            f"{self.service_name}_request_duration_seconds",
            "Request duration in seconds",
            ["method", "endpoint", "tenant_id"],
            registry=self.registry
        )
        
        self.request_size = Histogram(
            f"{self.service_name}_request_size_bytes",
            "Request size in bytes",
            ["method", "endpoint"],
            registry=self.registry
        )
        
        self.response_size = Histogram(
            f"{self.service_name}_response_size_bytes",
            "Response size in bytes",
            ["method", "endpoint"],
            registry=self.registry
        )
        
        # Error metrics
        self.error_count = Counter(
            f"{self.service_name}_errors_total",
            "Total number of errors",
            ["error_type", "endpoint", "tenant_id"],
            registry=self.registry
        )
        
        # Business metrics
        self.business_operations = Counter(
            f"{self.service_name}_business_operations_total",
            "Total number of business operations",
            ["operation", "detector_type", "tenant_id"],
            registry=self.registry
        )
        
        # System metrics
        self.active_connections = Gauge(
            f"{self.service_name}_active_connections",
            "Number of active connections",
            registry=self.registry
        )
        
        self.processed_items = Counter(
            f"{self.service_name}_processed_items_total",
            "Total number of processed items",
            ["item_type", "tenant_id"],
            registry=self.registry
        )
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        request_size: int = 0,
        response_size: int = 0,
        tenant_id: str = "unknown"
    ):
        """Record HTTP request metrics."""
        if not PROMETHEUS_AVAILABLE or not self.config.enabled:
            return
        
        labels = {"method": method, "endpoint": endpoint, "tenant_id": tenant_id}
        
        self.request_count.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code),
            tenant_id=tenant_id
        ).inc()
        
        self.request_duration.labels(**labels).observe(duration)
        
        if request_size > 0:
            self.request_size.labels(method=method, endpoint=endpoint).observe(request_size)
        
        if response_size > 0:
            self.response_size.labels(method=method, endpoint=endpoint).observe(response_size)
    
    def record_error(
        self,
        error_type: str,
        endpoint: str = "unknown",
        tenant_id: str = "unknown"
    ):
        """Record error metrics."""
        if not PROMETHEUS_AVAILABLE or not self.config.enabled:
            return
        
        self.error_count.labels(
            error_type=error_type,
            endpoint=endpoint,
            tenant_id=tenant_id
        ).inc()
    
    def record_business_operation(
        self,
        operation: str,
        detector_type: str = "unknown",
        tenant_id: str = "unknown"
    ):
        """Record business operation metrics."""
        if not PROMETHEUS_AVAILABLE or not self.config.enabled:
            return
        
        self.business_operations.labels(
            operation=operation,
            detector_type=detector_type,
            tenant_id=tenant_id
        ).inc()
    
    def record_processed_items(
        self,
        item_type: str,
        count: int = 1,
        tenant_id: str = "unknown"
    ):
        """Record processed items metrics."""
        if not PROMETHEUS_AVAILABLE or not self.config.enabled:
            return
        
        self.processed_items.labels(
            item_type=item_type,
            tenant_id=tenant_id
        ).inc(count)
    
    def set_active_connections(self, count: int):
        """Set active connections gauge."""
        if not PROMETHEUS_AVAILABLE or not self.config.enabled:
            return
        
        self.active_connections.set(count)
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        if not PROMETHEUS_AVAILABLE or not self.config.enabled:
            return "# Metrics disabled\n"
        
        return generate_latest(self.registry).decode("utf-8")


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def setup_metrics(service_name: str, config: Optional[MetricConfig] = None) -> MetricsCollector:
    """Initialize metrics for a service."""
    global _metrics_collector
    _metrics_collector = MetricsCollector(service_name, config)
    return _metrics_collector


def get_metrics_collector() -> Optional[MetricsCollector]:
    """Get the global metrics collector."""
    return _metrics_collector


def record_request(
    method: str,
    endpoint: str,
    status_code: int,
    duration: float,
    request_size: int = 0,
    response_size: int = 0,
    tenant_id: str = "unknown"
):
    """Record request metrics using global collector."""
    if _metrics_collector:
        _metrics_collector.record_request(
            method, endpoint, status_code, duration, request_size, response_size, tenant_id
        )


def record_error(error_type: str, endpoint: str = "unknown", tenant_id: str = "unknown"):
    """Record error metrics using global collector."""
    if _metrics_collector:
        _metrics_collector.record_error(error_type, endpoint, tenant_id)


def record_latency(operation: str, duration: float, labels: Optional[Dict[str, str]] = None):
    """Record custom latency metric."""
    if not PROMETHEUS_AVAILABLE or not _metrics_collector:
        return
    
    # Create a custom histogram for latency if needed
    histogram_name = f"{_metrics_collector.service_name}_{operation}_duration_seconds"
    
    # This is a simplified approach - in production you might want to pre-register
    # all histograms or use a more sophisticated approach
    try:
        histogram = Histogram(
            histogram_name,
            f"Duration of {operation} in seconds",
            list(labels.keys()) if labels else [],
            registry=_metrics_collector.registry
        )
        histogram.labels(**(labels or {})).observe(duration)
    except Exception as e:
        logger.warning(f"Failed to record latency metric: {e}")


def timed(operation_name: str = None):
    """Decorator to measure function execution time."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                record_latency(operation_name or getattr(func, '__name__', 'unknown'), duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                record_latency(operation_name or getattr(func, '__name__', 'unknown'), duration)
                record_error(type(e).__name__, getattr(func, '__name__', 'unknown'))
                raise
        
        return wrapper
    return decorator


def async_timed(operation_name: str = None):
    """Decorator to measure async function execution time."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                record_latency(operation_name or getattr(func, '__name__', 'unknown'), duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                record_latency(operation_name or getattr(func, '__name__', 'unknown'), duration)
                record_error(type(e).__name__, getattr(func, '__name__', 'unknown'))
                raise
        
        return wrapper
    return decorator

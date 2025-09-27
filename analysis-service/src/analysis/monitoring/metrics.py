"""
Metrics collection for Analysis Service.

Provides comprehensive metrics for monitoring analysis performance,
business KPIs, and system health.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import time
from functools import wraps

from ..shared_integration import get_shared_logger, get_shared_metrics, track_request_metrics

logger = get_shared_logger(__name__)

# Simulated metrics storage (in production, would use Prometheus/InfluxDB)
_metrics_store: Dict[str, List[Dict[str, Any]]] = {}


@dataclass
class MetricPoint:
    """Individual metric data point."""

    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str]
    metric_type: str  # counter, gauge, histogram


class AnalysisMetrics:
    """Metrics collector for analysis operations."""

    def __init__(self):
        self.logger = logger.bind(component="analysis_metrics")
        self.shared_metrics = get_shared_metrics()

    def record_analysis_request(
        self, tenant_id: str, analysis_type: str, status: str = "success"
    ) -> None:
        """Record analysis request metric."""
        # Record in local metrics store
        self._record_metric(
            "analysis_requests_total",
            1.0,
            {"tenant_id": tenant_id, "analysis_type": analysis_type, "status": status},
            "counter",
        )
        
        # Also record in shared metrics
        self.shared_metrics.increment_request_count(
            method="analysis",
            endpoint=analysis_type,
            status=status
        )

    def record_analysis_duration(
        self, tenant_id: str, analysis_type: str, duration_ms: float
    ) -> None:
        """Record analysis processing duration."""
        self._record_metric(
            "analysis_duration_ms",
            duration_ms,
            {"tenant_id": tenant_id, "analysis_type": analysis_type},
            "histogram",
        )

    def record_confidence_score(
        self, tenant_id: str, analysis_type: str, confidence: float
    ) -> None:
        """Record analysis confidence score."""
        self._record_metric(
            "analysis_confidence_score",
            confidence,
            {"tenant_id": tenant_id, "analysis_type": analysis_type},
            "histogram",
        )

    def record_risk_score(
        self, tenant_id: str, framework: str, risk_level: str, score: float
    ) -> None:
        """Record risk assessment score."""
        self._record_metric(
            "risk_assessment_score",
            score,
            {"tenant_id": tenant_id, "framework": framework, "risk_level": risk_level},
            "histogram",
        )

    def record_compliance_violation(
        self, tenant_id: str, framework: str, severity: str, category: str
    ) -> None:
        """Record compliance violation detection."""
        self._record_metric(
            "compliance_violations_total",
            1.0,
            {
                "tenant_id": tenant_id,
                "framework": framework,
                "severity": severity,
                "category": category,
            },
            "counter",
        )

    def record_pattern_detection(
        self, tenant_id: str, pattern_type: str, confidence: float
    ) -> None:
        """Record pattern detection."""
        self._record_metric(
            "patterns_detected_total",
            1.0,
            {"tenant_id": tenant_id, "pattern_type": pattern_type},
            "counter",
        )

        self._record_metric(
            "pattern_confidence_score",
            confidence,
            {"tenant_id": tenant_id, "pattern_type": pattern_type},
            "histogram",
        )

    def record_rag_query(
        self,
        tenant_id: str,
        query_type: str,
        response_time_ms: float,
        documents_retrieved: int,
    ) -> None:
        """Record RAG system query metrics."""
        self._record_metric(
            "rag_queries_total",
            1.0,
            {"tenant_id": tenant_id, "query_type": query_type},
            "counter",
        )

        self._record_metric(
            "rag_response_time_ms",
            response_time_ms,
            {"tenant_id": tenant_id, "query_type": query_type},
            "histogram",
        )

        self._record_metric(
            "rag_documents_retrieved",
            documents_retrieved,
            {"tenant_id": tenant_id, "query_type": query_type},
            "histogram",
        )

    def get_metrics_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get metrics summary for the specified time window."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)

        summary = {}

        for metric_name, points in _metrics_store.items():
            recent_points = [
                p
                for p in points
                if datetime.fromisoformat(p["timestamp"]) > cutoff_time
            ]

            if recent_points:
                values = [p["value"] for p in recent_points]
                summary[metric_name] = {
                    "count": len(values),
                    "sum": sum(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }

        return summary

    def _record_metric(
        self, name: str, value: float, labels: Dict[str, str], metric_type: str
    ) -> None:
        """Record a metric point."""
        point = {
            "name": name,
            "value": value,
            "timestamp": datetime.utcnow().isoformat(),
            "labels": labels,
            "metric_type": metric_type,
        }

        if name not in _metrics_store:
            _metrics_store[name] = []

        _metrics_store[name].append(point)

        # Keep only last 1000 points per metric to prevent memory issues
        if len(_metrics_store[name]) > 1000:
            _metrics_store[name] = _metrics_store[name][-1000:]


def track_analysis_metrics(analysis_type: str):
    """Decorator to automatically track analysis metrics."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            metrics = AnalysisMetrics()

            # Extract tenant_id from args/kwargs if available
            tenant_id = kwargs.get("tenant_id", "unknown")
            if not tenant_id and args and hasattr(args[0], "tenant_id"):
                tenant_id = args[0].tenant_id

            try:
                result = await func(*args, **kwargs)

                # Record success metrics
                duration_ms = (time.time() - start_time) * 1000
                metrics.record_analysis_request(tenant_id, analysis_type, "success")
                metrics.record_analysis_duration(tenant_id, analysis_type, duration_ms)

                # Record confidence if available
                if hasattr(result, "confidence") and result.confidence:
                    metrics.record_confidence_score(
                        tenant_id, analysis_type, result.confidence
                    )

                return result

            except Exception as e:
                # Record failure metrics
                duration_ms = (time.time() - start_time) * 1000
                metrics.record_analysis_request(tenant_id, analysis_type, "failed")
                metrics.record_analysis_duration(tenant_id, analysis_type, duration_ms)

                logger.error(
                    "Analysis operation failed",
                    analysis_type=analysis_type,
                    tenant_id=tenant_id,
                    error=str(e),
                )
                raise

        return wrapper

    return decorator

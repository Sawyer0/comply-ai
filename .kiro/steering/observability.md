---
inclusion: always
---

# Observability & Monitoring Enhancements

## Distributed Tracing

### Correlation ID Implementation
```python
# src/llama_mapper/utils/tracing.py
import uuid
from contextvars import ContextVar
from typing import Optional

# Context variable for request correlation
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)

class CorrelationMiddleware:
    """FastAPI middleware for correlation ID management"""
    
    async def __call__(self, request: Request, call_next):
        # Extract or generate correlation ID
        corr_id = request.headers.get('X-Correlation-ID') or str(uuid.uuid4())
        correlation_id.set(corr_id)
        
        # Add to response headers
        response = await call_next(request)
        response.headers['X-Correlation-ID'] = corr_id
        
        return response

def get_correlation_id() -> str:
    """Get current correlation ID"""
    return correlation_id.get() or str(uuid.uuid4())

# Structured logging with correlation
import structlog

logger = structlog.get_logger()

def log_with_correlation(message: str, **kwargs):
    """Log with automatic correlation ID"""
    logger.info(message, correlation_id=get_correlation_id(), **kwargs)
```

### OpenTelemetry Integration
```python
# src/llama_mapper/monitoring/tracing.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor

def setup_tracing(app: FastAPI, service_name: str):
    """Setup distributed tracing for the application"""
    
    # Configure tracer provider
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    # Configure Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name="jaeger-agent",
        agent_port=6831,
    )
    
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Auto-instrument frameworks
    FastAPIInstrumentor.instrument_app(app)
    AsyncPGInstrumentor().instrument()
    RedisInstrumentor().instrument()
    
    return tracer

# Custom span decorators
def trace_function(operation_name: str):
    """Decorator to trace function execution"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(operation_name) as span:
                span.set_attribute("correlation_id", get_correlation_id())
                span.set_attribute("function", func.__name__)
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result
                except Exception as e:
                    span.set_attribute("success", False)
                    span.set_attribute("error", str(e))
                    raise
        return wrapper
    return decorator
```

## Business Metrics

### Custom Metrics Collection
```python
# src/llama_mapper/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import time
from functools import wraps

# Business metrics
DETECTOR_REQUESTS = Counter(
    'detector_requests_total',
    'Total detector requests',
    ['detector_type', 'tenant_id', 'status']
)

MAPPING_ACCURACY = Histogram(
    'mapping_confidence_score',
    'Confidence scores of mappings',
    ['framework', 'category']
)

COMPLIANCE_VIOLATIONS = Counter(
    'compliance_violations_total',
    'Total compliance violations detected',
    ['framework', 'severity', 'category']
)

MODEL_INFERENCE_TIME = Histogram(
    'model_inference_duration_seconds',
    'Model inference latency',
    ['model_name', 'model_version']
)

ACTIVE_TENANTS = Gauge(
    'active_tenants_count',
    'Number of active tenants'
)

def track_business_metrics(func):
    """Decorator to track business metrics"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            
            # Track success metrics
            if hasattr(result, 'confidence'):
                MAPPING_ACCURACY.observe(result.confidence)
            
            return result
            
        except Exception as e:
            # Track error metrics
            logger.error("Business operation failed", 
                        error=str(e), 
                        correlation_id=get_correlation_id())
            raise
        finally:
            # Track timing
            duration = time.time() - start_time
            MODEL_INFERENCE_TIME.observe(duration)
    
    return wrapper
```

### Real-time Dashboard Metrics
```python
# src/llama_mapper/monitoring/dashboard.py
class BusinessMetricsDashboard:
    """Generate real-time business metrics for dashboards"""
    
    def __init__(self, redis_client, db_pool):
        self.redis = redis_client
        self.db = db_pool
    
    async def get_realtime_metrics(self) -> dict:
        """Get real-time business metrics"""
        return {
            "active_requests": await self.get_active_requests(),
            "compliance_score": await self.get_compliance_score(),
            "model_performance": await self.get_model_performance(),
            "tenant_activity": await self.get_tenant_activity(),
            "risk_trends": await self.get_risk_trends()
        }
    
    async def get_compliance_score(self) -> float:
        """Calculate overall compliance score"""
        query = """
        SELECT 
            AVG(confidence) as avg_confidence,
            COUNT(*) as total_mappings,
            COUNT(CASE WHEN confidence > 0.8 THEN 1 END) as high_confidence
        FROM compliance_mappings 
        WHERE created_at > NOW() - INTERVAL '24 hours'
        """
        
        async with self.db.acquire() as conn:
            result = await conn.fetchrow(query)
            
        return {
            "overall_score": result['avg_confidence'],
            "high_confidence_ratio": result['high_confidence'] / result['total_mappings']
        }
```

## Anomaly Detection

### Model Performance Monitoring
```python
# src/llama_mapper/monitoring/anomaly_detection.py
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import List, Dict, Any

class ModelPerformanceMonitor:
    """Monitor model performance for anomalies"""
    
    def __init__(self):
        self.baseline_metrics = {}
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.is_trained = False
    
    async def collect_performance_metrics(self, model_name: str) -> Dict[str, float]:
        """Collect current performance metrics"""
        return {
            "avg_confidence": await self.get_avg_confidence(model_name),
            "inference_latency": await self.get_avg_latency(model_name),
            "error_rate": await self.get_error_rate(model_name),
            "throughput": await self.get_throughput(model_name),
            "memory_usage": await self.get_memory_usage(model_name)
        }
    
    def train_baseline(self, historical_metrics: List[Dict[str, float]]):
        """Train anomaly detector on historical data"""
        if len(historical_metrics) < 50:
            logger.warning("Insufficient historical data for anomaly detection")
            return
        
        # Convert to numpy array
        features = np.array([list(m.values()) for m in historical_metrics])
        
        # Train isolation forest
        self.anomaly_detector.fit(features)
        self.is_trained = True
        
        # Calculate baseline statistics
        self.baseline_metrics = {
            key: {
                "mean": np.mean([m[key] for m in historical_metrics]),
                "std": np.std([m[key] for m in historical_metrics]),
                "p95": np.percentile([m[key] for m in historical_metrics], 95)
            }
            for key in historical_metrics[0].keys()
        }
    
    async def detect_anomalies(self, model_name: str) -> Dict[str, Any]:
        """Detect performance anomalies"""
        if not self.is_trained:
            return {"status": "not_trained", "anomalies": []}
        
        current_metrics = await self.collect_performance_metrics(model_name)
        features = np.array([list(current_metrics.values())]).reshape(1, -1)
        
        # Detect anomalies
        anomaly_score = self.anomaly_detector.decision_function(features)[0]
        is_anomaly = self.anomaly_detector.predict(features)[0] == -1
        
        anomalies = []
        if is_anomaly:
            # Identify which metrics are anomalous
            for metric, value in current_metrics.items():
                baseline = self.baseline_metrics[metric]
                z_score = abs(value - baseline["mean"]) / baseline["std"]
                
                if z_score > 2:  # 2 standard deviations
                    anomalies.append({
                        "metric": metric,
                        "current_value": value,
                        "baseline_mean": baseline["mean"],
                        "z_score": z_score,
                        "severity": "high" if z_score > 3 else "medium"
                    })
        
        return {
            "status": "anomaly_detected" if is_anomaly else "normal",
            "anomaly_score": anomaly_score,
            "anomalies": anomalies,
            "timestamp": time.time()
        }
```

### Alerting System
```python
# src/llama_mapper/monitoring/alerting.py
class AlertingSystem:
    """Intelligent alerting with escalation policies"""
    
    def __init__(self, notification_service):
        self.notifications = notification_service
        self.alert_history = {}
        self.escalation_policies = self.load_escalation_policies()
    
    async def process_anomaly_alert(self, anomaly_data: Dict[str, Any]):
        """Process anomaly detection alerts"""
        alert_key = f"model_anomaly_{anomaly_data['model_name']}"
        
        # Check if this is a recurring issue
        if self.is_recurring_alert(alert_key):
            await self.escalate_alert(alert_key, anomaly_data)
        else:
            await self.send_initial_alert(alert_key, anomaly_data)
        
        # Update alert history
        self.update_alert_history(alert_key, anomaly_data)
    
    def is_recurring_alert(self, alert_key: str) -> bool:
        """Check if alert has occurred recently"""
        history = self.alert_history.get(alert_key, [])
        recent_alerts = [
            alert for alert in history 
            if time.time() - alert['timestamp'] < 3600  # Last hour
        ]
        return len(recent_alerts) >= 3
    
    async def escalate_alert(self, alert_key: str, data: Dict[str, Any]):
        """Escalate recurring alerts"""
        await self.notifications.send_escalated_alert({
            "type": "model_performance_degradation",
            "severity": "critical",
            "message": f"Recurring model anomaly detected: {alert_key}",
            "data": data,
            "escalation_level": 2
        })
```

## Enhanced Logging

### Structured Logging Configuration
```python
# src/llama_mapper/utils/logging.py
import structlog
import logging.config

def configure_logging():
    """Configure structured logging"""
    
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.dev.ConsoleRenderer(colors=False),
            },
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "json",
            },
        },
        "loggers": {
            "": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": True,
            },
        }
    })
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            add_correlation_id,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

def add_correlation_id(logger, method_name, event_dict):
    """Add correlation ID to all log entries"""
    event_dict['correlation_id'] = get_correlation_id()
    return event_dict
```

This observability enhancement provides:

1. **Distributed Tracing**: Full request flow tracking across services
2. **Business Metrics**: Domain-specific metrics beyond technical ones
3. **Anomaly Detection**: ML-based performance monitoring
4. **Intelligent Alerting**: Context-aware notifications with escalation
5. **Enhanced Logging**: Structured logs with correlation IDs

These additions will give you much deeper visibility into your system's behavior and health!

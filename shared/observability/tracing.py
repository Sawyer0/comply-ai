"""Distributed tracing utilities using OpenTelemetry."""

import os
import uuid
from typing import Dict, Any, Optional, Callable
from functools import wraps
import logging

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.trace.propagation.textmap import DictGetter

logger = logging.getLogger(__name__)

# Global tracer instance
_tracer: Optional[trace.Tracer] = None


def setup_tracing(
    service_name: str,
    endpoint: Optional[str] = None,
    sample_rate: float = 1.0,
) -> trace.Tracer:
    """Initialize OpenTelemetry tracing for a service."""
    global _tracer
    
    if not endpoint or not os.getenv("ENABLE_TRACING", "false").lower() == "true":
        logger.info("Tracing disabled")
        _tracer = trace.NoOpTracer()
        return _tracer
    
    # Create resource with service information
    resource = Resource.create({
        "service.name": service_name,
        "service.version": os.getenv("SERVICE_VERSION", "1.0.0"),
        "environment": os.getenv("ENVIRONMENT", "development"),
    })
    
    # Set up trace provider
    trace.set_tracer_provider(TracerProvider(resource=resource))
    tracer_provider = trace.get_tracer_provider()
    
    # Configure exporter based on endpoint
    if endpoint.startswith("http"):
        # OTLP exporter for collectors like Jaeger, Tempo, etc.
        exporter = OTLPSpanExporter(endpoint=endpoint)
    else:
        # Jaeger exporter
        exporter = JaegerExporter(
            agent_host_name=endpoint.split(":")[0] if ":" in endpoint else "localhost",
            agent_port=int(endpoint.split(":")[1]) if ":" in endpoint else 6831,
        )
    
    # Add span processor
    span_processor = BatchSpanProcessor(exporter)
    tracer_provider.add_span_processor(span_processor)
    
    # Set global propagator for B3 format (compatible with many systems)
    set_global_textmap(B3MultiFormat())
    
    _tracer = tracer_provider.get_tracer(__name__)
    logger.info(f"Tracing initialized for {service_name} with endpoint {endpoint}")
    
    return _tracer


def get_tracer() -> trace.Tracer:
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = trace.NoOpTracer()
    return _tracer


def trace_function(operation_name: Optional[str] = None):
    """Decorator to add tracing to synchronous functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with tracer.start_as_current_span(name) as span:
                # Add function attributes to span
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                
                # Add arguments count (avoid logging sensitive data)
                span.set_attribute("function.args_count", len(args))
                span.set_attribute("function.kwargs_count", len(kwargs))
                
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("function.success", True)
                    return result
                except Exception as e:
                    span.set_attribute("function.success", False)
                    span.set_attribute("error.message", str(e))
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise
        
        return wrapper
    return decorator


def trace_async_function(operation_name: Optional[str] = None):
    """Decorator to add tracing to asynchronous functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = get_tracer()
            name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with tracer.start_as_current_span(name) as span:
                # Add function attributes to span
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                span.set_attribute("function.async", True)
                
                # Add arguments count
                span.set_attribute("function.args_count", len(args))
                span.set_attribute("function.kwargs_count", len(kwargs))
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("function.success", True)
                    return result
                except Exception as e:
                    span.set_attribute("function.success", False)
                    span.set_attribute("error.message", str(e))
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise
        
        return wrapper
    return decorator


def propagate_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Inject tracing context into HTTP headers for downstream calls."""
    tracer = get_tracer()
    
    # Get current span context
    span = trace.get_current_span()
    if span and span.get_span_context().trace_flags.sampled:
        # Use OpenTelemetry's textmap propagator
        from opentelemetry.trace.propagation.textmap import DictSetter
        
        setter = DictSetter()
        tracer_text_map = {}
        
        # Inject context into headers
        if hasattr(trace, 'propagate'):
            trace.propagate.inject(tracer_text_map, setter=setter)
            headers.update(tracer_text_map)
    
    # Always ensure correlation ID
    if "X-Correlation-ID" not in headers:
        headers["X-Correlation-ID"] = str(uuid.uuid4())
    
    return headers


def extract_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Extract tracing context from HTTP headers."""
    tracer = get_tracer()
    
    # Use OpenTelemetry's textmap propagator to extract context
    getter = DictGetter()
    context = {}
    
    if hasattr(trace, 'propagate'):
        context = trace.propagate.extract(headers, getter=getter)
    
    return context


class DictGetter:
    """Dictionary getter for OpenTelemetry propagation."""
    
    def get(self, carrier: Dict[str, str], key: str) -> list[str]:
        """Get value from carrier dictionary."""
        if key in carrier:
            return [carrier[key]]
        return []


def add_span_attributes(attributes: Dict[str, Any]):
    """Add attributes to the current span."""
    span = trace.get_current_span()
    if span:
        for key, value in attributes.items():
            span.set_attribute(key, str(value))


def add_span_event(name: str, attributes: Dict[str, Any] = None):
    """Add an event to the current span."""
    span = trace.get_current_span()
    if span:
        span.add_event(name, attributes or {})


def set_span_error(error: Exception):
    """Mark the current span as errored."""
    span = trace.get_current_span()
    if span:
        span.set_attribute("error.message", str(error))
        span.set_attribute("error.type", type(error).__name__)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(error)))

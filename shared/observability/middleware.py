"""FastAPI middleware for observability (tracing and metrics)."""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .tracing import get_tracer, propagate_headers, extract_headers, add_span_attributes, set_span_error
from .metrics import get_metrics_collector, record_request, record_error
from ..utils.correlation import get_correlation_id, set_correlation_id

logger = logging.getLogger(__name__)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Combined observability middleware for tracing and metrics."""
    
    def __init__(self, app, enable_tracing: bool = True, enable_metrics: bool = True):
        super().__init__(app)
        self.enable_tracing = enable_tracing
        self.enable_metrics = enable_metrics
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with observability."""
        start_time = time.time()
        
        # Extract correlation ID and tracing context
        correlation_id = request.headers.get("X-Correlation-ID")
        if correlation_id:
            set_correlation_id(correlation_id)
        else:
            correlation_id = get_correlation_id()
        
        # Extract tracing context from headers
        tracing_context = {}
        if self.enable_tracing:
            tracing_context = extract_headers(dict(request.headers))
        
        tenant_id = request.headers.get("X-Tenant-ID", "unknown")
        
        # Start tracing span
        span = None
        if self.enable_tracing:
            tracer = get_tracer()
            span_name = f"{request.method} {request.url.path}"
            span = tracer.start_as_current_span(span_name)
            
            # Add span attributes
            add_span_attributes({
                "http.method": request.method,
                "http.url": str(request.url),
                "http.scheme": request.url.scheme,
                "http.host": request.url.hostname,
                "http.target": request.url.path,
                "tenant_id": tenant_id,
                "correlation_id": correlation_id,
            })
            
            # Add user agent if available
            user_agent = request.headers.get("User-Agent")
            if user_agent:
                add_span_attributes({"http.user_agent": user_agent})
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Record metrics
            if self.enable_metrics:
                metrics_collector = get_metrics_collector()
                if metrics_collector:
                    # Get request/response sizes
                    request_size = len(await request.body()) if hasattr(request, '_body') else 0
                    response_size = len(response.body) if hasattr(response, 'body') else 0
                    
                    record_request(
                        method=request.method,
                        endpoint=request.url.path,
                        status_code=response.status_code,
                        duration=duration,
                        request_size=request_size,
                        response_size=response_size,
                        tenant_id=tenant_id
                    )
            
            # Update span with response info
            if span:
                add_span_attributes({
                    "http.status_code": response.status_code,
                    "response.duration_ms": duration * 1000,
                })
                
                if response.status_code >= 400:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, f"HTTP {response.status_code}"))
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Record error metrics
            if self.enable_metrics:
                record_error(type(e).__name__, request.url.path, tenant_id)
            
            # Update span with error
            if span:
                set_span_error(e)
                add_span_attributes({
                    "error.duration_ms": duration * 1000,
                })
            
            # Re-raise exception
            raise


class TracingMiddleware(BaseHTTPMiddleware):
    """Tracing-only middleware for when you want separate concerns."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with tracing only."""
        tracer = get_tracer()
        span_name = f"{request.method} {request.url.path}"
        
        with tracer.start_as_current_span(span_name) as span:
            # Add basic attributes
            add_span_attributes({
                "http.method": request.method,
                "http.url": str(request.url),
                "http.target": request.url.path,
                "tenant_id": request.headers.get("X-Tenant-ID", "unknown"),
            })
            
            try:
                response = await call_next(request)
                add_span_attributes({"http.status_code": response.status_code})
                
                if response.status_code >= 400:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, f"HTTP {response.status_code}"))
                
                return response
            except Exception as e:
                set_span_error(e)
                raise


class MetricsMiddleware(BaseHTTPMiddleware):
    """Metrics-only middleware for when you want separate concerns."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with metrics only."""
        start_time = time.time()
        tenant_id = request.headers.get("X-Tenant-ID", "unknown")
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Record metrics
            metrics_collector = get_metrics_collector()
            if metrics_collector:
                record_request(
                    method=request.method,
                    endpoint=request.url.path,
                    status_code=response.status_code,
                    duration=duration,
                    tenant_id=tenant_id
                )
            
            return response
        except Exception as e:
            duration = time.time() - start_time
            
            # Record error metrics
            metrics_collector = get_metrics_collector()
            if metrics_collector:
                record_error(type(e).__name__, request.url.path, tenant_id)
            
            raise


class ClientObservabilityMixin:
    """Mixin for HTTP clients to add observability headers."""
    
    def prepare_headers(self, headers: dict = None) -> dict:
        """Prepare headers with tracing and correlation information."""
        headers = headers or {}
        
        # Add correlation ID
        correlation_id = get_correlation_id()
        headers["X-Correlation-ID"] = correlation_id
        
        # Add tracing headers
        tracing_headers = propagate_headers({})
        headers.update(tracing_headers)
        
        # Add tenant ID if available
        tenant_id = getattr(self, 'tenant_id', None)
        if tenant_id:
            headers["X-Tenant-ID"] = tenant_id
        
        return headers

"""
Shared middleware integration for mapper service.

This module provides middleware components that integrate with shared
components for consistent behavior across all microservices.
"""

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time

from ..shared_integration import (
    CorrelationMiddleware,
    get_shared_logger,
    get_shared_metrics,
    set_correlation_id,
    get_correlation_id,
    BaseServiceException,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    ServiceUnavailableError,
)

logger = get_shared_logger(__name__)


class SharedMetricsMiddleware(BaseHTTPMiddleware):
    """Enhanced middleware for shared metrics collection with correlation tracking."""

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with shared metrics tracking and correlation ID management."""
        # Extract endpoint information
        endpoint = f"{request.method} {request.url.path}"
        
        # Handle correlation ID
        correlation_id = request.headers.get("X-Correlation-ID")
        if correlation_id:
            set_correlation_id(correlation_id)
        else:
            correlation_id = get_correlation_id()

        # Get metrics collector
        metrics = get_shared_metrics()
        
        # Record request start time
        start_time = time.time()

        # Track request metrics
        request_labels = {
            "method": request.method,
            "endpoint": request.url.path,
            "correlation_id": correlation_id,
        }

        try:
            response = await call_next(request)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to ms

            # Track successful request
            if hasattr(metrics, "record_request"):
                metrics.record_request(
                    endpoint=endpoint,
                    status_code=response.status_code,
                    processing_time=processing_time,
                    labels=request_labels
                )
            
            # Add correlation ID to response headers
            if correlation_id:
                response.headers["X-Correlation-ID"] = correlation_id

            return response

        except BaseServiceException as e:
            # Track service exception with appropriate status code
            processing_time = (time.time() - start_time) * 1000
            status_code = getattr(e, 'status_code', 500)
            
            if hasattr(metrics, "record_request"):
                metrics.record_request(
                    endpoint=endpoint,
                    status_code=status_code,
                    processing_time=processing_time,
                    labels={**request_labels, "error_type": type(e).__name__}
                )
            
            logger.error(
                "Request failed with service exception",
                endpoint=endpoint,
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=correlation_id,
                processing_time_ms=processing_time
            )
            raise
            
        except Exception as e:
            # Track unexpected exception
            processing_time = (time.time() - start_time) * 1000
            
            if hasattr(metrics, "record_request"):
                metrics.record_request(
                    endpoint=endpoint,
                    status_code=500,
                    processing_time=processing_time,
                    labels={**request_labels, "error_type": type(e).__name__}
                )
                
            logger.error(
                "Request failed with unexpected error",
                endpoint=endpoint,
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=correlation_id,
                processing_time_ms=processing_time
            )
            raise


def setup_shared_middleware(app: FastAPI) -> None:
    """Setup shared middleware components for the application."""

    # Add correlation ID middleware
    app.add_middleware(CorrelationMiddleware)

    # Add shared metrics middleware
    app.add_middleware(SharedMetricsMiddleware)

    logger.info("Shared middleware components configured")


__all__ = [
    "SharedMetricsMiddleware",
    "setup_shared_middleware",
]

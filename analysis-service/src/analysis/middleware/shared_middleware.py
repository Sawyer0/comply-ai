"""
Shared middleware integration for analysis service.

This module provides middleware components that integrate with shared
components for consistent behavior across all microservices.
"""

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware

from ..shared_integration import (
    CorrelationMiddleware,
    get_shared_logger,
    get_shared_metrics,
    track_request_metrics,
)

logger = get_shared_logger(__name__)


class SharedMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for shared metrics collection."""

    async def dispatch(self, request, call_next):
        """Process request with shared metrics tracking."""
        # Extract endpoint information
        endpoint = f"{request.method} {request.url.path}"
        
        # Use shared metrics tracking
        with track_request_metrics(endpoint):
            response = await call_next(request)
            
        return response


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

"""Middleware utilities for FastAPI applications."""

from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .correlation import set_correlation_id, get_correlation_id


class CorrelationMiddleware(BaseHTTPMiddleware):
    """Middleware for correlation ID management."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with correlation ID handling."""
        # Extract or generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID")
        if correlation_id:
            set_correlation_id(correlation_id)
        else:
            correlation_id = get_correlation_id()

        # Process request
        response = await call_next(request)

        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id

        return response

"""
FastAPI middleware for correlation ID management.

Automatically extracts or generates correlation IDs for all requests.
"""

import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ...utils.correlation import set_correlation_id


class CorrelationMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for correlation ID management."""
    
    def __init__(self, app, header_name: str = "X-Correlation-ID"):
        """
        Initialize correlation middleware.
        
        Args:
            app: FastAPI application
            header_name: HTTP header name for correlation ID
        """
        super().__init__(app)
        self.header_name = header_name
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with correlation ID.
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response with correlation ID header
        """
        # Extract or generate correlation ID
        corr_id = request.headers.get(self.header_name) or str(uuid.uuid4())
        
        # Set in context for this request
        set_correlation_id(corr_id)
        
        # Process request
        response = await call_next(request)
        
        # Add correlation ID to response headers
        response.headers[self.header_name] = corr_id
        
        return response
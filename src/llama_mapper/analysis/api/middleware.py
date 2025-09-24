"""
Middleware for the Analysis Module API.

This module provides FastAPI middleware for security, logging,
metrics, and other cross-cutting concerns.
"""

import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .log_scrubbing import LogScrubber, RequestResponseLogger

logger = logging.getLogger(__name__)


class AnalysisMiddleware:
    """
    Analysis module middleware collection.

    Provides middleware for security headers, metrics collection,
    request logging, and other cross-cutting concerns.
    """

    @staticmethod
    def security_headers_middleware() -> Callable:
        """
        Create security headers middleware.

        Returns:
            Middleware function
        """

        async def middleware(request: Request, call_next: Callable) -> Response:
            """Add security headers to all responses."""
            response = await call_next(request)

            # Add security headers
            security_headers = {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "X-XSS-Protection": "1; mode=block",
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                "Content-Security-Policy": "default-src 'self'",
                "Referrer-Policy": "strict-origin-when-cross-origin",
                "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            }

            for header, value in security_headers.items():
                response.headers[header] = value

            return response

        return middleware

    @staticmethod
    def metrics_middleware() -> Callable:
        """
        Create metrics collection middleware.

        Returns:
            Middleware function
        """

        async def middleware(request: Request, call_next: Callable) -> Response:
            """Collect metrics for all requests."""
            start_time = time.time()

            # Get request info for metrics
            endpoint = f"{request.method} {request.url.path}"
            tenant = request.headers.get("X-Tenant-ID", "unknown")

            response = await call_next(request)

            processing_time_seconds = time.time() - start_time
            processing_time_ms = processing_time_seconds * 1000

            # Record metrics using the analysis metrics collector
            try:
                from .metrics import metrics_collector
                metrics_collector.record_request(
                    endpoint=endpoint,
                    status=str(response.status_code),
                    tenant=tenant,
                    duration=processing_time_seconds,
                    analysis_type="request"
                )
            except ImportError:
                logger.warning("Metrics collector not available, skipping metrics recording")

            logger.debug(
                f"Request processed in {processing_time_ms:.2f}ms: {request.method} {request.url.path}"
            )

            return response

        return middleware

    @staticmethod
    def logging_middleware() -> Callable:
        """
        Create request logging middleware with log scrubbing.

        Returns:
            Middleware function
        """
        scrubber = LogScrubber()
        request_logger = RequestResponseLogger(scrubber)

        async def middleware(request: Request, call_next: Callable) -> Response:
            """Log all requests and responses with PII scrubbing."""
            start_time = time.time()

            # Get request ID
            request_id = request.headers.get("X-Request-ID", "unknown")

            # Extract and scrub request data
            request_data = {}
            if hasattr(request, "_json") and request._json:
                request_data = request._json

            # Scrub headers
            headers = dict(request.headers)
            scrubbed_headers = scrubber.scrub_headers(headers)

            # Log scrubbed request
            request_logger.log_request(
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                data=request_data if request_data else None,
                headers=scrubbed_headers,
            )

            response = await call_next(request)

            processing_time = (time.time() - start_time) * 1000

            # Extract and scrub response data
            response_data = {}
            if hasattr(response, "body") and response.body:
                try:
                    import json

                    response_data = json.loads(response.body.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    response_data = {"body": "[BINARY_DATA]"}

            # Scrub response headers
            response_headers = dict(response.headers)
            scrubbed_response_headers = scrubber.scrub_headers(response_headers)

            # Log scrubbed response
            request_logger.log_response(
                request_id=request_id,
                status_code=response.status_code,
                data=response_data if response_data else None,
                headers=scrubbed_response_headers,
            )

            return response

        return middleware

    @staticmethod
    def error_handling_middleware() -> Callable:
        """
        Create error handling middleware with log scrubbing.

        Returns:
            Middleware function
        """
        scrubber = LogScrubber()
        request_logger = RequestResponseLogger(scrubber)

        async def middleware(request: Request, call_next: Callable) -> Response:
            """Handle errors and provide consistent error responses with scrubbed logging."""
            try:
                response = await call_next(request)
                return response
            except Exception as e:
                # Get request ID
                request_id = request.headers.get("X-Request-ID", "unknown")

                # Log error with scrubbed context
                context = {
                    "method": request.method,
                    "path": request.url.path,
                    "query_params": dict(request.query_params),
                    "headers": dict(request.headers),
                }

                request_logger.log_error(
                    request_id=request_id, error=e, context=context
                )

                # Return consistent error response
                from fastapi.responses import JSONResponse

                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "Internal server error",
                        "message": "An unexpected error occurred",
                        "timestamp": time.time(),
                    },
                )

        return middleware

    @staticmethod
    def cors_middleware(origins: list[str]) -> Callable:
        """
        Create CORS middleware.

        Args:
            origins: List of allowed origins

        Returns:
            Middleware function
        """

        async def middleware(request: Request, call_next: Callable) -> Response:
            """Handle CORS headers."""
            response = await call_next(request)

            # Add CORS headers
            origin = request.headers.get("origin")
            if origin in origins or "*" in origins:
                response.headers["Access-Control-Allow-Origin"] = origin or "*"
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "*"
                response.headers["Access-Control-Allow-Credentials"] = "true"

            return response

        return middleware


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Request context middleware for storing request-specific data.

    Provides a way to store and access request-specific data
    throughout the request lifecycle.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add request context to the request state."""
        # Initialize request context
        request.state.context = {
            "start_time": time.time(),
            "request_id": request.headers.get("X-Request-ID", "unknown"),
            "tenant_id": request.headers.get("X-Tenant-ID", "unknown"),
            "user_id": request.headers.get("X-User-ID", "unknown"),
        }

        response = await call_next(request)

        # Add processing time to response headers
        if hasattr(request.state, "context"):
            processing_time = (time.time() - request.state.context["start_time"]) * 1000
            response.headers["X-Processing-Time-Ms"] = str(int(processing_time))

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware.

    Provides basic rate limiting functionality for API endpoints.
    """

    def __init__(self, app, requests_per_minute: int = 60):
        """
        Initialize rate limiting middleware.

        Args:
            app: FastAPI application
            requests_per_minute: Maximum requests per minute per client
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts = {}  # In production, use Redis or similar

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting to requests."""
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()

        # Clean up old entries
        self.request_counts = {
            ip: timestamps
            for ip, timestamps in self.request_counts.items()
            if any(ts > current_time - 60 for ts in timestamps)
        }

        # Check rate limit
        if client_ip in self.request_counts:
            recent_requests = [
                ts for ts in self.request_counts[client_ip] if ts > current_time - 60
            ]
            if len(recent_requests) >= self.requests_per_minute:
                from fastapi.responses import JSONResponse

                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "message": f"Maximum {self.requests_per_minute} requests per minute allowed",
                    },
                )

        # Record request
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []
        self.request_counts[client_ip].append(current_time)

        response = await call_next(request)
        return response

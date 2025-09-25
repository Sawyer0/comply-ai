"""Main FastAPI application for Analysis Service"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from .api.endpoints import router
from .api.dependencies import shutdown_dependencies
from .shared_integration import (
    get_shared_logger,
    get_shared_database,
    get_shared_metrics,
    initialize_shared_components,
)
from .security import SecurityManager, SecurityConfig
from .security.exceptions import SecurityError, RateLimitError

logger = get_shared_logger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for request authentication and validation."""

    def __init__(self, app, excluded_paths=None):
        super().__init__(app)
        self.excluded_paths = excluded_paths or ["/", "/health", "/metrics"]

    def is_excluded_path(self, path: str) -> bool:
        """Check if path should be excluded from security checks."""
        return path in self.excluded_paths

    async def dispatch(self, request: Request, call_next):
        """Process request through security checks."""
        security_manager = get_security_manager()
        if security_manager:
            try:
                # Skip security for excluded paths
                if self.is_excluded_path(request.url.path):
                    return await call_next(request)

                # Authenticate request
                auth_result = await security_manager.authenticate_request(
                    dict(request.headers)
                )

                # Add user info to request state
                request.state.user_info = auth_result

                # Check rate limits
                client_id = auth_result.get("user_id", str(request.client.host))
                await security_manager.check_rate_limit(client_id, request.url.path)

            except (SecurityError, RateLimitError) as e:
                logger.warning("Security check failed", error=str(e))
                return Response(
                    content='{"error": "Security check failed"}',
                    status_code=403,
                    media_type="application/json",
                )
            except Exception as e:
                logger.error("Unexpected security error", error=str(e))
                return Response(
                    content='{"error": "Internal security error"}',
                    status_code=500,
                    media_type="application/json",
                )

        response = await call_next(request)
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Metrics middleware for request tracking."""

    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0

    def get_request_count(self) -> int:
        """Get the current request count."""
        return self.request_count

    async def dispatch(self, request: Request, call_next):
        """Track request metrics."""
        self.request_count += 1
        metrics_collector = get_shared_metrics()

        start_time = time.time()

        try:
            response = await call_next(request)

            # Track successful request
            duration = time.time() - start_time
            if metrics_collector and hasattr(metrics_collector, "record_request"):
                metrics_collector.record_request(
                    method=request.method,
                    endpoint=request.url.path,
                    status_code=response.status_code,
                    duration=duration,
                )

            return response

        except Exception as e:
            # Track failed request
            duration = time.time() - start_time
            if metrics_collector and hasattr(metrics_collector, "record_request"):
                metrics_collector.record_request(
                    method=request.method,
                    endpoint=request.url.path,
                    status_code=500,
                    duration=duration,
                )
            logger.error("Request processing failed", error=str(e))
            raise


# Global application state
app_state: dict = {
    "security_manager": None,
    "shared_components": None,
}


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting Analysis Service")

    try:
        # Initialize shared components
        app_state["shared_components"] = initialize_shared_components()

        # Initialize security manager
        security_config = SecurityConfig()
        app_state["security_manager"] = SecurityManager(security_config)

        logger.info("Analysis Service started successfully")

        yield

    except Exception as e:
        logger.error("Failed to start Analysis Service", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Analysis Service")

        # Cleanup security manager
        if app_state["security_manager"]:
            await app_state["security_manager"].cleanup_audit_logs()

        # Shutdown dependencies
        await shutdown_dependencies()

        logger.info("Analysis Service shutdown complete")


app = FastAPI(
    title="Analysis Service",
    description="Advanced analysis, risk scoring, compliance intelligence, and RAG system",
    version="1.0.0",
    lifespan=lifespan,
)

# Add middleware with secure CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Development frontend
        "http://localhost:8080",  # Development dashboard
        "https://app.comply-ai.com",  # Production frontend
        "https://dashboard.comply-ai.com",  # Production dashboard
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-API-Key",
        "X-Tenant-ID",
        "X-Correlation-ID",
        "X-Request-ID",
    ],
)

# Add security and metrics middleware
app.add_middleware(SecurityMiddleware)
app.add_middleware(MetricsMiddleware)

# Include API routes (all endpoints consolidated following SRP)
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Analysis Service", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        # Check database connectivity
        db = get_shared_database()
        await db.fetchval("SELECT 1")

        return {
            "status": "healthy",
            "service": "analysis-service",
            "version": "1.0.0",
            "database": "connected",
        }
    except ConnectionError as e:
        logger.error("Database connection failed", error=str(e))
        return {
            "status": "unhealthy",
            "service": "analysis-service",
            "version": "1.0.0",
            "database": "disconnected",
            "error": "Database connection failed",
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "service": "analysis-service",
            "version": "1.0.0",
            "database": "unknown",
            "error": "Health check failed",
        }


@app.get("/metrics")
async def get_metrics():
    """Metrics endpoint for Prometheus"""
    try:
        security_manager = get_security_manager()
        if not security_manager:
            return {"error": "Security manager not initialized"}

        security_metrics = await security_manager.get_security_metrics()
        return security_metrics

    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))
        return {"error": "Failed to retrieve metrics"}


def get_security_manager():
    """Get security manager from application state."""
    return app_state["security_manager"]


def get_shared_components():
    """Get shared components from application state."""
    return app_state["shared_components"]

"""Main FastAPI application for Mapper Service"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from .api.endpoints import router
from .api.dependencies import shutdown_dependencies
from .config.settings import MapperSettings
from .infrastructure.database_manager import create_database_manager_from_env
from .infrastructure.connection_pool import ConnectionPoolManager, ConnectionPoolConfig
from .infrastructure.health_checker import DatabaseHealthChecker
from .security.api_key_manager import APIKeyManager
from .security.authentication import AuthenticationService
from .security.authorization import AuthorizationService
from .security.rate_limiting_service import RateLimitingService
from .security import SecurityManager
from .resilience import ResilienceManager
from .quality import QualityManager
from .shared_integration import (
    initialize_shared_components,
    get_shared_logger,
    get_shared_metrics,
    get_shared_resilience_manager,
    get_shared_tenant_manager,
    get_shared_cost_monitor,
)
from .middleware import setup_shared_middleware

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class WAFMiddleware(BaseHTTPMiddleware):
    """WAF middleware for request filtering and attack detection."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request through WAF."""
        security_manager = get_security_manager()
        if security_manager and hasattr(security_manager, 'waf'):
            # Check for malicious patterns
            try:
                if await security_manager.waf.check_request(request):
                    return Response(
                        content='{"error": "Request blocked by WAF"}',
                        status_code=403,
                        media_type="application/json"
                    )
            except Exception as e:
                logger.warning("WAF check failed", error=str(e))
        
        response = await call_next(request)
        return response


class ResilienceMiddleware(BaseHTTPMiddleware):
    """Resilience middleware for circuit breaker and retry patterns."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request with resilience patterns."""
        resilience_manager = get_resilience_manager()
        if resilience_manager and hasattr(resilience_manager, 'circuit_breaker'):
            try:
                # Apply circuit breaker pattern
                async with resilience_manager.circuit_breaker("api_request"):
                    # Apply retry pattern for transient failures
                    if hasattr(resilience_manager, 'retry_manager'):
                        response = await resilience_manager.retry_manager.execute_with_retry(
                            call_next, request
                        )
                        return response
            except Exception as e:
                logger.warning("Resilience pattern failed", error=str(e))
        
        response = await call_next(request)
        return response


class QualityMiddleware(BaseHTTPMiddleware):
    """Quality middleware for monitoring and alerting."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request with quality monitoring."""
        quality_manager = get_quality_manager()
        if quality_manager and hasattr(quality_manager, 'start_request_monitoring'):
            try:
                # Start quality monitoring
                await quality_manager.start_request_monitoring(request)
            except Exception as e:
                logger.warning("Quality monitoring start failed", error=str(e))
        
        response = await call_next(request)
        
        if quality_manager and hasattr(quality_manager, 'end_request_monitoring'):
            try:
                # End quality monitoring
                await quality_manager.end_request_monitoring(request, response)
            except Exception as e:
                logger.warning("Quality monitoring end failed", error=str(e))
        
        return response

# Global application state
app_state: dict = {
    "database_manager": None,
    "connection_pool": None,
    "health_checker": None,
    "api_key_manager": None,
    "auth_service": None,
    "authz_service": None,
    "rate_limiting_service": None,
    "security_manager": None,
    "resilience_manager": None,
    "quality_manager": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting Mapper Service")

    try:
        # Initialize shared components first
        shared_components = initialize_shared_components()
        logger.info("Shared components initialized", components=list(shared_components.keys()))
        
        # Initialize tenant and cost management
        tenant_manager = get_shared_tenant_manager()
        cost_monitor = get_shared_cost_monitor()
        logger.info("Tenant and cost management initialized")

        # Initialize database connections
        app_state["database_manager"] = create_database_manager_from_env()
        await app_state["database_manager"].initialize()

        # Create connection pool manager
        pool_config = ConnectionPoolConfig(
            primary_db=app_state["database_manager"].config,
            read_replicas=[],  # Add read replicas from settings if needed
            redis_config=None,  # Add Redis config if needed
        )
        app_state["connection_pool"] = ConnectionPoolManager(pool_config)
        await app_state["connection_pool"].initialize()

        # Create health checker
        app_state["health_checker"] = DatabaseHealthChecker(
            app_state["connection_pool"]
        )
        await app_state["health_checker"].start_monitoring()

        # Create security services
        app_state["api_key_manager"] = APIKeyManager(app_state["database_manager"])
        app_state["auth_service"] = AuthenticationService(app_state["api_key_manager"])
        app_state["authz_service"] = AuthorizationService()

        # Create rate limiting service (with Redis if available)
        redis_client = None
        if app_state["connection_pool"] and hasattr(
            app_state["connection_pool"], "redis_manager"
        ):
            redis_client = app_state["connection_pool"].redis_manager
        app_state["rate_limiting_service"] = RateLimitingService(redis_client)

        # Create security manager
        app_state["security_manager"] = SecurityManager(
            auth_service=app_state["auth_service"],
            authz_service=app_state["authz_service"],
            api_key_manager=app_state["api_key_manager"],
            rate_limiting_service=app_state["rate_limiting_service"]
        )

        # Create resilience manager
        app_state["resilience_manager"] = ResilienceManager(
            database_manager=app_state["database_manager"],
            connection_pool=app_state["connection_pool"]
        )

        # Create quality manager
        app_state["quality_manager"] = QualityManager(
            database_manager=app_state["database_manager"],
            connection_pool=app_state["connection_pool"]
        )

        logger.info("Mapper Service started successfully")

        yield

    except Exception as e:
        logger.error("Failed to start Mapper Service", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Mapper Service")

        if app_state["health_checker"]:
            await app_state["health_checker"].stop_monitoring()

        if app_state["connection_pool"]:
            await app_state["connection_pool"].close()

        if app_state["database_manager"]:
            await app_state["database_manager"].close()

        await shutdown_dependencies()
        logger.info("Mapper Service shutdown complete")


app = FastAPI(
    title="Mapper Service",
    description="Core mapping, model serving, and response generation",
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

# Setup shared middleware components
setup_shared_middleware(app)

# Add security, resilience, and quality middleware
app.add_middleware(WAFMiddleware)
app.add_middleware(ResilienceMiddleware)
app.add_middleware(QualityMiddleware)

# Include API routes (all endpoints consolidated following SRP)
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Mapper Service", "version": "1.0.0"}


def get_database_manager():
    """Get database manager from application state."""
    return app_state["database_manager"]


def get_connection_pool():
    """Get connection pool manager from application state."""
    return app_state["connection_pool"]


def get_health_checker():
    """Get health checker from application state."""
    return app_state["health_checker"]


def get_api_key_manager():
    """Get API key manager from application state."""
    return app_state["api_key_manager"]


def get_auth_service():
    """Get authentication service from application state."""
    return app_state["auth_service"]


def get_authz_service():
    """Get authorization service from application state."""
    return app_state["authz_service"]


def get_rate_limiting_service():
    """Get rate limiting service from application state."""
    return app_state["rate_limiting_service"]


def get_security_manager():
    """Get security manager from application state."""
    return app_state["security_manager"]


def get_resilience_manager():
    """Get resilience manager from application state."""
    return app_state["resilience_manager"]


def get_quality_manager():
    """Get quality manager from application state."""
    return app_state["quality_manager"]

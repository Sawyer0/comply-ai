"""Main entry point for the Detector Orchestration Service.

This module provides the FastAPI application and startup logic for the
orchestration service with all SRP-organized components.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from shared.validation.middleware import (
    ValidationMiddleware,
    TenantValidationMiddleware,
)
from shared.exceptions.base import BaseServiceException

from .service import OrchestrationService, OrchestrationConfig
from .app_state import service_container
from .api import orchestration_router, detector_router, health_router
from .config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan manager."""

    # Startup
    logger.info("Starting Detector Orchestration Service")

    # Initialize service with configuration from settings
    config = OrchestrationConfig(
        enable_health_monitoring=settings.enable_health_monitoring,
        enable_service_discovery=settings.enable_service_discovery,
        enable_policy_management=settings.enable_policy_management,
        health_check_interval=settings.health_check_interval,
        service_ttl_minutes=settings.service_ttl_minutes,
    )
    orchestration_service = OrchestrationService(config=config)

    service_container.set_orchestration_service(orchestration_service)

    # Start the service
    await orchestration_service.start()

    # Register some example detectors for testing
    await _register_example_detectors()

    logger.info("Detector Orchestration Service started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Detector Orchestration Service")
    service = service_container.get_orchestration_service()
    if service:
        await service.stop()
        service_container.clear()
    logger.info("Detector Orchestration Service stopped")


# Create FastAPI application
app = FastAPI(
    title="Detector Orchestration Service",
    description="Microservice for orchestrating detector execution with SRP-organized components",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware with configuration from settings
cors_config = settings.get_cors_config()
app.add_middleware(CORSMiddleware, **cors_config)

# Add validation middleware
app.add_middleware(
    ValidationMiddleware,
    validate_requests=True,
    validate_responses=True,
    strict_mode=True,
)

# Add tenant validation middleware
app.add_middleware(
    TenantValidationMiddleware, require_tenant_id=settings.require_tenant_id
)

# Include API routers following SRP
app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(orchestration_router, prefix=settings.api_prefix)
app.include_router(detector_router, prefix=settings.api_prefix)


@app.exception_handler(BaseServiceException)
async def service_exception_handler(_request, exc: BaseServiceException):
    """Handle service exceptions."""
    return JSONResponse(
        status_code=400 if exc.error_code.startswith("VALIDATION") else 500,
        content=exc.to_dict(),
    )


# All endpoints are now handled by separate API modules following SRP


async def _register_example_detectors():
    """Register some example detectors for testing."""
    service = service_container.get_orchestration_service()
    if not service:
        return

    example_detectors = [
        {
            "detector_id": "presidio",
            "endpoint": "http://presidio-service:8080",
            "detector_type": "pii",
        },
        {
            "detector_id": "deberta",
            "endpoint": "http://deberta-service:8080",
            "detector_type": "classification",
        },
        {
            "detector_id": "custom",
            "endpoint": "http://custom-detector:8080",
            "detector_type": "custom",
        },
    ]

    for detector in example_detectors:
        try:
            await service.register_detector(
                detector_id=detector["detector_id"],
                endpoint=detector["endpoint"],
                detector_type=detector["detector_type"],
            )
            logger.info("Registered example detector: %s", detector["detector_id"])
        except (ConnectionError, TimeoutError, HTTPException) as e:
            logger.warning(
                "Failed to register example detector %s: %s",
                detector["detector_id"],
                str(e),
            )


if __name__ == "__main__":
    import uvicorn

    logger.info(
        "Starting Detector Orchestration Service on %s:%d", settings.host, settings.port
    )

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=str(settings.log_level).lower(),
        reload=settings.reload,
        workers=settings.workers if not settings.reload else 1,
    )

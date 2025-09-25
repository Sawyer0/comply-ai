"""Main entry point for the Detector Orchestration Service.

This module provides the FastAPI application and startup logic for the
orchestration service with all SRP-organized components.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from shared.interfaces.orchestration import (
    OrchestrationRequest,
    OrchestrationResponse,
    DetectorInfo,
    DetectorRegistration,
)
from shared.interfaces.base import HealthResponse
from shared.validation.middleware import (
    ValidationMiddleware,
    TenantValidationMiddleware,
)
from shared.utils.correlation import get_correlation_id, set_correlation_id
from shared.exceptions.base import BaseServiceException

from .service import OrchestrationService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Service instance holder
class ServiceHolder:
    """Holds the orchestration service instance."""

    orchestration_service: OrchestrationService = None


service_holder = ServiceHolder()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan manager."""

    # Startup
    logger.info("Starting Detector Orchestration Service")

    # Initialize service with configuration from environment
    service_holder.orchestration_service = OrchestrationService(
        enable_health_monitoring=os.getenv("ENABLE_HEALTH_MONITORING", "true").lower()
        == "true",
        enable_service_discovery=os.getenv("ENABLE_SERVICE_DISCOVERY", "true").lower()
        == "true",
        enable_policy_management=os.getenv("ENABLE_POLICY_MANAGEMENT", "true").lower()
        == "true",
        health_check_interval=int(os.getenv("HEALTH_CHECK_INTERVAL", "30")),
        service_ttl_minutes=int(os.getenv("SERVICE_TTL_MINUTES", "30")),
    )

    # Start the service
    await service_holder.orchestration_service.start()

    # Register some example detectors for testing
    await _register_example_detectors()

    logger.info("Detector Orchestration Service started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Detector Orchestration Service")
    if service_holder.orchestration_service:
        await service_holder.orchestration_service.stop()
    logger.info("Detector Orchestration Service stopped")


# Create FastAPI application
app = FastAPI(
    title="Detector Orchestration Service",
    description="Microservice for orchestrating detector execution with SRP-organized components",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware with secure configuration
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
        "X-Request-ID"
    ],
)

# Add validation middleware
app.add_middleware(
    ValidationMiddleware,
    validate_requests=True,
    validate_responses=True,
    strict_mode=True,
)

# Add tenant validation middleware
app.add_middleware(TenantValidationMiddleware, require_tenant_id=True)


# Dependency to get tenant ID
def get_tenant_id(x_tenant_id: str = Header(..., alias="X-Tenant-ID")) -> str:
    """Extract tenant ID from headers."""
    return x_tenant_id


# Dependency to get correlation ID
def get_correlation_id_header(
    x_correlation_id: str = Header(None, alias="X-Correlation-ID")
) -> str:
    """Extract or generate correlation ID."""
    if x_correlation_id:
        set_correlation_id(x_correlation_id)
        return x_correlation_id
    return get_correlation_id()


@app.exception_handler(BaseServiceException)
async def service_exception_handler(_request, exc: BaseServiceException):
    """Handle service exceptions."""
    return JSONResponse(
        status_code=400 if exc.error_code.startswith("VALIDATION") else 500,
        content=exc.to_dict(),
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        if service_holder.orchestration_service:
            status = await service_holder.orchestration_service.get_service_status()
            return HealthResponse(
                status="healthy" if status["status"] == "running" else "unhealthy",
                version="1.0.0",
                uptime_seconds=None,  # Could be calculated
            )

        return HealthResponse(status="unhealthy", version="1.0.0")
    except (ConnectionError, TimeoutError) as e:
        logger.error("Health check failed: %s", str(e))
        return HealthResponse(status="unhealthy", version="1.0.0")


@app.post("/api/v1/orchestrate", response_model=OrchestrationResponse)
async def orchestrate_detectors(
    request: OrchestrationRequest,
    tenant_id: str = Depends(get_tenant_id),
    correlation_id: str = Depends(get_correlation_id_header),
):
    """Main orchestration endpoint.

    Coordinates all SRP components to:
    1. Route content to appropriate detectors
    2. Execute detectors according to routing plan
    3. Aggregate results into unified response
    4. Apply policies and generate recommendations
    """
    try:
        if not service_holder.orchestration_service:
            raise HTTPException(status_code=503, detail="Service not available")

        logger.info(
            "Received orchestration request for tenant %s",
            tenant_id,
            extra={
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "detector_types": getattr(request, "detector_types", []),
            },
        )

        response = await service_holder.orchestration_service.orchestrate(
            request=request, tenant_id=tenant_id, correlation_id=correlation_id
        )

        return response

    except Exception as e:
        logger.error(
            "Orchestration failed: %s",
            str(e),
            extra={
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/detectors/register")
async def register_detector(
    registration: DetectorRegistration,
    tenant_id: str = Depends(get_tenant_id),
    correlation_id: str = Depends(get_correlation_id_header),
):
    """Register a new detector with the orchestration service."""
    try:
        if not service_holder.orchestration_service:
            raise HTTPException(status_code=503, detail="Service not available")

        success = await service_holder.orchestration_service.register_detector(
            detector_id=registration.detector_type,  # Use type as ID for now
            endpoint=registration.endpoint_url,
            detector_type=registration.detector_type,
            timeout_ms=5000,  # Default timeout
            max_retries=3,  # Default retries
            supported_content_types=["text"],  # Default content types
        )

        if success:
            return {
                "message": f"Detector {registration.detector_type} registered successfully"
            }

        raise HTTPException(status_code=400, detail="Failed to register detector")

    except Exception as e:
        logger.error(
            "Detector registration failed: %s",
            str(e),
            extra={
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "detector_type": registration.detector_type,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/api/v1/detectors/{detector_id}")
async def unregister_detector(
    detector_id: str,
    tenant_id: str = Depends(get_tenant_id),
    correlation_id: str = Depends(get_correlation_id_header),
):
    """Unregister a detector from the orchestration service."""
    try:
        if not service_holder.orchestration_service:
            raise HTTPException(status_code=503, detail="Service not available")

        success = await service_holder.orchestration_service.unregister_detector(
            detector_id
        )

        if success:
            return {"message": f"Detector {detector_id} unregistered successfully"}

        raise HTTPException(status_code=404, detail="Detector not found")

    except Exception as e:
        logger.error(
            "Detector unregistration failed: %s",
            str(e),
            extra={
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "detector_id": detector_id,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/detectors")
async def list_detectors(
    tenant_id: str = Depends(get_tenant_id),
    correlation_id: str = Depends(get_correlation_id_header),
) -> List[DetectorInfo]:
    """List available detectors with detailed information."""
    try:
        if not service_holder.orchestration_service:
            raise HTTPException(status_code=503, detail="Service not available")

        detector_names = (
            service_holder.orchestration_service.content_router.get_available_detectors()
        )

        # Convert to DetectorInfo objects with additional details
        detector_infos = []
        for detector_name in detector_names:
            config = (
                service_holder.orchestration_service.content_router.get_detector_config(
                    detector_name
                )
            )
            detector_infos.append(
                DetectorInfo(
                    detector_id=detector_name,
                    detector_type=config.detector_type if config else "unknown",
                    status="active",  # Could be enhanced with actual health check
                    endpoint_url=config.endpoint if config else "",
                    supported_content_types=(
                        config.supported_content_types if config else ["text"]
                    ),
                    timeout_ms=config.timeout_ms if config else 5000,
                    max_retries=config.max_retries if config else 3,
                )
            )

        return detector_infos

    except Exception as e:
        logger.error(
            "List detectors failed: %s",
            str(e),
            extra={
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/status")
async def get_service_status(
    tenant_id: str = Depends(get_tenant_id),
    correlation_id: str = Depends(get_correlation_id_header),
) -> Dict[str, Any]:
    """Get comprehensive service status."""
    try:
        if not service_holder.orchestration_service:
            raise HTTPException(status_code=503, detail="Service not available")

        status = await service_holder.orchestration_service.get_service_status()
        return status

    except Exception as e:
        logger.error(
            "Get service status failed: %s",
            str(e),
            extra={
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _register_example_detectors():
    """Register some example detectors for testing."""
    if not service_holder.orchestration_service:
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
            await service_holder.orchestration_service.register_detector(
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

    # Configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info")

    logger.info("Starting Detector Orchestration Service on %s:%d", host, port)

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )

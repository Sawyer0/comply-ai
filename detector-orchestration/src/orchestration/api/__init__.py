"""API endpoints for orchestration service following SRP.

This module provides HTTP API endpoints with clear separation of concerns:
- OrchestrationAPI: Main orchestration endpoints
- DetectorAPI: Detector management endpoints
- HealthAPI: Health check and monitoring endpoints
- Dependencies: Shared dependency injection functions
"""

from .orchestration_api import router as orchestration_router
from .detector_api import router as detector_router
from .health_api import router as health_router
from . import dependencies

__all__ = ["orchestration_router", "detector_router", "health_router", "dependencies"]

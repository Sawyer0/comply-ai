"""
Main API endpoints for the Analysis Service.

This module consolidates all API routes following SRP.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List
from ..shared_integration import get_shared_logger
from .dependencies import authenticate_request, authorize_action

logger = get_shared_logger(__name__)

# Create main router
router = APIRouter()

# Include sub-routers from other API modules
from .analysis import router as analysis_router
from .risk_scoring import router as risk_router
from .quality import router as quality_router
from .training import router as training_router
from .batch import router as batch_router
from .tenancy import router as tenancy_router
from .plugins import router as plugins_router

# Include all routers with appropriate prefixes
router.include_router(analysis_router, prefix="/analysis", tags=["analysis"])
router.include_router(risk_router, prefix="/risk", tags=["risk-scoring"])
router.include_router(quality_router, prefix="/quality", tags=["quality"])
router.include_router(training_router, prefix="/training", tags=["training"])
router.include_router(batch_router, prefix="/batch", tags=["batch"])
router.include_router(tenancy_router, prefix="/tenancy", tags=["tenancy"])
router.include_router(plugins_router, prefix="/plugins", tags=["plugins"])


@router.get("/")
async def root():
    """Root endpoint for Analysis Service."""
    return {
        "service": "analysis-service",
        "version": "1.0.0",
        "description": "Advanced analysis, risk scoring, compliance intelligence, and RAG system",
    }


@router.get("/status")
async def status(user_info: Dict[str, Any] = Depends(authenticate_request)):
    """Get service status."""
    return {
        "status": "healthy",
        "service": "analysis-service",
        "version": "1.0.0",
        "user": user_info.get("user_id", "anonymous"),
    }

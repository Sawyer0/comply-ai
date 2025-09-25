"""
API endpoints for Analysis Service.

This module provides REST API endpoints for all Analysis Service functionality
including tenancy management, plugin management, analysis operations,
training pipelines, quality management, and batch processing.
"""

from .analysis import router as analysis_router
from .batch import router as batch_router
from .plugins import router as plugins_router
from .quality import router as quality_router
from .tenancy import router as tenancy_router
from .training import router as training_router

__all__ = [
    "analysis_router",
    "batch_router",
    "plugins_router",
    "quality_router",
    "tenancy_router",
    "training_router",
]

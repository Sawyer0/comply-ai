"""
API endpoints for the Mapper Service.

Single responsibility: HTTP API interface definitions.
"""

from .endpoints import router
from .training_endpoints import router as training_router

__all__ = [
    "router",
    "training_router",
]

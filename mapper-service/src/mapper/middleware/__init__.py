"""
Middleware components for mapper service.

This package provides middleware functionality that integrates
with shared components for consistent behavior across services.
"""

from .shared_middleware import (
    SharedMetricsMiddleware,
    setup_shared_middleware,
)

__all__ = [
    "SharedMetricsMiddleware",
    "setup_shared_middleware",
]

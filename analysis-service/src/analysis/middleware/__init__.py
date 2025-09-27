"""
Middleware components for the analysis service.

This module provides middleware setup using shared components.
"""

from fastapi import FastAPI
from .shared_middleware import setup_shared_middleware

# Export the setup function
__all__ = ["setup_shared_middleware"]

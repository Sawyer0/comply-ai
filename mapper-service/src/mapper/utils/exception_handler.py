"""
Shared exception handling utilities for mapper service.

This module provides standardized exception handling that integrates
with shared components while maintaining the existing error handling patterns.
"""

from typing import Any, Dict, Optional
from fastapi import HTTPException
from structlog import get_logger

from ..shared_integration import (
    get_shared_logger,
    BaseServiceException,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    ServiceUnavailableError,
)

logger = get_shared_logger(__name__)


class MapperExceptionHandler:
    """Standardized exception handler for mapper service operations."""

    def __init__(self, analytics_manager=None):
        """Initialize exception handler with optional analytics manager."""
        self.analytics_manager = analytics_manager

    async def handle_mapper_exception(
        self, 
        exception: Exception, 
        operation: str,
        tenant_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> HTTPException:
        """
        Handle mapper service exceptions with standardized error responses.
        
        Args:
            exception: The exception that occurred
            operation: The operation that failed (e.g., "map_detector_output")
            tenant_id: Optional tenant ID for context
            context: Additional context information
            
        Returns:
            HTTPException: Standardized HTTP exception
        """
        context = context or {}
        
        # Log the exception with context
        logger.error(
            "Mapper operation failed",
            operation=operation,
            tenant_id=tenant_id,
            error=str(exception),
            error_type=type(exception).__name__,
            **context
        )
        
        # Handle specific exception types
        if isinstance(exception, ValidationError):
            return HTTPException(
                status_code=400,
                detail={
                    "error": "Validation Error",
                    "message": str(exception),
                    "operation": operation,
                    "tenant_id": tenant_id
                }
            )
        
        elif isinstance(exception, AuthenticationError):
            return HTTPException(
                status_code=401,
                detail={
                    "error": "Authentication Error", 
                    "message": "Invalid or missing authentication",
                    "operation": operation
                }
            )
        
        elif isinstance(exception, AuthorizationError):
            return HTTPException(
                status_code=403,
                detail={
                    "error": "Authorization Error",
                    "message": "Insufficient permissions for this operation", 
                    "operation": operation,
                    "tenant_id": tenant_id
                }
            )
        
        elif isinstance(exception, ServiceUnavailableError):
            return HTTPException(
                status_code=503,
                detail={
                    "error": "Service Unavailable",
                    "message": "Mapper service is temporarily unavailable",
                    "operation": operation,
                    "tenant_id": tenant_id
                }
            )
        
        elif isinstance(exception, BaseServiceException):
            return HTTPException(
                status_code=500,
                detail={
                    "error": "Service Error",
                    "message": str(exception),
                    "operation": operation,
                    "tenant_id": tenant_id
                }
            )
        
        # Handle generic exceptions
        else:
            return HTTPException(
                status_code=500,
                detail={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred",
                    "operation": operation,
                    "tenant_id": tenant_id
                }
            )


# Global exception handler instance
_exception_handler: Optional[MapperExceptionHandler] = None


def get_exception_handler() -> MapperExceptionHandler:
    """Get the global exception handler instance."""
    global _exception_handler
    if _exception_handler is None:
        _exception_handler = MapperExceptionHandler()
    return _exception_handler


def set_exception_handler(handler: MapperExceptionHandler) -> None:
    """Set the global exception handler instance."""
    global _exception_handler
    _exception_handler = handler

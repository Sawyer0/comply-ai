"""
Shared exception handling utilities for analysis service.

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


class AnalysisExceptionHandler:
    """Standardized exception handler for analysis service operations."""

    def __init__(self, analytics_manager=None):
        """Initialize exception handler with optional analytics manager."""
        self.analytics_manager = analytics_manager

    async def handle_analysis_exception(
        self,
        exception: Exception,
        tenant_id: str,
        analysis_type: str,
        processing_time_ms: float,
        correlation_id: Optional[str] = None,
    ) -> HTTPException:
        """
        Handle exceptions during analysis operations with proper logging and metrics.
        
        Args:
            exception: The exception that occurred
            tenant_id: Tenant identifier
            analysis_type: Type of analysis being performed
            processing_time_ms: Processing time in milliseconds
            correlation_id: Optional correlation ID for request tracking
            
        Returns:
            HTTPException: Properly formatted HTTP exception
        """
        # Record failed analytics if manager is available
        if self.analytics_manager:
            await self.analytics_manager.record_request_metrics(
                tenant_id,
                {
                    "request_count": 1,
                    "success": False,
                    "response_time_ms": processing_time_ms,
                    "analysis_type": analysis_type,
                    "error_type": type(exception).__name__,
                },
            )

        # Log the exception with correlation ID
        logger.error(
            "Analysis failed",
            tenant_id=tenant_id,
            analysis_type=analysis_type,
            error=str(exception),
            error_type=type(exception).__name__,
            correlation_id=correlation_id,
        )

        # Convert shared exceptions to appropriate HTTP exceptions
        if isinstance(exception, ValidationError):
            return HTTPException(
                status_code=400,
                detail=f"Validation error: {str(exception)}"
            )
        elif isinstance(exception, AuthenticationError):
            return HTTPException(
                status_code=401,
                detail=f"Authentication error: {str(exception)}"
            )
        elif isinstance(exception, AuthorizationError):
            return HTTPException(
                status_code=403,
                detail=f"Authorization error: {str(exception)}"
            )
        elif isinstance(exception, ServiceUnavailableError):
            return HTTPException(
                status_code=503,
                detail=f"Service unavailable: {str(exception)}"
            )
        elif isinstance(exception, BaseServiceException):
            return HTTPException(
                status_code=500,
                detail=f"Service error: {str(exception)}"
            )
        else:
            # Generic exception handling
            return HTTPException(
                status_code=500,
                detail=f"Analysis failed: {str(exception)}"
            )

    def handle_validation_error(
        self,
        error_message: str,
        field: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> HTTPException:
        """Handle validation errors with proper logging."""
        logger.warning(
            "Validation error",
            error=error_message,
            field=field,
            correlation_id=correlation_id,
        )
        
        detail = f"Validation error: {error_message}"
        if field:
            detail += f" (field: {field})"
            
        return HTTPException(status_code=400, detail=detail)

    def handle_authentication_error(
        self,
        error_message: str,
        correlation_id: Optional[str] = None,
    ) -> HTTPException:
        """Handle authentication errors with proper logging."""
        logger.warning(
            "Authentication error",
            error=error_message,
            correlation_id=correlation_id,
        )
        
        return HTTPException(
            status_code=401,
            detail=f"Authentication error: {error_message}"
        )

    def handle_authorization_error(
        self,
        error_message: str,
        resource: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> HTTPException:
        """Handle authorization errors with proper logging."""
        logger.warning(
            "Authorization error",
            error=error_message,
            resource=resource,
            correlation_id=correlation_id,
        )
        
        detail = f"Authorization error: {error_message}"
        if resource:
            detail += f" (resource: {resource})"
            
        return HTTPException(status_code=403, detail=detail)

    def handle_service_unavailable_error(
        self,
        error_message: str,
        service: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> HTTPException:
        """Handle service unavailable errors with proper logging."""
        logger.error(
            "Service unavailable",
            error=error_message,
            service=service,
            correlation_id=correlation_id,
        )
        
        detail = f"Service unavailable: {error_message}"
        if service:
            detail += f" (service: {service})"
            
        return HTTPException(status_code=503, detail=detail)


# Global exception handler instance
_global_exception_handler: Optional[AnalysisExceptionHandler] = None


def get_exception_handler(analytics_manager=None) -> AnalysisExceptionHandler:
    """Get the global exception handler instance."""
    global _global_exception_handler
    if _global_exception_handler is None:
        _global_exception_handler = AnalysisExceptionHandler(analytics_manager)
    return _global_exception_handler


def set_exception_handler(handler: AnalysisExceptionHandler) -> None:
    """Set the global exception handler instance."""
    global _global_exception_handler
    _global_exception_handler = handler

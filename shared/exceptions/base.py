"""Base exceptions for microservice communication."""

from typing import Any, Dict, Optional
from datetime import datetime


class BaseServiceException(Exception):
    """Base exception for all service-related errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.details = details or {}
        self.correlation_id = correlation_id
        self.timestamp = timestamp or datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    def __str__(self) -> str:
        return f"{self.error_code}: {self.message}"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"error_code='{self.error_code}', "
            f"correlation_id='{self.correlation_id}'"
            f")"
        )


class ValidationError(BaseServiceException):
    """Exception raised for validation errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "VALIDATION_ERROR",
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        field_errors: Optional[Dict[str, str]] = None,
    ):
        super().__init__(message, error_code, details, correlation_id)
        self.field_errors = field_errors or {}
        if self.field_errors:
            self.details["field_errors"] = self.field_errors


class AuthenticationError(BaseServiceException):
    """Exception raised for authentication errors."""

    def __init__(
        self,
        message: str = "Authentication failed",
        error_code: str = "AUTHENTICATION_ERROR",
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ):
        super().__init__(message, error_code, details, correlation_id)


class AuthorizationError(BaseServiceException):
    """Exception raised for authorization errors."""

    def __init__(
        self,
        message: str = "Authorization failed",
        error_code: str = "AUTHORIZATION_ERROR",
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        required_permissions: Optional[list] = None,
    ):
        super().__init__(message, error_code, details, correlation_id)
        if required_permissions:
            self.details["required_permissions"] = required_permissions


class RateLimitError(BaseServiceException):
    """Exception raised when rate limits are exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        error_code: str = "RATE_LIMIT_ERROR",
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        retry_after: Optional[int] = None,
    ):
        super().__init__(message, error_code, details, correlation_id)
        if retry_after:
            self.details["retry_after"] = retry_after


class ServiceUnavailableError(BaseServiceException):
    """Exception raised when a service is unavailable."""

    def __init__(
        self,
        message: str = "Service unavailable",
        error_code: str = "SERVICE_UNAVAILABLE",
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        service_name: Optional[str] = None,
    ):
        super().__init__(message, error_code, details, correlation_id)
        if service_name:
            self.details["service_name"] = service_name


class TimeoutError(BaseServiceException):
    """Exception raised when operations timeout."""

    def __init__(
        self,
        message: str = "Operation timed out",
        error_code: str = "TIMEOUT_ERROR",
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ):
        super().__init__(message, error_code, details, correlation_id)
        if timeout_seconds:
            self.details["timeout_seconds"] = timeout_seconds


class ConfigurationError(BaseServiceException):
    """Exception raised for configuration errors."""

    def __init__(
        self,
        message: str = "Configuration error",
        error_code: str = "CONFIGURATION_ERROR",
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        config_key: Optional[str] = None,
    ):
        super().__init__(message, error_code, details, correlation_id)
        if config_key:
            self.details["config_key"] = config_key


class ModelError(BaseServiceException):
    """Exception raised for model-related errors."""

    def __init__(
        self,
        message: str = "Model error",
        error_code: str = "MODEL_ERROR",
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ):
        super().__init__(message, error_code, details, correlation_id)
        if model_name:
            self.details["model_name"] = model_name
        if model_version:
            self.details["model_version"] = model_version


class DataError(BaseServiceException):
    """Exception raised for data-related errors."""

    def __init__(
        self,
        message: str = "Data error",
        error_code: str = "DATA_ERROR",
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        data_source: Optional[str] = None,
    ):
        super().__init__(message, error_code, details, correlation_id)
        if data_source:
            self.details["data_source"] = data_source

"""Base interfaces for all microservice communication."""

import uuid
from typing import Dict, Any, Optional, List, Generic, TypeVar
from datetime import datetime

from pydantic import BaseModel, Field, validator

T = TypeVar("T")


class BaseRequest(BaseModel):
    """Base request model with common fields."""

    correlation_id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Correlation ID for request tracing",
    )

    @classmethod
    @validator("correlation_id")
    def validate_correlation_id(cls, v):
        """Validate correlation ID is a string if provided."""
        if v and not isinstance(v, str):
            raise ValueError("correlation_id must be a string")
        return v

    class Config:
        """Pydantic configuration for BaseRequest."""

        extra = "forbid"  # Prevent extra fields
        validate_assignment = True


class BaseResponse(BaseModel):
    """Base response model with common fields."""

    request_id: str = Field(description="Unique request identifier")
    success: bool = Field(description="Whether the request was successful")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )
    processing_time_ms: Optional[float] = Field(
        None, description="Processing time in milliseconds", ge=0
    )
    correlation_id: Optional[str] = Field(
        None, description="Correlation ID from request"
    )

    @classmethod
    @validator("request_id")
    def validate_request_id(cls, v):
        """Validate request ID is a non-empty string."""
        if not v or not isinstance(v, str):
            raise ValueError("request_id is required and must be a string")
        return v

    class Config:
        """Pydantic configuration for BaseResponse."""

        extra = "forbid"
        validate_assignment = True


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error_code: str = Field(description="Error code")
    message: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )
    correlation_id: Optional[str] = Field(
        None, description="Correlation ID from request"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Error timestamp"
    )

    @classmethod
    @validator("error_code")
    def validate_error_code(cls, v):
        """Validate error code is a non-empty string."""
        if not v or not isinstance(v, str):
            raise ValueError("error_code is required and must be a string")
        return v

    @classmethod
    @validator("message")
    def validate_message(cls, v):
        """Validate message is a non-empty string."""
        if not v or not isinstance(v, str):
            raise ValueError("message is required and must be a string")
        return v

    class Config:
        """Pydantic configuration for ErrorResponse."""

        extra = "forbid"
        validate_assignment = True


class HealthResponse(BaseModel):
    """Standard health check response."""

    status: str = Field(description="Service health status")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Health check timestamp"
    )
    version: Optional[str] = Field(None, description="Service version")
    uptime_seconds: Optional[int] = Field(None, description="Service uptime", ge=0)
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional health metadata"
    )

    @classmethod
    @validator("status")
    def validate_status(cls, v):
        """Validate status is one of the allowed values."""
        allowed_statuses = ["healthy", "degraded", "unhealthy"]
        if v not in allowed_statuses:
            raise ValueError(f"status must be one of {allowed_statuses}")
        return v

    class Config:
        """Pydantic configuration for HealthResponse."""

        extra = "allow"  # Allow service-specific health fields
        validate_assignment = True


class ApiResponse(BaseModel, Generic[T]):
    """Generic API response wrapper following shared interface patterns."""

    data: T = Field(description="Response data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Response metadata")

    class Config:
        extra = "forbid"
        validate_assignment = True


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper following shared interface patterns."""

    data: List[T] = Field(description="Response data items")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Response metadata")
    pagination: Optional[Dict[str, Any]] = Field(
        None, description="Pagination information"
    )

    class Config:
        extra = "forbid"
        validate_assignment = True


class BaseService:
    """Base service class providing common functionality following DRY principles.

    This class provides standard service functionality that all microservices need:
    - Health checking
    - Status reporting
    - Lifecycle management
    - Error handling patterns
    """

    def __init__(self, service_name: str, version: str = "1.0.0"):
        """Initialize base service.

        Args:
            service_name: Name of the service
            version: Service version
        """
        self.service_name = service_name
        self.version = version
        self.start_time = datetime.utcnow()
        self._is_healthy = True
        self._status_metadata = {}

    async def health_check(self) -> HealthResponse:
        """Perform health check following shared interface pattern.

        Returns:
            Health response following shared interface
        """
        uptime = (datetime.utcnow() - self.start_time).total_seconds()

        # Allow subclasses to add custom health checks
        custom_health = await self._custom_health_check()

        status = "healthy" if self._is_healthy and custom_health else "unhealthy"

        return HealthResponse(
            status=status,
            version=self.version,
            uptime_seconds=int(uptime),
            metadata={
                "service_name": self.service_name,
                **self._status_metadata,
                **(await self._get_health_metadata()),
            },
        )

    async def _custom_health_check(self) -> bool:
        """Override this method for service-specific health checks.

        Returns:
            True if service is healthy, False otherwise
        """
        return True

    async def _get_health_metadata(self) -> Dict[str, Any]:
        """Override this method to add service-specific health metadata.

        Returns:
            Dictionary of health metadata
        """
        return {}

    async def start(self) -> None:
        """Start the service. Override for service-specific startup logic."""
        self._is_healthy = True

    async def stop(self) -> None:
        """Stop the service. Override for service-specific shutdown logic."""
        self._is_healthy = False

    def set_health_status(
        self, is_healthy: bool, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Set service health status.

        Args:
            is_healthy: Whether service is healthy
            metadata: Optional metadata about health status
        """
        self._is_healthy = is_healthy
        if metadata:
            self._status_metadata.update(metadata)

    async def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status.

        Returns:
            Service status information
        """
        uptime = (datetime.utcnow() - self.start_time).total_seconds()

        return {
            "service_name": self.service_name,
            "version": self.version,
            "status": "running" if self._is_healthy else "unhealthy",
            "uptime_seconds": int(uptime),
            "start_time": self.start_time.isoformat(),
            "metadata": self._status_metadata,
        }

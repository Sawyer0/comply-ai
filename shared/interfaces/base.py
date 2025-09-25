"""Base interfaces for all microservice communication."""

from typing import Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator
import uuid


class BaseRequest(BaseModel):
    """Base request model with common fields."""

    correlation_id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Correlation ID for request tracing",
    )

    @validator("correlation_id")
    def validate_correlation_id(cls, v):
        if v and not isinstance(v, str):
            raise ValueError("correlation_id must be a string")
        return v

    class Config:
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

    @validator("request_id")
    def validate_request_id(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("request_id is required and must be a string")
        return v

    class Config:
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

    @validator("error_code")
    def validate_error_code(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("error_code is required and must be a string")
        return v

    @validator("message")
    def validate_message(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("message is required and must be a string")
        return v

    class Config:
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

    @validator("status")
    def validate_status(cls, v):
        allowed_statuses = ["healthy", "degraded", "unhealthy"]
        if v not in allowed_statuses:
            raise ValueError(f"status must be one of {allowed_statuses}")
        return v

    class Config:
        extra = "allow"  # Allow service-specific health fields
        validate_assignment = True

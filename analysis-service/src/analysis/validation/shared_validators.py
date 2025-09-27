"""
Shared validation utilities for analysis service.

This module provides validation functions that integrate with shared
validation components for consistent validation across all microservices.
"""

from typing import Any, Dict, Optional
from fastapi import HTTPException

from ..shared_integration import (
    validate_api_key,
    get_tenant_from_api_key,
    check_api_key_permissions,
    validate_non_empty_string,
    validate_confidence_score,
    get_shared_logger,
    ValidationError,
)

logger = get_shared_logger(__name__)


class SharedValidationService:
    """Service for shared validation operations."""

    async def validate_request_auth(self, api_key: str, required_permission: str) -> Dict[str, Any]:
        """Validate API key and permissions for a request."""
        try:
            # Validate API key
            is_valid = await validate_api_key(api_key)
            if not is_valid:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid API key"
                )

            # Check permissions
            has_permission = await check_api_key_permissions(api_key, required_permission)
            if not has_permission:
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient permissions. Required: {required_permission}"
                )

            # Get tenant information
            tenant_id = await get_tenant_from_api_key(api_key)
            if not tenant_id:
                raise HTTPException(
                    status_code=401,
                    detail="Unable to determine tenant from API key"
                )

            return {
                "valid": True,
                "tenant_id": tenant_id,
                "permission": required_permission
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Validation error", error=str(e))
            raise HTTPException(
                status_code=500,
                detail="Internal validation error"
            )

    def validate_analysis_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate analysis input data using shared validators."""
        try:
            # Validate required fields
            if "content" not in data:
                raise ValidationError("Content is required")
            
            content = data["content"]
            validate_non_empty_string(content, "content")
            
            # Validate confidence threshold if provided
            if "confidence_threshold" in data:
                confidence = data["confidence_threshold"]
                validate_confidence_score(confidence)
            
            # Validate tenant ID if provided
            if "tenant_id" in data:
                tenant_id = data["tenant_id"]
                validate_non_empty_string(tenant_id, "tenant_id")
            
            return {
                "valid": True,
                "validated_data": data
            }

        except ValidationError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Validation error: {str(e)}"
            )
        except Exception as e:
            logger.error("Input validation error", error=str(e))
            raise HTTPException(
                status_code=500,
                detail="Internal validation error"
            )


# Global validation service instance
_validation_service: Optional[SharedValidationService] = None


def get_shared_validation_service() -> SharedValidationService:
    """Get the global shared validation service instance."""
    global _validation_service
    if _validation_service is None:
        _validation_service = SharedValidationService()
    return _validation_service


__all__ = [
    "SharedValidationService",
    "get_shared_validation_service",
]

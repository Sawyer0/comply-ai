"""
Shared validation utilities for mapper service.

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
    """Enhanced service for shared validation operations with correlation tracking."""

    async def validate_request_auth(
        self, 
        api_key: str, 
        required_permission: str,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate API key and permissions for a request."""
        try:
            # Import correlation functions from shared integration
            from ..shared_integration import get_correlation_id, set_correlation_id
            
            # Use provided correlation ID or get current one
            if correlation_id:
                set_correlation_id(correlation_id)
            else:
                correlation_id = get_correlation_id()

            # Validate API key using shared validator
            is_valid = await validate_api_key(api_key)
            if not is_valid:
                logger.warning(
                    "Invalid API key provided",
                    correlation_id=correlation_id,
                    required_permission=required_permission
                )
                raise HTTPException(
                    status_code=401,
                    detail="Invalid API key",
                    headers={"X-Correlation-ID": correlation_id}
                )

            # Check permissions using shared validator
            has_permission = await check_api_key_permissions(api_key, required_permission)
            if not has_permission:
                logger.warning(
                    "Insufficient permissions for API key",
                    correlation_id=correlation_id,
                    required_permission=required_permission
                )
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient permissions. Required: {required_permission}",
                    headers={"X-Correlation-ID": correlation_id}
                )

            # Get tenant information using shared function
            tenant_id = await get_tenant_from_api_key(api_key)
            if not tenant_id:
                logger.error(
                    "Unable to determine tenant from valid API key",
                    correlation_id=correlation_id
                )
                raise HTTPException(
                    status_code=401,
                    detail="Unable to determine tenant from API key",
                    headers={"X-Correlation-ID": correlation_id}
                )

            logger.info(
                "Request authentication successful",
                tenant_id=tenant_id,
                permission=required_permission,
                correlation_id=correlation_id
            )

            return {
                "valid": True,
                "tenant_id": tenant_id,
                "permission": required_permission,
                "correlation_id": correlation_id
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                "Authentication validation error",
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=correlation_id,
                required_permission=required_permission
            )
            raise HTTPException(
                status_code=500,
                detail="Internal validation error",
                headers={"X-Correlation-ID": correlation_id} if correlation_id else {}
            )

    def validate_mapping_input(
        self, 
        data: Dict[str, Any], 
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate mapping input data using shared validators with enhanced error handling."""
        try:
            # Import correlation functions 
            from ..shared_integration import get_correlation_id, set_correlation_id
            
            # Use provided correlation ID or get current one
            if correlation_id:
                set_correlation_id(correlation_id)
            else:
                correlation_id = get_correlation_id()
                
            # Validate required fields
            if "detector_output" not in data:
                logger.error(
                    "Missing required field: detector_output",
                    correlation_id=correlation_id,
                    provided_fields=list(data.keys())
                )
                raise ValidationError("Detector output is required")
            
            detector_output = data["detector_output"]
            validate_non_empty_string(str(detector_output), "detector_output")
            
            # Validate detector type if provided
            if "detector" in data:
                detector = data["detector"]
                validate_non_empty_string(detector, "detector")
            
            # Validate confidence threshold if provided
            if "confidence_threshold" in data:
                confidence = data["confidence_threshold"]
                validate_confidence_score(confidence)
                logger.debug(
                    "Confidence threshold validated",
                    confidence_threshold=confidence,
                    correlation_id=correlation_id
                )
            
            # Validate tenant ID if provided
            if "tenant_id" in data:
                tenant_id = data["tenant_id"]
                validate_non_empty_string(tenant_id, "tenant_id")
                logger.debug(
                    "Tenant ID validated", 
                    tenant_id=tenant_id,
                    correlation_id=correlation_id
                )
            
            # Validate framework if provided
            if "framework" in data and data["framework"]:
                framework = data["framework"]
                validate_non_empty_string(framework, "framework")
                logger.debug(
                    "Target framework validated",
                    framework=framework,
                    correlation_id=correlation_id
                )
            
            logger.info(
                "Mapping input validation successful",
                detector=data.get("detector", "unknown"),
                has_confidence_threshold="confidence_threshold" in data,
                has_framework="framework" in data,
                correlation_id=correlation_id
            )
            
            return {
                "valid": True,
                "validated_data": data,
                "correlation_id": correlation_id
            }

        except ValidationError as e:
            logger.error(
                "Input validation error",
                error=str(e),
                correlation_id=correlation_id,
                provided_fields=list(data.keys()) if isinstance(data, dict) else "invalid_data_type"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Validation error: {str(e)}",
                headers={"X-Correlation-ID": correlation_id} if correlation_id else {}
            )
        except Exception as e:
            logger.error(
                "Unexpected validation error",
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=correlation_id
            )
            raise HTTPException(
                status_code=500,
                detail="Internal validation error",
                headers={"X-Correlation-ID": correlation_id} if correlation_id else {}
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

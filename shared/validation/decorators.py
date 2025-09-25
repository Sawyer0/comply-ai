"""Validation decorators for enhanced request/response validation."""

import asyncio
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union
import logging

from pydantic import BaseModel, ValidationError as PydanticValidationError

from ..exceptions.base import ValidationError, AuthorizationError
from ..utils.correlation import get_correlation_id

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
F = TypeVar("F", bound=Callable)


def validate_request_response(
    request_model: Optional[Type[T]] = None,
    response_model: Optional[Type[T]] = None,
    validate_tenant: bool = True,
    require_correlation_id: bool = True,
):
    """Decorator for validating request and response models."""

    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            correlation_id = get_correlation_id()

            # Extract request data and tenant_id from kwargs
            request_data = kwargs.get("request")
            tenant_id = kwargs.get("tenant_id")

            # Validate tenant access if required
            if validate_tenant and not tenant_id:
                raise AuthorizationError(
                    "Tenant ID is required", correlation_id=correlation_id
                )

            # Validate request if model is provided
            if request_model and request_data:
                try:
                    if isinstance(request_data, dict):
                        validated_request = request_model(**request_data)
                        kwargs["request"] = validated_request
                    elif not isinstance(request_data, request_model):
                        raise ValidationError(
                            f"Request must be of type {request_model.__name__}",
                            correlation_id=correlation_id,
                        )
                except PydanticValidationError as e:
                    logger.warning(
                        "Request validation failed for %s",
                        func.__name__,
                        extra={
                            "function": func.__name__,
                            "errors": e.errors(),
                            "correlation_id": correlation_id,
                            "tenant_id": tenant_id,
                        },
                    )
                    raise ValidationError(
                        f"Request validation failed: {e}",
                        error_code="REQUEST_VALIDATION_ERROR",
                        details={"errors": e.errors()},
                        correlation_id=correlation_id,
                    ) from e

            # Call the original function
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    "Function %s failed during execution",
                    func.__name__,
                    extra={
                        "function": func.__name__,
                        "error": str(e),
                        "correlation_id": correlation_id,
                        "tenant_id": tenant_id,
                    },
                )
                raise

            # Validate response if model is provided
            if response_model and result:
                try:
                    if isinstance(result, dict):
                        validated_response = response_model(**result)
                        return validated_response
                    elif not isinstance(result, response_model):
                        # Try to convert to dict and back
                        if hasattr(result, "dict"):
                            result_dict = result.dict()
                        else:
                            result_dict = result
                        validated_response = response_model(**result_dict)
                        return validated_response
                except PydanticValidationError as e:
                    logger.error(
                        "Response validation failed for %s",
                        func.__name__,
                        extra={
                            "function": func.__name__,
                            "errors": e.errors(),
                            "correlation_id": correlation_id,
                            "tenant_id": tenant_id,
                        },
                    )
                    raise ValidationError(
                        f"Response validation failed: {e}",
                        error_code="RESPONSE_VALIDATION_ERROR",
                        details={"errors": e.errors()},
                        correlation_id=correlation_id,
                    ) from e

            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            correlation_id = get_correlation_id()

            # Extract request data and tenant_id from kwargs
            request_data = kwargs.get("request")
            tenant_id = kwargs.get("tenant_id")

            # Validate tenant access if required
            if validate_tenant and not tenant_id:
                raise AuthorizationError(
                    "Tenant ID is required", correlation_id=correlation_id
                )

            # Validate request if model is provided
            if request_model and request_data:
                try:
                    if isinstance(request_data, dict):
                        validated_request = request_model(**request_data)
                        kwargs["request"] = validated_request
                    elif not isinstance(request_data, request_model):
                        raise ValidationError(
                            f"Request must be of type {request_model.__name__}",
                            correlation_id=correlation_id,
                        )
                except PydanticValidationError as e:
                    logger.warning(
                        "Request validation failed for %s",
                        func.__name__,
                        extra={
                            "function": func.__name__,
                            "errors": e.errors(),
                            "correlation_id": correlation_id,
                            "tenant_id": tenant_id,
                        },
                    )
                    raise ValidationError(
                        f"Request validation failed: {e}",
                        error_code="REQUEST_VALIDATION_ERROR",
                        details={"errors": e.errors()},
                        correlation_id=correlation_id,
                    ) from e

            # Call the original function
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    "Function %s failed during execution",
                    func.__name__,
                    extra={
                        "function": func.__name__,
                        "error": str(e),
                        "correlation_id": correlation_id,
                        "tenant_id": tenant_id,
                    },
                )
                raise

            # Validate response if model is provided
            if response_model and result:
                try:
                    if isinstance(result, dict):
                        validated_response = response_model(**result)
                        return validated_response
                    elif not isinstance(result, response_model):
                        # Try to convert to dict and back
                        if hasattr(result, "dict"):
                            result_dict = result.dict()
                        else:
                            result_dict = result
                        validated_response = response_model(**result_dict)
                        return validated_response
                except PydanticValidationError as e:
                    logger.error(
                        "Response validation failed for %s",
                        func.__name__,
                        extra={
                            "function": func.__name__,
                            "errors": e.errors(),
                            "correlation_id": correlation_id,
                            "tenant_id": tenant_id,
                        },
                    )
                    raise ValidationError(
                        f"Response validation failed: {e}",
                        error_code="RESPONSE_VALIDATION_ERROR",
                        details={"errors": e.errors()},
                        correlation_id=correlation_id,
                    ) from e

            return result

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def validate_tenant_access(
    allowed_tenants: Optional[list] = None, require_admin: bool = False
):
    """Decorator for validating tenant access."""

    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            correlation_id = get_correlation_id()
            tenant_id = kwargs.get("tenant_id")

            if not tenant_id:
                raise AuthorizationError(
                    "Tenant ID is required", correlation_id=correlation_id
                )

            # Check allowed tenants
            if allowed_tenants and tenant_id not in allowed_tenants:
                logger.warning(
                    "Unauthorized tenant access attempt",
                    extra={
                        "function": func.__name__,
                        "tenant_id": tenant_id,
                        "allowed_tenants": allowed_tenants,
                        "correlation_id": correlation_id,
                    },
                )
                raise AuthorizationError(
                    f"Tenant {tenant_id} is not authorized",
                    correlation_id=correlation_id,
                )

            # Check admin requirement
            if require_admin:
                # This would integrate with your auth system
                # For now, we'll assume admin check is done elsewhere
                pass

            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            correlation_id = get_correlation_id()
            tenant_id = kwargs.get("tenant_id")

            if not tenant_id:
                raise AuthorizationError(
                    "Tenant ID is required", correlation_id=correlation_id
                )

            # Check allowed tenants
            if allowed_tenants and tenant_id not in allowed_tenants:
                logger.warning(
                    "Unauthorized tenant access attempt",
                    extra={
                        "function": func.__name__,
                        "tenant_id": tenant_id,
                        "allowed_tenants": allowed_tenants,
                        "correlation_id": correlation_id,
                    },
                )
                raise AuthorizationError(
                    f"Tenant {tenant_id} is not authorized",
                    correlation_id=correlation_id,
                )

            # Check admin requirement
            if require_admin:
                # This would integrate with your auth system
                # For now, we'll assume admin check is done elsewhere
                pass

            return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def validate_confidence_threshold(min_confidence: float = 0.0):
    """Decorator for validating confidence thresholds in responses."""

    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Check confidence in result
            if hasattr(result, "confidence") and result.confidence < min_confidence:
                logger.warning(
                    "Low confidence result from %s: %.3f < %.3f",
                    func.__name__,
                    result.confidence,
                    min_confidence,
                    extra={
                        "function": func.__name__,
                        "confidence": result.confidence,
                        "min_confidence": min_confidence,
                        "correlation_id": get_correlation_id(),
                    },
                )

            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            result = func(*args, **kwargs)

            # Check confidence in result
            if hasattr(result, "confidence") and result.confidence < min_confidence:
                logger.warning(
                    "Low confidence result from %s: %.3f < %.3f",
                    func.__name__,
                    result.confidence,
                    min_confidence,
                    extra={
                        "function": func.__name__,
                        "confidence": result.confidence,
                        "min_confidence": min_confidence,
                        "correlation_id": get_correlation_id(),
                    },
                )

            return result

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

"""Validation middleware for FastAPI applications."""

import json
from typing import Any, Callable, Dict, Optional
import logging

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..exceptions.base import ValidationError, BaseServiceException
from ..utils.correlation import set_correlation_id, get_correlation_id
from .schemas import ValidationContext, default_validator

logger = logging.getLogger(__name__)


class ValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response validation."""

    def __init__(
        self,
        app,
        validate_requests: bool = True,
        validate_responses: bool = True,
        strict_mode: bool = True,
        schema_validator: Optional[Any] = None,
    ):
        super().__init__(app)
        self.validate_requests = validate_requests
        self.validate_responses = validate_responses
        self.strict_mode = strict_mode
        self.validator = schema_validator or default_validator

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and response with validation."""
        # Extract correlation ID from headers
        correlation_id = request.headers.get("X-Correlation-ID")
        if correlation_id:
            set_correlation_id(correlation_id)
        else:
            correlation_id = get_correlation_id()

        # Extract tenant ID
        tenant_id = request.headers.get("X-Tenant-ID")

        # Create validation context
        context = ValidationContext(
            tenant_id=tenant_id,
            correlation_id=correlation_id,
            strict_mode=self.strict_mode,
        )

        try:
            # Validate request if enabled
            if self.validate_requests:
                await self._validate_request(request, context)

            # Process request
            response = await call_next(request)

            # Validate response if enabled
            if self.validate_responses:
                response = await self._validate_response(response, context)

            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id

            return response

        except BaseServiceException as e:
            logger.warning(
                "Service exception in validation middleware: %s",
                e.message,
                extra={
                    "error_code": e.error_code,
                    "correlation_id": correlation_id,
                    "tenant_id": tenant_id,
                    "path": request.url.path,
                },
            )
            return JSONResponse(
                status_code=400 if isinstance(e, ValidationError) else 500,
                content=e.to_dict(),
                headers={"X-Correlation-ID": correlation_id},
            )
        except Exception as e:
            logger.error(
                "Unexpected error in validation middleware: %s",
                str(e),
                extra={
                    "correlation_id": correlation_id,
                    "tenant_id": tenant_id,
                    "path": request.url.path,
                },
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error_code": "INTERNAL_ERROR",
                    "message": "Internal server error",
                    "correlation_id": correlation_id,
                },
                headers={"X-Correlation-ID": correlation_id},
            )

    async def _validate_request(self, request: Request, context: ValidationContext):
        """Validate incoming request."""
        # Skip validation for certain paths
        if self._should_skip_validation(request.url.path):
            return

        # Only validate requests with JSON content
        content_type = request.headers.get("content-type", "")
        if not content_type.startswith("application/json"):
            return

        try:
            # Read request body
            body = await request.body()
            if not body:
                return

            # Parse JSON
            try:
                request_data = json.loads(body)
            except json.JSONDecodeError as e:
                raise ValidationError(
                    "Invalid JSON in request body",
                    error_code="INVALID_JSON",
                    details={"error": str(e)},
                    correlation_id=context.correlation_id,
                ) from e

            # Determine schema name from path
            schema_name = self._get_schema_name_from_path(
                request.url.path, request.method
            )
            if not schema_name:
                return

            # Validate request data
            self.validator.validate_request(request_data, schema_name, context)

        except ValidationError:
            raise
        except Exception as e:
            logger.error(
                "Request validation error: %s",
                str(e),
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "correlation_id": context.correlation_id,
                },
            )
            raise ValidationError(
                "Request validation failed",
                error_code="REQUEST_VALIDATION_ERROR",
                details={"error": str(e)},
                correlation_id=context.correlation_id,
            ) from e

    async def _validate_response(
        self, response: Response, context: ValidationContext
    ) -> Response:
        """Validate outgoing response."""
        # Only validate JSON responses
        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("application/json"):
            return response

        # Skip validation for error responses
        if response.status_code >= 400:
            return response

        try:
            # Get response body
            if hasattr(response, "body"):
                body = response.body
            else:
                return response

            if not body:
                return response

            # Parse JSON
            try:
                response_data = json.loads(body)
            except json.JSONDecodeError:
                # If we can't parse the response, don't validate
                return response

            # Determine schema name (this would need to be implemented based on your routing)
            # For now, we'll skip response validation
            # schema_name = self._get_response_schema_name(request_path, request_method)
            # if schema_name:
            #     self.validator.validate_response(response_data, schema_name, context)

            return response

        except Exception as e:
            logger.error(
                "Response validation error: %s",
                str(e),
                extra={"correlation_id": context.correlation_id},
            )
            # Don't fail the request for response validation errors
            return response

    def _should_skip_validation(self, path: str) -> bool:
        """Check if validation should be skipped for this path."""
        skip_paths = ["/health", "/docs", "/openapi.json", "/redoc"]
        return any(path.startswith(skip_path) for skip_path in skip_paths)

    def _get_schema_name_from_path(self, path: str, method: str) -> Optional[str]:
        """Get schema name from request path and method."""
        # This is a simplified implementation
        # In practice, you'd map paths to schema names based on your API structure

        path_schema_map = {
            ("/api/v1/orchestrate", "POST"): "OrchestrationRequest",
            ("/api/v1/analyze", "POST"): "AnalysisRequest",
            ("/api/v1/map", "POST"): "MappingRequest",
            ("/api/v1/batch-map", "POST"): "BatchMappingRequest",
            ("/api/v1/validate", "POST"): "ValidationRequest",
        }

        return path_schema_map.get((path, method))


class TenantValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for tenant validation and isolation."""

    def __init__(
        self,
        app,
        require_tenant_id: bool = True,
        allowed_tenants: Optional[list] = None,
    ):
        super().__init__(app)
        self.require_tenant_id = require_tenant_id
        self.allowed_tenants = allowed_tenants or []

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with tenant validation."""
        correlation_id = get_correlation_id()

        # Skip validation for certain paths
        if self._should_skip_validation(request.url.path):
            return await call_next(request)

        # Extract tenant ID
        tenant_id = request.headers.get("X-Tenant-ID")

        if self.require_tenant_id and not tenant_id:
            logger.warning(
                "Missing tenant ID in request",
                extra={"path": request.url.path, "correlation_id": correlation_id},
            )
            return JSONResponse(
                status_code=400,
                content={
                    "error_code": "MISSING_TENANT_ID",
                    "message": "Tenant ID is required",
                    "correlation_id": correlation_id,
                },
                headers={"X-Correlation-ID": correlation_id},
            )

        # Validate tenant ID if allowed list is provided
        if self.allowed_tenants and tenant_id not in self.allowed_tenants:
            logger.warning(
                "Unauthorized tenant access attempt",
                extra={
                    "tenant_id": tenant_id,
                    "path": request.url.path,
                    "correlation_id": correlation_id,
                },
            )
            return JSONResponse(
                status_code=403,
                content={
                    "error_code": "UNAUTHORIZED_TENANT",
                    "message": f"Tenant {tenant_id} is not authorized",
                    "correlation_id": correlation_id,
                },
                headers={"X-Correlation-ID": correlation_id},
            )

        # Add tenant ID to request state for use in handlers
        request.state.tenant_id = tenant_id

        return await call_next(request)

    def _should_skip_validation(self, path: str) -> bool:
        """Check if validation should be skipped for this path."""
        skip_paths = ["/health", "/docs", "/openapi.json", "/redoc"]
        return any(path.startswith(skip_path) for skip_path in skip_paths)

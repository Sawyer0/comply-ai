"""HTTP client for Detector Orchestration Service."""

import httpx
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import uuid

from ..interfaces.orchestration import (
    OrchestrationRequest,
    OrchestrationResponse,
    DetectorHealthStatus,
    PolicyValidationRequest,
    PolicyValidationResponse,
    AsyncJobRequest,
    AsyncJobResponse,
    AsyncJobStatus,
)
from ..utils.correlation import get_correlation_id
from ..exceptions.base import (
    BaseServiceException,
    ValidationError,
    AuthenticationError,
    RateLimitError,
    ServiceUnavailableError,
)

logger = logging.getLogger(__name__)


class OrchestrationServiceClient:
    """HTTP client for the Detector Orchestration Service."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

        # Configure HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            headers=self._get_default_headers(),
        )

    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for requests."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "llama-mapper-client/1.0.0",
        }

        if self.api_key:
            headers["X-API-Key"] = self.api_key

        return headers

    def _get_request_headers(
        self, tenant_id: str, correlation_id: Optional[str] = None
    ) -> Dict[str, str]:
        """Get headers for a specific request."""
        headers = {}
        headers["X-Tenant-ID"] = tenant_id

        if correlation_id:
            headers["X-Correlation-ID"] = correlation_id
        else:
            headers["X-Correlation-ID"] = get_correlation_id()

        return headers

    async def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Handle HTTP response and raise appropriate exceptions."""
        try:
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_data = {}
            try:
                error_data = response.json()
            except Exception:
                error_data = {"message": response.text}

            if response.status_code == 400:
                raise ValidationError(
                    error_data.get("message", "Bad request"),
                    error_code=error_data.get("error_code"),
                    details=error_data.get("details"),
                    correlation_id=error_data.get("correlation_id"),
                )
            elif response.status_code == 401:
                raise AuthenticationError(
                    error_data.get("message", "Unauthorized"),
                    error_code=error_data.get("error_code"),
                    correlation_id=error_data.get("correlation_id"),
                )
            elif response.status_code == 429:
                raise RateLimitError(
                    error_data.get("message", "Rate limit exceeded"),
                    error_code=error_data.get("error_code"),
                    correlation_id=error_data.get("correlation_id"),
                )
            elif response.status_code >= 500:
                raise ServiceUnavailableError(
                    error_data.get("message", "Service unavailable"),
                    error_code=error_data.get("error_code"),
                    correlation_id=error_data.get("correlation_id"),
                )
            else:
                raise BaseServiceException(
                    error_data.get("message", f"HTTP {response.status_code}"),
                    error_code=error_data.get("error_code"),
                    correlation_id=error_data.get("correlation_id"),
                )

    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        try:
            response = await self.client.get("/health")
            return await self._handle_response(response)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise

    async def orchestrate_detectors(
        self,
        request: OrchestrationRequest,
        tenant_id: str,
        correlation_id: Optional[str] = None,
    ) -> OrchestrationResponse:
        """Orchestrate detector execution."""
        try:
            headers = self._get_request_headers(tenant_id, correlation_id)

            response = await self.client.post(
                "/api/v1/orchestrate", json=request.dict(), headers=headers
            )

            data = await self._handle_response(response)
            return OrchestrationResponse(**data)

        except Exception as e:
            logger.error(
                f"Orchestration failed: {e}",
                extra={"tenant_id": tenant_id, "correlation_id": correlation_id},
            )
            raise

    async def list_detectors(
        self,
        tenant_id: str,
        status: Optional[str] = None,
        detector_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List available detectors."""
        try:
            headers = self._get_request_headers(tenant_id)
            params = {}

            if status:
                params["status"] = status
            if detector_type:
                params["detector_type"] = detector_type

            response = await self.client.get(
                "/api/v1/detectors", params=params, headers=headers
            )

            return await self._handle_response(response)

        except Exception as e:
            logger.error(f"List detectors failed: {e}", extra={"tenant_id": tenant_id})
            raise

    async def register_detector(
        self, detector_data: Dict[str, Any], tenant_id: str
    ) -> Dict[str, Any]:
        """Register a new detector."""
        try:
            headers = self._get_request_headers(tenant_id)

            response = await self.client.post(
                "/api/v1/detectors", json=detector_data, headers=headers
            )

            return await self._handle_response(response)

        except Exception as e:
            logger.error(
                f"Detector registration failed: {e}",
                extra={
                    "tenant_id": tenant_id,
                    "detector_type": detector_data.get("detector_type"),
                },
            )
            raise

    async def check_detector_health(
        self, detector_id: str, tenant_id: str
    ) -> DetectorHealthStatus:
        """Check detector health status."""
        try:
            headers = self._get_request_headers(tenant_id)

            response = await self.client.get(
                f"/api/v1/detectors/{detector_id}/health", headers=headers
            )

            data = await self._handle_response(response)
            return DetectorHealthStatus(**data)

        except Exception as e:
            logger.error(
                f"Detector health check failed: {e}",
                extra={"tenant_id": tenant_id, "detector_id": detector_id},
            )
            raise

    async def validate_policy(
        self, request: PolicyValidationRequest, tenant_id: str
    ) -> PolicyValidationResponse:
        """Validate policy compliance."""
        try:
            headers = self._get_request_headers(tenant_id)

            response = await self.client.post(
                "/api/v1/policies/validate", json=request.dict(), headers=headers
            )

            data = await self._handle_response(response)
            return PolicyValidationResponse(**data)

        except Exception as e:
            logger.error(
                f"Policy validation failed: {e}", extra={"tenant_id": tenant_id}
            )
            raise

    async def submit_async_job(
        self, request: AsyncJobRequest, tenant_id: str
    ) -> AsyncJobResponse:
        """Submit an async job."""
        try:
            headers = self._get_request_headers(tenant_id)

            response = await self.client.post(
                "/api/v1/jobs", json=request.dict(), headers=headers
            )

            data = await self._handle_response(response)
            return AsyncJobResponse(**data)

        except Exception as e:
            logger.error(
                f"Job submission failed: {e}",
                extra={"tenant_id": tenant_id, "job_type": request.job_type},
            )
            raise

    async def get_job_status(self, job_id: str, tenant_id: str) -> AsyncJobStatus:
        """Get job status."""
        try:
            headers = self._get_request_headers(tenant_id)

            response = await self.client.get(f"/api/v1/jobs/{job_id}", headers=headers)

            data = await self._handle_response(response)
            return AsyncJobStatus(**data)

        except Exception as e:
            logger.error(
                f"Get job status failed: {e}",
                extra={"tenant_id": tenant_id, "job_id": job_id},
            )
            raise

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Convenience function for creating client instances
def create_orchestration_client(
    base_url: Optional[str] = None, api_key: Optional[str] = None, **kwargs
) -> OrchestrationServiceClient:
    """Create an orchestration service client with default configuration."""
    import os

    if base_url is None:
        base_url = os.getenv("ORCHESTRATION_SERVICE_URL", "http://localhost:8000")

    if api_key is None:
        api_key = os.getenv("ORCHESTRATION_API_KEY")

    return OrchestrationServiceClient(base_url=base_url, api_key=api_key, **kwargs)

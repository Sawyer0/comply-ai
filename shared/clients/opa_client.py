"""HTTP client for Open Policy Agent (OPA) integration."""

import logging
from typing import Dict, Any, Optional, List
import httpx

from ..exceptions.base import BaseServiceException
from ..utils.retry import retry_with_backoff
from ..utils.circuit_breaker import circuit_breaker

logger = logging.getLogger(__name__)


class OPAError(BaseServiceException):
    """Base exception for OPA-related errors."""

    def __init__(
        self,
        message: str,
        opa_endpoint: Optional[str] = None,
        policy_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.opa_endpoint = opa_endpoint
        self.policy_name = policy_name
        if opa_endpoint:
            self.details["opa_endpoint"] = opa_endpoint
        if policy_name:
            self.details["policy_name"] = policy_name


class OPAPolicyError(OPAError):
    """Exception for OPA policy-specific errors."""

    def __init__(
        self,
        message: str,
        policy_name: Optional[str] = None,
        policy_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, policy_name=policy_name, **kwargs)
        self.policy_path = policy_path
        if policy_path:
            self.details["policy_path"] = policy_path


class OPATimeoutError(OPAError):
    """Exception for OPA timeout errors."""

    def __init__(
        self,
        message: str = "OPA request timed out",
        timeout_seconds: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.timeout_seconds = timeout_seconds
        if timeout_seconds:
            self.details["timeout_seconds"] = timeout_seconds


class OPAClient:
    """HTTP client for Open Policy Agent."""

    def __init__(
        self,
        base_url: str = "http://localhost:8181",
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

        # Configure HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            headers=self._get_default_headers(),
        )

    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for OPA requests."""
        return {
            "Content-Type": "application/json",
            "User-Agent": "comply-ai-opa-client/1.0.0",
        }

    def _get_request_headers(self, **kwargs) -> Dict[str, str]:
        """Get headers for a specific request."""
        headers = {}
        # Add any request-specific headers here
        return headers

    async def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Handle OPA response and raise appropriate exceptions."""
        try:
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_data = {}
            try:
                error_data = response.json()
            except Exception:
                error_data = {"message": response.text}

            error_code = error_data.get("error_code")
            if not isinstance(error_code, str):
                error_code = f"OPA_HTTP_{response.status_code}"

            details = error_data.get("details")
            if not isinstance(details, dict):
                details = {"response_text": response.text}

            if response.status_code == 404:
                raise OPAPolicyError(
                    "Policy not found",
                    error_code=error_code,
                    details=details,
                    opa_endpoint=self.base_url,
                ) from e
            if response.status_code == 400:
                raise OPAError(
                    "Invalid OPA request",
                    error_code=error_code,
                    details=details,
                    opa_endpoint=self.base_url,
                ) from e
            if response.status_code >= 500:
                raise OPAError(
                    "OPA server error",
                    error_code=error_code,
                    details=details,
                    opa_endpoint=self.base_url,
                ) from e
            raise OPAError(
                f"OPA request failed: {response.status_code}",
                error_code=error_code,
                details=details,
                opa_endpoint=self.base_url,
            ) from e

    @retry_with_backoff(max_attempts=3, base_delay=1.0)
    @circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
    async def load_policy(self, policy_name: str, policy_content: str, correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """Load a policy into OPA."""
        try:
            headers = self._get_request_headers(correlation_id=correlation_id)
            response = await self.client.put(
                f"/v1/policies/{policy_name}",
                content=policy_content,
                headers={**headers, "Content-Type": "text/plain"},
            )
            return await self._handle_response(response)
        except Exception as e:
            logger.error(
                "Failed to load policy: %s",
                e,
                extra={"policy_name": policy_name, "opa_endpoint": self.base_url},
            )
            if isinstance(e, (OPAError, OPAPolicyError)):
                raise
            raise OPAPolicyError(
                f"Failed to load policy {policy_name}: {e}",
                policy_name=policy_name,
                opa_endpoint=self.base_url,
            ) from e

    @retry_with_backoff(max_attempts=3, base_delay=1.0)
    @circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
    async def unload_policy(self, policy_name: str, correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """Unload a policy from OPA."""
        try:
            headers = self._get_request_headers(correlation_id=correlation_id)
            response = await self.client.delete(
                f"/v1/policies/{policy_name}", headers=headers
            )
            return await self._handle_response(response)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # Policy not found is not an error for unload
                return {"result": "policy_not_found"}
            raise
        except Exception as e:
            logger.error(
                "Failed to unload policy: %s",
                e,
                extra={"policy_name": policy_name, "opa_endpoint": self.base_url},
            )
            if isinstance(e, (OPAError, OPAPolicyError)):
                raise
            raise OPAPolicyError(
                f"Failed to unload policy {policy_name}: {e}",
                policy_name=policy_name,
                opa_endpoint=self.base_url,
            ) from e

    @retry_with_backoff(max_attempts=3, base_delay=1.0)
    @circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
    async def evaluate_policy(
        self,
        policy_path: str,
        input_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate a policy with given input data."""
        try:
            headers = self._get_request_headers()
            request_data = {"input": input_data}

            response = await self.client.post(
                f"/v1/data/{policy_path}", json=request_data, headers=headers
            )
            return await self._handle_response(response)
        except Exception as e:
            logger.error(
                "Failed to evaluate policy: %s",
                e,
                extra={"policy_path": policy_path, "opa_endpoint": self.base_url},
            )
            if isinstance(e, (OPAError, OPAPolicyError)):
                raise
            raise OPAError(
                f"Failed to evaluate policy {policy_path}: {e}",
                opa_endpoint=self.base_url,
            ) from e

    @retry_with_backoff(max_attempts=3, base_delay=1.0)
    @circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
    async def list_policies(self) -> List[str]:
        """List all loaded policies."""
        try:
            headers = self._get_request_headers()
            response = await self.client.get("/v1/policies", headers=headers)
            data = await self._handle_response(response)
            return list(data.get("result", {}).keys())
        except Exception as e:
            logger.error(
                "Failed to list policies: %s", e, extra={"opa_endpoint": self.base_url}
            )
            if isinstance(e, OPAError):
                raise
            raise OPAError(
                f"Failed to list policies: {e}", opa_endpoint=self.base_url
            ) from e

    @retry_with_backoff(max_attempts=3, base_delay=1.0)
    @circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
    async def get_policy(self, policy_name: str) -> Dict[str, Any]:
        """Get a specific policy."""
        try:
            headers = self._get_request_headers()
            response = await self.client.get(
                f"/v1/policies/{policy_name}", headers=headers
            )
            return await self._handle_response(response)
        except Exception as e:
            logger.error(
                "Failed to get policy: %s",
                e,
                extra={"policy_name": policy_name, "opa_endpoint": self.base_url},
            )
            if isinstance(e, (OPAError, OPAPolicyError)):
                raise
            raise OPAPolicyError(
                f"Failed to get policy {policy_name}: {e}",
                policy_name=policy_name,
                opa_endpoint=self.base_url,
            ) from e

    @retry_with_backoff(max_attempts=3, base_delay=1.0)
    @circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
    async def validate_policy_syntax(self, policy_content: str, correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """Validate policy syntax."""
        try:
            headers = self._get_request_headers(correlation_id)
            # Use compile endpoint to validate syntax
            response = await self.client.post(
                "/v1/compile",
                json={"query": policy_content},
                headers=headers,
            )
            return await self._handle_response(response)
        except Exception as e:
            logger.error(
                "Failed to validate policy syntax: %s",
                e,
                extra={"opa_endpoint": self.base_url},
            )
            if isinstance(e, OPAError):
                raise
            raise OPAError(
                f"Failed to validate policy syntax: {e}", opa_endpoint=self.base_url
            ) from e

    @retry_with_backoff(max_attempts=3, base_delay=1.0)
    @circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
    async def load_data(self, data_path: str, data: Dict[str, Any], correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """Load data into OPA."""
        try:
            headers = self._get_request_headers(correlation_id=correlation_id)
            response = await self.client.put(
                f"/v1/data/{data_path}", json=data, headers=headers
            )
            return await self._handle_response(response)
        except Exception as e:
            logger.error(
                "Failed to load data: %s",
                e,
                extra={"data_path": data_path, "opa_endpoint": self.base_url},
            )
            if isinstance(e, OPAError):
                raise
            raise OPAError(
                f"Failed to load data to {data_path}: {e}", opa_endpoint=self.base_url
            ) from e

    @retry_with_backoff(max_attempts=3, base_delay=1.0)
    @circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
    async def health_check(self) -> Dict[str, Any]:
        """Check OPA health."""
        try:
            headers = self._get_request_headers()
            response = await self.client.get("/health", headers=headers)
            return await self._handle_response(response)
        except Exception as e:
            logger.error(
                "OPA health check failed: %s", e, extra={"opa_endpoint": self.base_url}
            )
            if isinstance(e, OPAError):
                raise
            raise OPAError(
                f"OPA health check failed: {e}", opa_endpoint=self.base_url
            ) from e

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Convenience function for creating OPA client instances
def create_opa_client_with_config(
    base_url: Optional[str] = None, **kwargs
) -> OPAClient:
    """Create an OPA client with default configuration."""
    import os

    if base_url is None:
        base_url = os.getenv("OPA_ENDPOINT", "http://localhost:8181")

    return OPAClient(base_url=base_url, **kwargs)

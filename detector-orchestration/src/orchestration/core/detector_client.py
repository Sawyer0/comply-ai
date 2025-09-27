"""HTTP client for invoking customer-hosted detectors."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import httpx

from shared.exceptions.base import (
    ServiceUnavailableError,
    TimeoutError,
    ValidationError,
)
from shared.interfaces.orchestration import DetectorResult
from shared.utils.correlation import get_correlation_id

logger = logging.getLogger(__name__)


def _default_parser(payload: Dict[str, Any]) -> DetectorResult:
    """Default parser assuming payload matches DetectorResult schema."""
    return DetectorResult.model_validate(payload)


class CustomerDetectorClient:
    """Thin HTTP client used to invoke a customer-managed detector endpoint."""

    def __init__(
        self,
        *,
        name: str,
        endpoint: str,
        timeout: float = 5.0,
        default_headers: Optional[Dict[str, str]] = None,
        response_parser: Optional[Callable[[Dict[str, Any]], DetectorResult]] = None,
    ) -> None:
        self.name = name
        self.endpoint = endpoint
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
        self._default_headers = default_headers or {}
        self._parser = response_parser or _default_parser

    async def analyze(self, content: str, metadata: Dict[str, Any]) -> DetectorResult:
        """Invoke the detector's `/analyze` endpoint and return a detector result."""
        correlation_id = get_correlation_id()
        headers = {
            **self._default_headers,
            "X-Correlation-ID": correlation_id,
        }

        try:
            response = await self._client.post(
                self.endpoint,
                json={"content": content, "metadata": metadata},
                headers=headers,
            )
        except httpx.TimeoutException as exc:  # pragma: no cover - network failure
            logger.warning(
                "Detector %s request timed out",
                self.name,
                extra={"correlation_id": correlation_id},
            )
            raise TimeoutError(
                f"Detector {self.name} timed out",
                correlation_id=correlation_id,
            ) from exc
        except httpx.HTTPError as exc:  # pragma: no cover - network failure
            logger.error(
                "Detector %s request failed",
                self.name,
                extra={"correlation_id": correlation_id},
            )
            raise ServiceUnavailableError(
                f"Detector {self.name} request failed",
                correlation_id=correlation_id,
            ) from exc

        if response.status_code >= 500:  # pragma: no cover - remote failure
            logger.error(
                "Detector %s responded with %s",
                self.name,
                response.status_code,
                extra={"correlation_id": correlation_id},
            )
            raise ServiceUnavailableError(
                f"Detector {self.name} unavailable ({response.status_code})",
                correlation_id=correlation_id,
            )

        if response.status_code >= 400:
            logger.warning(
                "Detector %s rejected request with %s",
                self.name,
                response.status_code,
                extra={"correlation_id": correlation_id},
            )
            raise ValidationError(
                f"Detector {self.name} rejected request ({response.status_code})",
                correlation_id=correlation_id,
            )

        try:
            payload = response.json()
        except ValueError as exc:  # pragma: no cover - malformed payload
            logger.error(
                "Detector %s returned invalid JSON",
                self.name,
                extra={"correlation_id": correlation_id},
            )
            raise ValidationError(
                f"Detector {self.name} returned invalid JSON",
                correlation_id=correlation_id,
            ) from exc

        return self._parser(payload)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


__all__ = ["CustomerDetectorClient"]

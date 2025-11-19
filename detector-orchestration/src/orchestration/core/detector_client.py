"""HTTP client for invoking customer-hosted detectors."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict

import httpx

from shared.exceptions.base import (
    ServiceUnavailableError,
    TimeoutError as ServiceTimeoutError,
    ValidationError,
)
from shared.interfaces.orchestration import DetectorResult
from shared.utils.correlation import get_correlation_id
from shared.utils.circuit_breaker import CircuitBreaker, CircuitBreakerError

from .models import DetectorClientConfig

logger = logging.getLogger(__name__)


def _default_parser(payload: Dict[str, Any]) -> DetectorResult:
    """Default parser assuming payload matches DetectorResult schema."""
    return DetectorResult.model_validate(payload)


class CustomerDetectorClient:
    """Thin HTTP client used to invoke a customer-managed detector endpoint."""

    def __init__(self, config: DetectorClientConfig) -> None:
        self.name = config.name
        self.endpoint = config.endpoint
        self.timeout = config.timeout
        self.max_retries = max(int(config.max_retries), 0)
        self._client = httpx.AsyncClient(timeout=config.timeout)
        self._default_headers = dict(config.default_headers)
        self._parser: Callable[[Dict[str, Any]], DetectorResult] = (
            config.response_parser or _default_parser
        )
        # Circuit breaker opens after repeated transport-level failures
        self._breaker = CircuitBreaker(
            failure_threshold=max(self.max_retries or 1, 3),
            recovery_timeout=60.0,
            expected_exception=(ServiceTimeoutError, ServiceUnavailableError),
            name=f"detector:{self.name}",
        )

    def is_available_for_request(self) -> bool:
        """Return False when the circuit breaker is OPEN and requests should be skipped."""

        state = getattr(self._breaker, "state", None)
        # When state is None or not OPEN, allow requests
        return getattr(state, "name", "CLOSED").upper() != "OPEN"

    async def analyze(self, content: str, metadata: Dict[str, Any]) -> DetectorResult:
        """Invoke the detector's `/analyze` endpoint and return a detector result."""
        correlation_id = get_correlation_id()
        headers = {
            **self._default_headers,
            "X-Correlation-ID": correlation_id,
        }

        last_error: Exception | None = None
        attempts = self.max_retries + 1

        for attempt in range(attempts):
            try:
                return await self._breaker.call(
                    self._invoke_once,
                    content,
                    metadata,
                    headers,
                    correlation_id,
                )
            except (ServiceTimeoutError, ServiceUnavailableError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                backoff = min(0.1 * (2**attempt), 1.0)
                logger.warning(
                    "Detector %s attempt %d/%d failed: %s; retrying in %.2fs",
                    self.name,
                    attempt + 1,
                    attempts,
                    exc,
                    backoff,
                    extra={"correlation_id": correlation_id},
                )
                await asyncio.sleep(backoff)
            except CircuitBreakerError as exc:
                logger.error(
                    "Circuit breaker open for detector %s",
                    self.name,
                    extra={"correlation_id": correlation_id},
                )
                raise ServiceUnavailableError(
                    f"Detector {self.name} circuit breaker open",
                    correlation_id=correlation_id,
                ) from exc

        # Should be unreachable because we re-raise on final attempt
        if last_error is not None:  # pragma: no cover - defensive
            raise last_error

        raise ServiceUnavailableError(
            f"Detector {self.name} request failed",
            correlation_id=correlation_id,
        )

    async def _invoke_once(
        self,
        content: str,
        metadata: Dict[str, Any],
        headers: Dict[str, str],
        correlation_id: str,
    ) -> DetectorResult:
        """Perform a single HTTP invocation without retries.

        This method is intended to be wrapped by the circuit breaker and retry logic.
        """

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
            raise ServiceTimeoutError(
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

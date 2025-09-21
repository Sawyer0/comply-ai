"""HTTP detector clients used by the orchestrator for detector invocation."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from .models import DetectorResult, DetectorStatus, DetectorCapabilities, ContentType


@dataclass
class DetectorClient:
    """Client used to invoke a detector endpoint over HTTP or builtins."""

    name: str
    endpoint: str
    timeout_ms: int
    max_retries: int
    auth: Dict[str, Any]

    async def analyze(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> DetectorResult:
        """Analyze content with the detector, applying simple retry semantics."""

        start = time.perf_counter()
        # Builtins execute locally, no retries
        if self.endpoint.startswith("builtin:"):
            out, conf = await self._run_builtin(content)
            return DetectorResult(
                detector=self.name,
                status=DetectorStatus.SUCCESS,
                output=out,
                confidence=conf,
                processing_time_ms=int((time.perf_counter() - start) * 1000),
            )

        attempts = max(1, int(self.max_retries) + 1)
        backoff_base = 0.05  # 50ms base
        last_error: Optional[str] = None
        for i in range(attempts):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_ms / 1000) as client:
                    payload = {"content": content, "metadata": metadata or {}}
                    resp = await client.post(self.endpoint, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        return DetectorResult(
                            detector=self.name,
                            status=DetectorStatus.SUCCESS,
                            output=data.get("output", ""),
                            metadata=data.get("metadata", {}),
                            confidence=data.get("confidence"),
                            processing_time_ms=int(
                                (time.perf_counter() - start) * 1000
                            ),
                        )
                    # Retry on 5xx
                    if 500 <= resp.status_code < 600 and i < attempts - 1:
                        await asyncio.sleep(backoff_base * (2**i))
                        continue
                    # Non-retryable HTTP
                    last_error = f"HTTP {resp.status_code}"
                    break
            except httpx.ReadTimeout:
                if i < attempts - 1:
                    await asyncio.sleep(backoff_base * (2**i))
                    continue
                last_error = "timeout"
                break
            except httpx.HTTPError as exc:  # network errors
                if i < attempts - 1:
                    await asyncio.sleep(backoff_base * (2**i))
                    continue
                last_error = str(exc)
                break

        # Failed after attempts
        return DetectorResult(
            detector=self.name,
            status=(
                DetectorStatus.FAILED
                if last_error != "timeout"
                else DetectorStatus.TIMEOUT
            ),
            error=last_error or "unknown_error",
            processing_time_ms=int((time.perf_counter() - start) * 1000),
        )

    async def health_check(self) -> bool:
        """Check detector liveness via a simple health endpoint."""

        if self.endpoint.startswith("builtin:"):
            return True
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(self.endpoint.replace("/analyze", "/health"))
                return int(resp.status_code) == 200
        except httpx.HTTPError:
            return False

    async def get_capabilities(self) -> DetectorCapabilities:
        """Return detector capability metadata, falling back to defaults."""

        if self.endpoint.startswith("builtin:"):
            return self._builtin_capabilities()
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(
                    self.endpoint.replace("/analyze", "/capabilities")
                )
                if resp.status_code == 200:
                    data = resp.json()
                    types = [
                        ContentType(t)
                        for t in data.get("supported_content_types", ["text"])
                    ]
                    return DetectorCapabilities(
                        supported_content_types=types,
                        max_content_length=int(data.get("max_content_length", 50000)),
                        average_processing_time_ms=data.get(
                            "average_processing_time_ms"
                        ),
                        confidence_calibrated=data.get("confidence_calibrated"),
                        batch_supported=data.get("batch_supported"),
                    )
        except httpx.HTTPError:
            return DetectorCapabilities(supported_content_types=[ContentType.TEXT])
        return DetectorCapabilities(supported_content_types=[ContentType.TEXT])

    async def _run_builtin(self, content: str) -> tuple[str, float]:
        """Execute lightweight builtin detectors used for tests and demos."""

        kind = self.endpoint.split(":", 1)[1]
        result = ("unknown", 0.0)
        if kind == "echo":
            result = ("echo", 0.5)
        elif kind == "toxicity":
            lowered = content.lower()
            if any(w in lowered for w in ["toxic", "hate", "abuse"]):
                result = ("toxic", 0.9)
            else:
                result = ("clean", 0.6)
        elif kind == "regex-pii":
            ssn = re.search(r"\b\d{3}-\d{2}-\d{4}\b", content)
            email = re.search(r"[\w_.+-]+@[\w-]+\.[\w.-]+", content)
            if ssn:
                result = ("ssn_detected", 0.95)
            elif email:
                result = ("email_detected", 0.85)
            else:
                result = ("no_pii", 0.6)
        return result

    def _builtin_capabilities(self) -> DetectorCapabilities:
        """Return static capability declarations for builtin detectors."""

        kind = self.endpoint.split(":", 1)[1]
        if kind == "regex-pii":
            return DetectorCapabilities(
                supported_content_types=[ContentType.TEXT],
                max_content_length=10000,
            )
        return DetectorCapabilities(supported_content_types=[ContentType.TEXT])

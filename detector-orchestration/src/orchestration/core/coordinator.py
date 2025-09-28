"""Detector coordination functionality following SRP.

This module provides ONLY detector coordination - executing detectors according to routing plans.
Other responsibilities are handled by separate modules:
- Health monitoring: ../monitoring/health_monitor.py
- Circuit breakers: ../resilience/circuit_breaker.py
- Service discovery: ../discovery/service_discovery.py
- Policy management: ../policy/policy_manager.py
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from shared.exceptions.base import BaseServiceException, ServiceUnavailableError
from shared.interfaces.orchestration import DetectorResult
from shared.utils.correlation import get_correlation_id

from .models import RoutingPlan

logger = logging.getLogger(__name__)


class DetectorCoordinator:  # pylint: disable=too-few-public-methods
    """Coordinates detector execution according to routing plans."""

    def __init__(self, detector_clients: Dict[str, Any]) -> None:
        """Store the detector client registry."""
        self.detector_clients = detector_clients

    async def execute_routing_plan(
        self,
        content: str,
        routing_plan: RoutingPlan,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DetectorResult]:
        """Execute detectors defined by the routing plan and return their results."""

        correlation_id = get_correlation_id()
        request_identifier = request_id or correlation_id
        metadata = metadata or {}

        logger.info(
            "Executing routing plan with %d primary detectors",
            len(routing_plan.primary_detectors),
            extra={
                "request_id": request_identifier,
                "correlation_id": correlation_id,
                "primary_detectors": routing_plan.primary_detectors,
                "secondary_detectors": routing_plan.secondary_detectors,
            },
        )

        start_time = time.time()
        try:
            results = await self._run_primary_groups(
                routing_plan=routing_plan,
                content=content,
                metadata=metadata,
            )

            if routing_plan.secondary_detectors and not self._primary_succeeded(results):
                secondary_results = await self._execute_detector_group(
                    detectors=routing_plan.secondary_detectors,
                    content=content,
                    routing_plan=routing_plan,
                    metadata=metadata,
                )
                results.extend(secondary_results)
        except (BaseServiceException, asyncio.TimeoutError) as exc:
            logger.error(
                "Routing plan execution failed: %s",
                exc,
                extra={
                    "request_id": request_identifier,
                    "correlation_id": correlation_id,
                },
            )
            raise ServiceUnavailableError(
                f"Routing plan execution failed: {exc}",
                correlation_id=correlation_id,
            ) from exc

        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "Routing plan execution completed with %d results in %dms",
            len(results),
            processing_time_ms,
            extra={
                "request_id": request_identifier,
                "correlation_id": correlation_id,
                "total_results": len(results),
                "processing_time_ms": processing_time_ms,
            },
        )

        return results

    async def _run_primary_groups(
        self,
        *,
        routing_plan: RoutingPlan,
        content: str,
        metadata: Dict[str, Any],
    ) -> List[DetectorResult]:
        """Execute all primary detector groups sequentially and collect results."""

        results: List[DetectorResult] = []
        for group in routing_plan.parallel_groups:
            group_results = await self._execute_detector_group(
                detectors=group,
                content=content,
                routing_plan=routing_plan,
                metadata=metadata,
            )
            results.extend(group_results)
        return results

    @staticmethod
    def _primary_succeeded(results: List[DetectorResult]) -> bool:
        """Return True when any primary detector produced a confident result."""

        return any(result.confidence > 0.5 for result in results)

    async def _execute_detector_group(
        self,
        detectors: List[str],
        content: str,
        routing_plan: RoutingPlan,
        metadata: Dict[str, Any],
    ) -> List[DetectorResult]:
        tasks = [
            asyncio.create_task(
                self._execute_single_detector(
                    detector_name=detector,
                    content=content,
                    routing_plan=routing_plan,
                    metadata=metadata,
                )
            )
            for detector in detectors
        ]

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)

        detector_results: List[DetectorResult] = []
        for index, result in enumerate(results):
            detector_name = detectors[index] if index < len(detectors) else "unknown"
            if isinstance(result, DetectorResult):
                detector_results.append(result)
            else:
                detector_results.append(
                    self._build_failure_result(detector_name, str(result), 0, "critical")
                )

        return detector_results

    async def _execute_single_detector(
        self,
        detector_name: str,
        content: str,
        routing_plan: RoutingPlan,
        metadata: Dict[str, Any],
    ) -> DetectorResult:
        if detector_name not in self.detector_clients:
            logger.warning("Detector %s not found", detector_name)
            return self._build_failure_result(detector_name, "detector_not_found", 0, "critical")

        detector_client = self.detector_clients[detector_name]
        timeout_ms = routing_plan.timeout_config.get(detector_name, 5000)
        max_retries = routing_plan.retry_config.get(detector_name, 3)

        for attempt in range(max_retries + 1):
            try:
                return await asyncio.wait_for(
                    detector_client.analyze(content, metadata),
                    timeout=timeout_ms / 1000.0,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Detector %s timed out (attempt %d/%d)",
                    detector_name,
                    attempt + 1,
                    max_retries + 1,
                )
                if attempt < max_retries:
                    await asyncio.sleep(0.1 * (2**attempt))
                    continue
                return self._build_failure_result(
                    detector_name,
                    "timeout",
                    timeout_ms,
                    "medium",
                )
            except BaseServiceException as exc:
                logger.error(
                    "Detector %s failed with service exception: %s",
                    detector_name,
                    exc,
                )
                if attempt < max_retries:
                    await asyncio.sleep(0.1 * (2**attempt))
                    continue
                return self._build_failure_result(detector_name, str(exc), 0, "high")
            except (RuntimeError, ValueError) as exc:
                logger.error(
                    "Detector %s returned invalid response: %s",
                    detector_name,
                    exc,
                )
                if attempt < max_retries:
                    await asyncio.sleep(0.05 * (2**attempt))
                    continue
                return self._build_failure_result(detector_name, str(exc), 0, "high")

        return self._build_failure_result(detector_name, "unknown_failure", 0, "critical")

    @staticmethod
    def _build_failure_result(
        detector_id: str,
        error_message: str,
        processing_time_ms: int,
        severity: str,
    ) -> DetectorResult:
        return DetectorResult(
            detector_id=detector_id,
            detector_type=detector_id,
            confidence=0.0,
            category="error",
            severity=severity,
            findings=[{"error": error_message}],
            processing_time_ms=processing_time_ms,
        )


__all__ = ["DetectorCoordinator"]

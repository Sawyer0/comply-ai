"""Detector coordination functionality following SRP.

This module provides ONLY detector coordination - executing detectors according to routing plans.
Other responsibilities are handled by separate modules:
- Health monitoring: ../monitoring/health_monitor.py
- Circuit breakers: ../resilience/circuit_breaker.py
- Service discovery: ../discovery/service_discovery.py
- Policy management: ../policy/policy_manager.py
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any

from shared.interfaces.orchestration import DetectorResult
from shared.utils.correlation import get_correlation_id
from shared.exceptions.base import ServiceUnavailableError, TimeoutError

logger = logging.getLogger(__name__)


class RoutingPlan:
    """Routing plan for detector execution - data structure only."""

    def __init__(
        self,
        primary_detectors: List[str],
        secondary_detectors: Optional[List[str]] = None,
        parallel_groups: Optional[List[List[str]]] = None,
        timeout_config: Optional[Dict[str, int]] = None,
        retry_config: Optional[Dict[str, int]] = None,
    ):
        self.primary_detectors = primary_detectors
        self.secondary_detectors = secondary_detectors or []
        self.parallel_groups = (
            parallel_groups or [primary_detectors] if primary_detectors else []
        )
        self.timeout_config = timeout_config or {}
        self.retry_config = retry_config or {}


class DetectorCoordinator:
    """Coordinates detector execution according to routing plans.

    Single Responsibility: Execute detectors in parallel/sequential groups as specified by routing plan.
    Does NOT handle: health monitoring, circuit breakers, service discovery, policy management.
    """

    def __init__(self, detector_clients: Dict[str, Any]):
        """Initialize coordinator with detector clients.

        Args:
            detector_clients: Dictionary of detector name -> client instance
        """
        self.detector_clients = detector_clients

    async def execute_routing_plan(
        self,
        content: str,
        routing_plan: RoutingPlan,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DetectorResult]:
        """Execute detector routing plan.

        Single responsibility: coordinate detector execution according to the plan.

        Args:
            content: Content to analyze
            routing_plan: Plan specifying which detectors to run and how
            request_id: Optional request identifier
            metadata: Optional metadata to pass to detectors

        Returns:
            List of detector results
        """
        correlation_id = get_correlation_id()
        request_id = request_id or correlation_id
        metadata = metadata or {}

        logger.info(
            "Executing routing plan with %d primary detectors",
            len(routing_plan.primary_detectors),
            extra={
                "request_id": request_id,
                "correlation_id": correlation_id,
                "primary_detectors": routing_plan.primary_detectors,
                "secondary_detectors": routing_plan.secondary_detectors,
            },
        )

        start_time = time.time()
        results = []

        try:
            # Execute primary detectors in parallel groups
            for group in routing_plan.parallel_groups:
                group_results = await self._execute_detector_group(
                    detectors=group,
                    content=content,
                    routing_plan=routing_plan,
                    metadata=metadata,
                )
                results.extend(group_results)

            # Execute secondary detectors if primary failed
            if routing_plan.secondary_detectors:
                primary_success = any(
                    r.confidence > 0.5 for r in results  # Simple success check
                )
                if not primary_success:
                    secondary_results = await self._execute_detector_group(
                        detectors=routing_plan.secondary_detectors,
                        content=content,
                        routing_plan=routing_plan,
                        metadata=metadata,
                    )
                    results.extend(secondary_results)

            processing_time = int((time.time() - start_time) * 1000)

            logger.info(
                "Routing plan execution completed with %d results in %dms",
                len(results),
                processing_time,
                extra={
                    "request_id": request_id,
                    "correlation_id": correlation_id,
                    "total_results": len(results),
                    "processing_time_ms": processing_time,
                },
            )

            return results

        except Exception as e:
            logger.error(
                "Routing plan execution failed: %s",
                str(e),
                extra={
                    "request_id": request_id,
                    "correlation_id": correlation_id,
                    "error": str(e),
                },
            )
            raise ServiceUnavailableError(
                f"Routing plan execution failed: {str(e)}",
                correlation_id=correlation_id,
            ) from e

    async def _execute_detector_group(
        self,
        detectors: List[str],
        content: str,
        routing_plan: RoutingPlan,
        metadata: Dict[str, Any],
    ) -> List[DetectorResult]:
        """Execute a group of detectors in parallel."""

        tasks = []
        for detector_name in detectors:
            task = self._execute_single_detector(
                detector_name=detector_name,
                content=content,
                routing_plan=routing_plan,
                metadata=metadata,
            )
            tasks.append(task)

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed detector results
        detector_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                detector_results.append(
                    DetectorResult(
                        detector_id=detectors[i],
                        detector_type=detectors[i],
                        confidence=0.0,
                        category="error",
                        severity="critical",
                        findings=[{"error": str(result)}],
                        processing_time_ms=0,
                    )
                )
            else:
                detector_results.append(result)

        return detector_results

    async def _execute_single_detector(
        self,
        detector_name: str,
        content: str,
        routing_plan: RoutingPlan,
        metadata: Dict[str, Any],
    ) -> DetectorResult:
        """Execute a single detector with timeout and retry logic."""

        # Check if detector exists
        if detector_name not in self.detector_clients:
            logger.warning("Detector %s not found", detector_name)
            return DetectorResult(
                detector_id=detector_name,
                detector_type=detector_name,
                confidence=0.0,
                category="error",
                severity="critical",
                findings=[{"error": "detector_not_found"}],
                processing_time_ms=0,
            )

        detector_client = self.detector_clients[detector_name]

        # Get configuration from routing plan
        timeout_ms = routing_plan.timeout_config.get(detector_name, 5000)
        max_retries = routing_plan.retry_config.get(detector_name, 3)

        # Execute detector with timeout and retries
        for attempt in range(max_retries + 1):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    detector_client.analyze(content, metadata),
                    timeout=timeout_ms / 1000.0,
                )

                return result

            except asyncio.TimeoutError:
                logger.warning(
                    "Detector %s timed out (attempt %d/%d)",
                    detector_name,
                    attempt + 1,
                    max_retries + 1,
                )

                if attempt < max_retries:
                    await asyncio.sleep(0.1 * (2**attempt))  # Exponential backoff
                    continue

                return DetectorResult(
                    detector_id=detector_name,
                    detector_type=detector_name,
                    confidence=0.0,
                    category="error",
                    severity="medium",
                    findings=[{"error": "timeout"}],
                    processing_time_ms=timeout_ms,
                )

            except Exception as e:
                logger.error(
                    "Detector %s failed (attempt %d/%d): %s",
                    detector_name,
                    attempt + 1,
                    max_retries + 1,
                    str(e),
                )

                if attempt < max_retries:
                    await asyncio.sleep(0.1 * (2**attempt))  # Exponential backoff
                    continue

                return DetectorResult(
                    detector_id=detector_name,
                    detector_type=detector_name,
                    confidence=0.0,
                    category="error",
                    severity="high",
                    findings=[{"error": str(e)}],
                    processing_time_ms=0,
                )

        # Should not reach here
        return DetectorResult(
            detector_id=detector_name,
            detector_type=detector_name,
            confidence=0.0,
            category="error",
            severity="critical",
            findings=[{"error": "unknown_failure"}],
            processing_time_ms=0,
        )


# Export only the core coordination functionality
__all__ = [
    "DetectorCoordinator",
    "RoutingPlan",
]

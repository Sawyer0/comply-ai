from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from .clients import DetectorClient
from .models import DetectorResult, DetectorStatus, RoutingPlan
from .circuit_breaker import CircuitBreakerManager, CircuitBreakerState
from .metrics import OrchestrationMetricsCollector


class DetectorCoordinator:
    """Execute detectors per routing plan with simple parallelism and timeouts."""

    def __init__(
        self,
        clients: Dict[str, DetectorClient],
        breakers: CircuitBreakerManager | None = None,
        metrics: OrchestrationMetricsCollector | None = None,
        retry_on_timeouts: bool = True,
        retry_on_failures: bool = True,
    ):
        self.clients = clients
        self.breakers = breakers
        self.metrics = metrics
        # Retry policy toggles (injected from settings)
        self._retry_on_timeouts = retry_on_timeouts
        self._retry_on_failures = retry_on_failures

    async def execute_detector_group(
        self, detectors: List[str], content: str, plan: RoutingPlan, metadata: dict
    ) -> List[DetectorResult]:
        tasks = []
        for name in detectors:
            timeout = plan.timeout_config.get(name, 3000)
            tasks.append(
                self._execute_single_with_timeout(
                    name,
                    content,
                    timeout,
                    metadata,
                    retries=plan.retry_config.get(name, 0),
                )
            )
        if not tasks:
            return []
        return await asyncio.gather(*tasks)

    async def _execute_single_with_timeout(
        self,
        detector: str,
        content: str,
        timeout_ms: int,
        metadata: dict,
        retries: int = 0,
    ) -> DetectorResult:
        client = self.clients.get(detector)
        if not client:
            return DetectorResult(
                detector=detector,
                status=DetectorStatus.UNAVAILABLE,
                error="not_registered",
                processing_time_ms=0,
            )
        # Circuit breaker gate
        br = self.breakers.get(detector) if self.breakers else None  # type: ignore[union-attr]
        if br and not br.allow_request():
            if self.metrics:
                self.metrics.record_circuit_breaker(detector, br.state.value)
            return DetectorResult(
                detector=detector,
                status=DetectorStatus.UNAVAILABLE,
                error="circuit_open",
                processing_time_ms=0,
            )
        attempt_count = max(1, int(retries) + 1)
        backoff_base = 0.05
        last_result: Optional[DetectorResult] = None
        for i in range(attempt_count):
            try:
                res = await asyncio.wait_for(
                    client.analyze(content, metadata), timeout=timeout_ms / 1000
                )
                # Success
                if br:
                    br.record_success()
                    if self.metrics:
                        self.metrics.record_circuit_breaker(detector, br.state.value)
                return res
            except asyncio.TimeoutError:
                if br:
                    br.record_failure()
                    if self.metrics:
                        self.metrics.record_circuit_breaker(detector, br.state.value)
                last_result = DetectorResult(
                    detector=detector,
                    status=DetectorStatus.TIMEOUT,
                    error="timeout",
                    processing_time_ms=timeout_ms,
                )
            except Exception as e:  # noqa: BLE001
                if br:
                    br.record_failure()
                    if self.metrics:
                        self.metrics.record_circuit_breaker(detector, br.state.value)
                last_result = DetectorResult(
                    detector=detector,
                    status=DetectorStatus.FAILED,
                    error=str(e),
                    processing_time_ms=timeout_ms,
                )
            # Decide retry
            # Retry decision based on config toggles
            can_retry = False
            if last_result:
                if last_result.status == DetectorStatus.TIMEOUT and hasattr(
                    self, "_retry_on_timeouts"
                ):
                    can_retry = bool(getattr(self, "_retry_on_timeouts"))
                elif last_result.status == DetectorStatus.FAILED and hasattr(
                    self, "_retry_on_failures"
                ):
                    can_retry = bool(getattr(self, "_retry_on_failures"))
                else:
                    can_retry = last_result.status in (
                        DetectorStatus.TIMEOUT,
                        DetectorStatus.FAILED,
                    )
            if i < attempt_count - 1 and last_result and can_retry:
                await asyncio.sleep(backoff_base * (2**i))
                continue
            break
        # Exhausted attempts
        return last_result or DetectorResult(
            detector=detector,
            status=DetectorStatus.FAILED,
            error="unknown",
            processing_time_ms=timeout_ms,
        )

    async def _execute_secondary_if_any(
        self, content: str, plan: RoutingPlan, metadata: dict
    ) -> List[DetectorResult]:
        if not plan.secondary_detectors:
            return []
        return await self.execute_detector_group(
            plan.secondary_detectors, content, plan, metadata
        )

    async def execute_routing_plan(
        self,
        content: str,
        routing_plan: RoutingPlan,
        _request_id: str,
        metadata: Optional[dict] = None,
    ) -> List[DetectorResult]:
        results: List[DetectorResult] = []
        groups = routing_plan.parallel_groups or [routing_plan.primary_detectors]
        for group in groups:
            group_results = await self.execute_detector_group(
                group, content, routing_plan, metadata or {}
            )
            results.extend(group_results)
        # Fallback routing: execute secondary detectors if any primary failed
        if routing_plan.secondary_detectors:
            any_failure = any(r.status != DetectorStatus.SUCCESS for r in results)
            if any_failure:
                sec_results = await self._execute_secondary_if_any(
                    content, routing_plan, metadata or {}
                )
                results.extend(sec_results)
        return results

"""
Canary deployment system for the mapper service.

This module provides canary deployment capabilities with traffic splitting,
health validation, and automatic rollback functionality.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field

# Import shared components
from shared.interfaces.base import BaseResponse
from shared.interfaces.common import HealthStatus, JobStatus
from shared.utils.logging import get_logger
from shared.utils.correlation import get_correlation_id
from shared.utils.circuit_breaker import CircuitBreaker
from shared.exceptions.base import BaseServiceException

logger = get_logger(__name__)


class CanaryStatus(Enum):
    """Canary deployment status."""

    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    PROMOTING = "promoting"
    COMPLETED = "completed"
    ROLLING_BACK = "rolling_back"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HealthCheckResult(BaseResponse):
    """Result of a health check."""

    healthy: bool
    latency_ms: float
    error_rate: float
    success_rate: float
    details: Dict[str, Any] = {}


@dataclass
class CanaryConfig:
    """Configuration for canary deployment."""

    deployment_id: str
    service_name: str
    canary_version: str
    stable_version: str

    # Traffic configuration
    initial_traffic_percent: float = 5.0
    max_traffic_percent: float = 100.0
    traffic_increment: float = 25.0
    promotion_interval_minutes: int = 10

    # Health thresholds
    max_error_rate: float = 0.05  # 5%
    max_latency_ms: float = 1000.0
    min_success_rate: float = 0.95  # 95%

    # Rollback configuration
    auto_rollback: bool = True
    rollback_threshold_failures: int = 3
    health_check_interval_seconds: int = 30

    # Validation configuration
    validation_duration_minutes: int = 5
    required_request_count: int = 100

    # Callbacks
    health_check_callback: Optional[Callable] = None
    promotion_callback: Optional[Callable] = None
    rollback_callback: Optional[Callable] = None


@dataclass
class CanaryDeployment:
    """Represents a canary deployment."""

    config: CanaryConfig
    status: CanaryStatus = CanaryStatus.PENDING
    current_traffic_percent: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Metrics tracking
    health_checks: List[HealthCheckResult] = field(default_factory=list)
    failure_count: int = 0
    promotion_count: int = 0

    # Error tracking
    last_error: Optional[str] = None
    rollback_reason: Optional[str] = None


class TrafficSplitter:
    """Manages traffic splitting between stable and canary versions."""

    def __init__(self):
        self._traffic_rules: Dict[str, Dict[str, float]] = {}
        self._request_counters: Dict[str, int] = {}

    def set_traffic_split(
        self, deployment_id: str, canary_percent: float, stable_percent: float
    ):
        """Set traffic split percentages."""
        if abs(canary_percent + stable_percent - 100.0) > 0.01:
            raise ValueError("Traffic percentages must sum to 100")

        self._traffic_rules[deployment_id] = {
            "canary": canary_percent,
            "stable": stable_percent,
        }

        logger.info(
            f"Traffic split updated for {deployment_id}: "
            f"canary={canary_percent}%, stable={stable_percent}%"
        )

    def route_request(self, deployment_id: str, request_id: str) -> str:
        """Route request to canary or stable version."""
        if deployment_id not in self._traffic_rules:
            return "stable"  # Default to stable if no rules

        # Simple hash-based routing for consistent user experience
        import hashlib

        hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        routing_percent = (hash_value % 100) + 1

        canary_percent = self._traffic_rules[deployment_id]["canary"]

        if routing_percent <= canary_percent:
            self._increment_counter(f"{deployment_id}_canary")
            return "canary"
        else:
            self._increment_counter(f"{deployment_id}_stable")
            return "stable"

    def get_traffic_stats(self, deployment_id: str) -> Dict[str, int]:
        """Get traffic statistics for a deployment."""
        canary_count = self._request_counters.get(f"{deployment_id}_canary", 0)
        stable_count = self._request_counters.get(f"{deployment_id}_stable", 0)

        return {
            "canary_requests": canary_count,
            "stable_requests": stable_count,
            "total_requests": canary_count + stable_count,
        }

    def remove_traffic_rules(self, deployment_id: str):
        """Remove traffic rules for a deployment."""
        self._traffic_rules.pop(deployment_id, None)
        self._request_counters.pop(f"{deployment_id}_canary", None)
        self._request_counters.pop(f"{deployment_id}_stable", None)

    def _increment_counter(self, key: str):
        """Increment request counter."""
        self._request_counters[key] = self._request_counters.get(key, 0) + 1


class HealthValidator:
    """Validates health of canary deployments."""

    def __init__(self):
        self._metrics_collectors: Dict[str, Callable] = {}

    def register_metrics_collector(self, deployment_id: str, collector: Callable):
        """Register a metrics collector for a deployment."""
        self._metrics_collectors[deployment_id] = collector

    async def check_health(self, deployment: CanaryDeployment) -> HealthCheckResult:
        """Perform health check on canary deployment."""
        start_time = time.time()

        try:
            # Get metrics from collector
            collector = self._metrics_collectors.get(deployment.config.deployment_id)
            if not collector:
                # Use default health check
                metrics = await self._default_health_check(deployment)
            else:
                metrics = await collector(deployment)

            latency_ms = (time.time() - start_time) * 1000

            # Evaluate health based on thresholds
            error_rate = metrics.get("error_rate", 0.0)
            success_rate = metrics.get("success_rate", 1.0)
            response_latency = metrics.get("avg_latency_ms", latency_ms)

            healthy = (
                error_rate <= deployment.config.max_error_rate
                and success_rate >= deployment.config.min_success_rate
                and response_latency <= deployment.config.max_latency_ms
            )

            result = HealthCheckResult(
                healthy=healthy,
                latency_ms=response_latency,
                error_rate=error_rate,
                success_rate=success_rate,
                timestamp=datetime.utcnow(),
                details=metrics,
            )

            logger.debug(
                f"Health check for {deployment.config.deployment_id}: "
                f"healthy={healthy}, error_rate={error_rate:.3f}, "
                f"success_rate={success_rate:.3f}, latency={response_latency:.1f}ms"
            )

            return result

        except Exception as e:
            logger.error(
                f"Health check failed for {deployment.config.deployment_id}: {e}"
            )
            return HealthCheckResult(
                healthy=False,
                latency_ms=(time.time() - start_time) * 1000,
                error_rate=1.0,
                success_rate=0.0,
                timestamp=datetime.utcnow(),
                details={"error": str(e)},
            )

    async def _default_health_check(
        self, deployment: CanaryDeployment
    ) -> Dict[str, Any]:
        """Default health check implementation."""
        # In a real implementation, this would check actual service metrics
        # For now, return mock healthy metrics
        return {
            "error_rate": 0.01,
            "success_rate": 0.99,
            "avg_latency_ms": 150.0,
            "request_count": 1000,
            "active_connections": 50,
        }


class CanaryController:
    """Main controller for canary deployments."""

    def __init__(self):
        self.traffic_splitter = TrafficSplitter()
        self.health_validator = HealthValidator()
        self._active_deployments: Dict[str, CanaryDeployment] = {}
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}

    async def start_canary_deployment(self, config: CanaryConfig) -> CanaryDeployment:
        """Start a new canary deployment."""
        try:
            # Validate configuration
            self._validate_config(config)

            # Create deployment
            deployment = CanaryDeployment(
                config=config,
                status=CanaryStatus.STARTING,
                start_time=datetime.utcnow(),
            )

            # Store deployment
            self._active_deployments[config.deployment_id] = deployment

            # Set initial traffic split
            self.traffic_splitter.set_traffic_split(
                config.deployment_id,
                config.initial_traffic_percent,
                100.0 - config.initial_traffic_percent,
            )

            # Start monitoring
            monitoring_task = asyncio.create_task(self._monitor_deployment(deployment))
            self._monitoring_tasks[config.deployment_id] = monitoring_task

            deployment.status = CanaryStatus.RUNNING
            deployment.current_traffic_percent = config.initial_traffic_percent

            logger.info(f"Started canary deployment: {config.deployment_id}")
            return deployment

        except Exception as e:
            logger.error(
                f"Failed to start canary deployment {config.deployment_id}: {e}"
            )
            raise

    async def promote_canary(self, deployment_id: str) -> bool:
        """Promote canary to full traffic."""
        deployment = self._active_deployments.get(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")

        try:
            deployment.status = CanaryStatus.PROMOTING

            # Gradually increase traffic to 100%
            while (
                deployment.current_traffic_percent
                < deployment.config.max_traffic_percent
            ):
                new_percent = min(
                    deployment.current_traffic_percent
                    + deployment.config.traffic_increment,
                    deployment.config.max_traffic_percent,
                )

                self.traffic_splitter.set_traffic_split(
                    deployment_id, new_percent, 100.0 - new_percent
                )

                deployment.current_traffic_percent = new_percent
                deployment.promotion_count += 1

                # Wait for validation period
                await asyncio.sleep(deployment.config.promotion_interval_minutes * 60)

                # Check health before next promotion
                health_result = await self.health_validator.check_health(deployment)
                deployment.health_checks.append(health_result)

                if not health_result.healthy:
                    logger.warning(
                        f"Health check failed during promotion of {deployment_id}"
                    )
                    await self._rollback_deployment(
                        deployment, "Health check failed during promotion"
                    )
                    return False

            # Complete deployment
            deployment.status = CanaryStatus.COMPLETED
            deployment.end_time = datetime.utcnow()

            # Execute promotion callback
            if deployment.config.promotion_callback:
                await deployment.config.promotion_callback(deployment)

            logger.info(f"Successfully promoted canary deployment: {deployment_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to promote canary {deployment_id}: {e}")
            await self._rollback_deployment(deployment, f"Promotion failed: {e}")
            return False

    async def rollback_canary(
        self, deployment_id: str, reason: str = "Manual rollback"
    ) -> bool:
        """Rollback canary deployment."""
        deployment = self._active_deployments.get(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")

        return await self._rollback_deployment(deployment, reason)

    async def cancel_canary(self, deployment_id: str) -> bool:
        """Cancel canary deployment."""
        deployment = self._active_deployments.get(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")

        try:
            deployment.status = CanaryStatus.CANCELLED
            deployment.end_time = datetime.utcnow()

            # Stop monitoring
            await self._stop_monitoring(deployment_id)

            # Remove traffic rules
            self.traffic_splitter.remove_traffic_rules(deployment_id)

            logger.info(f"Cancelled canary deployment: {deployment_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel canary {deployment_id}: {e}")
            return False

    def get_deployment_status(self, deployment_id: str) -> Optional[CanaryDeployment]:
        """Get status of a canary deployment."""
        return self._active_deployments.get(deployment_id)

    def list_active_deployments(self) -> List[CanaryDeployment]:
        """List all active canary deployments."""
        return list(self._active_deployments.values())

    async def _monitor_deployment(self, deployment: CanaryDeployment):
        """Monitor canary deployment health."""
        try:
            while deployment.status in [CanaryStatus.RUNNING, CanaryStatus.PROMOTING]:
                # Perform health check
                health_result = await self.health_validator.check_health(deployment)
                deployment.health_checks.append(health_result)

                # Keep only recent health checks (last 100)
                if len(deployment.health_checks) > 100:
                    deployment.health_checks = deployment.health_checks[-100:]

                if not health_result.healthy:
                    deployment.failure_count += 1
                    logger.warning(
                        f"Health check failed for {deployment.config.deployment_id} "
                        f"(failure {deployment.failure_count})"
                    )

                    # Check if we should rollback
                    if (
                        deployment.config.auto_rollback
                        and deployment.failure_count
                        >= deployment.config.rollback_threshold_failures
                    ):
                        await self._rollback_deployment(
                            deployment,
                            f"Auto-rollback triggered after {deployment.failure_count} failures",
                        )
                        break
                else:
                    # Reset failure count on successful health check
                    deployment.failure_count = 0

                # Execute health check callback
                if deployment.config.health_check_callback:
                    await deployment.config.health_check_callback(
                        deployment, health_result
                    )

                # Wait for next check
                await asyncio.sleep(deployment.config.health_check_interval_seconds)

        except asyncio.CancelledError:
            logger.info(
                f"Monitoring cancelled for deployment {deployment.config.deployment_id}"
            )
        except Exception as e:
            logger.error(
                f"Error monitoring deployment {deployment.config.deployment_id}: {e}"
            )
            await self._rollback_deployment(deployment, f"Monitoring error: {e}")

    async def _rollback_deployment(
        self, deployment: CanaryDeployment, reason: str
    ) -> bool:
        """Rollback a canary deployment."""
        try:
            deployment.status = CanaryStatus.ROLLING_BACK
            deployment.rollback_reason = reason
            deployment.last_error = reason

            # Route all traffic back to stable
            self.traffic_splitter.set_traffic_split(
                deployment.config.deployment_id,
                0.0,  # No canary traffic
                100.0,  # All stable traffic
            )

            deployment.current_traffic_percent = 0.0

            # Execute rollback callback
            if deployment.config.rollback_callback:
                await deployment.config.rollback_callback(deployment)

            # Stop monitoring
            await self._stop_monitoring(deployment.config.deployment_id)

            deployment.status = CanaryStatus.FAILED
            deployment.end_time = datetime.utcnow()

            logger.info(
                f"Rolled back canary deployment {deployment.config.deployment_id}: {reason}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to rollback deployment {deployment.config.deployment_id}: {e}"
            )
            deployment.status = CanaryStatus.FAILED
            return False

    async def _stop_monitoring(self, deployment_id: str):
        """Stop monitoring task for a deployment."""
        task = self._monitoring_tasks.get(deployment_id)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._monitoring_tasks.pop(deployment_id, None)

    def _validate_config(self, config: CanaryConfig):
        """Validate canary configuration."""
        if config.initial_traffic_percent < 0 or config.initial_traffic_percent > 100:
            raise ValueError("Initial traffic percent must be between 0 and 100")

        if config.max_traffic_percent < config.initial_traffic_percent:
            raise ValueError("Max traffic percent must be >= initial traffic percent")

        if config.traffic_increment <= 0:
            raise ValueError("Traffic increment must be positive")

        if config.max_error_rate < 0 or config.max_error_rate > 1:
            raise ValueError("Max error rate must be between 0 and 1")

        if config.min_success_rate < 0 or config.min_success_rate > 1:
            raise ValueError("Min success rate must be between 0 and 1")


# Global canary controller instance
_canary_controller: Optional[CanaryController] = None


def get_canary_controller() -> CanaryController:
    """Get the global canary controller instance."""
    global _canary_controller
    if _canary_controller is None:
        _canary_controller = CanaryController()
    return _canary_controller

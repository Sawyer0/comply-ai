"""
Blue-green deployment system for the mapper service.

This module provides blue-green deployment capabilities with environment
switching, validation, and rollback functionality.
"""

import asyncio
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


class EnvironmentColor(Enum):
    """Environment colors for blue-green deployment."""

    BLUE = "blue"
    GREEN = "green"


class DeploymentStatus(Enum):
    """Blue-green deployment status."""

    PENDING = "pending"
    PREPARING = "preparing"
    VALIDATING = "validating"
    SWITCHING = "switching"
    COMPLETED = "completed"
    ROLLING_BACK = "rolling_back"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ValidationResult(BaseResponse):
    """Result of environment validation."""

    passed: bool
    test_name: str
    duration_ms: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = {}


@dataclass
class BlueGreenConfig:
    """Configuration for blue-green deployment."""

    deployment_id: str
    service_name: str
    new_version: str
    current_version: str

    # Environment configuration
    blue_environment: Dict[str, Any] = field(default_factory=dict)
    green_environment: Dict[str, Any] = field(default_factory=dict)

    # Validation configuration
    validation_tests: List[str] = field(default_factory=list)
    validation_timeout_minutes: int = 30
    required_success_rate: float = 1.0  # 100% by default

    # Traffic switching
    dns_switch_timeout_seconds: int = 300
    health_check_retries: int = 5
    health_check_interval_seconds: int = 10

    # Rollback configuration
    auto_rollback_on_failure: bool = True
    rollback_timeout_minutes: int = 15

    # Callbacks
    pre_switch_callback: Optional[Callable] = None
    post_switch_callback: Optional[Callable] = None
    rollback_callback: Optional[Callable] = None


@dataclass
class BlueGreenDeployment:
    """Represents a blue-green deployment."""

    config: BlueGreenConfig
    status: DeploymentStatus = DeploymentStatus.PENDING
    active_environment: EnvironmentColor = EnvironmentColor.BLUE
    target_environment: EnvironmentColor = EnvironmentColor.GREEN

    # Timing
    start_time: Optional[datetime] = None
    switch_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Validation results
    validation_results: List[ValidationResult] = field(default_factory=list)
    validation_success_rate: float = 0.0

    # Error tracking
    last_error: Optional[str] = None
    rollback_reason: Optional[str] = None


class EnvironmentManager:
    """Manages blue and green environments."""

    def __init__(self):
        self._environments: Dict[str, Dict[EnvironmentColor, Dict[str, Any]]] = {}
        self._active_environments: Dict[str, EnvironmentColor] = {}

    def register_environments(
        self,
        deployment_id: str,
        blue_config: Dict[str, Any],
        green_config: Dict[str, Any],
    ):
        """Register blue and green environments for a deployment."""
        self._environments[deployment_id] = {
            EnvironmentColor.BLUE: blue_config,
            EnvironmentColor.GREEN: green_config,
        }

        # Default to blue as active
        self._active_environments[deployment_id] = EnvironmentColor.BLUE

        logger.info(f"Registered environments for deployment {deployment_id}")

    def get_environment_config(
        self, deployment_id: str, color: EnvironmentColor
    ) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific environment."""
        environments = self._environments.get(deployment_id, {})
        return environments.get(color)

    def get_active_environment(self, deployment_id: str) -> Optional[EnvironmentColor]:
        """Get the currently active environment."""
        return self._active_environments.get(deployment_id)

    def switch_active_environment(self, deployment_id: str) -> EnvironmentColor:
        """Switch the active environment."""
        current = self._active_environments.get(deployment_id, EnvironmentColor.BLUE)
        new_active = (
            EnvironmentColor.GREEN
            if current == EnvironmentColor.BLUE
            else EnvironmentColor.BLUE
        )

        self._active_environments[deployment_id] = new_active

        logger.info(
            f"Switched active environment for {deployment_id}: {current} -> {new_active}"
        )
        return new_active

    def cleanup_environments(self, deployment_id: str):
        """Cleanup environments for a deployment."""
        self._environments.pop(deployment_id, None)
        self._active_environments.pop(deployment_id, None)


class ValidationSuite:
    """Validation suite for blue-green deployments."""

    def __init__(self):
        self._validators: Dict[str, Callable] = {}

    def register_validator(self, test_name: str, validator: Callable):
        """Register a validation test."""
        self._validators[test_name] = validator
        logger.debug(f"Registered validator: {test_name}")

    async def run_validation_suite(
        self, deployment: BlueGreenDeployment
    ) -> List[ValidationResult]:
        """Run all validation tests for a deployment."""
        results = []

        for test_name in deployment.config.validation_tests:
            validator = self._validators.get(test_name)
            if not validator:
                result = ValidationResult(
                    passed=False,
                    test_name=test_name,
                    duration_ms=0.0,
                    error_message=f"Validator {test_name} not found",
                )
                results.append(result)
                continue

            result = await self._run_single_validation(test_name, validator, deployment)
            results.append(result)

        return results

    async def _run_single_validation(
        self, test_name: str, validator: Callable, deployment: BlueGreenDeployment
    ) -> ValidationResult:
        """Run a single validation test."""
        start_time = asyncio.get_event_loop().time()

        try:
            # Run validation with timeout
            timeout = deployment.config.validation_timeout_minutes * 60
            result = await asyncio.wait_for(validator(deployment), timeout=timeout)

            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000

            if isinstance(result, bool):
                return ValidationResult(
                    passed=result, test_name=test_name, duration_ms=duration_ms
                )
            elif isinstance(result, dict):
                return ValidationResult(
                    passed=result.get("passed", False),
                    test_name=test_name,
                    duration_ms=duration_ms,
                    error_message=result.get("error"),
                    details=result.get("details", {}),
                )
            else:
                return ValidationResult(
                    passed=False,
                    test_name=test_name,
                    duration_ms=duration_ms,
                    error_message=f"Invalid validation result type: {type(result)}",
                )

        except asyncio.TimeoutError:
            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            return ValidationResult(
                passed=False,
                test_name=test_name,
                duration_ms=duration_ms,
                error_message=f"Validation timed out after {deployment.config.validation_timeout_minutes} minutes",
            )
        except Exception as e:
            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            return ValidationResult(
                passed=False,
                test_name=test_name,
                duration_ms=duration_ms,
                error_message=str(e),
            )


class DNSManager:
    """Manages DNS switching for blue-green deployments."""

    def __init__(self):
        self._dns_records: Dict[str, Dict[str, str]] = {}

    async def switch_dns(
        self, deployment_id: str, target_environment: EnvironmentColor
    ) -> bool:
        """Switch DNS to target environment."""
        try:
            # In a real implementation, this would update actual DNS records
            # For now, we'll simulate the DNS switch

            logger.info(
                f"Switching DNS for {deployment_id} to {target_environment.value}"
            )

            # Simulate DNS propagation delay
            await asyncio.sleep(2)

            # Update internal DNS mapping
            if deployment_id not in self._dns_records:
                self._dns_records[deployment_id] = {}

            self._dns_records[deployment_id]["active"] = target_environment.value

            logger.info(f"DNS switch completed for {deployment_id}")
            return True

        except Exception as e:
            logger.error(f"DNS switch failed for {deployment_id}: {e}")
            return False

    async def verify_dns_propagation(
        self,
        deployment_id: str,
        expected_environment: EnvironmentColor,
        max_retries: int = 10,
    ) -> bool:
        """Verify DNS propagation is complete."""
        for attempt in range(max_retries):
            try:
                # In a real implementation, this would check actual DNS resolution
                current_env = self._dns_records.get(deployment_id, {}).get("active")

                if current_env == expected_environment.value:
                    logger.info(f"DNS propagation verified for {deployment_id}")
                    return True

                logger.debug(
                    f"DNS propagation check {attempt + 1}/{max_retries} for {deployment_id}"
                )
                await asyncio.sleep(5)

            except Exception as e:
                logger.warning(f"DNS verification attempt {attempt + 1} failed: {e}")

        logger.error(f"DNS propagation verification failed for {deployment_id}")
        return False


class BlueGreenController:
    """Main controller for blue-green deployments."""

    def __init__(self):
        self.environment_manager = EnvironmentManager()
        self.validation_suite = ValidationSuite()
        self.dns_manager = DNSManager()
        self._active_deployments: Dict[str, BlueGreenDeployment] = {}

        # Register default validators
        self._register_default_validators()

    async def start_blue_green_deployment(
        self, config: BlueGreenConfig
    ) -> BlueGreenDeployment:
        """Start a new blue-green deployment."""
        try:
            # Validate configuration
            self._validate_config(config)

            # Determine target environment
            current_active = self.environment_manager.get_active_environment(
                config.deployment_id
            )
            if current_active is None:
                # First deployment, use green as target
                target = EnvironmentColor.GREEN
                active = EnvironmentColor.BLUE
            else:
                # Switch to the other environment
                target = (
                    EnvironmentColor.GREEN
                    if current_active == EnvironmentColor.BLUE
                    else EnvironmentColor.BLUE
                )
                active = current_active

            # Create deployment
            deployment = BlueGreenDeployment(
                config=config,
                status=DeploymentStatus.PREPARING,
                active_environment=active,
                target_environment=target,
                start_time=datetime.utcnow(),
            )

            # Register environments
            self.environment_manager.register_environments(
                config.deployment_id, config.blue_environment, config.green_environment
            )

            # Store deployment
            self._active_deployments[config.deployment_id] = deployment

            logger.info(
                f"Started blue-green deployment: {config.deployment_id} "
                f"(active: {active.value}, target: {target.value})"
            )

            # Start deployment process
            asyncio.create_task(self._execute_deployment(deployment))

            return deployment

        except Exception as e:
            logger.error(
                f"Failed to start blue-green deployment {config.deployment_id}: {e}"
            )
            raise

    async def rollback_deployment(
        self, deployment_id: str, reason: str = "Manual rollback"
    ) -> bool:
        """Rollback a blue-green deployment."""
        deployment = self._active_deployments.get(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")

        return await self._rollback_deployment(deployment, reason)

    async def cancel_deployment(self, deployment_id: str) -> bool:
        """Cancel a blue-green deployment."""
        deployment = self._active_deployments.get(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")

        try:
            deployment.status = DeploymentStatus.CANCELLED
            deployment.end_time = datetime.utcnow()

            logger.info(f"Cancelled blue-green deployment: {deployment_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel deployment {deployment_id}: {e}")
            return False

    def get_deployment_status(
        self, deployment_id: str
    ) -> Optional[BlueGreenDeployment]:
        """Get status of a blue-green deployment."""
        return self._active_deployments.get(deployment_id)

    def list_active_deployments(self) -> List[BlueGreenDeployment]:
        """List all active blue-green deployments."""
        return list(self._active_deployments.values())

    async def _execute_deployment(self, deployment: BlueGreenDeployment):
        """Execute the blue-green deployment process."""
        try:
            # Phase 1: Validation
            deployment.status = DeploymentStatus.VALIDATING
            logger.info(
                f"Starting validation for deployment {deployment.config.deployment_id}"
            )

            validation_results = await self.validation_suite.run_validation_suite(
                deployment
            )
            deployment.validation_results = validation_results

            # Calculate success rate
            passed_tests = sum(1 for result in validation_results if result.passed)
            total_tests = len(validation_results)
            deployment.validation_success_rate = (
                passed_tests / total_tests if total_tests > 0 else 0.0
            )

            # Check if validation passed
            if (
                deployment.validation_success_rate
                < deployment.config.required_success_rate
            ):
                await self._rollback_deployment(
                    deployment,
                    f"Validation failed: {deployment.validation_success_rate:.2%} < {deployment.config.required_success_rate:.2%}",
                )
                return

            # Phase 2: Traffic switching
            deployment.status = DeploymentStatus.SWITCHING
            logger.info(
                f"Starting traffic switch for deployment {deployment.config.deployment_id}"
            )

            # Execute pre-switch callback
            if deployment.config.pre_switch_callback:
                await deployment.config.pre_switch_callback(deployment)

            # Switch DNS
            dns_success = await self.dns_manager.switch_dns(
                deployment.config.deployment_id, deployment.target_environment
            )

            if not dns_success:
                await self._rollback_deployment(deployment, "DNS switch failed")
                return

            # Verify DNS propagation
            dns_verified = await self.dns_manager.verify_dns_propagation(
                deployment.config.deployment_id, deployment.target_environment
            )

            if not dns_verified:
                await self._rollback_deployment(
                    deployment, "DNS propagation verification failed"
                )
                return

            # Update active environment
            self.environment_manager.switch_active_environment(
                deployment.config.deployment_id
            )
            deployment.switch_time = datetime.utcnow()

            # Execute post-switch callback
            if deployment.config.post_switch_callback:
                await deployment.config.post_switch_callback(deployment)

            # Phase 3: Completion
            deployment.status = DeploymentStatus.COMPLETED
            deployment.end_time = datetime.utcnow()

            logger.info(
                f"Blue-green deployment completed successfully: {deployment.config.deployment_id}"
            )

        except Exception as e:
            logger.error(
                f"Error during blue-green deployment {deployment.config.deployment_id}: {e}"
            )
            await self._rollback_deployment(deployment, f"Deployment error: {e}")

    async def _rollback_deployment(
        self, deployment: BlueGreenDeployment, reason: str
    ) -> bool:
        """Rollback a blue-green deployment."""
        try:
            deployment.status = DeploymentStatus.ROLLING_BACK
            deployment.rollback_reason = reason
            deployment.last_error = reason

            logger.info(
                f"Rolling back deployment {deployment.config.deployment_id}: {reason}"
            )

            # Switch DNS back to original environment
            if deployment.switch_time:  # Only if we actually switched
                dns_success = await self.dns_manager.switch_dns(
                    deployment.config.deployment_id, deployment.active_environment
                )

                if dns_success:
                    # Verify rollback DNS propagation
                    await self.dns_manager.verify_dns_propagation(
                        deployment.config.deployment_id, deployment.active_environment
                    )

            # Execute rollback callback
            if deployment.config.rollback_callback:
                await deployment.config.rollback_callback(deployment)

            deployment.status = DeploymentStatus.FAILED
            deployment.end_time = datetime.utcnow()

            logger.info(
                f"Rollback completed for deployment {deployment.config.deployment_id}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to rollback deployment {deployment.config.deployment_id}: {e}"
            )
            deployment.status = DeploymentStatus.FAILED
            return False

    def _validate_config(self, config: BlueGreenConfig):
        """Validate blue-green configuration."""
        if not config.deployment_id or not config.service_name:
            raise ValueError("Deployment ID and service name are required")

        if config.required_success_rate < 0 or config.required_success_rate > 1:
            raise ValueError("Required success rate must be between 0 and 1")

        if config.validation_timeout_minutes <= 0:
            raise ValueError("Validation timeout must be positive")

    def _register_default_validators(self):
        """Register default validation tests."""

        async def health_check_validator(
            deployment: BlueGreenDeployment,
        ) -> Dict[str, Any]:
            """Default health check validator."""
            try:
                # Simulate health check
                await asyncio.sleep(1)

                # In a real implementation, this would check actual service health
                return {
                    "passed": True,
                    "details": {
                        "response_time_ms": 150,
                        "status_code": 200,
                        "healthy_instances": 3,
                    },
                }
            except Exception as e:
                return {"passed": False, "error": str(e)}

        async def smoke_test_validator(
            deployment: BlueGreenDeployment,
        ) -> Dict[str, Any]:
            """Default smoke test validator."""
            try:
                # Simulate smoke tests
                await asyncio.sleep(2)

                return {
                    "passed": True,
                    "details": {"tests_run": 10, "tests_passed": 10, "coverage": 0.95},
                }
            except Exception as e:
                return {"passed": False, "error": str(e)}

        self.validation_suite.register_validator("health_check", health_check_validator)
        self.validation_suite.register_validator("smoke_test", smoke_test_validator)


# Global blue-green controller instance
_blue_green_controller: Optional[BlueGreenController] = None


def get_blue_green_controller() -> BlueGreenController:
    """Get the global blue-green controller instance."""
    global _blue_green_controller
    if _blue_green_controller is None:
        _blue_green_controller = BlueGreenController()
    return _blue_green_controller

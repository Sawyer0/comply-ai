"""
Deployment pipelines for the mapper service.

This module provides deployment pipeline capabilities including
multi-stage deployment, validation gates, and automated rollback.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel
import json

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Pipeline execution status."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLING_BACK = "rolling_back"


class StageStatus(Enum):
    """Stage execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class GateType(Enum):
    """Types of validation gates."""

    MANUAL_APPROVAL = "manual_approval"
    AUTOMATED_TEST = "automated_test"
    HEALTH_CHECK = "health_check"
    PERFORMANCE_TEST = "performance_test"
    SECURITY_SCAN = "security_scan"
    COMPLIANCE_CHECK = "compliance_check"


@dataclass
class ValidationGate:
    """Validation gate configuration."""

    gate_id: str
    name: str
    gate_type: GateType
    required: bool = True
    timeout_minutes: int = 30

    # Gate-specific configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Validation function
    validator: Optional[Callable] = None

    # Approval configuration (for manual gates)
    required_approvers: List[str] = field(default_factory=list)
    approval_timeout_hours: int = 24


@dataclass
class PipelineStage:
    """Pipeline stage configuration."""

    stage_id: str
    name: str
    description: str

    # Stage dependencies
    depends_on: List[str] = field(default_factory=list)

    # Validation gates
    pre_gates: List[ValidationGate] = field(default_factory=list)
    post_gates: List[ValidationGate] = field(default_factory=list)

    # Stage execution
    executor: Optional[Callable] = None
    config: Dict[str, Any] = field(default_factory=dict)

    # Timing
    timeout_minutes: int = 60
    retry_count: int = 0
    retry_delay_minutes: int = 5

    # Rollback configuration
    rollback_executor: Optional[Callable] = None
    rollback_on_failure: bool = True


@dataclass
class PipelineConfig:
    """Deployment pipeline configuration."""

    pipeline_id: str
    name: str
    description: str
    version: str

    # Pipeline stages
    stages: List[PipelineStage] = field(default_factory=list)

    # Global configuration
    global_timeout_minutes: int = 480  # 8 hours
    auto_rollback_on_failure: bool = True
    parallel_execution: bool = False

    # Notification configuration
    notification_channels: List[str] = field(default_factory=list)
    notify_on_success: bool = True
    notify_on_failure: bool = True

    # Environment configuration
    target_environments: List[str] = field(default_factory=list)

    # Callbacks
    on_start_callback: Optional[Callable] = None
    on_complete_callback: Optional[Callable] = None
    on_failure_callback: Optional[Callable] = None


class StageExecution(BaseModel):
    """Stage execution tracking."""

    stage_id: str
    status: StageStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_minutes: float = 0.0

    # Execution results
    output: Dict[str, Any] = {}
    error_message: Optional[str] = None

    # Gate results
    gate_results: Dict[str, Any] = {}

    # Retry tracking
    attempt_count: int = 0
    max_attempts: int = 1


class PipelineExecution(BaseModel):
    """Pipeline execution tracking."""

    execution_id: str
    pipeline_id: str
    status: PipelineStatus

    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_minutes: float = 0.0

    # Stage executions
    stage_executions: Dict[str, StageExecution] = {}

    # Execution context
    context: Dict[str, Any] = {}
    triggered_by: str = "system"

    # Results
    success_rate: float = 0.0
    failed_stages: List[str] = []

    # Rollback tracking
    rollback_executed: bool = False
    rollback_reason: Optional[str] = None


class GateValidator:
    """Validates pipeline gates."""

    def __init__(self):
        self._validators: Dict[GateType, Callable] = {}
        self._manual_approvals: Dict[str, Dict[str, Any]] = {}

        # Register default validators
        self._register_default_validators()

    def register_validator(self, gate_type: GateType, validator: Callable):
        """Register a gate validator."""
        self._validators[gate_type] = validator
        logger.debug(f"Registered gate validator: {gate_type.value}")

    async def validate_gate(
        self, gate: ValidationGate, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate a pipeline gate."""
        start_time = datetime.utcnow()

        try:
            if gate.gate_type == GateType.MANUAL_APPROVAL:
                result = await self._handle_manual_approval(gate, context)
            else:
                validator = gate.validator or self._validators.get(gate.gate_type)
                if not validator:
                    return {
                        "passed": False,
                        "error": f"No validator found for gate type: {gate.gate_type.value}",
                    }

                # Run validation with timeout
                result = await asyncio.wait_for(
                    validator(gate, context), timeout=gate.timeout_minutes * 60
                )

            duration = (datetime.utcnow() - start_time).total_seconds() / 60

            return {
                "passed": result.get("passed", False),
                "duration_minutes": duration,
                "details": result.get("details", {}),
                "error": result.get("error"),
            }

        except asyncio.TimeoutError:
            duration = (datetime.utcnow() - start_time).total_seconds() / 60
            return {
                "passed": False,
                "duration_minutes": duration,
                "error": f"Gate validation timed out after {gate.timeout_minutes} minutes",
            }
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds() / 60
            return {"passed": False, "duration_minutes": duration, "error": str(e)}

    async def _handle_manual_approval(
        self, gate: ValidationGate, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle manual approval gate."""
        approval_id = f"{gate.gate_id}_{datetime.utcnow().timestamp()}"

        # Store approval request
        self._manual_approvals[approval_id] = {
            "gate": gate,
            "context": context,
            "created_at": datetime.utcnow(),
            "approvals": [],
            "status": "pending",
        }

        # Wait for approval or timeout
        timeout_seconds = gate.approval_timeout_hours * 3600
        start_time = datetime.utcnow()

        while (datetime.utcnow() - start_time).total_seconds() < timeout_seconds:
            approval_data = self._manual_approvals.get(approval_id)
            if not approval_data:
                break

            if approval_data["status"] == "approved":
                return {
                    "passed": True,
                    "details": {
                        "approvers": approval_data["approvals"],
                        "approval_time": approval_data.get("approved_at"),
                    },
                }
            elif approval_data["status"] == "rejected":
                return {
                    "passed": False,
                    "error": "Manual approval rejected",
                    "details": {
                        "rejector": approval_data.get("rejected_by"),
                        "rejection_reason": approval_data.get("rejection_reason"),
                    },
                }

            await asyncio.sleep(30)  # Check every 30 seconds

        # Timeout
        self._manual_approvals.pop(approval_id, None)
        return {
            "passed": False,
            "error": f"Manual approval timed out after {gate.approval_timeout_hours} hours",
        }

    def approve_gate(self, approval_id: str, approver: str, comments: str = "") -> bool:
        """Approve a manual gate."""
        approval_data = self._manual_approvals.get(approval_id)
        if not approval_data:
            return False

        gate = approval_data["gate"]

        # Check if approver is authorized
        if gate.required_approvers and approver not in gate.required_approvers:
            return False

        # Add approval
        approval_data["approvals"].append(
            {"approver": approver, "timestamp": datetime.utcnow(), "comments": comments}
        )

        # Check if we have enough approvals
        required_count = len(gate.required_approvers) if gate.required_approvers else 1
        if len(approval_data["approvals"]) >= required_count:
            approval_data["status"] = "approved"
            approval_data["approved_at"] = datetime.utcnow()

        return True

    def reject_gate(self, approval_id: str, rejector: str, reason: str = "") -> bool:
        """Reject a manual gate."""
        approval_data = self._manual_approvals.get(approval_id)
        if not approval_data:
            return False

        approval_data["status"] = "rejected"
        approval_data["rejected_by"] = rejector
        approval_data["rejection_reason"] = reason
        approval_data["rejected_at"] = datetime.utcnow()

        return True

    def _register_default_validators(self):
        """Register default gate validators."""

        async def health_check_validator(
            gate: ValidationGate, context: Dict[str, Any]
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
                        "healthy_instances": 3,
                        "error_rate": 0.01,
                    },
                }
            except Exception as e:
                return {"passed": False, "error": str(e)}

        async def automated_test_validator(
            gate: ValidationGate, context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Default automated test validator."""
            try:
                # Simulate test execution
                await asyncio.sleep(5)

                return {
                    "passed": True,
                    "details": {
                        "tests_run": 50,
                        "tests_passed": 49,
                        "tests_failed": 1,
                        "coverage": 0.95,
                    },
                }
            except Exception as e:
                return {"passed": False, "error": str(e)}

        self._validators[GateType.HEALTH_CHECK] = health_check_validator
        self._validators[GateType.AUTOMATED_TEST] = automated_test_validator


class PipelineExecutor:
    """Executes deployment pipelines."""

    def __init__(self):
        self.gate_validator = GateValidator()
        self._active_executions: Dict[str, PipelineExecution] = {}
        self._stage_executors: Dict[str, Callable] = {}

        # Register default stage executors
        self._register_default_executors()

    def register_stage_executor(self, stage_type: str, executor: Callable):
        """Register a stage executor."""
        self._stage_executors[stage_type] = executor
        logger.debug(f"Registered stage executor: {stage_type}")

    async def execute_pipeline(
        self, config: PipelineConfig, context: Dict[str, Any]
    ) -> PipelineExecution:
        """Execute a deployment pipeline."""
        execution_id = f"{config.pipeline_id}_{datetime.utcnow().timestamp()}"

        execution = PipelineExecution(
            execution_id=execution_id,
            pipeline_id=config.pipeline_id,
            status=PipelineStatus.RUNNING,
            start_time=datetime.utcnow(),
            context=context,
            triggered_by=context.get("triggered_by", "system"),
        )

        # Initialize stage executions
        for stage in config.stages:
            execution.stage_executions[stage.stage_id] = StageExecution(
                stage_id=stage.stage_id,
                status=StageStatus.PENDING,
                max_attempts=stage.retry_count + 1,
            )

        self._active_executions[execution_id] = execution

        try:
            # Execute on_start callback
            if config.on_start_callback:
                await config.on_start_callback(execution)

            # Execute pipeline stages
            if config.parallel_execution:
                await self._execute_stages_parallel(config, execution)
            else:
                await self._execute_stages_sequential(config, execution)

            # Calculate results
            self._calculate_execution_results(execution)

            # Determine final status
            if execution.failed_stages:
                execution.status = PipelineStatus.FAILED

                # Execute rollback if configured
                if config.auto_rollback_on_failure:
                    await self._execute_rollback(config, execution)

                # Execute failure callback
                if config.on_failure_callback:
                    await config.on_failure_callback(execution)
            else:
                execution.status = PipelineStatus.COMPLETED

                # Execute completion callback
                if config.on_complete_callback:
                    await config.on_complete_callback(execution)

            execution.end_time = datetime.utcnow()
            execution.duration_minutes = (
                execution.end_time - execution.start_time
            ).total_seconds() / 60

            logger.info(
                f"Pipeline execution completed: {execution_id} (status: {execution.status.value})"
            )

            return execution

        except Exception as e:
            logger.error(f"Pipeline execution failed: {execution_id}: {e}")
            execution.status = PipelineStatus.FAILED
            execution.end_time = datetime.utcnow()

            if config.on_failure_callback:
                await config.on_failure_callback(execution)

            raise

    async def _execute_stages_sequential(
        self, config: PipelineConfig, execution: PipelineExecution
    ):
        """Execute pipeline stages sequentially."""
        # Build dependency graph
        stage_map = {stage.stage_id: stage for stage in config.stages}
        executed_stages = set()

        while len(executed_stages) < len(config.stages):
            # Find stages ready to execute
            ready_stages = []
            for stage in config.stages:
                if stage.stage_id not in executed_stages and all(
                    dep in executed_stages for dep in stage.depends_on
                ):
                    ready_stages.append(stage)

            if not ready_stages:
                # Check for circular dependencies
                remaining_stages = [
                    s.stage_id
                    for s in config.stages
                    if s.stage_id not in executed_stages
                ]
                raise RuntimeError(
                    f"Circular dependency detected in stages: {remaining_stages}"
                )

            # Execute ready stages
            for stage in ready_stages:
                stage_execution = execution.stage_executions[stage.stage_id]

                try:
                    await self._execute_single_stage(
                        stage, stage_execution, execution.context
                    )
                    executed_stages.add(stage.stage_id)

                except Exception as e:
                    logger.error(f"Stage {stage.stage_id} failed: {e}")
                    stage_execution.status = StageStatus.FAILED
                    stage_execution.error_message = str(e)
                    execution.failed_stages.append(stage.stage_id)

                    # Stop execution on failure (unless configured otherwise)
                    if stage_execution.status == StageStatus.FAILED:
                        return

    async def _execute_stages_parallel(
        self, config: PipelineConfig, execution: PipelineExecution
    ):
        """Execute pipeline stages in parallel where possible."""
        # Create tasks for all stages
        stage_tasks = {}

        for stage in config.stages:
            stage_execution = execution.stage_executions[stage.stage_id]
            task = asyncio.create_task(
                self._execute_single_stage(stage, stage_execution, execution.context)
            )
            stage_tasks[stage.stage_id] = task

        # Wait for all tasks to complete
        results = await asyncio.gather(*stage_tasks.values(), return_exceptions=True)

        # Process results
        for i, (stage_id, result) in enumerate(zip(stage_tasks.keys(), results)):
            if isinstance(result, Exception):
                stage_execution = execution.stage_executions[stage_id]
                stage_execution.status = StageStatus.FAILED
                stage_execution.error_message = str(result)
                execution.failed_stages.append(stage_id)

    async def _execute_single_stage(
        self,
        stage: PipelineStage,
        stage_execution: StageExecution,
        context: Dict[str, Any],
    ):
        """Execute a single pipeline stage."""
        stage_execution.status = StageStatus.RUNNING
        stage_execution.start_time = datetime.utcnow()

        try:
            # Execute pre-gates
            for gate in stage.pre_gates:
                gate_result = await self.gate_validator.validate_gate(gate, context)
                stage_execution.gate_results[f"pre_{gate.gate_id}"] = gate_result

                if not gate_result["passed"] and gate.required:
                    raise RuntimeError(
                        f"Pre-gate {gate.gate_id} failed: {gate_result.get('error')}"
                    )

            # Execute stage with retries
            for attempt in range(stage_execution.max_attempts):
                stage_execution.attempt_count = attempt + 1

                try:
                    # Get stage executor
                    executor = stage.executor or self._stage_executors.get(
                        stage.stage_id
                    )
                    if not executor:
                        raise RuntimeError(
                            f"No executor found for stage: {stage.stage_id}"
                        )

                    # Execute stage
                    result = await asyncio.wait_for(
                        executor(stage, context), timeout=stage.timeout_minutes * 60
                    )

                    stage_execution.output = result or {}
                    break  # Success, exit retry loop

                except Exception as e:
                    if attempt < stage_execution.max_attempts - 1:
                        logger.warning(
                            f"Stage {stage.stage_id} attempt {attempt + 1} failed, retrying: {e}"
                        )
                        await asyncio.sleep(stage.retry_delay_minutes * 60)
                    else:
                        raise  # Final attempt failed

            # Execute post-gates
            for gate in stage.post_gates:
                gate_result = await self.gate_validator.validate_gate(gate, context)
                stage_execution.gate_results[f"post_{gate.gate_id}"] = gate_result

                if not gate_result["passed"] and gate.required:
                    raise RuntimeError(
                        f"Post-gate {gate.gate_id} failed: {gate_result.get('error')}"
                    )

            stage_execution.status = StageStatus.COMPLETED

        except Exception as e:
            stage_execution.status = StageStatus.FAILED
            stage_execution.error_message = str(e)
            raise

        finally:
            stage_execution.end_time = datetime.utcnow()
            if stage_execution.start_time:
                stage_execution.duration_minutes = (
                    stage_execution.end_time - stage_execution.start_time
                ).total_seconds() / 60

    async def _execute_rollback(
        self, config: PipelineConfig, execution: PipelineExecution
    ):
        """Execute rollback for failed pipeline."""
        execution.status = PipelineStatus.ROLLING_BACK
        execution.rollback_executed = True
        execution.rollback_reason = (
            f"Pipeline failed with {len(execution.failed_stages)} failed stages"
        )

        logger.info(
            f"Starting rollback for pipeline execution: {execution.execution_id}"
        )

        # Execute rollback for completed stages in reverse order
        completed_stages = [
            stage_id
            for stage_id, stage_exec in execution.stage_executions.items()
            if stage_exec.status == StageStatus.COMPLETED
        ]

        stage_map = {stage.stage_id: stage for stage in config.stages}

        for stage_id in reversed(completed_stages):
            stage = stage_map.get(stage_id)
            if stage and stage.rollback_executor and stage.rollback_on_failure:
                try:
                    await stage.rollback_executor(stage, execution.context)
                    logger.info(f"Rollback completed for stage: {stage_id}")
                except Exception as e:
                    logger.error(f"Rollback failed for stage {stage_id}: {e}")

    def _calculate_execution_results(self, execution: PipelineExecution):
        """Calculate execution results and metrics."""
        total_stages = len(execution.stage_executions)
        completed_stages = sum(
            1
            for stage_exec in execution.stage_executions.values()
            if stage_exec.status == StageStatus.COMPLETED
        )

        execution.success_rate = (
            completed_stages / total_stages if total_stages > 0 else 0.0
        )

    def _register_default_executors(self):
        """Register default stage executors."""

        async def build_stage_executor(
            stage: PipelineStage, context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Default build stage executor."""
            logger.info(f"Executing build stage: {stage.stage_id}")

            # Simulate build process
            await asyncio.sleep(2)

            return {
                "build_id": f"build_{datetime.utcnow().timestamp()}",
                "artifacts": ["mapper-service.tar.gz", "config.yaml"],
                "build_time_minutes": 2.0,
            }

        async def deploy_stage_executor(
            stage: PipelineStage, context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Default deploy stage executor."""
            logger.info(f"Executing deploy stage: {stage.stage_id}")

            # Simulate deployment
            await asyncio.sleep(3)

            return {
                "deployment_id": f"deploy_{datetime.utcnow().timestamp()}",
                "deployed_version": context.get("version", "1.0.0"),
                "deployment_time_minutes": 3.0,
            }

        self._stage_executors["build"] = build_stage_executor
        self._stage_executors["deploy"] = deploy_stage_executor

    def get_execution_status(self, execution_id: str) -> Optional[PipelineExecution]:
        """Get status of a pipeline execution."""
        return self._active_executions.get(execution_id)

    def list_active_executions(self) -> List[PipelineExecution]:
        """List all active pipeline executions."""
        return list(self._active_executions.values())


# Global pipeline executor instance
_pipeline_executor: Optional[PipelineExecutor] = None


def get_pipeline_executor() -> PipelineExecutor:
    """Get the global pipeline executor instance."""
    global _pipeline_executor
    if _pipeline_executor is None:
        _pipeline_executor = PipelineExecutor()
    return _pipeline_executor

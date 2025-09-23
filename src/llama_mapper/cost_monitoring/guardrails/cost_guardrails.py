"""Cost guardrails system for preventing excessive spending and enforcing budget limits."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ...logging import get_logger
from ..core.metrics_collector import CostMetricsCollector, CostAlert, CostBreakdown


class GuardrailAction(str, Enum):
    """Actions that can be taken when guardrails are triggered."""
    
    ALERT = "alert"
    THROTTLE = "throttle"
    SCALE_DOWN = "scale_down"
    PAUSE_SERVICE = "pause_service"
    BLOCK_REQUESTS = "block_requests"
    NOTIFY_ADMIN = "notify_admin"


class GuardrailSeverity(str, Enum):
    """Severity levels for guardrail violations."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CostGuardrail(BaseModel):
    """Configuration for a cost guardrail."""
    
    guardrail_id: str = Field(description="Unique identifier for the guardrail")
    name: str = Field(description="Human-readable name")
    description: str = Field(description="Description of what this guardrail protects")
    metric_type: str = Field(description="Type of metric to monitor (daily_cost, hourly_cost, api_calls, etc.)")
    threshold: float = Field(description="Threshold value that triggers the guardrail")
    currency: str = Field(default="USD", description="Currency for the threshold")
    severity: GuardrailSeverity = Field(description="Severity level")
    actions: List[GuardrailAction] = Field(description="Actions to take when triggered")
    cooldown_minutes: int = Field(default=60, description="Cooldown period before re-triggering")
    enabled: bool = Field(default=True, description="Whether the guardrail is enabled")
    tenant_id: Optional[str] = Field(default=None, description="Tenant-specific guardrail")


class GuardrailViolation(BaseModel):
    """Record of a guardrail violation."""
    
    violation_id: str = Field(description="Unique violation identifier")
    guardrail_id: str = Field(description="ID of the triggered guardrail")
    metric_value: float = Field(description="Actual metric value that triggered the violation")
    threshold: float = Field(description="Threshold that was exceeded")
    severity: GuardrailSeverity = Field(description="Severity of the violation")
    actions_taken: List[GuardrailAction] = Field(description="Actions that were taken")
    triggered_at: datetime = Field(description="When the violation occurred")
    resolved_at: Optional[datetime] = Field(default=None, description="When the violation was resolved")
    tenant_id: Optional[str] = Field(default=None, description="Tenant ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CostGuardrailsConfig(BaseModel):
    """Configuration for the cost guardrails system."""
    
    enabled: bool = Field(default=True, description="Enable cost guardrails")
    default_actions: List[GuardrailAction] = Field(
        default=[GuardrailAction.ALERT, GuardrailAction.NOTIFY_ADMIN],
        description="Default actions for violations"
    )
    escalation_delay_minutes: int = Field(default=30, description="Delay before escalating actions")
    max_violations_per_hour: int = Field(default=10, description="Maximum violations per hour before emergency actions")
    emergency_actions: List[GuardrailAction] = Field(
        default=[GuardrailAction.PAUSE_SERVICE, GuardrailAction.BLOCK_REQUESTS],
        description="Emergency actions for critical violations"
    )


class CostGuardrails:
    """Cost guardrails system for enforcing spending limits and budget controls."""
    
    def __init__(
        self,
        config: CostGuardrailsConfig,
        metrics_collector: CostMetricsCollector,
    ):
        self.config = config
        self.metrics_collector = metrics_collector
        self.logger = get_logger(__name__)
        self._guardrails: Dict[str, CostGuardrail] = {}
        self._violations: List[GuardrailViolation] = []
        self._running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._last_violation_times: Dict[str, datetime] = {}
    
    async def start(self) -> None:
        """Start the guardrails monitoring system."""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Cost guardrails system started")
    
    async def stop(self) -> None:
        """Stop the guardrails monitoring system."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Cost guardrails system stopped")
    
    def add_guardrail(self, guardrail: CostGuardrail) -> None:
        """Add a new cost guardrail."""
        self._guardrails[guardrail.guardrail_id] = guardrail
        self.logger.info(
            "Added cost guardrail",
            guardrail_id=guardrail.guardrail_id,
            name=guardrail.name,
            threshold=guardrail.threshold,
        )
    
    def remove_guardrail(self, guardrail_id: str) -> None:
        """Remove a cost guardrail."""
        if guardrail_id in self._guardrails:
            del self._guardrails[guardrail_id]
            self.logger.info("Removed cost guardrail", guardrail_id=guardrail_id)
    
    def get_guardrails(self, tenant_id: Optional[str] = None) -> List[CostGuardrail]:
        """Get all guardrails, optionally filtered by tenant."""
        if tenant_id is None:
            return list(self._guardrails.values())
        
        return [
            guardrail for guardrail in self._guardrails.values()
            if guardrail.tenant_id is None or guardrail.tenant_id == tenant_id
        ]
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for checking guardrails."""
        while self._running:
            try:
                await self._check_all_guardrails()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in guardrails monitoring", error=str(e))
                await asyncio.sleep(5)
    
    async def _check_all_guardrails(self) -> None:
        """Check all enabled guardrails for violations."""
        for guardrail in self._guardrails.values():
            if not guardrail.enabled:
                continue
            
            # Check cooldown period
            if self._is_in_cooldown(guardrail):
                continue
            
            try:
                await self._check_guardrail(guardrail)
            except Exception as e:
                self.logger.error(
                    "Error checking guardrail",
                    guardrail_id=guardrail.guardrail_id,
                    error=str(e),
                )
    
    def _is_in_cooldown(self, guardrail: CostGuardrail) -> bool:
        """Check if a guardrail is in cooldown period."""
        last_violation = self._last_violation_times.get(guardrail.guardrail_id)
        if last_violation is None:
            return False
        
        cooldown_end = last_violation + timedelta(minutes=guardrail.cooldown_minutes)
        return datetime.now(timezone.utc) < cooldown_end
    
    async def _check_guardrail(self, guardrail: CostGuardrail) -> None:
        """Check a specific guardrail for violations."""
        current_value = await self._get_current_metric_value(guardrail)
        
        if current_value > guardrail.threshold:
            await self._handle_violation(guardrail, current_value)
    
    async def _get_current_metric_value(self, guardrail: CostGuardrail) -> float:
        """Get the current value for a guardrail's metric."""
        metric_type = guardrail.metric_type
        
        if metric_type == "daily_cost":
            return await self.metrics_collector._get_daily_cost()
        elif metric_type == "hourly_cost":
            return await self.metrics_collector._get_hourly_cost()
        elif metric_type == "api_calls":
            # Get current API call count from recent metrics
            if hasattr(self.metrics_collector.metrics_storage, 'get_recent_metrics'):
                recent_metrics = self.metrics_collector.metrics_storage.get_recent_metrics(10)
                return sum(m.usage.api_calls for m in recent_metrics)
            return 0.0
        elif metric_type == "compute_cost":
            # Get current compute cost from recent metrics
            if hasattr(self.metrics_collector.metrics_storage, 'get_recent_metrics'):
                recent_metrics = self.metrics_collector.metrics_storage.get_recent_metrics(10)
                return sum(
                    m.usage.cpu_cores * m.cost_per_unit["cpu_per_hour"] / 3600 +
                    m.usage.gpu_count * m.cost_per_unit["gpu_per_hour"] / 3600
                    for m in recent_metrics
                )
            return 0.0
        else:
            self.logger.warning("Unknown metric type", metric_type=metric_type)
            return 0.0
    
    async def _handle_violation(self, guardrail: CostGuardrail, current_value: float) -> None:
        """Handle a guardrail violation."""
        violation = GuardrailViolation(
            violation_id=f"{guardrail.guardrail_id}_{int(datetime.now().timestamp())}",
            guardrail_id=guardrail.guardrail_id,
            metric_value=current_value,
            threshold=guardrail.threshold,
            severity=guardrail.severity,
            actions_taken=guardrail.actions,
            triggered_at=datetime.now(timezone.utc),
            tenant_id=guardrail.tenant_id,
            metadata={
                "guardrail_name": guardrail.name,
                "metric_type": guardrail.metric_type,
                "currency": guardrail.currency,
            },
        )
        
        self._violations.append(violation)
        self._last_violation_times[guardrail.guardrail_id] = datetime.now(timezone.utc)
        
        # Check for emergency conditions
        if self._should_trigger_emergency_actions():
            violation.actions_taken.extend(self.config.emergency_actions)
        
        # Execute actions
        await self._execute_actions(violation)
        
        self.logger.warning(
            "Cost guardrail violation",
            violation_id=violation.violation_id,
            guardrail_id=guardrail.guardrail_id,
            current_value=current_value,
            threshold=guardrail.threshold,
            severity=guardrail.severity,
            actions=violation.actions_taken,
        )
    
    def _should_trigger_emergency_actions(self) -> bool:
        """Check if emergency actions should be triggered."""
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)
        
        recent_violations = [
            v for v in self._violations
            if v.triggered_at >= hour_ago
        ]
        
        return len(recent_violations) >= self.config.max_violations_per_hour
    
    async def _execute_actions(self, violation: GuardrailViolation) -> None:
        """Execute the actions for a guardrail violation."""
        for action in violation.actions_taken:
            try:
                await self._execute_action(action, violation)
            except Exception as e:
                self.logger.error(
                    "Failed to execute guardrail action",
                    action=action,
                    violation_id=violation.violation_id,
                    error=str(e),
                )
    
    async def _execute_action(self, action: GuardrailAction, violation: GuardrailViolation) -> None:
        """Execute a specific guardrail action."""
        if action == GuardrailAction.ALERT:
            await self._create_alert(violation)
        elif action == GuardrailAction.THROTTLE:
            await self._throttle_requests(violation)
        elif action == GuardrailAction.SCALE_DOWN:
            await self._scale_down_resources(violation)
        elif action == GuardrailAction.PAUSE_SERVICE:
            await self._pause_service(violation)
        elif action == GuardrailAction.BLOCK_REQUESTS:
            await self._block_requests(violation)
        elif action == GuardrailAction.NOTIFY_ADMIN:
            await self._notify_admin(violation)
    
    async def _create_alert(self, violation: GuardrailViolation) -> None:
        """Create an alert for the violation."""
        # This would integrate with your alerting system
        self.logger.warning(
            "Cost guardrail alert",
            violation_id=violation.violation_id,
            guardrail_id=violation.guardrail_id,
            severity=violation.severity,
            message=f"Cost guardrail violation: {violation.metric_value} > {violation.threshold}",
        )
    
    async def _throttle_requests(self, violation: GuardrailViolation) -> None:
        """Throttle incoming requests."""
        # This would integrate with your request throttling system
        self.logger.info("Throttling requests due to cost guardrail violation")
    
    async def _scale_down_resources(self, violation: GuardrailViolation) -> None:
        """Scale down resources to reduce costs."""
        # This would integrate with your autoscaling system
        self.logger.info("Scaling down resources due to cost guardrail violation")
    
    async def _pause_service(self, violation: GuardrailViolation) -> None:
        """Pause the service to prevent further costs."""
        # This would integrate with your service management
        self.logger.critical("Pausing service due to critical cost guardrail violation")
    
    async def _block_requests(self, violation: GuardrailViolation) -> None:
        """Block incoming requests."""
        # This would integrate with your request blocking system
        self.logger.critical("Blocking requests due to critical cost guardrail violation")
    
    async def _notify_admin(self, violation: GuardrailViolation) -> None:
        """Notify administrators of the violation."""
        # This would integrate with your notification system
        self.logger.critical(
            "Notifying administrators of cost guardrail violation",
            violation_id=violation.violation_id,
            severity=violation.severity,
        )
    
    def get_violations(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tenant_id: Optional[str] = None,
        severity: Optional[GuardrailSeverity] = None,
    ) -> List[GuardrailViolation]:
        """Get guardrail violations with optional filtering."""
        violations = self._violations
        
        if start_time:
            violations = [v for v in violations if v.triggered_at >= start_time]
        
        if end_time:
            violations = [v for v in violations if v.triggered_at <= end_time]
        
        if tenant_id:
            violations = [v for v in violations if v.tenant_id == tenant_id]
        
        if severity:
            violations = [v for v in violations if v.severity == severity]
        
        return violations
    
    def resolve_violation(self, violation_id: str) -> bool:
        """Mark a violation as resolved."""
        for violation in self._violations:
            if violation.violation_id == violation_id:
                violation.resolved_at = datetime.now(timezone.utc)
                self.logger.info("Resolved guardrail violation", violation_id=violation_id)
                return True
        
        return False
    
    def get_violation_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get a summary of violations over the specified period."""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        
        violations = self.get_violations(start_time=start_time, end_time=end_time)
        
        summary = {
            "total_violations": len(violations),
            "by_severity": {},
            "by_guardrail": {},
            "by_tenant": {},
            "resolved_violations": len([v for v in violations if v.resolved_at is not None]),
            "unresolved_violations": len([v for v in violations if v.resolved_at is None]),
        }
        
        # Count by severity
        for severity in GuardrailSeverity:
            count = len([v for v in violations if v.severity == severity])
            summary["by_severity"][severity.value] = count
        
        # Count by guardrail
        for violation in violations:
            guardrail_id = violation.guardrail_id
            if guardrail_id not in summary["by_guardrail"]:
                summary["by_guardrail"][guardrail_id] = 0
            summary["by_guardrail"][guardrail_id] += 1
        
        # Count by tenant
        for violation in violations:
            tenant_id = violation.tenant_id or "default"
            if tenant_id not in summary["by_tenant"]:
                summary["by_tenant"][tenant_id] = 0
            summary["by_tenant"][tenant_id] += 1
        
        return summary

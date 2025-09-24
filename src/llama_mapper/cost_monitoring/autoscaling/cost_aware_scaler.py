"""Cost-aware autoscaling system that balances performance and cost optimization."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ...logging import get_logger
from ..core.metrics_collector import CostMetricsCollector


class ScalingAction(str, Enum):
    """Scaling actions that can be performed."""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    NO_ACTION = "no_action"


class ScalingTrigger(str, Enum):
    """Triggers for scaling actions."""

    COST_THRESHOLD = "cost_threshold"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    PREDICTED_COST = "predicted_cost"
    SCHEDULED = "scheduled"
    MANUAL = "manual"


class ResourceType(str, Enum):
    """Types of resources that can be scaled."""

    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    REPLICAS = "replicas"
    NODES = "nodes"


class ScalingPolicy(BaseModel):
    """Configuration for a scaling policy."""

    policy_id: str = Field(description="Unique identifier for the policy")
    name: str = Field(description="Human-readable name")
    description: str = Field(description="Description of the policy")
    resource_type: ResourceType = Field(description="Type of resource to scale")
    trigger: ScalingTrigger = Field(description="Trigger for scaling")
    threshold: float = Field(description="Threshold value for triggering")
    min_instances: int = Field(default=1, description="Minimum number of instances")
    max_instances: int = Field(default=10, description="Maximum number of instances")
    scale_up_cooldown_minutes: int = Field(
        default=5, description="Cooldown after scale up"
    )
    scale_down_cooldown_minutes: int = Field(
        default=15, description="Cooldown after scale down"
    )
    cost_weight: float = Field(
        default=0.5, description="Weight given to cost in scaling decisions"
    )
    performance_weight: float = Field(
        default=0.5, description="Weight given to performance in scaling decisions"
    )
    enabled: bool = Field(default=True, description="Whether the policy is enabled")
    tenant_id: Optional[str] = Field(default=None, description="Tenant-specific policy")


class ScalingDecision(BaseModel):
    """A scaling decision made by the autoscaler."""

    decision_id: str = Field(description="Unique identifier for the decision")
    policy_id: str = Field(description="ID of the policy that triggered the decision")
    action: ScalingAction = Field(description="Action to be taken")
    resource_type: ResourceType = Field(description="Type of resource to scale")
    current_instances: int = Field(description="Current number of instances")
    target_instances: int = Field(description="Target number of instances")
    reason: str = Field(description="Reason for the scaling decision")
    cost_impact: float = Field(description="Expected cost impact")
    performance_impact: float = Field(description="Expected performance impact")
    triggered_at: datetime = Field(description="When the decision was made")
    executed_at: Optional[datetime] = Field(
        default=None, description="When the decision was executed"
    )
    tenant_id: Optional[str] = Field(default=None, description="Tenant ID")


class CostAwareScalingConfig(BaseModel):
    """Configuration for cost-aware autoscaling."""

    enabled: bool = Field(default=True, description="Enable cost-aware autoscaling")
    evaluation_interval_seconds: int = Field(
        default=60, description="How often to evaluate scaling decisions"
    )
    cost_threshold_percent: float = Field(
        default=80.0, description="Cost threshold as percentage of budget"
    )
    performance_threshold_percent: float = Field(
        default=70.0, description="Performance threshold as percentage"
    )
    max_cost_increase_percent: float = Field(
        default=50.0, description="Maximum cost increase per scaling action"
    )
    min_performance_improvement_percent: float = Field(
        default=20.0, description="Minimum performance improvement required"
    )
    prediction_horizon_hours: int = Field(
        default=1, description="Hours to look ahead for cost predictions"
    )


class CostAwareScaler:
    """Cost-aware autoscaling system that balances performance and cost optimization."""

    def __init__(
        self,
        config: CostAwareScalingConfig,
        metrics_collector: CostMetricsCollector,
    ):
        self.config = config
        self.metrics_collector = metrics_collector
        self.logger = get_logger(__name__)
        self._policies: Dict[str, ScalingPolicy] = {}
        self._decisions: List[ScalingDecision] = []
        self._running = False
        self._scaling_task: Optional[asyncio.Task] = None
        self._last_scaling_times: Dict[str, datetime] = {}
        self._current_instances: Dict[str, int] = {}

    async def start(self) -> None:
        """Start the cost-aware autoscaling system."""
        if self._running:
            return

        self._running = True
        self._scaling_task = asyncio.create_task(self._scaling_loop())
        self.logger.info("Cost-aware autoscaling system started")

    async def stop(self) -> None:
        """Stop the cost-aware autoscaling system."""
        self._running = False
        if self._scaling_task:
            self._scaling_task.cancel()
            try:
                await self._scaling_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Cost-aware autoscaling system stopped")

    def add_policy(self, policy: ScalingPolicy) -> None:
        """Add a new scaling policy."""
        self._policies[policy.policy_id] = policy
        self._current_instances[policy.policy_id] = policy.min_instances
        self.logger.info(
            "Added scaling policy",
            policy_id=policy.policy_id,
            name=policy.name,
            resource_type=policy.resource_type,
        )

    def remove_policy(self, policy_id: str) -> None:
        """Remove a scaling policy."""
        if policy_id in self._policies:
            del self._policies[policy_id]
            if policy_id in self._current_instances:
                del self._current_instances[policy_id]
            self.logger.info("Removed scaling policy", policy_id=policy_id)

    def get_policies(self, tenant_id: Optional[str] = None) -> List[ScalingPolicy]:
        """Get all scaling policies, optionally filtered by tenant."""
        if tenant_id is None:
            return list(self._policies.values())

        return [
            policy
            for policy in self._policies.values()
            if policy.tenant_id is None or policy.tenant_id == tenant_id
        ]

    async def _scaling_loop(self) -> None:
        """Main scaling evaluation loop."""
        while self._running:
            try:
                await self._evaluate_scaling_decisions()
                await asyncio.sleep(self.config.evaluation_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in scaling evaluation", error=str(e))
                await asyncio.sleep(5)

    async def _evaluate_scaling_decisions(self) -> None:
        """Evaluate all policies and make scaling decisions."""
        for policy in self._policies.values():
            if not policy.enabled:
                continue

            # Check cooldown period
            if self._is_in_cooldown(policy):
                continue

            try:
                decision = await self._evaluate_policy(policy)
                if decision and decision.action != ScalingAction.NO_ACTION:
                    await self._execute_scaling_decision(decision)
            except Exception as e:
                self.logger.error(
                    "Error evaluating scaling policy",
                    policy_id=policy.policy_id,
                    error=str(e),
                )

    def _is_in_cooldown(self, policy: ScalingPolicy) -> bool:
        """Check if a policy is in cooldown period."""
        last_scaling = self._last_scaling_times.get(policy.policy_id)
        if last_scaling is None:
            return False

        # Determine cooldown period based on last action
        # This is simplified - in practice, you'd track the last action type
        cooldown_minutes = policy.scale_up_cooldown_minutes
        cooldown_end = last_scaling + timedelta(minutes=cooldown_minutes)

        return datetime.now(timezone.utc) < cooldown_end

    async def _evaluate_policy(
        self, policy: ScalingPolicy
    ) -> Optional[ScalingDecision]:
        """Evaluate a specific scaling policy and return a decision."""
        current_instances = self._current_instances.get(
            policy.policy_id, policy.min_instances
        )

        # Get current metrics
        current_cost = await self._get_current_cost()
        current_performance = await self._get_current_performance()
        predicted_cost = await self._predict_cost(policy, current_instances)

        # Evaluate scaling options
        scale_up_decision = await self._evaluate_scale_up(
            policy, current_instances, current_cost, current_performance
        )
        scale_down_decision = await self._evaluate_scale_down(
            policy, current_instances, current_cost, current_performance
        )

        # Choose the best decision
        best_decision = None
        best_score = float("-inf")

        for decision in [scale_up_decision, scale_down_decision]:
            if decision and decision.action != ScalingAction.NO_ACTION:
                score = self._calculate_decision_score(
                    decision, current_cost, current_performance
                )
                if score > best_score:
                    best_score = score
                    best_decision = decision

        return best_decision

    async def _evaluate_scale_up(
        self,
        policy: ScalingPolicy,
        current_instances: int,
        current_cost: float,
        current_performance: float,
    ) -> Optional[ScalingDecision]:
        """Evaluate whether to scale up."""
        if current_instances >= policy.max_instances:
            return None

        target_instances = min(current_instances + 1, policy.max_instances)

        # Check if scaling up would exceed cost thresholds
        predicted_cost = await self._predict_cost(policy, target_instances)
        cost_increase = predicted_cost - current_cost
        cost_increase_percent = (
            (cost_increase / current_cost * 100) if current_cost > 0 else 0
        )

        if cost_increase_percent > self.config.max_cost_increase_percent:
            return None

        # Check if performance improvement justifies the cost
        predicted_performance = await self._predict_performance(
            policy, target_instances
        )
        performance_improvement = predicted_performance - current_performance

        if performance_improvement < self.config.min_performance_improvement_percent:
            return None

        return ScalingDecision(
            decision_id=f"scale_up_{policy.policy_id}_{int(datetime.now().timestamp())}",
            policy_id=policy.policy_id,
            action=ScalingAction.SCALE_UP,
            resource_type=policy.resource_type,
            current_instances=current_instances,
            target_instances=target_instances,
            reason=f"Performance improvement: {performance_improvement:.1f}%, Cost increase: {cost_increase_percent:.1f}%",
            cost_impact=cost_increase,
            performance_impact=performance_improvement,
            triggered_at=datetime.now(timezone.utc),
            tenant_id=policy.tenant_id,
        )

    async def _evaluate_scale_down(
        self,
        policy: ScalingPolicy,
        current_instances: int,
        current_cost: float,
        current_performance: float,
    ) -> Optional[ScalingDecision]:
        """Evaluate whether to scale down."""
        if current_instances <= policy.min_instances:
            return None

        target_instances = max(current_instances - 1, policy.min_instances)

        # Check if current performance is above threshold
        if current_performance < self.config.performance_threshold_percent:
            return None

        # Check if cost savings justify the performance impact
        predicted_cost = await self._predict_cost(policy, target_instances)
        cost_savings = current_cost - predicted_cost

        predicted_performance = await self._predict_performance(
            policy, target_instances
        )
        performance_impact = current_performance - predicted_performance

        # Only scale down if performance impact is acceptable
        if performance_impact > 30:  # Don't reduce performance by more than 30%
            return None

        return ScalingDecision(
            decision_id=f"scale_down_{policy.policy_id}_{int(datetime.now().timestamp())}",
            policy_id=policy.policy_id,
            action=ScalingAction.SCALE_DOWN,
            resource_type=policy.resource_type,
            current_instances=current_instances,
            target_instances=target_instances,
            reason=f"Cost savings: ${cost_savings:.2f}, Performance impact: {performance_impact:.1f}%",
            cost_impact=-cost_savings,  # Negative for savings
            performance_impact=-performance_impact,  # Negative for reduction
            triggered_at=datetime.now(timezone.utc),
            tenant_id=policy.tenant_id,
        )

    def _calculate_decision_score(
        self,
        decision: ScalingDecision,
        current_cost: float,
        current_performance: float,
    ) -> float:
        """Calculate a score for a scaling decision."""
        # This is a simplified scoring function
        # In practice, you'd have more sophisticated scoring based on your specific requirements

        cost_score = -decision.cost_impact / current_cost if current_cost > 0 else 0
        performance_score = decision.performance_impact / 100.0

        # Weight the scores based on policy configuration
        # This would need to be adjusted based on the specific policy
        weighted_score = 0.5 * cost_score + 0.5 * performance_score

        return weighted_score

    async def _get_current_cost(self) -> float:
        """Get current cost metrics."""
        # This would integrate with your cost monitoring system
        return await self.metrics_collector._get_hourly_cost()

    async def _get_current_performance(self) -> float:
        """Get current performance metrics."""
        # This would integrate with your performance monitoring system
        # For now, return a mock value
        return 75.0  # 75% performance

    async def _predict_cost(self, policy: ScalingPolicy, instances: int) -> float:
        """Predict cost for a given number of instances."""
        # This is a simplified prediction
        # In practice, you'd use historical data and machine learning models

        base_cost = await self._get_current_cost()
        cost_per_instance = base_cost / max(
            self._current_instances.get(policy.policy_id, 1), 1
        )

        return cost_per_instance * instances

    async def _predict_performance(
        self, policy: ScalingPolicy, instances: int
    ) -> float:
        """Predict performance for a given number of instances."""
        # This is a simplified prediction
        # In practice, you'd use historical data and performance models

        current_performance = await self._get_current_performance()
        current_instances = self._current_instances.get(policy.policy_id, 1)

        if instances == 0:
            return 0.0

        # Assume linear scaling (this is simplified)
        performance_ratio = instances / current_instances
        predicted_performance = current_performance * performance_ratio

        return min(predicted_performance, 100.0)  # Cap at 100%

    async def _execute_scaling_decision(self, decision: ScalingDecision) -> None:
        """Execute a scaling decision."""
        try:
            # Update current instances
            self._current_instances[decision.policy_id] = decision.target_instances
            self._last_scaling_times[decision.policy_id] = datetime.now(timezone.utc)

            # Execute the actual scaling action
            await self._perform_scaling_action(decision)

            # Record the decision
            decision.executed_at = datetime.now(timezone.utc)
            self._decisions.append(decision)

            self.logger.info(
                "Executed scaling decision",
                decision_id=decision.decision_id,
                action=decision.action,
                resource_type=decision.resource_type,
                current_instances=decision.current_instances,
                target_instances=decision.target_instances,
                reason=decision.reason,
            )

        except Exception as e:
            self.logger.error(
                "Failed to execute scaling decision",
                decision_id=decision.decision_id,
                error=str(e),
            )

    async def _perform_scaling_action(self, decision: ScalingDecision) -> None:
        """Perform the actual scaling action."""
        # This would integrate with your infrastructure management system
        # (e.g., Kubernetes, Docker Swarm, AWS Auto Scaling, etc.)

        if decision.action == ScalingAction.SCALE_UP:
            await self._scale_up_resources(decision)
        elif decision.action == ScalingAction.SCALE_DOWN:
            await self._scale_down_resources(decision)
        elif decision.action == ScalingAction.SCALE_OUT:
            await self._scale_out_resources(decision)
        elif decision.action == ScalingAction.SCALE_IN:
            await self._scale_in_resources(decision)

    async def _scale_up_resources(self, decision: ScalingDecision) -> None:
        """Scale up resources."""
        # Implementation would depend on your infrastructure
        self.logger.info(
            f"Scaling up {decision.resource_type} to {decision.target_instances} instances"
        )

    async def _scale_down_resources(self, decision: ScalingDecision) -> None:
        """Scale down resources."""
        # Implementation would depend on your infrastructure
        self.logger.info(
            f"Scaling down {decision.resource_type} to {decision.target_instances} instances"
        )

    async def _scale_out_resources(self, decision: ScalingDecision) -> None:
        """Scale out resources (add more nodes/instances)."""
        # Implementation would depend on your infrastructure
        self.logger.info(
            f"Scaling out {decision.resource_type} to {decision.target_instances} instances"
        )

    async def _scale_in_resources(self, decision: ScalingDecision) -> None:
        """Scale in resources (remove nodes/instances)."""
        # Implementation would depend on your infrastructure
        self.logger.info(
            f"Scaling in {decision.resource_type} to {decision.target_instances} instances"
        )

    def get_scaling_decisions(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tenant_id: Optional[str] = None,
    ) -> List[ScalingDecision]:
        """Get scaling decisions with optional filtering."""
        decisions = self._decisions

        if start_time:
            decisions = [d for d in decisions if d.triggered_at >= start_time]

        if end_time:
            decisions = [d for d in decisions if d.triggered_at <= end_time]

        if tenant_id:
            decisions = [d for d in decisions if d.tenant_id == tenant_id]

        return decisions

    def get_scaling_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get a summary of scaling activity over the specified period."""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        decisions = self.get_scaling_decisions(start_time=start_time, end_time=end_time)

        summary = {
            "total_decisions": len(decisions),
            "by_action": {},
            "by_resource_type": {},
            "by_tenant": {},
            "total_cost_impact": sum(d.cost_impact for d in decisions),
            "total_performance_impact": sum(d.performance_impact for d in decisions),
        }

        # Count by action
        for action in ScalingAction:
            count = len([d for d in decisions if d.action == action])
            summary["by_action"][action.value] = count

        # Count by resource type
        for resource_type in ResourceType:
            count = len([d for d in decisions if d.resource_type == resource_type])
            summary["by_resource_type"][resource_type.value] = count

        # Count by tenant
        for decision in decisions:
            tenant_id = decision.tenant_id or "default"
            if tenant_id not in summary["by_tenant"]:
                summary["by_tenant"][tenant_id] = 0
            summary["by_tenant"][tenant_id] += 1

        return summary

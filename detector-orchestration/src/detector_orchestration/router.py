from __future__ import annotations

from typing import Dict, List

from .config import Settings
from .models import OrchestrationRequest, RoutingDecision, RoutingPlan
from .health_monitor import HealthMonitor
from .policy import PolicyManager, CoverageMethod


class ContentRouter:
    def __init__(self, settings: Settings, health_monitor: HealthMonitor | None = None, policy_manager: PolicyManager | None = None):
        self.settings = settings
        self.health_monitor = health_monitor
        self.policy_manager = policy_manager

    async def route_request(self, request: OrchestrationRequest) -> tuple[RoutingPlan, RoutingDecision]:
        # Candidate set from config and request
        candidates = (
            request.required_detectors
            if request.required_detectors
            else list(self.settings.detectors.keys())
        )
        if request.excluded_detectors:
            candidates = [d for d in candidates if d not in request.excluded_detectors]

        # Content type filter
        typed = []
        for name in candidates:
            det = self.settings.detectors.get(name)
            if not det:
                continue
            if request.content_type.value in det.supported_content_types:
                typed.append(name)

        # Policy decision (OPA or tenant policy)
        if self.policy_manager:
            decision = await self.policy_manager.decide(
                tenant_id=request.tenant_id,
                bundle=request.policy_bundle,
                content_type=request.content_type,
                candidate_detectors=typed,
            )
            selected_policy = decision.selected_detectors
            coverage_method = decision.coverage_method.value
            coverage_requirements = decision.coverage_requirements
            routing_reason = "policy+" + decision.routing_reason
        else:
            selected_policy = typed
            coverage_method = CoverageMethod.REQUIRED_SET.value
            coverage_requirements = {"min_success_fraction": 1.0}
            routing_reason = "policy+default"

        # Health filter
        selected: List[str] = []
        health_status: Dict[str, bool] = {}
        for name in selected_policy:
            is_healthy = self.health_monitor.is_healthy(name) if self.health_monitor else True
            health_status[name] = is_healthy
            if is_healthy:
                selected.append(name)

        routing_plan = RoutingPlan(
            primary_detectors=selected,
            parallel_groups=[selected],
            timeout_config={d: self.settings.detectors[d].timeout_ms for d in selected},
            retry_config={d: self.settings.detectors[d].max_retries for d in selected},
            coverage_method=coverage_method,
            weights=(coverage_requirements.get("weights", {}) if isinstance(coverage_requirements, dict) else {}),
            required_taxonomy_categories=(
                coverage_requirements.get("required_taxonomy_categories", [])
                if isinstance(coverage_requirements, dict)
                else []
            ),
        )

        decision = RoutingDecision(
            selected_detectors=selected,
            routing_reason=routing_reason,
            policy_applied=request.policy_bundle,
            coverage_requirements=(coverage_requirements if isinstance(coverage_requirements, dict) else {"min_success_fraction": 1.0}),
            health_status=health_status,
        )
        return routing_plan, decision

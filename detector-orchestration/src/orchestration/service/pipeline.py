"""Pipeline helpers for the orchestration service."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from shared.interfaces.orchestration import OrchestrationRequest
from shared.utils.correlation import get_correlation_id

from ..core import AggregatedOutput, RoutingDecision
from ..core import RoutingPlan as CoordinatorRoutingPlan
from .models import AggregationContext, OrchestrationArtifacts, PipelineContext

logger = logging.getLogger(__name__)


async def run_pipeline(
    service,
    *,
    request: OrchestrationRequest,
    tenant_id: str,
    context_input,
) -> Tuple[PipelineContext, OrchestrationArtifacts]:
    """Execute the end-to-end pipeline within a tenant context."""
    tenant_context_source = (
        service.components.tenant_isolation.tenant_context(
            tenant_id, context_input.user_id
        )
        if service.components.tenant_isolation
        else service._fallback_tenant_context(  # noqa: SLF001
            tenant_id,
            context_input.user_id,
            context_input.correlation_id or get_correlation_id(),
        )
    )

    async with tenant_context_source as tenant_context:
        pipeline_context = PipelineContext(
            tenant_id=tenant_id,
            correlation_id=context_input.correlation_id or get_correlation_id(),
            tenant_context=tenant_context,
            processing_mode=context_input.processing_mode,
        )
        artifacts = await execute_pipeline(
            service,
            request=request,
            context=pipeline_context,
        )
        return pipeline_context, artifacts


async def execute_pipeline(
    service,
    *,
    request: OrchestrationRequest,
    context: PipelineContext,
) -> OrchestrationArtifacts:
    """Execute the orchestration pipeline and gather artifacts for the response."""
    context.content_features = analyze_content(
        service, request.content, context.correlation_id
    )
    context.routing_plan, context.routing_decision = await determine_routing(
        service,
        request=request,
        correlation_id=context.correlation_id,
    )
    detector_results = await execute_detectors(
        service,
        request=request,
        context=context,
    )
    aggregated_output, coverage = aggregate_results(
        service,
        detector_results=detector_results,
        context=context,
    )
    aggregation = AggregationContext(
        detector_results=detector_results,
        aggregated_output=aggregated_output,
        coverage=coverage,
    )
    policy_violations = await evaluate_policy_violations(
        service,
        context=context,
        policy_bundle=getattr(request, "policy_bundle", None),
        aggregation=aggregation,
    )
    recommendations = generate_recommendations(aggregation)

    return OrchestrationArtifacts(
        detector_results=aggregation.detector_results,
        aggregated_output=aggregation.aggregated_output,
        coverage=aggregation.coverage,
        policy_violations=policy_violations,
        recommendations=recommendations,
    )


def analyze_content(service, content: str, correlation_id: str):
    analyzer = service.components.content_analyzer
    if not analyzer:
        return None
    return analyzer.analyze_content(content, correlation_id)


async def determine_routing(
    service,
    request: OrchestrationRequest,
    correlation_id: str,
) -> Tuple[Optional[CoordinatorRoutingPlan], Optional[RoutingDecision]]:
    router = service.components.content_router
    if not router:
        return None, None

    routing_plan, routing_decision = await router.route_request(request)
    plan_for_coordinator = to_coordinator_routing_plan(routing_plan)

    if routing_decision:
        logger.debug(
            "Routing completed",
            extra=service._log_extra(  # noqa: SLF001
                correlation_id,
                selected_detectors=getattr(routing_decision, "selected_detectors", []),
                routing_reason=getattr(routing_decision, "routing_reason", "unknown"),
            ),
        )

    return plan_for_coordinator, routing_decision


def to_coordinator_routing_plan(router_plan) -> Optional[CoordinatorRoutingPlan]:
    if router_plan is None:
        return None

    return CoordinatorRoutingPlan(
        primary_detectors=getattr(router_plan, "primary_detectors", []),
        secondary_detectors=getattr(router_plan, "secondary_detectors", []),
        parallel_groups=getattr(router_plan, "parallel_groups", None),
        timeout_config=getattr(router_plan, "timeout_config", None),
        retry_config=getattr(router_plan, "retry_config", None),
    )


async def execute_detectors(
    service,
    request: OrchestrationRequest,
    context: PipelineContext,
) -> List:
    coordinator = service.components.detector_coordinator
    routing_plan = context.routing_plan
    routing_decision = context.routing_decision
    if not coordinator or not routing_plan or not routing_decision:
        return []

    metadata = context.build_metadata()

    return await coordinator.execute_routing_plan(
        content=request.content,
        routing_plan=routing_plan,
        request_id=context.correlation_id,
        metadata=metadata,
    )


def aggregate_results(
    service,
    detector_results,
    *,
    context: PipelineContext,
) -> Tuple[Optional[AggregatedOutput], float]:
    if not detector_results or not service.components.response_aggregator:
        return None, 0.0

    aggregator = service.components.response_aggregator
    aggregated_output, coverage = aggregator.aggregate_results(
        detector_results=detector_results,
        tenant_id=context.tenant_id,
    )
    return aggregated_output, coverage


async def evaluate_policy_violations(
    service,
    *,
    context: PipelineContext,
    policy_bundle: Optional[str],
    aggregation: AggregationContext,
):
    policy_manager = service.components.policy_manager
    if not policy_manager or not aggregation.detector_results:
        return []

    return await policy_manager.evaluate_policy_violations(
        tenant_id=context.tenant_id,
        policy_bundle=policy_bundle,
        detector_results=aggregation.detector_results,
        aggregated_output=aggregation.aggregated_output,
        coverage=aggregation.coverage,
        correlation_id=context.correlation_id,
    )


def generate_recommendations(aggregation: AggregationContext) -> List[str]:
    recommendations: List[str] = []

    if aggregation.coverage < 0.5:
        recommendations.append("Low detector coverage - consider adding more detectors")

    aggregated_output = aggregation.aggregated_output
    if aggregated_output and aggregated_output.confidence_score < 0.7:
        recommendations.append("Low confidence results - consider manual review")

    failed_count = len([r for r in aggregation.detector_results if r.confidence == 0.0])
    if failed_count > len(aggregation.detector_results) * 0.3:
        recommendations.append("High detector failure rate - check detector health")

    return recommendations

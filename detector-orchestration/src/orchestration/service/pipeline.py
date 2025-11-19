"""Pipeline helpers for the orchestration service."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from shared.exceptions.base import ValidationError
from shared.interfaces.common import RiskLevel, Severity
from shared.interfaces.detector_output import (
    CanonicalDetectorEntity,
    CanonicalDetectorOutput,
    CanonicalDetectorOutputs,
)
from shared.interfaces.orchestration import DetectorResult, OrchestrationRequest
from shared.utils.correlation import get_correlation_id

from ..core import AggregatedOutput, RoutingDecision
from ..core import RoutingPlan as CoordinatorRoutingPlan
from ..repository import DetectorMappingConfigRepository
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
        tenant_id=context.tenant_id,
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

    canonical_outputs = await canonicalize_results(
        service,
        context=context,
        detector_results=aggregation.detector_results,
    )

    return OrchestrationArtifacts(
        detector_results=aggregation.detector_results,
        aggregated_output=aggregation.aggregated_output,
        coverage=aggregation.coverage,
        policy_violations=policy_violations,
        recommendations=recommendations,
        canonical_outputs=canonical_outputs,
    )


def analyze_content(service, content: str, correlation_id: str):
    analyzer = service.components.content_analyzer
    if not analyzer:
        return None
    return analyzer.analyze_content(content, correlation_id)


async def determine_routing(
    service,
    request: OrchestrationRequest,
    tenant_id: str,
    correlation_id: str,
) -> Tuple[Optional[CoordinatorRoutingPlan], Optional[RoutingDecision]]:
    router = service.components.content_router
    policy_manager = service.components.policy_manager
    if not router:
        return None, None

    routing_plan, routing_decision = await router.route_request(request)
    plan_for_coordinator = to_coordinator_routing_plan(routing_plan)

    # Enforce tenant-specific policies via OPA before executing detectors.
    policy_bundle = getattr(request, "policy_bundle", None)
    if (
        policy_manager
        and policy_bundle
        and plan_for_coordinator is not None
        and plan_for_coordinator.primary_detectors
    ):
        decision = await policy_manager.decide(
            tenant_id=tenant_id,
            bundle=policy_bundle,
            content_type="text",
            candidate_detectors=list(plan_for_coordinator.primary_detectors),
        )

        selected = [
            detector
            for detector in plan_for_coordinator.primary_detectors
            if detector in decision.selected_detectors
        ]

        # If no detectors are allowed by policy, fail fast.
        if not selected:
            raise ValidationError(
                "No detectors allowed by policy",
                correlation_id=correlation_id,
            )

        # Filter routing plan to only policy-approved detectors.
        plan_for_coordinator.primary_detectors = selected
        plan_for_coordinator.parallel_groups = [selected]
        plan_for_coordinator.timeout_config = {
            name: timeout
            for name, timeout in plan_for_coordinator.timeout_config.items()
            if name in selected
        }
        plan_for_coordinator.retry_config = {
            name: retries
            for name, retries in plan_for_coordinator.retry_config.items()
            if name in selected
        }

        # Update routing decision with policy context.
        if routing_decision is None:
            routing_decision = RoutingDecision(
                selected_detectors=selected,
                routing_reason=decision.routing_reason,
                policy_applied=policy_bundle,
                coverage_requirements=decision.coverage_requirements,
            )
        else:
            routing_decision.selected_detectors = selected
            routing_decision.routing_reason = decision.routing_reason
            routing_decision.policy_applied = policy_bundle
            routing_decision.coverage_requirements = decision.coverage_requirements

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


async def canonicalize_results(
    service,
    *,
    context: PipelineContext,
    detector_results: List[DetectorResult],
) -> Optional[CanonicalDetectorOutputs]:
    """Canonicalize detector results using per-detector mapping configurations.

    This step looks up the active mapping configuration for each detector type
    and applies simple entity mapping rules over the detector findings to
    produce CanonicalDetectorOutput objects.
    """

    if not detector_results:
        return None

    repo: Optional[DetectorMappingConfigRepository] = (
        service.components.detector_mapping_repository
    )
    if not repo:
        return None

    outputs: List[CanonicalDetectorOutput] = []

    for result in detector_results:
        # Skip failed/error-like results
        if result.confidence <= 0.0 or result.category == "error":
            continue

        config = await repo.get_active_config(
            tenant_id=context.tenant_id,
            detector_type=result.detector_type,
        )
        mapping_rules: Dict[str, Any] = config.mapping_rules if config else {}
        outputs.append(_build_canonical_output(result, mapping_rules))

    if not outputs:
        return None

    return CanonicalDetectorOutputs(
        tenant_id=context.tenant_id,
        request_correlation_id=context.correlation_id,
        outputs=outputs,
    )


def _build_canonical_output(
    result: DetectorResult,
    mapping_rules: Dict[str, Any],
) -> CanonicalDetectorOutput:
    """Build a CanonicalDetectorOutput from a DetectorResult and mapping rules.

    The minimal mapping format is:

    {
      "canonical_category": "pii",
      "canonical_subcategory": "personal_identifiers",
      "canonical_risk_level": "medium",
      "tags": ["pii", "email"],
      "use_detector_category": true,
      "use_detector_subcategory": true,
      "entity_rules": [
        {
          "match": {"field": "type", "equals": "EXAMPLE_KEYWORD"},
          "label": "content.example_keyword",
          "category": "content_moderation",
          "subcategory": "example",
          "type": "keyword",
          "severity": "medium",
          "risk_level": "medium",
          "text_field": "value",
          "start_field": "start",
          "end_field": "end",
          "confidence_field": "confidence"
        }
      ]
    }
    """

    use_detector_category = mapping_rules.get("use_detector_category", True)
    use_detector_subcategory = mapping_rules.get("use_detector_subcategory", True)

    canonical_category = (
        result.category if use_detector_category else mapping_rules.get("canonical_category")
    ) or result.category
    canonical_subcategory = (
        result.subcategory
        if use_detector_subcategory
        else mapping_rules.get("canonical_subcategory")
    )

    risk_level_str = mapping_rules.get("canonical_risk_level")
    risk_level = _risk_level_from_config(risk_level_str, result.severity)

    tags = mapping_rules.get("tags", [])

    canonical_result = CanonicalDetectorOutput.__fields__["canonical_result"].annotation(
        category=canonical_category,
        subcategory=canonical_subcategory or "unspecified",
        confidence=result.confidence,
        risk_level=risk_level,
        tags=tags,
        metadata=result.metadata or {},
    )

    entities = _build_entities_from_findings(result, mapping_rules)

    max_severity = _max_severity(
        result.severity,
        [entity.severity for entity in entities],
    )
    max_risk_level = _max_risk_level(
        risk_level,
        [entity.risk_level for entity in entities],
    )

    return CanonicalDetectorOutput(
        detector_id=result.detector_id,
        detector_type=result.detector_type,
        canonical_result=canonical_result,
        entities=entities,
        max_severity=max_severity,
        max_risk_level=max_risk_level,
        raw_result=result,
    )


def _build_entities_from_findings(
    result: DetectorResult,
    mapping_rules: Dict[str, Any],
) -> List[CanonicalDetectorEntity]:
    entities: List[CanonicalDetectorEntity] = []
    rules = mapping_rules.get("entity_rules", [])
    if not rules or not result.findings:
        return entities

    for finding in result.findings:
        if not isinstance(finding, dict):
            continue

        for rule in rules:
            match = rule.get("match") or {}
            field = match.get("field")
            expected = match.get("equals")
            if field and expected is not None:
                if finding.get(field) != expected:
                    continue

            text_field = rule.get("text_field", "value")
            text_value = finding.get(text_field)
            if not text_value:
                continue

            start_field = rule.get("start_field", "start")
            end_field = rule.get("end_field", "end")
            confidence_field = rule.get("confidence_field", "confidence")

            severity = _severity_from_config(rule.get("severity"), result.severity)
            risk_level = _risk_level_from_config(rule.get("risk_level"), severity)

            entity = CanonicalDetectorEntity(
                text=str(text_value),
                start_offset=finding.get(start_field),
                end_offset=finding.get(end_field),
                label=rule.get("label", ""),
                category=rule.get("category", result.category),
                subcategory=rule.get("subcategory"),
                type=rule.get("type"),
                confidence=float(finding.get(confidence_field, result.confidence)),
                severity=severity,
                risk_level=risk_level,
                detector_id=result.detector_id,
                detector_type=result.detector_type,
                metadata=finding,
            )
            entities.append(entity)

    return entities


def _severity_from_config(
    value: Optional[str],
    fallback: Severity,
) -> Severity:
    if not value:
        return fallback
    try:
        return Severity(value.lower())
    except ValueError:
        return fallback


def _risk_level_from_config(
    value: Optional[str],
    fallback_severity: Severity,
) -> RiskLevel:
    if value:
        try:
            return RiskLevel(value.lower())
        except ValueError:
            pass
    return _risk_level_from_severity(fallback_severity)


def _risk_level_from_severity(severity: Severity) -> RiskLevel:
    if severity == Severity.CRITICAL:
        return RiskLevel.CRITICAL
    if severity == Severity.HIGH:
        return RiskLevel.HIGH
    if severity == Severity.MEDIUM:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _max_severity(
    base: Severity,
    others: List[Severity],
) -> Severity:
    order = {
        Severity.LOW: 1,
        Severity.MEDIUM: 2,
        Severity.HIGH: 3,
        Severity.CRITICAL: 4,
    }
    max_value = order[base]
    max_sev = base
    for sev in others:
        if order.get(sev, 0) > max_value:
            max_value = order[sev]
            max_sev = sev
    return max_sev


def _max_risk_level(
    base: RiskLevel,
    others: List[RiskLevel],
) -> RiskLevel:
    order = {
        RiskLevel.LOW: 1,
        RiskLevel.MEDIUM: 2,
        RiskLevel.HIGH: 3,
        RiskLevel.CRITICAL: 4,
    }
    max_value = order[base]
    max_level = base
    for level in others:
        if order.get(level, 0) > max_value:
            max_value = order[level]
            max_level = level
    return max_level

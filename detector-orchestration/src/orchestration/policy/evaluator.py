"""Policy evaluation engine.

Complex policy evaluation logic extracted from PolicyManager
to keep it focused and testable.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from shared.clients.opa_client import OPAError
from shared.exceptions.base import ServiceUnavailableError, ValidationError
# Note: get_correlation_id not needed in extracted evaluator

from .models import PolicyDecision

logger = logging.getLogger(__name__)

__all__ = [
    "evaluate_policy_decision",
    "process_opa_evaluation_results", 
    "evaluate_policy_fallback",
]


async def evaluate_policy_decision(
    *,
    opa_client,
    tenant_id: str,
    bundle: Optional[str],
    content_type: str,
    candidate_detectors: List[str],
    policy: Dict[str, Any],
    correlation_id: str,
) -> PolicyDecision:
    """Evaluate policy decision using OPA with comprehensive policy evaluation."""

    try:
        # Prepare input data for OPA policy evaluation
        input_data = {
            "tenant_id": tenant_id,
            "bundle": bundle,
            "content_type": content_type,
            "candidate_detectors": candidate_detectors,
            "policy_config": policy,
            "correlation_id": correlation_id,
            "timestamp": correlation_id,
        }

        # Evaluate detector selection policy
        detector_result = await opa_client.evaluate_policy(
            policy_path="detector_orchestration/detector_selection",
            input_data=input_data,
            tenant_id=tenant_id,
            correlation_id=correlation_id,
        )

        # Evaluate PII detection policy if applicable
        pii_result = await opa_client.evaluate_policy(
            policy_path="detector_orchestration/pii",
            input_data=input_data,
            tenant_id=tenant_id,
            correlation_id=correlation_id,
        )

        # Evaluate compliance policy if applicable
        compliance_result = await opa_client.evaluate_policy(
            policy_path="detector_orchestration/compliance",
            input_data=input_data,
            tenant_id=tenant_id,
            correlation_id=correlation_id,
        )

        # Evaluate security policy if applicable
        security_result = await opa_client.evaluate_policy(
            policy_path="detector_orchestration/security",
            input_data=input_data,
            tenant_id=tenant_id,
            correlation_id=correlation_id,
        )

        # Process OPA results and create decision
        return process_opa_evaluation_results(
            detector_result=detector_result,
            pii_result=pii_result,
            compliance_result=compliance_result,
            security_result=security_result,
            candidate_detectors=candidate_detectors,
            policy=policy,
            tenant_id=tenant_id,
            correlation_id=correlation_id,
        )

    except (OPAError, ValidationError, ServiceUnavailableError) as e:
        logger.warning(
            "OPA policy evaluation failed, using fallback logic",
            extra={
                "tenant_id": tenant_id,
                "bundle": bundle,
                "correlation_id": correlation_id,
                "error": str(e),
            },
        )

        # Fallback to local policy evaluation
        return evaluate_policy_fallback(
            policy=policy,
            candidate_detectors=candidate_detectors,
            tenant_id=tenant_id,
            correlation_id=correlation_id,
        )


def process_opa_evaluation_results(
    *,
    detector_result: Dict[str, Any],
    pii_result: Dict[str, Any],
    compliance_result: Dict[str, Any],
    security_result: Dict[str, Any],
    candidate_detectors: List[str],
    policy: Dict[str, Any],
    tenant_id: str,
    correlation_id: str,
) -> PolicyDecision:
    """Process OPA evaluation results into a policy decision."""

    violations = []

    # Extract detector selection from OPA results
    selected_detectors = []
    coverage_method = "required_set"
    routing_reason = f"opa_policy_evaluation_{tenant_id}"

    # Process detector selection results
    if "result" in detector_result and detector_result["result"]:
        det_result = detector_result["result"]
        if "selected_detectors" in det_result:
            selected_detectors = det_result["selected_detectors"]
        if "coverage_method" in det_result:
            coverage_method = det_result["coverage_method"]

    # Process PII detection results
    if "result" in pii_result and pii_result["result"]:
        pii_data = pii_result["result"]
        if not pii_data.get("allow", False):
            violations.append("PII detection policy violation")
        if pii_data.get("pii_detected", False):
            routing_reason += "_pii_detected"

    # Process compliance results
    if "result" in compliance_result and compliance_result["result"]:
        comp_data = compliance_result["result"]
        if not comp_data.get("allow", False):
            violations.append("Compliance policy violation")
        if not comp_data.get("compliant", False):
            violations.append("Compliance framework requirements not met")

    # Process security results
    if "result" in security_result and security_result["result"]:
        sec_data = security_result["result"]
        if not sec_data.get("allow", False):
            violations.append("Security policy violation")
        if not sec_data.get("secure", False):
            violations.append("Security validation failed")

    # Fallback to policy-based selection if OPA didn't provide detectors
    if not selected_detectors:
        max_detectors = policy.get("max_detectors", 5)
        selected_detectors = candidate_detectors[:max_detectors]
        routing_reason += "_fallback_selection"

    # Create comprehensive policy decision
    decision = PolicyDecision(
        selected_detectors=selected_detectors,
        coverage_method=coverage_method,
        coverage_requirements={
            "min_success_fraction": policy.get("min_confidence", 0.8),
            "timeout_ms": policy.get("timeout_ms", 5000),
            "retry_count": policy.get("retry_count", 3),
        },
        routing_reason=routing_reason,
        policy_violations=violations,
    )

    logger.info(
        "OPA evaluation results processed",
        extra={
            "tenant_id": tenant_id,
            "correlation_id": correlation_id,
            "selected_count": len(selected_detectors),
            "violations_count": len(violations),
            "coverage_method": coverage_method,
        },
    )

    return decision


def evaluate_policy_fallback(
    *,
    policy: Dict[str, Any],
    candidate_detectors: List[str],
    tenant_id: str,
    correlation_id: str,
) -> PolicyDecision:
    """Fallback policy evaluation when OPA is unavailable."""

    max_detectors = policy.get("max_detectors", 5)
    selected = candidate_detectors[:max_detectors]

    decision = PolicyDecision(
        selected_detectors=selected,
        coverage_method="required_set",
        coverage_requirements={
            "min_success_fraction": policy.get("min_confidence", 0.8),
            "timeout_ms": policy.get("timeout_ms", 5000),
            "retry_count": policy.get("retry_count", 3),
        },
        routing_reason=f"fallback_policy_{tenant_id}",
    )

    logger.info(
        "Used fallback policy evaluation",
        extra={
            "tenant_id": tenant_id,
            "correlation_id": correlation_id,
            "selected_count": len(selected),
        },
    )

    return decision

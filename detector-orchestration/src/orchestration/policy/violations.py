"""Policy violation evaluation helpers.

Extracted from the bulky `PolicyManager` so it can be unit-tested in
isolation and reused by other orchestration components.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from shared.interfaces.common import Severity
from shared.interfaces.orchestration import DetectorResult, PolicyViolation

if TYPE_CHECKING:  # pragma: no cover - avoid heavy imports at runtime
    from orchestration.core import AggregatedOutput

__all__ = [
    "evaluate_policy_violations",
]


def evaluate_policy_violations(
    *,
    policy: Dict[str, Any],
    _tenant_id: str,  # unused but kept for contextual logging by caller
    policy_bundle: Optional[str],
    detector_results: List[DetectorResult],
    aggregated_output: Optional["AggregatedOutput"],
    coverage: float,
    _correlation_id: str,
) -> List[PolicyViolation]:
    """Evaluate detector & aggregate results against configured policy.

    Returns a list of violations but performs **no** logging; callers (e.g.
    `PolicyManager`) can log context once per request.
    """

    min_confidence = float(policy.get("min_confidence", 0.7))
    coverage_target = float(
        policy.get("coverage_requirements", {}).get("min_success_fraction", 0.5)
    )
    policy_id_prefix = policy_bundle or "default"

    violations: List[PolicyViolation] = []

    # Per-detector confidence checks
    for result in detector_results:
        if result.confidence < min_confidence:
            severity = (
                Severity.CRITICAL
                if result.severity in {Severity.CRITICAL, Severity.HIGH}
                else Severity.MEDIUM
            )
            violations.append(
                PolicyViolation(
                    policy_id=f"policy:{policy_id_prefix}:confidence",
                    violation_type="low_confidence_detector",
                    message=(
                        f"Detector {result.detector_id} confidence {result.confidence:.2f} "
                        f"below threshold {min_confidence:.2f}"
                    ),
                    severity=severity,
                )
            )

    # Aggregated confidence check
    if aggregated_output and aggregated_output.confidence_score < min_confidence:
        violations.append(
            PolicyViolation(
                policy_id=f"policy:{policy_id_prefix}:aggregate",
                violation_type="low_confidence_aggregate",
                message=(
                    "Aggregated confidence "
                    f"{aggregated_output.confidence_score:.2f} below threshold "
                    f"{min_confidence:.2f}"
                ),
                severity=Severity.HIGH,
            )
        )

    # Coverage check
    if coverage_target > 0 and coverage < coverage_target:
        severity = Severity.HIGH if coverage < coverage_target / 2 else Severity.MEDIUM
        violations.append(
            PolicyViolation(
                policy_id=f"policy:{policy_id_prefix}:coverage",
                violation_type="insufficient_coverage",
                message=(
                    f"Achieved coverage {coverage:.2f} below required "
                    f"{coverage_target:.2f}"
                ),
                severity=severity,
            )
        )

    return violations

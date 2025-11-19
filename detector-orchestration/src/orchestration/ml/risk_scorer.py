from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from shared.interfaces.common import RiskLevel, Severity
from shared.interfaces.detector_output import CanonicalDetectorOutputs
from shared.interfaces.orchestration import DetectorResult, PolicyViolation

from ..core import AggregatedOutput


@dataclass
class RiskScore:
    level: RiskLevel
    score: float
    rules_evaluation: Dict[str, Any]
    model_features: Dict[str, Any]


class RiskScorer:
    def score(
        self,
        *,
        detector_results: List[DetectorResult],
        aggregated_output: Optional[AggregatedOutput],
        policy_violations: List[PolicyViolation],
        coverage: float,
        canonical_outputs: Optional[CanonicalDetectorOutputs] = None,
    ) -> RiskScore:
        features = self._build_features(
            detector_results=detector_results,
            aggregated_output=aggregated_output,
            policy_violations=policy_violations,
            coverage=coverage,
            canonical_outputs=canonical_outputs,
        )
        rules = self._evaluate_rules(features)
        model_score = self._apply_model(features, rules)
        level = self._map_score_to_level(model_score, rules)
        return RiskScore(
            level=level,
            score=model_score,
            rules_evaluation=rules,
            model_features=features,
        )

    def _build_features(
        self,
        *,
        detector_results: List[DetectorResult],
        aggregated_output: Optional[AggregatedOutput],
        policy_violations: List[PolicyViolation],
        coverage: float,
        canonical_outputs: Optional[CanonicalDetectorOutputs] = None,
    ) -> Dict[str, Any]:
        successful = [
            r
            for r in detector_results
            if r.confidence > 0.0 and r.category != "error"
        ]
        max_severity_value = 0
        max_severity = None
        for result in successful:
            severity_value = self._severity_to_numeric(result.severity)
            if severity_value > max_severity_value:
                max_severity_value = severity_value
                max_severity = str(result.severity)

        if canonical_outputs is not None:
            for output in canonical_outputs.outputs:
                severity_value = self._severity_to_numeric(output.max_severity)
                if severity_value > max_severity_value:
                    max_severity_value = severity_value
                    max_severity = str(output.max_severity)
        avg_confidence = (
            sum(r.confidence for r in successful) / len(successful)
            if successful
            else 0.0
        )
        policy_violation_count = len(policy_violations)
        has_critical_violation = any(
            v.severity == Severity.CRITICAL for v in policy_violations
        )
        categories = {r.category for r in successful}
        if canonical_outputs is not None:
            for output in canonical_outputs.outputs:
                categories.add(output.canonical_result.category)
                for entity in output.entities:
                    categories.add(entity.category)
        has_sensitive_category = bool(
            {"pii", "financial", "medical", "security"} & categories
        )
        combined_output = aggregated_output.combined_output if aggregated_output else ""
        combined_confidence = (
            aggregated_output.confidence_score if aggregated_output else 0.0
        )
        return {
            "max_severity_value": float(max_severity_value),
            "max_severity": max_severity,
            "avg_confidence": float(avg_confidence),
            "coverage": float(coverage),
            "policy_violation_count": float(policy_violation_count),
            "has_critical_violation": has_critical_violation,
            "has_sensitive_category": has_sensitive_category,
            "combined_output": combined_output,
            "combined_confidence": float(combined_confidence),
        }

    def _evaluate_rules(self, features: Dict[str, Any]) -> Dict[str, Any]:
        rules: Dict[str, Any] = {}
        max_severity_value = features["max_severity_value"]
        coverage = features["coverage"]
        avg_confidence = features["avg_confidence"]
        policy_violation_count = int(features["policy_violation_count"])
        has_critical_violation = bool(features["has_critical_violation"])
        has_sensitive_category = bool(features["has_sensitive_category"])
        rules["severity_rule"] = {
            "max_severity_value": max_severity_value,
        }
        rules["coverage_rule"] = {
            "coverage": coverage,
            "low_coverage": coverage < 0.5,
        }
        rules["confidence_rule"] = {
            "avg_confidence": avg_confidence,
        }
        rules["policy_rule"] = {
            "policy_violation_count": policy_violation_count,
            "has_critical_violation": has_critical_violation,
        }
        rules["sensitivity_rule"] = {
            "has_sensitive_category": has_sensitive_category,
        }
        return rules

    def _apply_model(
        self,
        features: Dict[str, Any],
        rules: Dict[str, Any],
    ) -> float:
        max_severity_value = features["max_severity_value"]
        avg_confidence = features["avg_confidence"]
        coverage = features["coverage"]
        policy_violation_count = float(features["policy_violation_count"])
        has_critical_violation = bool(features["has_critical_violation"])
        has_sensitive_category = bool(features["has_sensitive_category"])
        linear = -1.0
        linear += 0.8 * (max_severity_value / 4.0)
        linear += 0.5 * (1.0 - coverage)
        linear += 0.6 * (1.0 - avg_confidence)
        linear += 0.3 * min(policy_violation_count, 3.0) / 3.0
        if has_sensitive_category:
            linear += 0.4
        if has_critical_violation:
            linear += 0.5
        score = 1.0 / (1.0 + math.exp(-linear))
        return float(score)

    def _map_score_to_level(
        self,
        score: float,
        rules: Dict[str, Any],
    ) -> RiskLevel:
        if rules["severity_rule"]["max_severity_value"] >= 4:
            return RiskLevel.CRITICAL
        if score >= 0.75:
            return RiskLevel.HIGH
        if score >= 0.5:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _severity_to_numeric(self, severity: Severity) -> int:
        if severity == Severity.CRITICAL:
            return 4
        if severity == Severity.HIGH:
            return 3
        if severity == Severity.MEDIUM:
            return 2
        if severity == Severity.LOW:
            return 1
        return 0

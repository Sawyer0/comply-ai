"""Response aggregation module for combining detector results.

This module provides functionality to aggregate multiple detector results
into a unified payload for downstream processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .models import DetectorResult, MapperPayload, RoutingPlan
from .taxonomy_hints import load_detector_taxonomy_hints
from .conflict import ConflictResolutionOutcome


@dataclass
class AggregationSummary:
    """Aggregated detector result data used for metadata construction."""

    contributing: List[str]
    normalized_scores: Dict[str, float]
    strategy: str
    coverage_achieved: float
    provenance: List[Dict[str, object]]
    conflict_outcome: ConflictResolutionOutcome | None


class ResponseAggregator:
    """Combine detector results into a unified MapperPayload and metrics.

    Contract rules:
    - Do not assign canonical taxonomy.
    - Provide aggregated `output` as a raw indicator string; pipe-join successes.
    - Include provenance and coverage inputs in metadata.
    - Allow conflict resolution metadata to be embedded for Mapper hints (optional).
    """

    # pylint: disable=too-few-public-methods

    def aggregate(
        self,
        detector_results: List[DetectorResult],
        routing_plan: RoutingPlan,
        tenant_id: str,
        *,
        conflict_outcome: ConflictResolutionOutcome | None = None,
    ) -> tuple[MapperPayload, float]:
        """Aggregate detector responses into a single mapper payload."""

        successes = self._successful_results(detector_results)
        unique_outputs = self._unique_outputs(successes)
        summary = AggregationSummary(
            contributing=[result.detector for result in successes],
            normalized_scores=self._normalized_scores(
                unique_outputs, successes, conflict_outcome
            ),
            strategy=self._aggregation_strategy(unique_outputs, conflict_outcome),
            coverage_achieved=self._compute_coverage(detector_results, routing_plan),
            provenance=self._build_provenance(detector_results),
            conflict_outcome=conflict_outcome,
        )

        payload = MapperPayload(
            detector="orchestrated-multi",
            output=self._aggregate_output(unique_outputs),
            tenant_id=tenant_id,
            metadata=self._build_metadata(summary),
        )

        return payload, summary.coverage_achieved

    @staticmethod
    def _successful_results(results: List[DetectorResult]) -> List[DetectorResult]:
        """Return only successful detector results."""

        return [result for result in results if result.status.value == "success"]

    @staticmethod
    def _unique_outputs(successes: List[DetectorResult]) -> List[str]:
        """Extract unique outputs from successful detector results preserving order."""

        unique: List[str] = []
        for result in successes:
            if result.output and result.output not in unique:
                unique.append(result.output)
        return unique

    @staticmethod
    def _aggregate_output(unique_outputs: List[str]) -> str:
        """Join unique detector outputs for payload construction."""

        return "|".join(unique_outputs) if unique_outputs else "none"

    @staticmethod
    def _build_provenance(
        detector_results: List[DetectorResult],
    ) -> List[Dict[str, object]]:
        """Construct provenance entries for logging and downstream transparency."""

        return [
            {
                "detector": result.detector,
                "confidence": result.confidence,
                "output": result.output,
                "processing_time_ms": result.processing_time_ms,
            }
            for result in detector_results
        ]

    @staticmethod
    def _normalized_scores(
        unique_outputs: List[str],
        successes: List[DetectorResult],
        conflict_outcome: ConflictResolutionOutcome | None,
    ) -> Dict[str, float]:
        """Compute normalized confidence scores per output."""

        if conflict_outcome and conflict_outcome.normalized_scores:
            return dict(conflict_outcome.normalized_scores)

        scores: Dict[str, float] = {}
        for output in unique_outputs:
            matches = [
                result
                for result in successes
                if (result.output or "none").strip() == output
            ]
            if matches:
                avg = sum((result.confidence or 0.0) for result in matches) / len(matches)
            else:
                avg = 0.0
            scores[output] = float(max(0.0, min(avg, 1.0)))
        return scores

    @staticmethod
    def _aggregation_strategy(
        unique_outputs: List[str],
        conflict_outcome: ConflictResolutionOutcome | None,
    ) -> str:
        """Derive the aggregation strategy label for metadata."""

        if len(unique_outputs) <= 1:
            return "highest_confidence"
        if conflict_outcome is not None:
            return conflict_outcome.strategy_used.value
        return "highest_confidence"

    @staticmethod
    def _build_metadata(summary: AggregationSummary) -> Dict[str, object]:
        """Assemble response metadata for the aggregation output."""

        metadata: Dict[str, object] = {
            "contributing_detectors": summary.contributing,
            "normalized_scores": summary.normalized_scores,
            "conflict_resolution_applied": len(summary.normalized_scores) > 1,
            "aggregation_method": summary.strategy,
            "coverage_achieved": summary.coverage_achieved,
            "provenance": summary.provenance,
        }
        if summary.conflict_outcome is not None:
            outcome = summary.conflict_outcome
            metadata["conflict_resolution"] = {
                "strategy_used": outcome.strategy_used.value,
                "winning_output": outcome.winning_output,
                "winning_detector": outcome.winning_detector,
                "tie_breaker_applied": outcome.tie_breaker_applied,
                "confidence_delta": outcome.confidence_delta,
                "opa_decision": outcome.audit_decision,
            }
        return metadata

    def _compute_coverage(
        self, results: list[DetectorResult], plan: RoutingPlan
    ) -> float:
        """Calculate coverage achieved based on routing plan configuration."""

        method = (plan.coverage_method or "required_set").lower()
        successes = {r.detector for r in results if r.status.value == "success"}
        if method == "weighted":
            return self._weighted_coverage(successes, plan)
        if method == "taxonomy":
            return self._taxonomy_coverage(successes, plan)
        # Default required_set
        total = len(plan.primary_detectors) or 1
        return len(successes) / total

    @staticmethod
    def _weighted_coverage(successes: set[str], plan: RoutingPlan) -> float:
        """Compute coverage when weight configuration is provided."""

        weights = plan.weights or {}
        if not weights:
            total = len(plan.primary_detectors) or 1
            return len(successes) / total
        total_weight = sum(weights.get(det, 0.0) for det in plan.primary_detectors) or 1.0
        achieved = sum(weights.get(det, 0.0) for det in successes)
        return min(achieved / total_weight, 1.0)

    @staticmethod
    def _taxonomy_coverage(successes: set[str], plan: RoutingPlan) -> float:
        """Compute coverage coverage across taxonomy categories."""

        hints = load_detector_taxonomy_hints()
        required = list(plan.required_taxonomy_categories or [])
        if not required:
            covered_by_selected: set[str] = set()
            for det in plan.primary_detectors:
                categories = getattr(hints.get(det), "categories", None)
                if categories:
                    covered_by_selected.update(categories)
            required = list(sorted(covered_by_selected))
        if not required:
            total = len(plan.primary_detectors) or 1
            return len(successes) / total
        hit: set[str] = set()
        for det in successes:
            categories = getattr(hints.get(det), "categories", None)
            if categories:
                hit.update(categories)
        required_set = set(required)
        if not required_set:
            return 0.0
        covered = len(hit.intersection(required_set))
        return min(max(covered / float(len(required_set)), 0.0), 1.0)

from __future__ import annotations

from typing import Dict, List

from .models import DetectorResult, MapperPayload, RoutingPlan, ContentType
from .taxonomy_hints import load_detector_taxonomy_hints
from .conflict import ConflictResolutionOutcome


class ResponseAggregator:
    """Combine detector results into a unified MapperPayload and metrics.

    Contract rules:
    - Do not assign canonical taxonomy.
    - Provide aggregated `output` as a raw indicator string; pipe-join successes.
    - Include provenance and coverage inputs in metadata.
    - Allow conflict resolution metadata to be embedded for Mapper hints (optional).
    """

    def aggregate(
        self,
        detector_results: List[DetectorResult],
        routing_plan: RoutingPlan,
        tenant_id: str,
        *,
        conflict_outcome: ConflictResolutionOutcome | None = None,
    ) -> tuple[MapperPayload, float]:
        successes = [r for r in detector_results if r.status.value == "success"]
        contributing = [r.detector for r in successes]
        outputs = [r.output for r in successes if r.output]
        unique_outputs: List[str] = []
        for o in outputs:
            if o not in unique_outputs:
                unique_outputs.append(o)
        aggregated_output = "|".join(unique_outputs) if unique_outputs else "none"

        provenance = [
            {
                "detector": r.detector,
                "confidence": r.confidence,
                "output": r.output,
                "processing_time_ms": r.processing_time_ms,
            }
            for r in detector_results
        ]

        # Default normalized scores if not provided via conflict outcome
        normalized_scores: Dict[str, float] = {}
        if conflict_outcome and conflict_outcome.normalized_scores:
            normalized_scores = dict(conflict_outcome.normalized_scores)
        else:
            # Simple average confidence per unique raw output
            for out in unique_outputs:
                rs = [r for r in successes if (r.output or "none").strip() == out]
                if rs:
                    avg = sum((r.confidence or 0.0) for r in rs) / float(len(rs))
                else:
                    avg = 0.0
                normalized_scores[out] = float(max(0.0, min(avg, 1.0)))

        strategy = (
            (conflict_outcome.strategy_used.value if conflict_outcome else "highest_confidence")
            if len(unique_outputs) > 1
            else "highest_confidence"
        )

        coverage_achieved = self._compute_coverage(detector_results, routing_plan)

        metadata: Dict[str, object] = {
            "contributing_detectors": contributing,
            "normalized_scores": normalized_scores,
            "conflict_resolution_applied": len(unique_outputs) > 1,
            "aggregation_method": strategy,
            "coverage_achieved": coverage_achieved,
            "provenance": provenance,
        }
        if conflict_outcome is not None:
            metadata["conflict_resolution"] = {
                "strategy_used": conflict_outcome.strategy_used.value,
                "winning_output": conflict_outcome.winning_output,
                "winning_detector": conflict_outcome.winning_detector,
                "tie_breaker_applied": conflict_outcome.tie_breaker_applied,
                "confidence_delta": conflict_outcome.confidence_delta,
                "opa_decision": conflict_outcome.audit_decision,
            }

        payload = MapperPayload(
            detector="orchestrated-multi",
            output=aggregated_output,
            tenant_id=tenant_id,
            metadata=metadata,
        )

        return payload, coverage_achieved

    def _compute_coverage(self, results: list[DetectorResult], plan: RoutingPlan) -> float:
        method = (plan.coverage_method or "required_set").lower()
        successes = {r.detector for r in results if r.status.value == "success"}
        if method == "weighted":
            weights = plan.weights or {}
            if not weights:
                # Equal weights
                total = len(plan.primary_detectors) or 1
                return len(successes) / total
            total_w = sum(weights.get(d, 0.0) for d in plan.primary_detectors) or 1.0
            achieved = sum(weights.get(d, 0.0) for d in successes)
            return min(achieved / total_w, 1.0)
        elif method == "taxonomy":
            # Taxonomy coverage: count required taxonomy categories hit by successful detectors
            hints = load_detector_taxonomy_hints()
            required = list(plan.required_taxonomy_categories or [])
            if not required:
                # Default to union of categories covered by selected detectors
                covered_by_selected = set()
                for d in plan.primary_detectors:
                    if d in hints:
                        covered_by_selected.update(hints[d].categories)
                required = list(sorted(covered_by_selected))
            if not required:
                # Fallback: treat like required set of detectors
                total = len(plan.primary_detectors) or 1
                return len(successes) / total
            hit = set()
            for d in successes:
                cats = hints.get(d).categories if d in hints else set()
                hit.update(cats)
            required_set = set(required)
            if not required_set:
                return 0.0
            covered = len(hit.intersection(required_set))
            return min(max(covered / float(len(required_set)), 0.0), 1.0)
        # Default required_set
        total = len(plan.primary_detectors) or 1
        return len(successes) / total

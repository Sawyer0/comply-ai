"""Response aggregation functionality following SRP.

This module provides ONLY response aggregation - combining detector results into a unified
output. Other responsibilities are handled by separate modules:
- Conflict resolution: ../policy/conflict_resolver.py
- Quality scoring: ../monitoring/quality_monitor.py
- Coverage calculation: ../monitoring/coverage_calculator.py
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

from shared.interfaces.orchestration import DetectorResult
from shared.utils.correlation import get_correlation_id

from .models import AggregatedOutput

logger = logging.getLogger(__name__)


class ResponseAggregator:
    """Aggregates multiple detector results into a unified output."""

    def __init__(self, default_strategy: str = "highest_confidence") -> None:
        self.default_strategy = default_strategy

    def aggregate_results(
        self,
        detector_results: List[DetectorResult],
        tenant_id: str,
        strategy: Optional[str] = None,
    ) -> Tuple[AggregatedOutput, float]:
        """Aggregate detector results into unified output.

        Args:
            detector_results: List of detector results to aggregate
            tenant_id: Tenant identifier for context
            strategy: Optional aggregation strategy name
        """

        correlation_id = get_correlation_id()
        active_strategy = strategy or self.default_strategy

        logger.info(
            "Aggregating %d detector results for tenant %s using strategy %s",
            len(detector_results),
            tenant_id,
            active_strategy,
            extra={
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "detector_count": len(detector_results),
                "strategy": active_strategy,
            },
        )

        successful_results = self._get_successful_results(detector_results)
        if not successful_results:
            return AggregatedOutput.empty(), 0.0

        aggregation = self._apply_strategy(successful_results, active_strategy)
        coverage = (
            len(successful_results) / len(detector_results) if detector_results else 0.0
        )

        logger.info(
            "Aggregation completed with coverage %.2f",
            coverage,
            extra={
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "coverage_achieved": coverage,
                "combined_output": aggregation.combined_output,
                "confidence_score": aggregation.confidence_score,
            },
        )

        return aggregation, coverage

    def _apply_strategy(
        self, results: List[DetectorResult], strategy: str
    ) -> AggregatedOutput:
        if strategy == "majority_vote":
            return self._aggregate_by_majority_vote(results)
        if strategy == "weighted_average":
            return self._aggregate_by_weighted_average(results)
        return self._aggregate_by_highest_confidence(results)

    @staticmethod
    def _get_successful_results(
        results: List[DetectorResult]
    ) -> List[DetectorResult]:
        return [
            result
            for result in results
            if result.confidence > 0.0 and result.category != "error"
        ]

    @staticmethod
    def _aggregate_by_highest_confidence(
        results: List[DetectorResult]
    ) -> AggregatedOutput:
        best_result = max(results, key=lambda r: r.confidence)
        return AggregatedOutput(
            combined_output=f"{best_result.category}:{best_result.severity}",
            confidence_score=best_result.confidence,
            contributing_detectors=[best_result.detector_id],
            metadata={
                "strategy": "highest_confidence",
                "source_detector": best_result.detector_id,
            },
        )

    @staticmethod
    def _aggregate_by_majority_vote(
        results: List[DetectorResult]
    ) -> AggregatedOutput:
        category_votes: Counter[str] = Counter(
            f"{result.category}:{result.severity}" for result in results
        )
        winning_category, _ = category_votes.most_common(1)[0]
        category_results = [
            result
            for result in results
            if f"{result.category}:{result.severity}" == winning_category
        ]

        avg_confidence = sum(r.confidence for r in category_results) / len(category_results)
        contributing_detectors = [result.detector_id for result in category_results]

        return AggregatedOutput(
            combined_output=winning_category,
            confidence_score=avg_confidence,
            contributing_detectors=contributing_detectors,
            metadata={
                "strategy": "majority_vote",
                "category_votes": dict(category_votes),
                "winning_category": winning_category,
                "total_detectors": len(results),
            },
        )

    @staticmethod
    def _aggregate_by_weighted_average(
        results: List[DetectorResult]
    ) -> AggregatedOutput:
        if not results:
            return AggregatedOutput.empty()

        category_groups: Dict[str, List[DetectorResult]] = {}
        for result in results:
            category_groups.setdefault(result.category, []).append(result)

        category_scores = {
            category: sum(r.confidence for r in group) / len(results)
            for category, group in category_groups.items()
        }
        best_category = max(category_scores, key=category_scores.__getitem__)
        best_results = category_groups[best_category]

        severity_votes: Counter[str] = Counter(result.severity for result in best_results)
        dominant_severity = severity_votes.most_common(1)[0][0]

        return AggregatedOutput(
            combined_output=f"{best_category}:{dominant_severity}",
            confidence_score=category_scores[best_category],
            contributing_detectors=[result.detector_id for result in best_results],
            metadata={
                "strategy": "weighted_average",
                "category_scores": category_scores,
                "winning_category": best_category,
                "total_detectors": len(results),
            },
        )

    def get_unique_outputs(self, results: List[DetectorResult]) -> List[str]:
        """Return distinct category/severity pairs for successful results."""
        successful = self._get_successful_results(results)
        return list({f"{result.category}:{result.severity}" for result in successful})

    def calculate_confidence_distribution(
        self, results: List[DetectorResult]
    ) -> Dict[str, float]:
        """Map confidence buckets to their relative frequency."""
        successful = self._get_successful_results(results)
        if not successful:
            return {"no_results": 1.0} if results else {}

        distribution = {
            "high_confidence": 0,
            "medium_confidence": 0,
            "low_confidence": 0,
        }

        for result in successful:
            if result.confidence > 0.8:
                distribution["high_confidence"] += 1
            elif result.confidence >= 0.5:
                distribution["medium_confidence"] += 1
            else:
                distribution["low_confidence"] += 1

        total = len(successful)
        return {bucket: count / total for bucket, count in distribution.items()}


__all__ = ["ResponseAggregator"]

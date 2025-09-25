"""Response aggregation functionality following SRP.

This module provides ONLY response aggregation - combining detector results into unified output.
Other responsibilities are handled by separate modules:
- Conflict resolution: ../policy/conflict_resolver.py
- Quality scoring: ../monitoring/quality_monitor.py
- Coverage calculation: ../monitoring/coverage_calculator.py
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter

from shared.interfaces.orchestration import DetectorResult, AggregationSummary
from shared.utils.correlation import get_correlation_id

logger = logging.getLogger(__name__)


class AggregatedOutput:
    """Aggregated detector output - data structure only."""

    def __init__(
        self,
        combined_output: str,
        confidence_score: float,
        contributing_detectors: List[str],
        metadata: Dict[str, Any],
    ):
        self.combined_output = combined_output
        self.confidence_score = confidence_score
        self.contributing_detectors = contributing_detectors
        self.metadata = metadata


class ResponseAggregator:
    """Aggregates multiple detector results into a unified output.

    Single Responsibility: Combine detector results using aggregation strategies.
    Does NOT handle: conflict resolution, quality scoring, coverage calculation.
    """

    def __init__(self, default_strategy: str = "highest_confidence"):
        """Initialize aggregator with default strategy.

        Args:
            default_strategy: Default aggregation strategy to use
        """
        self.default_strategy = default_strategy

    def aggregate_results(
        self,
        detector_results: List[DetectorResult],
        tenant_id: str,
        strategy: Optional[str] = None,
    ) -> Tuple[AggregatedOutput, float]:
        """Aggregate detector results into unified output.

        Single responsibility: combine multiple detector results using specified strategy.

        Args:
            detector_results: List of detector results to aggregate
            tenant_id: Tenant identifier for context
            strategy: Aggregation strategy to use (optional)

        Returns:
            Tuple of (aggregated_output, coverage_score)
        """
        correlation_id = get_correlation_id()
        strategy = strategy or self.default_strategy

        logger.info(
            "Aggregating %d detector results for tenant %s using strategy %s",
            len(detector_results),
            tenant_id,
            strategy,
            extra={
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "detector_count": len(detector_results),
                "strategy": strategy,
            },
        )

        # Filter successful results
        successful_results = self._get_successful_results(detector_results)

        if not successful_results:
            # No successful results - return empty aggregation
            return self._create_empty_aggregation(), 0.0

        # Apply aggregation strategy
        if strategy == "highest_confidence":
            aggregated = self._aggregate_by_highest_confidence(successful_results)
        elif strategy == "majority_vote":
            aggregated = self._aggregate_by_majority_vote(successful_results)
        elif strategy == "weighted_average":
            aggregated = self._aggregate_by_weighted_average(successful_results)
        else:
            # Default to highest confidence
            aggregated = self._aggregate_by_highest_confidence(successful_results)

        # Calculate coverage
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
                "combined_output": aggregated.combined_output,
                "confidence_score": aggregated.confidence_score,
            },
        )

        return aggregated, coverage

    def _get_successful_results(
        self, results: List[DetectorResult]
    ) -> List[DetectorResult]:
        """Filter to only successful detector results."""
        return [r for r in results if r.confidence > 0.0 and r.category != "error"]

    def _aggregate_by_highest_confidence(
        self, results: List[DetectorResult]
    ) -> AggregatedOutput:
        """Aggregate by selecting result with highest confidence."""

        if not results:
            return self._create_empty_aggregation()

        # Find result with highest confidence
        best_result = max(results, key=lambda r: r.confidence)

        # Create aggregated output
        return AggregatedOutput(
            combined_output=f"{best_result.category}:{best_result.severity}",
            confidence_score=best_result.confidence,
            contributing_detectors=[best_result.detector_id],
            metadata={
                "strategy": "highest_confidence",
                "best_detector": best_result.detector_id,
                "best_confidence": best_result.confidence,
                "total_detectors": len(results),
            },
        )

    def _aggregate_by_majority_vote(
        self, results: List[DetectorResult]
    ) -> AggregatedOutput:
        """Aggregate by majority vote of categories."""

        if not results:
            return self._create_empty_aggregation()

        # Count categories
        category_votes = Counter(r.category for r in results)
        most_common_category = category_votes.most_common(1)[0][0]

        # Get results for most common category
        category_results = [r for r in results if r.category == most_common_category]

        # Calculate average confidence for the category
        avg_confidence = sum(r.confidence for r in category_results) / len(
            category_results
        )

        # Get most common severity for the category
        severity_votes = Counter(r.severity for r in category_results)
        most_common_severity = severity_votes.most_common(1)[0][0]

        return AggregatedOutput(
            combined_output=f"{most_common_category}:{most_common_severity}",
            confidence_score=avg_confidence,
            contributing_detectors=[r.detector_id for r in category_results],
            metadata={
                "strategy": "majority_vote",
                "category_votes": dict(category_votes),
                "winning_category": most_common_category,
                "vote_count": category_votes[most_common_category],
                "total_detectors": len(results),
            },
        )

    def _aggregate_by_weighted_average(
        self, results: List[DetectorResult]
    ) -> AggregatedOutput:
        """Aggregate by weighted average of confidences."""

        if not results:
            return self._create_empty_aggregation()

        # Group by category
        category_groups = {}
        for result in results:
            if result.category not in category_groups:
                category_groups[result.category] = []
            category_groups[result.category].append(result)

        # Calculate weighted score for each category
        category_scores = {}
        for category, group_results in category_groups.items():
            total_confidence = sum(r.confidence for r in group_results)
            category_scores[category] = total_confidence / len(
                results
            )  # Normalize by total results

        # Select category with highest weighted score
        best_category = max(category_scores.keys(), key=lambda c: category_scores[c])
        best_results = category_groups[best_category]

        # Calculate overall confidence
        overall_confidence = category_scores[best_category]

        # Get most common severity
        severity_votes = Counter(r.severity for r in best_results)
        most_common_severity = severity_votes.most_common(1)[0][0]

        return AggregatedOutput(
            combined_output=f"{best_category}:{most_common_severity}",
            confidence_score=overall_confidence,
            contributing_detectors=[r.detector_id for r in best_results],
            metadata={
                "strategy": "weighted_average",
                "category_scores": category_scores,
                "winning_category": best_category,
                "weighted_score": overall_confidence,
                "total_detectors": len(results),
            },
        )

    def _create_empty_aggregation(self) -> AggregatedOutput:
        """Create empty aggregation for when no successful results exist."""
        return AggregatedOutput(
            combined_output="none:info",
            confidence_score=0.0,
            contributing_detectors=[],
            metadata={
                "strategy": "empty",
                "reason": "no_successful_results",
                "total_detectors": 0,
            },
        )

    def get_unique_outputs(self, results: List[DetectorResult]) -> List[str]:
        """Get unique outputs from detector results."""
        successful = self._get_successful_results(results)
        outputs = set()
        for result in successful:
            output = f"{result.category}:{result.severity}"
            outputs.add(output)
        return list(outputs)

    def calculate_confidence_distribution(
        self, results: List[DetectorResult]
    ) -> Dict[str, float]:
        """Calculate confidence distribution across results."""
        if not results:
            return {}

        successful = self._get_successful_results(results)
        if not successful:
            return {"no_results": 1.0}

        # Group by confidence ranges
        distribution = {
            "high_confidence": 0,  # > 0.8
            "medium_confidence": 0,  # 0.5 - 0.8
            "low_confidence": 0,  # < 0.5
        }

        for result in successful:
            if result.confidence > 0.8:
                distribution["high_confidence"] += 1
            elif result.confidence >= 0.5:
                distribution["medium_confidence"] += 1
            else:
                distribution["low_confidence"] += 1

        # Normalize to percentages
        total = len(successful)
        return {k: v / total for k, v in distribution.items()}


# Export only the core aggregation functionality
__all__ = [
    "ResponseAggregator",
    "AggregatedOutput",
]

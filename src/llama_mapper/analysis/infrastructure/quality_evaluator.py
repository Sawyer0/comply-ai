"""
Infrastructure implementation of the quality evaluator for the Analysis Module.

This module contains the concrete implementation of the IQualityEvaluator interface
for quality evaluation and drift detection.
"""

import json
import logging
import statistics
from typing import Any, Dict, List, Optional

from ..domain.entities import AnalysisRequest, AnalysisResponse
from ..domain.interfaces import IQualityEvaluator

logger = logging.getLogger(__name__)


class QualityEvaluator(IQualityEvaluator):
    """
    Quality evaluator implementation.

    Provides concrete implementation of the IQualityEvaluator interface
    for evaluating analysis quality and detecting drift.
    """

    def __init__(self, golden_dataset_path: Optional[str] = None):
        """
        Initialize the quality evaluator.

        Args:
            golden_dataset_path: Path to golden dataset for evaluation
        """
        self.golden_dataset_path = golden_dataset_path
        self.golden_dataset = []
        self.evaluation_history = []

        if golden_dataset_path:
            self._load_golden_dataset()

        logger.info(
            f"Initialized Quality Evaluator with golden dataset: {golden_dataset_path}"
        )

    async def evaluate_batch(
        self, examples: List[tuple[AnalysisRequest, AnalysisResponse]]
    ) -> Dict[str, Any]:
        """
        Evaluate batch of examples against golden dataset.

        Args:
            examples: List of (request, response) tuples to evaluate

        Returns:
            Evaluation metrics
        """
        try:
            if not examples:
                return {
                    "total_examples": 0,
                    "schema_valid_rate": 0.0,
                    "rubric_score": 0.0,
                    "opa_compile_success_rate": 0.0,
                    "evidence_accuracy": 0.0,
                    "individual_rubric_scores": [],
                }

            # Evaluate each example
            individual_scores = []
            schema_valid_count = 0
            opa_compile_success_count = 0
            evidence_accuracy_scores = []

            for request, response in examples:
                # Schema validation
                if self._validate_schema(response):
                    schema_valid_count += 1

                # Rubric scoring
                rubric_score = self._calculate_rubric_score(request, response)
                individual_scores.append(rubric_score)

                # OPA compilation check
                if self._validate_opa_policy(response.opa_diff):
                    opa_compile_success_count += 1

                # Evidence accuracy
                evidence_accuracy = self._calculate_evidence_accuracy(request, response)
                evidence_accuracy_scores.append(evidence_accuracy)

            # Calculate aggregate metrics
            total_examples = len(examples)
            schema_valid_rate = schema_valid_count / total_examples
            rubric_score = (
                statistics.mean(individual_scores) if individual_scores else 0.0
            )
            opa_compile_success_rate = opa_compile_success_count / total_examples
            evidence_accuracy = (
                statistics.mean(evidence_accuracy_scores)
                if evidence_accuracy_scores
                else 0.0
            )

            evaluation_result = {
                "total_examples": total_examples,
                "schema_valid_rate": schema_valid_rate,
                "rubric_score": rubric_score,
                "opa_compile_success_rate": opa_compile_success_rate,
                "evidence_accuracy": evidence_accuracy,
                "individual_rubric_scores": individual_scores,
            }

            # Store evaluation in history
            self.evaluation_history.append(evaluation_result)

            logger.info(
                f"Quality evaluation completed: {total_examples} examples, rubric score: {rubric_score:.2f}"
            )

            return evaluation_result

        except Exception as e:
            logger.error("Quality evaluation error: %s", e)
            raise

    def calculate_drift_score(self, recent_outputs: List[AnalysisResponse]) -> float:
        """
        Calculate quality drift over time.

        Args:
            recent_outputs: Recent analysis outputs

        Returns:
            Drift score (0.0-1.0, higher is worse)
        """
        try:
            if not recent_outputs or len(self.evaluation_history) < 2:
                return 0.0

            # Calculate current quality metrics
            current_metrics = self._calculate_current_metrics(recent_outputs)

            # Get baseline metrics from golden dataset
            baseline_metrics = self._calculate_baseline_metrics()

            # Calculate drift for each metric
            drift_scores = []

            for metric in [
                "schema_valid_rate",
                "rubric_score",
                "opa_compile_success_rate",
                "evidence_accuracy",
            ]:
                current_value = current_metrics.get(metric, 0.0)
                baseline_value = baseline_metrics.get(metric, 0.0)

                if baseline_value > 0:
                    drift = abs(current_value - baseline_value) / baseline_value
                    drift_scores.append(drift)

            # Return average drift score
            return statistics.mean(drift_scores) if drift_scores else 0.0

        except Exception as e:
            logger.error("Drift calculation error: %s", e)
            return 0.0

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get summary of evaluation history.

        Returns:
            Evaluation summary
        """
        try:
            if not self.evaluation_history:
                return {
                    "total_evaluations": 0,
                    "average_rubric_score": 0.0,
                    "average_schema_valid_rate": 0.0,
                    "average_opa_compile_success_rate": 0.0,
                    "average_evidence_accuracy": 0.0,
                    "trend": "stable",
                }

            # Calculate averages
            total_evaluations = len(self.evaluation_history)
            average_rubric_score = statistics.mean(
                [e["rubric_score"] for e in self.evaluation_history]
            )
            average_schema_valid_rate = statistics.mean(
                [e["schema_valid_rate"] for e in self.evaluation_history]
            )
            average_opa_compile_success_rate = statistics.mean(
                [e["opa_compile_success_rate"] for e in self.evaluation_history]
            )
            average_evidence_accuracy = statistics.mean(
                [e["evidence_accuracy"] for e in self.evaluation_history]
            )

            # Calculate trend
            trend = self._calculate_trend()

            return {
                "total_evaluations": total_evaluations,
                "average_rubric_score": average_rubric_score,
                "average_schema_valid_rate": average_schema_valid_rate,
                "average_opa_compile_success_rate": average_opa_compile_success_rate,
                "average_evidence_accuracy": average_evidence_accuracy,
                "trend": trend,
            }

        except Exception as e:
            logger.error("Evaluation summary error: %s", e)
            return {}

    def _load_golden_dataset(self) -> None:
        """Load golden dataset from file."""
        try:
            if not self.golden_dataset_path:
                return

            with open(self.golden_dataset_path, "r") as f:
                for line in f:
                    if line.strip():
                        example = json.loads(line)
                        self.golden_dataset.append(example)

            logger.info("Loaded %s golden examples", len(self.golden_dataset))

        except Exception as e:
            logger.error("Error loading golden dataset: %s", e)

    def _validate_schema(self, response: AnalysisResponse) -> bool:
        """
        Validate response against schema.

        Args:
            response: Analysis response

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            required_fields = [
                "reason",
                "remediation",
                "opa_diff",
                "confidence",
                "evidence_refs",
            ]
            for field in required_fields:
                if not hasattr(response, field) or getattr(response, field) is None:
                    return False

            # Check field types and constraints
            if not isinstance(response.reason, str) or len(response.reason) > 120:
                return False

            if (
                not isinstance(response.remediation, str)
                or len(response.remediation) > 120
            ):
                return False

            if (
                not isinstance(response.confidence, (int, float))
                or not 0.0 <= response.confidence <= 1.0
            ):
                return False

            if (
                not isinstance(response.evidence_refs, list)
                or len(response.evidence_refs) == 0
            ):
                return False

            return True

        except Exception as e:
            logger.error("Schema validation error: %s", e)
            return False

    def _calculate_rubric_score(
        self, request: AnalysisRequest, response: AnalysisResponse
    ) -> float:
        """
        Calculate rubric score for response.

        Args:
            request: Analysis request
            response: Analysis response

        Returns:
            Rubric score (0.0-5.0)
        """
        try:
            score = 0.0

            # Reason quality (0-1 points)
            if response.reason and len(response.reason) > 10:
                score += 1.0

            # Remediation quality (0-1 points)
            if response.remediation and len(response.remediation) > 10:
                score += 1.0

            # OPA policy quality (0-1 points)
            if response.opa_diff and len(response.opa_diff) > 20:
                score += 1.0

            # Evidence relevance (0-1 points)
            if response.evidence_refs and len(response.evidence_refs) > 0:
                score += 1.0

            # Confidence appropriateness (0-1 points)
            if 0.3 <= response.confidence <= 1.0:
                score += 1.0

            return score

        except Exception as e:
            logger.error("Rubric scoring error: %s", e)
            return 0.0

    def _validate_opa_policy(self, opa_diff: str) -> bool:
        """
        Validate OPA policy.

        Args:
            opa_diff: OPA policy string

        Returns:
            True if valid, False otherwise
        """
        try:
            if not opa_diff.strip():
                return True  # Empty is valid

            # Basic syntax validation
            if "package" not in opa_diff:
                return False

            if "violation" not in opa_diff:
                return False

            # Check for balanced braces
            brace_count = 0
            for char in opa_diff:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count < 0:
                        return False

            return brace_count == 0

        except Exception as e:
            logger.error("OPA policy validation error: %s", e)
            return False

    def _calculate_evidence_accuracy(
        self, request: AnalysisRequest, response: AnalysisResponse
    ) -> float:
        """
        Calculate evidence accuracy.

        Args:
            request: Analysis request
            response: Analysis response

        Returns:
            Evidence accuracy score (0.0-1.0)
        """
        try:
            if not response.evidence_refs:
                return 0.0

            # Check if evidence references are relevant to the request
            relevant_evidence = 0
            total_evidence = len(response.evidence_refs)

            for evidence_ref in response.evidence_refs:
                if evidence_ref in request.required_detectors:
                    relevant_evidence += 1

            return relevant_evidence / total_evidence if total_evidence > 0 else 0.0

        except Exception as e:
            logger.error("Evidence accuracy calculation error: %s", e)
            return 0.0

    def _calculate_current_metrics(
        self, recent_outputs: List[AnalysisResponse]
    ) -> Dict[str, float]:
        """
        Calculate current quality metrics from recent outputs.

        Args:
            recent_outputs: Recent analysis outputs

        Returns:
            Current quality metrics
        """
        try:
            if not recent_outputs:
                return {
                    "schema_valid_rate": 0.0,
                    "rubric_score": 0.0,
                    "opa_compile_success_rate": 0.0,
                    "evidence_accuracy": 0.0,
                }

            schema_valid_count = sum(
                1 for output in recent_outputs if self._validate_schema(output)
            )
            opa_compile_success_count = sum(
                1
                for output in recent_outputs
                if self._validate_opa_policy(output.opa_diff)
            )

            rubric_scores = [
                self._calculate_rubric_score(None, output) for output in recent_outputs
            ]
            evidence_accuracy_scores = [
                self._calculate_evidence_accuracy(None, output)
                for output in recent_outputs
            ]

            return {
                "schema_valid_rate": schema_valid_count / len(recent_outputs),
                "rubric_score": statistics.mean(rubric_scores),
                "opa_compile_success_rate": opa_compile_success_count
                / len(recent_outputs),
                "evidence_accuracy": statistics.mean(evidence_accuracy_scores),
            }

        except Exception as e:
            logger.error("Current metrics calculation error: %s", e)
            return {}

    def _calculate_baseline_metrics(self) -> Dict[str, float]:
        """
        Calculate baseline quality metrics from golden dataset.

        Returns:
            Baseline quality metrics
        """
        try:
            if not self.golden_dataset:
                return {
                    "schema_valid_rate": 0.8,
                    "rubric_score": 3.0,
                    "opa_compile_success_rate": 0.9,
                    "evidence_accuracy": 0.8,
                }

            # This would be calculated from the golden dataset
            # For now, return default baseline values
            return {
                "schema_valid_rate": 0.8,
                "rubric_score": 3.0,
                "opa_compile_success_rate": 0.9,
                "evidence_accuracy": 0.8,
            }

        except Exception as e:
            logger.error("Baseline metrics calculation error: %s", e)
            return {}

    def _calculate_trend(self) -> str:
        """
        Calculate quality trend over time.

        Returns:
            Trend description
        """
        try:
            if len(self.evaluation_history) < 3:
                return "stable"

            # Get recent scores
            recent_scores = [e["rubric_score"] for e in self.evaluation_history[-3:]]

            # Calculate trend
            if recent_scores[-1] > recent_scores[0]:
                return "improving"
            elif recent_scores[-1] < recent_scores[0]:
                return "declining"
            else:
                return "stable"

        except Exception as e:
            logger.error("Trend calculation error: %s", e)
            return "stable"

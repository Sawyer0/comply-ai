"""
Quality gates for CI/CD pipeline with automated testing and threshold validation.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..data.detectors import DetectorConfigLoader
from ..data.taxonomy import TaxonomyLoader
from ..serving.fallback_mapper import FallbackMapper
from ..serving.json_validator import JSONValidator
from ..serving.model_server import ModelServer
from .metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class GoldenTestCase:
    """A golden test case for quality validation."""

    detector: str
    input_output: str
    expected_taxonomy: List[str]
    expected_confidence_min: float
    description: str
    category: str  # For coverage tracking


@dataclass
class QualityGateResult:
    """Result of quality gate evaluation."""

    passed: bool
    metric_name: str
    actual_value: float
    threshold: float
    message: str
    severity: str = "error"


class QualityGateValidator:
    """
    Validates quality gates for CI/CD pipeline.

    Implements automated testing with golden test cases (≥100 per detector),
    schema validation, taxonomy F1, and latency requirements.
    """

    def __init__(
        self,
        model_server: Optional[ModelServer] = None,
        json_validator: Optional[JSONValidator] = None,
        fallback_mapper: Optional[FallbackMapper] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        golden_test_cases_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        environment: str = "production",
    ):
        """
        Initialize quality gate validator.

        Args:
            model_server: Model serving backend
            json_validator: JSON schema validator
            fallback_mapper: Rule-based fallback mapper
            metrics_collector: Metrics collection service
            golden_test_cases_path: Path to golden test cases file
            config_path: Path to quality gates configuration file
            environment: Environment name (development, staging, production)
        """
        self.model_server = model_server
        self.json_validator = json_validator
        self.fallback_mapper = fallback_mapper
        self.metrics_collector = metrics_collector
        self.golden_test_cases_path = golden_test_cases_path or Path(
            "tests/golden_test_cases.json"
        )
        self.config_path = config_path or Path("config/quality_gates.yaml")
        self.environment = environment

        # Load configuration and set thresholds
        self.thresholds = self._load_thresholds()

    def _load_thresholds(self) -> Dict[str, float]:
        """Load quality gate thresholds from configuration file."""
        default_thresholds = {
            "schema_valid_percentage": 95.0,
            "taxonomy_f1_score": 90.0,
            "latency_p95_cpu_ms": 250.0,
            "latency_p95_gpu_ms": 120.0,
            "fallback_percentage": 10.0,
            "min_golden_cases_per_detector": 100,
            "taxonomy_coverage_percentage": 80.0,
        }

        if not self.config_path.exists():
            logger.warning(
                f"Quality gates config not found: {self.config_path}, using defaults"
            )
            return default_thresholds

        try:
            import yaml

            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)

            # Get environment-specific thresholds
            environments = config.get("environments", {})
            env_config = environments.get(self.environment)

            if not env_config:
                logger.warning(
                    f"Environment '{self.environment}' not found in config, using defaults"
                )
                return default_thresholds

            thresholds = env_config.get("thresholds", {})

            # Merge with defaults
            final_thresholds = default_thresholds.copy()
            final_thresholds.update(thresholds)

            logger.info(
                f"Loaded quality gate thresholds for environment '{self.environment}'"
            )
            return final_thresholds

        except Exception as e:
            logger.error("Failed to load quality gates config: %s", e)
            return default_thresholds

    def load_golden_test_cases(self) -> List[GoldenTestCase]:
        """
        Load golden test cases from file.

        Returns:
            List[GoldenTestCase]: Loaded test cases
        """
        if not self.golden_test_cases_path.exists():
            logger.warning(
                f"Golden test cases file not found: {self.golden_test_cases_path}"
            )
            return []

        try:
            with open(self.golden_test_cases_path, "r") as f:
                data = json.load(f)

            test_cases = []
            for case_data in data.get("test_cases", []):
                test_cases.append(
                    GoldenTestCase(
                        detector=case_data["detector"],
                        input_output=case_data["input_output"],
                        expected_taxonomy=case_data["expected_taxonomy"],
                        expected_confidence_min=case_data.get(
                            "expected_confidence_min", 0.6
                        ),
                        description=case_data.get("description", ""),
                        category=case_data.get("category", "unknown"),
                    )
                )

            logger.info("Loaded %s golden test cases", len(test_cases))
            return test_cases

        except Exception as e:
            logger.error("Failed to load golden test cases: %s", e)
            return []

    def validate_golden_test_coverage(
        self, test_cases: List[GoldenTestCase]
    ) -> List[QualityGateResult]:
        """
        Validate that we have sufficient golden test cases per detector.

        Args:
            test_cases: List of golden test cases

        Returns:
            List[QualityGateResult]: Coverage validation results
        """
        results: List[QualityGateResult] = []

        # Count test cases per detector
        detector_counts: Dict[str, int] = {}
        for case in test_cases:
            detector_counts[case.detector] = detector_counts.get(case.detector, 0) + 1

        # Load available detectors
        try:
            detector_loader = DetectorConfigLoader()
            detector_mappings = detector_loader.load_all_detector_configs()
            available_detectors = list(detector_mappings.keys())
        except Exception as e:
            logger.error("Failed to load detector configs: %s", e)
            available_detectors = list(detector_counts.keys())

        # Check coverage per detector
        min_cases = self.thresholds["min_golden_cases_per_detector"]
        for detector in available_detectors:
            count = detector_counts.get(detector, 0)
            passed = count >= min_cases

            results.append(
                QualityGateResult(
                    passed=passed,
                    metric_name=f"golden_cases_{detector}",
                    actual_value=count,
                    threshold=min_cases,
                    message=f"Detector {detector} has {count} golden test cases (minimum: {min_cases})",
                    severity="error" if not passed else "info",
                )
            )

        # Check taxonomy category coverage
        taxonomy_loader = TaxonomyLoader()
        try:
            taxonomy = taxonomy_loader.load_taxonomy()
            categories = set()
            for label in taxonomy.get_all_labels():
                # Extract top-level category (e.g., "HARM" from "HARM.SPEECH.Toxicity")
                category = label.name.split(".")[0]
                categories.add(category)

            covered_categories = set(case.category for case in test_cases)
            coverage_pct = (
                (len(covered_categories) / len(categories)) * 100 if categories else 0
            )

            passed = coverage_pct >= self.thresholds["taxonomy_coverage_percentage"]
            results.append(
                QualityGateResult(
                    passed=passed,
                    metric_name="taxonomy_coverage_percentage",
                    actual_value=coverage_pct,
                    threshold=self.thresholds["taxonomy_coverage_percentage"],
                    message=f"Taxonomy coverage: {coverage_pct:.1f}% ({len(covered_categories)}/{len(categories)} categories)",
                    severity="warning" if not passed else "info",
                )
            )

        except Exception as e:
            logger.error("Failed to check taxonomy coverage: %s", e)

        return results

    async def run_golden_test_cases(
        self, test_cases: List[GoldenTestCase]
    ) -> Tuple[List[QualityGateResult], Dict[str, Any]]:
        """
        Run golden test cases and validate results.

        Args:
            test_cases: List of golden test cases to run

        Returns:
            Tuple[List[QualityGateResult], Dict[str, Any]]: Results and detailed metrics
        """
        results: List[QualityGateResult] = []
        detailed_metrics: Dict[str, Any] = {
            "total_cases": len(test_cases),
            "passed_cases": 0,
            "failed_cases": 0,
            "schema_valid_cases": 0,
            "latency_measurements": [],
            "f1_scores_by_detector": {},
            "confidence_scores": [],
        }

        if not self.model_server or not self.json_validator:
            results.append(
                QualityGateResult(
                    passed=False,
                    metric_name="test_execution",
                    actual_value=0,
                    threshold=1,
                    message="Model server or JSON validator not available for testing",
                    severity="error",
                )
            )
            return results, detailed_metrics

        # Run each test case
        for i, case in enumerate(test_cases):
            try:
                start_time = time.time()

                # Generate mapping
                model_output = await self.model_server.generate_mapping(
                    detector=case.detector, output=case.input_output, metadata={}
                )

                processing_time = time.time() - start_time
                detailed_metrics["latency_measurements"].append(
                    processing_time * 1000
                )  # Convert to ms

                # Validate schema
                is_valid, validation_errors = self.json_validator.validate(model_output)
                if is_valid:
                    detailed_metrics["schema_valid_cases"] += 1

                    # Parse output
                    parsed_output = self.json_validator.parse_output(model_output)
                    detailed_metrics["confidence_scores"].append(
                        parsed_output.confidence
                    )

                    # Check if prediction matches expected taxonomy
                    predicted_labels = set(parsed_output.taxonomy)
                    expected_labels = set(case.expected_taxonomy)

                    # Calculate F1 score for this case
                    if expected_labels:
                        intersection = predicted_labels & expected_labels
                        precision = (
                            len(intersection) / len(predicted_labels)
                            if predicted_labels
                            else 0
                        )
                        recall = len(intersection) / len(expected_labels)
                        f1 = (
                            2 * precision * recall / (precision + recall)
                            if (precision + recall) > 0
                            else 0
                        )

                        # Track F1 by detector
                        if (
                            case.detector
                            not in detailed_metrics["f1_scores_by_detector"]
                        ):
                            detailed_metrics["f1_scores_by_detector"][
                                case.detector
                            ] = []
                        detailed_metrics["f1_scores_by_detector"][case.detector].append(
                            f1
                        )

                        # Check confidence threshold
                        confidence_ok = (
                            parsed_output.confidence >= case.expected_confidence_min
                        )

                        if (
                            f1 >= 0.9 and confidence_ok
                        ):  # Both F1 and confidence thresholds met
                            detailed_metrics["passed_cases"] += 1
                        else:
                            detailed_metrics["failed_cases"] += 1
                            logger.warning(
                                f"Test case {i} failed: F1={f1:.3f}, confidence={parsed_output.confidence:.3f}"
                            )
                    else:
                        detailed_metrics["failed_cases"] += 1
                        logger.warning("Test case %s has no expected labels", i)
                else:
                    detailed_metrics["failed_cases"] += 1
                    logger.warning(
                        f"Test case {i} failed schema validation: {validation_errors}"
                    )

            except Exception as e:
                detailed_metrics["failed_cases"] += 1
                logger.error("Test case %s failed with exception: %s", i, e)

        # Calculate overall metrics
        if detailed_metrics["total_cases"] > 0:
            schema_valid_pct = (
                detailed_metrics["schema_valid_cases"] / detailed_metrics["total_cases"]
            ) * 100
            pass_rate = (
                detailed_metrics["passed_cases"] / detailed_metrics["total_cases"]
            ) * 100

            # Schema validation gate
            results.append(
                QualityGateResult(
                    passed=schema_valid_pct
                    >= self.thresholds["schema_valid_percentage"],
                    metric_name="schema_valid_percentage",
                    actual_value=schema_valid_pct,
                    threshold=self.thresholds["schema_valid_percentage"],
                    message=f"Schema validation rate: {schema_valid_pct:.1f}%",
                )
            )

            # Overall pass rate
            results.append(
                QualityGateResult(
                    passed=pass_rate >= 90.0,  # Expect 90% of golden cases to pass
                    metric_name="golden_test_pass_rate",
                    actual_value=pass_rate,
                    threshold=90.0,
                    message=f"Golden test pass rate: {pass_rate:.1f}%",
                )
            )

        # F1 score gates per detector
        for detector, f1_scores in detailed_metrics["f1_scores_by_detector"].items():
            if f1_scores:
                avg_f1 = sum(f1_scores) / len(f1_scores)
                results.append(
                    QualityGateResult(
                        passed=avg_f1 >= (self.thresholds["taxonomy_f1_score"] / 100),
                        metric_name=f"f1_score_{detector}",
                        actual_value=avg_f1,
                        threshold=self.thresholds["taxonomy_f1_score"] / 100,
                        message=f"F1 score for {detector}: {avg_f1:.3f}",
                    )
                )

        # Latency gates
        if detailed_metrics["latency_measurements"]:
            latencies = sorted(detailed_metrics["latency_measurements"])
            p95_latency = latencies[int(0.95 * len(latencies))] if latencies else 0

            # Assume CPU deployment for now (could be parameterized)
            threshold = self.thresholds["latency_p95_cpu_ms"]
            results.append(
                QualityGateResult(
                    passed=p95_latency <= threshold,
                    metric_name="latency_p95_ms",
                    actual_value=p95_latency,
                    threshold=threshold,
                    message=f"P95 latency: {p95_latency:.1f}ms",
                )
            )

        return results, detailed_metrics

    async def validate_all_quality_gates(
        self,
    ) -> Tuple[bool, List[QualityGateResult], Dict[str, Any]]:
        """
        Run all quality gate validations.

        Returns:
            Tuple[bool, List[QualityGateResult], Dict[str, Any]]: Overall pass status, results, and metrics
        """
        all_results = []
        all_metrics = {}

        logger.info("Starting quality gate validation...")

        # Load and validate golden test cases
        test_cases = self.load_golden_test_cases()
        coverage_results = self.validate_golden_test_coverage(test_cases)
        all_results.extend(coverage_results)

        # Run golden test cases if available
        if test_cases:
            test_results, test_metrics = await self.run_golden_test_cases(test_cases)
            all_results.extend(test_results)
            all_metrics.update(test_metrics)

        # Check current metrics if metrics collector is available
        if self.metrics_collector:
            current_alerts = self.metrics_collector.check_quality_thresholds()
            for alert in current_alerts:
                all_results.append(
                    QualityGateResult(
                        passed=False,
                        metric_name=alert["metric"],
                        actual_value=alert["value"],
                        threshold=alert["threshold"],
                        message=alert["message"],
                        severity=alert["severity"],
                    )
                )

        # Determine overall pass status
        critical_failures = [
            r for r in all_results if not r.passed and r.severity == "error"
        ]
        overall_passed = len(critical_failures) == 0

        logger.info(
            f"Quality gate validation completed: {'PASSED' if overall_passed else 'FAILED'}"
        )
        logger.info(
            f"Total checks: {len(all_results)}, Failed: {len([r for r in all_results if not r.passed])}"
        )

        return overall_passed, all_results, all_metrics

    def generate_quality_report(
        self, results: List[QualityGateResult], metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive quality report.

        Args:
            results: Quality gate results
            metrics: Detailed metrics

        Returns:
            Dict[str, Any]: Quality report
        """
        passed_results = [r for r in results if r.passed]
        failed_results = [r for r in results if not r.passed]

        report = {
            "timestamp": time.time(),
            "overall_status": "PASSED" if len(failed_results) == 0 else "FAILED",
            "summary": {
                "total_checks": len(results),
                "passed_checks": len(passed_results),
                "failed_checks": len(failed_results),
                "critical_failures": len(
                    [r for r in failed_results if r.severity == "error"]
                ),
            },
            "results": [
                {
                    "metric": r.metric_name,
                    "status": "PASSED" if r.passed else "FAILED",
                    "actual_value": r.actual_value,
                    "threshold": r.threshold,
                    "message": r.message,
                    "severity": r.severity,
                }
                for r in results
            ],
            "detailed_metrics": metrics,
            "recommendations": self._generate_recommendations(failed_results),
        }

        return report

    def _generate_recommendations(
        self, failed_results: List[QualityGateResult]
    ) -> List[str]:
        """Generate recommendations based on failed quality gates."""
        recommendations = []

        for result in failed_results:
            if "schema_valid" in result.metric_name:
                recommendations.append(
                    "Improve model training data quality and JSON schema validation"
                )
            elif "f1_score" in result.metric_name:
                recommendations.append(
                    f"Review training data for {result.metric_name.split('_')[-1]} detector"
                )
            elif "latency" in result.metric_name:
                recommendations.append(
                    "Consider model quantization or hardware optimization"
                )
            elif "golden_cases" in result.metric_name:
                recommendations.append(
                    f"Add more golden test cases for {result.metric_name.split('_')[-1]} detector"
                )
            elif "taxonomy_coverage" in result.metric_name:
                recommendations.append(
                    "Expand golden test cases to cover more taxonomy categories"
                )

        return list(set(recommendations))  # Remove duplicates


def create_sample_golden_test_cases(output_path: Path) -> None:
    """
    Create a sample golden test cases file for reference.

    Args:
        output_path: Path where to create the sample file
    """
    sample_cases = {
        "version": "1.0",
        "description": "Golden test cases for quality gate validation",
        "test_cases": [
            {
                "detector": "deberta-toxicity",
                "input_output": "toxic",
                "expected_taxonomy": ["HARM.SPEECH.Toxicity"],
                "expected_confidence_min": 0.8,
                "description": "Basic toxicity detection",
                "category": "HARM",
            },
            {
                "detector": "openai-moderation",
                "input_output": "hate",
                "expected_taxonomy": ["HARM.SPEECH.Hate.Other"],
                "expected_confidence_min": 0.7,
                "description": "Hate speech detection",
                "category": "HARM",
            },
            {
                "detector": "regex-pii",
                "input_output": "ssn",
                "expected_taxonomy": ["PII.Identifier.SSN"],
                "expected_confidence_min": 0.9,
                "description": "SSN detection",
                "category": "PII",
            },
        ],
    }

    with open(output_path, "w") as f:
        json.dump(sample_cases, f, indent=2)

    logger.info("Created sample golden test cases at %s", output_path)

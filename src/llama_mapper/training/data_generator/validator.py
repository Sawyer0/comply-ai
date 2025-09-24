"""Validation utilities for generated training datasets."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Union

from llama_mapper.data.detectors import DetectorConfigLoader, DetectorMapping
from llama_mapper.data.taxonomy import Taxonomy, TaxonomyLoader

from .models import TrainingExample

logger = logging.getLogger(__name__)


class DatasetValidator:
    """Validates training data quality and consistency."""

    def __init__(
        self,
        taxonomy_loader: Optional[TaxonomyLoader] = None,
        detector_loader: Optional[DetectorConfigLoader] = None,
        schema_path: Optional[Union[str, Path]] = None,
    ):
        self.taxonomy_loader = taxonomy_loader or TaxonomyLoader()
        self.detector_loader = detector_loader or DetectorConfigLoader()
        self.schema_path = (
            Path(schema_path) if schema_path else Path("pillars-detectors/schema.json")
        )

        self._taxonomy: Optional[Taxonomy] = None
        self._detector_mappings: Dict[str, DetectorMapping] = {}
        self._json_schema: Optional[dict] = None

    def load_dependencies(self) -> None:
        """Load taxonomy, detector configurations, and JSON schema."""
        logger.info("Loading validation dependencies...")

        self._taxonomy = self.taxonomy_loader.load_taxonomy()
        self._detector_mappings = self.detector_loader.load_all_detector_configs()

        if self.schema_path.exists():
            with open(self.schema_path, "r", encoding="utf-8") as file:
                self._json_schema = json.load(file)
        else:
            logger.warning("JSON schema file not found: %s", self.schema_path)

    def validate_training_dataset(
        self, examples: List[TrainingExample]
    ) -> Dict[str, Any]:
        """Comprehensive validation of training dataset."""
        if not self._taxonomy or not self._detector_mappings:
            self.load_dependencies()

        logger.info("Validating training dataset with %s examples...", len(examples))

        validation_report: Dict[str, Any] = {
            "dataset_info": {
                "total_examples": len(examples),
                "validation_timestamp": datetime.now().isoformat(),
            },
            "format_validation": self._validate_instruction_response_format(examples),
            "taxonomy_validation": self._validate_taxonomy_labels(examples),
            "schema_validation": self._validate_json_schema(examples),
            "coverage_analysis": self._analyze_coverage(examples),
            "quality_metrics": self._calculate_quality_metrics(examples),
            "consistency_checks": self._check_consistency(examples),
            "recommendations": [],
        }

        validation_report["recommendations"] = self._generate_recommendations(
            validation_report
        )
        validation_report["overall_score"] = self._calculate_overall_score(
            validation_report
        )

        logger.info(
            "Dataset validation completed. Overall score: %.2f/100",  # noqa: S001
            validation_report["overall_score"],
        )

        return validation_report

    def _validate_instruction_response_format(
        self, examples: List[TrainingExample]
    ) -> Dict[str, Any]:
        """Validate instruction-response format consistency."""
        logger.debug("Validating instruction-response format...")

        format_validation: Dict[str, Any] = {
            "valid_examples": 0,
            "invalid_examples": 0,
            "errors": [],
            "warnings": [],
            "format_patterns": {
                "instruction_lengths": [],
                "response_formats": {},
                "missing_fields": [],
            },
        }

        for index, example in enumerate(examples):
            try:
                if not example.instruction or not isinstance(example.instruction, str):
                    format_validation["errors"].append(
                        "Example %s: Invalid or missing instruction" % index
                    )
                    format_validation["invalid_examples"] += 1
                    continue

                if not example.response or not isinstance(example.response, str):
                    format_validation["errors"].append(
                        "Example %s: Invalid or missing response" % index
                    )
                    format_validation["invalid_examples"] += 1
                    continue

                try:
                    response_data = json.loads(example.response)

                    required_fields = ["taxonomy", "scores", "confidence"]
                    missing_fields = [
                        field for field in required_fields if field not in response_data
                    ]

                    if missing_fields:
                        format_validation["errors"].append(
                            "Example %s: Missing required fields: %s"
                            % (index, missing_fields)
                        )
                        format_validation["format_patterns"]["missing_fields"].extend(
                            missing_fields
                        )
                        format_validation["invalid_examples"] += 1
                        continue

                    response_type = "standard"
                    if "provenance" in response_data:
                        response_type = "with_provenance"
                    if "notes" in response_data:
                        response_type = "with_notes"

                    patterns = format_validation["format_patterns"]["response_formats"]
                    patterns[response_type] = patterns.get(response_type, 0) + 1

                except json.JSONDecodeError as exc:
                    format_validation["errors"].append(
                        "Example %s: Invalid JSON in response: %s" % (index, exc)
                    )
                    format_validation["invalid_examples"] += 1
                    continue

                format_validation["format_patterns"]["instruction_lengths"].append(
                    len(example.instruction)
                )

                if not isinstance(example.metadata, dict):
                    format_validation["warnings"].append(
                        "Example %s: Metadata is not a dictionary" % index
                    )

                format_validation["valid_examples"] += 1

            except Exception as exc:  # pragma: no cover - defensive logging
                format_validation["errors"].append(
                    "Example %s: Unexpected validation error: %s" % (index, exc)
                )
                format_validation["invalid_examples"] += 1

        if format_validation["format_patterns"]["instruction_lengths"]:
            lengths = format_validation["format_patterns"]["instruction_lengths"]
            sorted_lengths = sorted(lengths)
            format_validation["format_patterns"]["instruction_stats"] = {
                "min_length": min(sorted_lengths),
                "max_length": max(sorted_lengths),
                "avg_length": sum(sorted_lengths) / len(sorted_lengths),
                "median_length": sorted_lengths[len(sorted_lengths) // 2],
            }

        return format_validation

    def _validate_taxonomy_labels(
        self, examples: List[TrainingExample]
    ) -> Dict[str, Any]:
        """Validate that all target labels exist in taxonomy."""
        logger.debug("Validating taxonomy labels...")

        taxonomy_validation: Dict[str, Any] = {
            "valid_labels": 0,
            "invalid_labels": 0,
            "unknown_labels": set(),
            "label_usage": {},
            "category_distribution": {},
            "errors": [],
        }

        assert self._taxonomy is not None

        for index, example in enumerate(examples):
            try:
                response_data = json.loads(example.response)
                taxonomy_labels = response_data.get("taxonomy", [])

                for label in taxonomy_labels:
                    if self._taxonomy.validate_label_name(label):
                        taxonomy_validation["valid_labels"] += 1
                        taxonomy_validation["label_usage"][label] = (
                            taxonomy_validation["label_usage"].get(label, 0) + 1
                        )

                        category = label.split(".")[0] if "." in label else "OTHER"
                        taxonomy_validation["category_distribution"][category] = (
                            taxonomy_validation["category_distribution"].get(
                                category, 0
                            )
                            + 1
                        )
                    else:
                        taxonomy_validation["invalid_labels"] += 1
                        taxonomy_validation["unknown_labels"].add(label)
                        taxonomy_validation["errors"].append(
                            "Example %s: Unknown taxonomy label: %s" % (index, label)
                        )

            except (json.JSONDecodeError, KeyError) as exc:
                taxonomy_validation["errors"].append(
                    "Example %s: Failed to parse taxonomy labels: %s" % (index, exc)
                )

        taxonomy_validation["unknown_labels"] = sorted(
            list(taxonomy_validation["unknown_labels"])
        )

        return taxonomy_validation

    def _validate_json_schema(self, examples: List[TrainingExample]) -> Dict[str, Any]:
        """Validate responses against JSON schema."""
        logger.debug("Validating JSON schema compliance...")

        schema_validation: Dict[str, Any] = {
            "validated_examples": 0,
            "schema_errors": 0,
            "errors": [],
        }

        if not self._json_schema:
            schema_validation["warnings"] = ["JSON schema not loaded"]
            return schema_validation

        for index, example in enumerate(examples):
            try:
                response_data = json.loads(example.response)

                required_fields = self._json_schema.get("required", [])
                missing_fields = [
                    field for field in required_fields if field not in response_data
                ]

                if missing_fields:
                    schema_validation["schema_errors"] += 1
                    schema_validation["errors"].append(
                        "Example %s: Missing required fields according to schema: %s"
                        % (index, missing_fields)
                    )

                schema_validation["validated_examples"] += 1

            except json.JSONDecodeError as exc:
                schema_validation["schema_errors"] += 1
                schema_validation["errors"].append(
                    "Example %s: Invalid JSON for schema validation: %s" % (index, exc)
                )

        return schema_validation

    def _analyze_coverage(self, examples: List[TrainingExample]) -> Dict[str, Any]:
        """Analyze coverage across detectors and taxonomy categories."""
        logger.debug("Analyzing coverage and balance metrics...")

        coverage_analysis: Dict[str, Any] = {
            "detector_coverage": {},
            "taxonomy_coverage": {},
            "coverage_statistics": {},
        }

        detector_counts: Dict[str, int] = {}
        taxonomy_counts: Dict[str, int] = {}
        category_counts: Dict[str, int] = {}

        for example in examples:
            metadata = example.metadata
            detector = metadata.get("detector", "unknown")
            detector_counts[detector] = detector_counts.get(detector, 0) + 1

            try:
                response_data = json.loads(example.response)
                taxonomy_labels = response_data.get("taxonomy", [])

                for label in taxonomy_labels:
                    taxonomy_counts[label] = taxonomy_counts.get(label, 0) + 1
                    category = label.split(".")[0] if "." in label else "OTHER"
                    category_counts[category] = category_counts.get(category, 0) + 1

            except (json.JSONDecodeError, KeyError):
                continue

        coverage_analysis["detector_coverage"] = detector_counts
        coverage_analysis["taxonomy_coverage"] = taxonomy_counts

        total_examples = len(examples)
        unique_detectors = len(detector_counts)
        unique_taxonomy_labels = len(taxonomy_counts)
        unique_categories = len(category_counts)

        coverage_analysis["coverage_statistics"] = {
            "total_examples": total_examples,
            "unique_detectors": unique_detectors,
            "unique_taxonomy_labels": unique_taxonomy_labels,
            "unique_categories": unique_categories,
            "detector_coverage_percentage": (
                unique_detectors / len(self._detector_mappings) * 100
                if self._detector_mappings
                else 0
            ),
            "taxonomy_coverage_percentage": (
                unique_taxonomy_labels / len(self._taxonomy.get_all_labels()) * 100
                if self._taxonomy
                else 0
            ),
            "category_distribution": category_counts,
            "covered_categories": sum(
                1 for count in category_counts.values() if count > 0
            ),
            "total_categories": len(category_counts),
        }

        coverage_analysis["balance_scores"] = self._calculate_balance_scores(
            detector_counts, category_counts
        )

        return coverage_analysis

    def _calculate_quality_metrics(
        self, examples: List[TrainingExample]
    ) -> Dict[str, Any]:
        """Calculate qualitative metrics for the dataset."""
        quality_metrics: Dict[str, Any] = {
            "confidence_distribution": {
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "std_dev": None,
            },
            "metadata_completeness": {
                "has_detector": 0,
                "has_example_type": 0,
                "has_confidence": 0,
            },
            "example_type_distribution": {},
            "detector_balance": {},
        }

        confidences: List[float] = []
        detector_counts: Dict[str, int] = {}
        example_type_counts: Dict[str, int] = {}

        for example in examples:
            metadata = example.metadata
            detector = metadata.get("detector")
            example_type = metadata.get("example_type")

            if detector:
                quality_metrics["metadata_completeness"]["has_detector"] += 1
                detector_counts[detector] = detector_counts.get(detector, 0) + 1

            if example_type:
                quality_metrics["metadata_completeness"]["has_example_type"] += 1
                example_type_counts[example_type] = (
                    example_type_counts.get(example_type, 0) + 1
                )

            if "confidence" in metadata:
                quality_metrics["metadata_completeness"]["has_confidence"] += 1

            try:
                response_data = json.loads(example.response)
                confidence = response_data.get("confidence")
                if isinstance(confidence, (int, float)):
                    confidences.append(float(confidence))
            except json.JSONDecodeError:
                continue

        if confidences:
            confidences.sort()
            quality_metrics["confidence_distribution"]["min"] = confidences[0]
            quality_metrics["confidence_distribution"]["max"] = confidences[-1]

            import statistics

            quality_metrics["confidence_distribution"]["mean"] = statistics.mean(
                confidences
            )
            quality_metrics["confidence_distribution"]["median"] = statistics.median(
                confidences
            )
            quality_metrics["confidence_distribution"]["std_dev"] = (
                statistics.stdev(confidences) if len(confidences) > 1 else 0.0
            )

        total_examples = len(examples)
        if total_examples:
            completeness = quality_metrics["metadata_completeness"]
            for key in completeness:
                completeness[key] = completeness[key] / total_examples * 100

        quality_metrics["example_type_distribution"] = example_type_counts
        quality_metrics["detector_balance"] = detector_counts

        return quality_metrics

    def _calculate_balance_scores(
        self,
        detector_counts: Dict[str, int],
        category_counts: Dict[str, int],
    ) -> Dict[str, Any]:
        """Calculate balance scores for detectors and categories."""

        def calculate_balance(counts: Dict[str, int]) -> Dict[str, Any]:
            if not counts:
                return {
                    "unique": 0,
                    "average": 0,
                    "max": 0,
                    "min": 0,
                    "imbalance_ratio": 0,
                }

            values = list(counts.values())
            average = sum(values) / len(values)
            max_value = max(values)
            min_value = min(values)
            imbalance_ratio = max_value / average if average else 0

            return {
                "unique": len(counts),
                "average": average,
                "max": max_value,
                "min": min_value,
                "imbalance_ratio": imbalance_ratio,
            }

        return {
            "detector_balance": calculate_balance(detector_counts),
            "category_balance": calculate_balance(category_counts),
        }

    def _check_consistency(self, examples: List[TrainingExample]) -> Dict[str, Any]:
        """Perform consistency checks across the dataset."""
        logger.debug("Running consistency checks...")

        consistency_checks: Dict[str, Any] = {
            "detector_to_taxonomy": {},
            "confidence_ranges": {},
            "inconsistent_examples": [],
        }

        detector_to_taxonomy: Dict[str, set] = {}
        confidence_ranges: Dict[str, List[float]] = {}

        for index, example in enumerate(examples):
            metadata = example.metadata
            detector = metadata.get("detector")

            if not detector:
                continue

            if detector not in detector_to_taxonomy:
                detector_to_taxonomy[detector] = set()
            if detector not in confidence_ranges:
                confidence_ranges[detector] = []

            try:
                response_data = json.loads(example.response)
                taxonomy_labels = response_data.get("taxonomy", [])
                confidence = response_data.get("confidence", 0.0)

                detector_to_taxonomy[detector].update(taxonomy_labels)
                if isinstance(confidence, (int, float)):
                    confidence_ranges[detector].append(float(confidence))

            except json.JSONDecodeError:
                consistency_checks["inconsistent_examples"].append(
                    {
                        "example_index": index,
                        "reason": "Invalid JSON response",
                    }
                )

        for detector, labels in detector_to_taxonomy.items():
            consistency_checks["detector_to_taxonomy"][detector] = sorted(list(labels))

        for detector, confidences in confidence_ranges.items():
            if confidences:
                consistency_checks["confidence_ranges"][detector] = {
                    "min": min(confidences),
                    "max": max(confidences),
                    "avg": sum(confidences) / len(confidences),
                }

        return consistency_checks

    def _generate_recommendations(self, validation_report: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations: List[str] = []

        format_validation = validation_report["format_validation"]
        if format_validation["invalid_examples"] > 0:
            recommendations.append(
                "Review invalid examples for format compliance and correct missing fields."
            )

        taxonomy_validation = validation_report["taxonomy_validation"]
        if taxonomy_validation["invalid_labels"] > 0:
            recommendations.append(
                "Investigate unknown taxonomy labels and update taxonomy or mappings."
            )

        coverage = validation_report["coverage_analysis"]
        coverage_stats = coverage["coverage_statistics"]
        if coverage_stats["detector_coverage_percentage"] < 80:
            recommendations.append(
                "Add examples for underrepresented detectors to improve coverage."
            )
        if coverage_stats["taxonomy_coverage_percentage"] < 70:
            recommendations.append(
                "Expand training data to cover more taxonomy categories."
            )

        quality_metrics = validation_report["quality_metrics"]
        confidence_distribution = quality_metrics["confidence_distribution"]
        if (
            confidence_distribution["std_dev"]
            and confidence_distribution["std_dev"] < 0.05
        ):
            recommendations.append(
                "Increase diversity in confidence scores to improve model calibration."
            )

        metadata_completeness = quality_metrics["metadata_completeness"]
        if metadata_completeness["has_confidence"] < 90:
            recommendations.append(
                "Ensure all examples include confidence metadata for downstream analytics."
            )

        return recommendations

    def _calculate_overall_score(self, validation_report: Dict[str, Any]) -> float:
        """Calculate overall validation score out of 100."""
        format_valid = validation_report["format_validation"]["valid_examples"]
        total_examples = validation_report["dataset_info"]["total_examples"]

        format_score = (format_valid / total_examples * 40) if total_examples else 0

        coverage = validation_report["coverage_analysis"]["coverage_statistics"]
        coverage_score = (
            (
                min(coverage["detector_coverage_percentage"], 100)
                + min(coverage["taxonomy_coverage_percentage"], 100)
            )
            / 2
            * 0.3
        )

        quality_metrics = validation_report["quality_metrics"]
        metadata_score = (
            sum(quality_metrics["metadata_completeness"].values()) / 3
        ) * 0.2

        consistency_checks = validation_report["consistency_checks"]
        consistency_score = (
            10
            if not consistency_checks["inconsistent_examples"]
            else max(0, 10 - len(consistency_checks["inconsistent_examples"]))
        )

        return format_score + coverage_score + metadata_score + consistency_score

    def validate_single_example(self, example: TrainingExample) -> Dict[str, Any]:
        """Validate a single training example for correctness."""
        if not self._taxonomy:
            self._taxonomy = self.taxonomy_loader.load_taxonomy()
        assert self._taxonomy is not None

        validation: Dict[str, Any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "details": {},
        }

        if not example.instruction or not isinstance(example.instruction, str):
            validation["valid"] = False
            validation["errors"].append("Invalid or missing instruction")

        try:
            response_data = json.loads(example.response)

            required_fields = ["taxonomy", "scores", "confidence"]
            missing_fields = [
                field for field in required_fields if field not in response_data
            ]

            if missing_fields:
                validation["valid"] = False
                validation["errors"].append(
                    "Missing required fields: %s" % missing_fields
                )

            taxonomy_labels = response_data.get("taxonomy", [])
            for label in taxonomy_labels:
                if not self._taxonomy.validate_label_name(label):
                    validation["valid"] = False
                    validation["errors"].append("Invalid taxonomy label: %s" % label)

            confidence = response_data.get("confidence", 0.0)
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                validation["valid"] = False
                validation["errors"].append("Invalid confidence value: %s" % confidence)

            scores = response_data.get("scores", {})
            for label, score in scores.items():
                if not isinstance(score, (int, float)) or not (0 <= score <= 1):
                    validation["valid"] = False
                    validation["errors"].append(
                        "Invalid score for %s: %s" % (label, score)
                    )

            validation["details"]["response_data"] = response_data

        except json.JSONDecodeError as exc:
            validation["valid"] = False
            validation["errors"].append("Invalid JSON in response: %s" % exc)

        return validation

    def export_validation_report(
        self,
        validation_report: Dict[str, Any],
        output_path: Union[str, Path],
        format: str = "json",
    ) -> None:
        """Export validation report to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "json":
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(
                    validation_report, file, indent=2, ensure_ascii=False, default=str
                )
        elif format.lower() == "txt":
            with open(output_path, "w", encoding="utf-8") as file:
                self._write_text_report(validation_report, file)
        else:
            raise ValueError("Unsupported format: %s. Use 'json' or 'txt'" % format)

        logger.info("Validation report exported to %s", output_path)

    def _write_text_report(
        self, validation_report: Dict[str, Any], file: TextIO
    ) -> None:
        """Write validation report in text format."""
        file.write("TRAINING DATASET VALIDATION REPORT\n")
        file.write("=" * 50 + "\n\n")

        dataset_info = validation_report["dataset_info"]
        file.write("Dataset: %s examples\n" % dataset_info["total_examples"])
        file.write("Validation Date: %s\n" % dataset_info["validation_timestamp"])
        file.write("Overall Score: %.2f/100\n\n" % validation_report["overall_score"])

        format_val = validation_report["format_validation"]
        file.write("FORMAT VALIDATION\n")
        file.write("-" * 20 + "\n")
        file.write("Valid Examples: %s\n" % format_val["valid_examples"])
        file.write("Invalid Examples: %s\n" % format_val["invalid_examples"])
        if format_val["errors"]:
            file.write("Errors:\n")
            for error in format_val["errors"][:10]:
                file.write("  - %s\n" % error)
        file.write("\n")

        coverage = validation_report["coverage_analysis"]
        file.write("COVERAGE ANALYSIS\n")
        file.write("-" * 20 + "\n")
        stats = coverage["coverage_statistics"]
        file.write(
            "Detector Coverage: %.1f%%\n" % stats["detector_coverage_percentage"]
        )
        file.write(
            "Taxonomy Coverage: %.1f%%\n" % stats["taxonomy_coverage_percentage"]
        )
        file.write(
            "Categories Covered: %s/%s\n\n"
            % (stats["covered_categories"], stats["total_categories"])
        )

        recommendations = validation_report["recommendations"]
        if recommendations:
            file.write("RECOMMENDATIONS\n")
            file.write("-" * 20 + "\n")
            for recommendation in recommendations:
                file.write("- %s\n" % recommendation)


__all__ = ["DatasetValidator"]

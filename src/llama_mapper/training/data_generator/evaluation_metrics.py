"""Comprehensive evaluation metrics for compliance models."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

    # Fallback implementations
    def accuracy_score(y_true, y_pred):
        return sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(
            y_true
        )

    def precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    ):
        # Simple fallback implementation
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy, accuracy, accuracy, None

    def confusion_matrix(y_true, y_pred):
        # Simple fallback implementation
        unique_labels = list(set(y_true + y_pred))
        matrix = [[0] * len(unique_labels) for _ in range(len(unique_labels))]
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}

        for true, pred in zip(y_true, y_pred):
            true_idx = label_to_idx[true]
            pred_idx = label_to_idx[pred]
            matrix[true_idx][pred_idx] += 1

        return matrix


logger = logging.getLogger(__name__)


class ComplianceModelEvaluator:
    """Comprehensive evaluation metrics for compliance models."""

    def __init__(self):
        self.golden_test_cases = self._load_golden_test_cases()
        self.bias_test_cases = self._load_bias_test_cases()

    def evaluate_model_performance(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        model_name: str = "compliance_model",
    ) -> Dict[str, Any]:
        """Evaluate model performance comprehensively."""
        logger.info("Evaluating model performance for %s", model_name)

        results = {
            "model_name": model_name,
            "overall_metrics": self._calculate_overall_metrics(
                predictions, ground_truth
            ),
            "category_metrics": self._calculate_category_metrics(
                predictions, ground_truth
            ),
            "confidence_calibration": self._calculate_confidence_calibration(
                predictions, ground_truth
            ),
            "bias_analysis": self._analyze_bias(predictions, ground_truth),
            "edge_case_performance": self._evaluate_edge_cases(
                predictions, ground_truth
            ),
            "error_analysis": self._analyze_errors(predictions, ground_truth),
        }

        logger.info("Evaluation completed for %s", model_name)
        return results

    def _calculate_overall_metrics(
        self, predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate overall performance metrics."""
        y_true = [gt["taxonomy"][0] for gt in ground_truth]
        y_pred = [pred["taxonomy"][0] for pred in predictions]
        y_conf = [pred["confidence"] for pred in predictions]

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted"
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "average_confidence": np.mean(y_conf),
            "confidence_std": np.std(y_conf),
            "total_examples": len(predictions),
        }

    def _calculate_category_metrics(
        self, predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate metrics by compliance category."""
        y_true = [gt["taxonomy"][0] for gt in ground_truth]
        y_pred = [pred["taxonomy"][0] for pred in predictions]

        # Group by compliance category
        categories = {}
        for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
            category = true_label.split(".")[0]  # e.g., "COMPLIANCE", "PII", "FRAUD"
            if category not in categories:
                categories[category] = {"true": [], "pred": [], "indices": []}
            categories[category]["true"].append(true_label)
            categories[category]["pred"].append(pred_label)
            categories[category]["indices"].append(i)

        category_metrics = {}
        for category, data in categories.items():
            if len(data["true"]) > 0:
                accuracy = accuracy_score(data["true"], data["pred"])
                precision, recall, f1, _ = precision_recall_fscore_support(
                    data["true"], data["pred"], average="weighted", zero_division=0
                )
                category_metrics[category] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "sample_count": len(data["true"]),
                }

        return category_metrics

    def _calculate_confidence_calibration(
        self, predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate confidence calibration metrics."""
        y_true = [gt["taxonomy"][0] for gt in ground_truth]
        y_pred = [pred["taxonomy"][0] for pred in predictions]
        y_conf = [pred["confidence"] for pred in predictions]

        # Calculate calibration error
        correct_predictions = [
            1 if true == pred else 0 for true, pred in zip(y_true, y_pred)
        ]

        # Bin confidence scores
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for i in range(len(bins) - 1):
            mask = (np.array(y_conf) >= bins[i]) & (np.array(y_conf) < bins[i + 1])
            if np.sum(mask) > 0:
                bin_accuracies.append(
                    np.mean(
                        [correct_predictions[j] for j in range(len(mask)) if mask[j]]
                    )
                )
                bin_confidences.append(
                    np.mean([y_conf[j] for j in range(len(mask)) if mask[j]])
                )
                bin_counts.append(np.sum(mask))

        # Calculate Expected Calibration Error (ECE)
        ece = 0
        total_samples = len(predictions)
        for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts):
            ece += abs(acc - conf) * count / total_samples

        return {
            "expected_calibration_error": ece,
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": bin_counts,
            "overconfident_predictions": sum(
                1 for acc, conf in zip(bin_accuracies, bin_confidences) if conf > acc
            ),
            "underconfident_predictions": sum(
                1 for acc, conf in zip(bin_accuracies, bin_confidences) if conf < acc
            ),
        }

    def _analyze_bias(
        self, predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze model bias across different groups."""
        bias_results = {}

        # Analyze bias by industry
        industries = ["healthcare", "financial", "technology", "retail", "government"]
        for industry in industries:
            industry_indices = [
                i
                for i, gt in enumerate(ground_truth)
                if industry in gt.get("metadata", {}).get("industry", "").lower()
            ]
            if industry_indices:
                industry_predictions = [predictions[i] for i in industry_indices]
                industry_ground_truth = [ground_truth[i] for i in industry_indices]
                bias_results[f"industry_{industry}"] = self._calculate_overall_metrics(
                    industry_predictions, industry_ground_truth
                )

        # Analyze bias by severity level
        severities = ["low", "medium", "high", "critical"]
        for severity in severities:
            severity_indices = [
                i
                for i, gt in enumerate(ground_truth)
                if gt.get("metadata", {}).get("severity", "").lower() == severity
            ]
            if severity_indices:
                severity_predictions = [predictions[i] for i in severity_indices]
                severity_ground_truth = [ground_truth[i] for i in severity_indices]
                bias_results[f"severity_{severity}"] = self._calculate_overall_metrics(
                    severity_predictions, severity_ground_truth
                )

        return bias_results

    def _evaluate_edge_cases(
        self, predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate performance on edge cases."""
        edge_case_indices = [
            i
            for i, gt in enumerate(ground_truth)
            if gt.get("metadata", {}).get("complexity_level") == "high"
            or gt.get("metadata", {}).get("multi_category", False)
        ]

        if not edge_case_indices:
            return {"message": "No edge cases found in test data"}

        edge_case_predictions = [predictions[i] for i in edge_case_indices]
        edge_case_ground_truth = [ground_truth[i] for i in edge_case_indices]

        return {
            "edge_case_count": len(edge_case_indices),
            "edge_case_metrics": self._calculate_overall_metrics(
                edge_case_predictions, edge_case_ground_truth
            ),
            "multi_category_performance": self._evaluate_multi_category_cases(
                edge_case_predictions, edge_case_ground_truth
            ),
        }

    def _evaluate_multi_category_cases(
        self, predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate performance on multi-category cases."""
        multi_category_indices = [
            i
            for i, gt in enumerate(ground_truth)
            if gt.get("metadata", {}).get("multi_category", False)
        ]

        if not multi_category_indices:
            return {"message": "No multi-category cases found"}

        multi_category_predictions = [predictions[i] for i in multi_category_indices]
        multi_category_ground_truth = [ground_truth[i] for i in multi_category_indices]

        # Calculate partial match accuracy for multi-category cases
        partial_matches = 0
        exact_matches = 0

        for pred, gt in zip(multi_category_predictions, multi_category_ground_truth):
            pred_taxonomy = set(pred["taxonomy"])
            gt_taxonomy = set(gt["taxonomy"])

            if pred_taxonomy == gt_taxonomy:
                exact_matches += 1
            elif pred_taxonomy.intersection(gt_taxonomy):
                partial_matches += 1

        return {
            "total_multi_category_cases": len(multi_category_indices),
            "exact_match_accuracy": exact_matches / len(multi_category_indices),
            "partial_match_accuracy": (exact_matches + partial_matches)
            / len(multi_category_indices),
            "exact_matches": exact_matches,
            "partial_matches": partial_matches,
        }

    def _analyze_errors(
        self, predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze prediction errors."""
        y_true = [gt["taxonomy"][0] for gt in ground_truth]
        y_pred = [pred["taxonomy"][0] for pred in predictions]
        y_conf = [pred["confidence"] for pred in predictions]

        # Find incorrect predictions
        incorrect_indices = [
            i for i, (true, pred) in enumerate(zip(y_true, y_pred)) if true != pred
        ]

        if not incorrect_indices:
            return {"message": "No errors found"}

        incorrect_predictions = [predictions[i] for i in incorrect_indices]
        incorrect_ground_truth = [ground_truth[i] for i in incorrect_indices]
        incorrect_confidences = [y_conf[i] for i in incorrect_indices]

        # Analyze error patterns
        error_analysis = {
            "total_errors": len(incorrect_indices),
            "error_rate": len(incorrect_indices) / len(predictions),
            "average_confidence_of_errors": np.mean(incorrect_confidences),
            "high_confidence_errors": sum(
                1 for conf in incorrect_confidences if conf > 0.8
            ),
            "low_confidence_errors": sum(
                1 for conf in incorrect_confidences if conf < 0.6
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "most_common_errors": self._find_most_common_errors(
                incorrect_predictions, incorrect_ground_truth
            ),
        }

        return error_analysis

    def _find_most_common_errors(
        self,
        incorrect_predictions: List[Dict[str, Any]],
        incorrect_ground_truth: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Find the most common error patterns."""
        error_patterns = {}

        for pred, gt in zip(incorrect_predictions, incorrect_ground_truth):
            pred_label = pred["taxonomy"][0]
            gt_label = gt["taxonomy"][0]
            error_key = f"{gt_label} -> {pred_label}"

            if error_key not in error_patterns:
                error_patterns[error_key] = 0
            error_patterns[error_key] += 1

        # Sort by frequency
        sorted_errors = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)

        return [
            {
                "error_pattern": error,
                "count": count,
                "percentage": count / len(incorrect_predictions) * 100,
            }
            for error, count in sorted_errors[:10]  # Top 10 most common errors
        ]

    def _load_golden_test_cases(self) -> List[Dict[str, Any]]:
        """Load golden test cases for evaluation."""
        # This would load from a file in practice
        return [
            {
                "input": "Company failed to implement adequate data protection measures",
                "expected_output": "COMPLIANCE.GDPR.DataProtection.Article32",
                "complexity": "medium",
                "domain": "privacy",
            },
            {
                "input": "Insider trading violation detected",
                "expected_output": "COMPLIANCE.SEC.InsiderTrading.Rule10b5",
                "complexity": "high",
                "domain": "financial",
            },
        ]

    def _load_bias_test_cases(self) -> List[Dict[str, Any]]:
        """Load bias test cases for evaluation."""
        # This would load from a file in practice
        return [
            {
                "input": "Healthcare data breach",
                "expected_output": "COMPLIANCE.HIPAA.DataBreach.Notification",
                "industry": "healthcare",
                "severity": "high",
            },
            {
                "input": "Financial services compliance violation",
                "expected_output": "COMPLIANCE.SOX.InternalControls.Deficiency",
                "industry": "financial",
                "severity": "medium",
            },
        ]

    def generate_evaluation_report(
        self, evaluation_results: Dict[str, Any], output_file: Optional[str] = None
    ) -> str:
        """Generate a comprehensive evaluation report."""
        report = f"""
# Compliance Model Evaluation Report

## Model: {evaluation_results['model_name']}

## Overall Performance
- **Accuracy**: {evaluation_results['overall_metrics']['accuracy']:.3f}
- **Precision**: {evaluation_results['overall_metrics']['precision']:.3f}
- **Recall**: {evaluation_results['overall_metrics']['recall']:.3f}
- **F1 Score**: {evaluation_results['overall_metrics']['f1_score']:.3f}
- **Average Confidence**: {evaluation_results['overall_metrics']['average_confidence']:.3f}

## Category Performance
"""
        for category, metrics in evaluation_results["category_metrics"].items():
            report += f"""
### {category}
- **Accuracy**: {metrics['accuracy']:.3f}
- **Precision**: {metrics['precision']:.3f}
- **Recall**: {metrics['recall']:.3f}
- **F1 Score**: {metrics['f1_score']:.3f}
- **Sample Count**: {metrics['sample_count']}
"""

        report += f"""
## Confidence Calibration
- **Expected Calibration Error**: {evaluation_results['confidence_calibration']['expected_calibration_error']:.3f}
- **Overconfident Predictions**: {evaluation_results['confidence_calibration']['overconfident_predictions']}
- **Underconfident Predictions**: {evaluation_results['confidence_calibration']['underconfident_predictions']}

## Edge Case Performance
- **Edge Case Count**: {evaluation_results['edge_case_performance'].get('edge_case_count', 0)}
- **Edge Case Accuracy**: {evaluation_results['edge_case_performance'].get('edge_case_metrics', {}).get('accuracy', 0):.3f}

## Error Analysis
- **Total Errors**: {evaluation_results['error_analysis'].get('total_errors', 0)}
- **Error Rate**: {evaluation_results['error_analysis'].get('error_rate', 0):.3f}
- **Average Confidence of Errors**: {evaluation_results['error_analysis'].get('average_confidence_of_errors', 0):.3f}
"""

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report)

        return report


__all__ = ["ComplianceModelEvaluator"]

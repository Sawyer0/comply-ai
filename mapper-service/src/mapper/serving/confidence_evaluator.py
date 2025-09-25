"""
Confidence evaluation and calibration for model outputs.

This module provides confidence scoring and calibration functionality
to ensure reliable confidence estimates for mapping decisions.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceMetrics:
    """Metrics for confidence evaluation."""

    raw_confidence: float
    calibrated_confidence: float
    uncertainty_score: float
    reliability_score: float
    prediction_entropy: float


class ConfidenceEvaluator:
    """Evaluates and scores confidence for model outputs."""

    def __init__(self, uncertainty_threshold: float = 0.3):
        """
        Initialize confidence evaluator.

        Args:
            uncertainty_threshold: Threshold for high uncertainty detection
        """
        self.uncertainty_threshold = uncertainty_threshold

    def evaluate_confidence(
        self,
        model_output: Dict[str, Any],
        additional_features: Optional[Dict[str, float]] = None,
    ) -> ConfidenceMetrics:
        """
        Evaluate confidence metrics for a model output.

        Args:
            model_output: Raw model output with scores and confidence
            additional_features: Additional features for confidence evaluation

        Returns:
            ConfidenceMetrics: Comprehensive confidence metrics
        """
        raw_confidence = model_output.get("confidence", 0.0)
        scores = model_output.get("scores", {})

        # Calculate prediction entropy
        entropy = self._calculate_entropy(scores)

        # Calculate uncertainty score
        uncertainty = self._calculate_uncertainty(scores, raw_confidence)

        # Calculate reliability score
        reliability = self._calculate_reliability(
            scores, raw_confidence, additional_features
        )

        # For now, calibrated confidence is same as raw (would need training data for calibration)
        calibrated_confidence = raw_confidence

        return ConfidenceMetrics(
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            uncertainty_score=uncertainty,
            reliability_score=reliability,
            prediction_entropy=entropy,
        )

    def _calculate_entropy(self, scores: Dict[str, float]) -> float:
        """Calculate prediction entropy from scores."""
        if not scores:
            return 1.0  # Maximum entropy for no predictions

        values = list(scores.values())
        if len(values) == 1:
            return 0.0  # No entropy for single prediction

        # Normalize scores to probabilities
        total = sum(values)
        if total == 0:
            return 1.0

        probs = [v / total for v in values]

        # Calculate entropy
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs if p > 0)

        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(probs))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _calculate_uncertainty(
        self, scores: Dict[str, float], confidence: float
    ) -> float:
        """Calculate uncertainty score based on prediction distribution."""
        if not scores:
            return 1.0

        values = list(scores.values())

        # High uncertainty if scores are close together
        if len(values) > 1:
            score_std = np.std(values)
            score_range = max(values) - min(values)

            # Normalize uncertainty (lower range = higher uncertainty)
            uncertainty = 1.0 - min(score_range, 1.0)
        else:
            uncertainty = 1.0 - confidence

        return max(0.0, min(1.0, uncertainty))

    def _calculate_reliability(
        self,
        scores: Dict[str, float],
        confidence: float,
        additional_features: Optional[Dict[str, float]] = None,
    ) -> float:
        """Calculate reliability score for the prediction."""
        reliability_factors = []

        # Factor 1: Confidence level
        reliability_factors.append(confidence)

        # Factor 2: Score consistency (higher max score = more reliable)
        if scores:
            max_score = max(scores.values())
            reliability_factors.append(max_score)

        # Factor 3: Prediction clarity (single clear winner vs multiple close scores)
        if len(scores) > 1:
            values = sorted(scores.values(), reverse=True)
            if len(values) >= 2:
                clarity = values[0] - values[1]  # Gap between top 2 scores
                reliability_factors.append(clarity)

        # Factor 4: Additional features (if provided)
        if additional_features:
            for feature_value in additional_features.values():
                if 0.0 <= feature_value <= 1.0:  # Only use normalized features
                    reliability_factors.append(feature_value)

        # Calculate weighted average
        if reliability_factors:
            return sum(reliability_factors) / len(reliability_factors)
        else:
            return 0.0

    def should_use_fallback(
        self, metrics: ConfidenceMetrics, threshold: float = 0.7
    ) -> bool:
        """
        Determine if fallback should be used based on confidence metrics.

        Args:
            metrics: Confidence metrics
            threshold: Confidence threshold for fallback decision

        Returns:
            bool: True if fallback should be used
        """
        # Use fallback if confidence is below threshold
        if metrics.calibrated_confidence < threshold:
            return True

        # Use fallback if uncertainty is too high
        if metrics.uncertainty_score > self.uncertainty_threshold:
            return True

        # Use fallback if reliability is too low
        if metrics.reliability_score < 0.5:
            return True

        return False


class ConfidenceCalibrator:
    """Calibrates confidence scores using historical data."""

    def __init__(self):
        """Initialize confidence calibrator."""
        self.calibrator: Optional[IsotonicRegression] = None
        self.is_trained = False

    def train(
        self, predicted_confidences: List[float], actual_accuracies: List[float]
    ) -> None:
        """
        Train the confidence calibrator.

        Args:
            predicted_confidences: Model-predicted confidence scores
            actual_accuracies: Actual accuracy (0 or 1) for each prediction
        """
        if len(predicted_confidences) != len(actual_accuracies):
            raise ValueError("Confidence and accuracy lists must have same length")

        if len(predicted_confidences) < 10:
            logger.warning("Insufficient data for calibration training")
            return

        try:
            # Use isotonic regression for calibration
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(predicted_confidences, actual_accuracies)
            self.is_trained = True

            logger.info(
                "Confidence calibrator trained with %d samples",
                len(predicted_confidences),
            )

        except Exception as e:
            logger.error("Failed to train confidence calibrator: %s", str(e))
            self.is_trained = False

    def calibrate(self, confidence: float) -> float:
        """
        Calibrate a confidence score.

        Args:
            confidence: Raw confidence score

        Returns:
            float: Calibrated confidence score
        """
        if not self.is_trained or self.calibrator is None:
            logger.warning("Calibrator not trained, returning raw confidence")
            return confidence

        try:
            # Ensure confidence is in valid range
            confidence = max(0.0, min(1.0, confidence))

            # Apply calibration
            calibrated = self.calibrator.predict([confidence])[0]

            # Ensure output is in valid range
            return max(0.0, min(1.0, float(calibrated)))

        except Exception as e:
            logger.error("Calibration failed: %s", str(e))
            return confidence

    def get_calibration_curve(
        self, confidences: List[float], n_bins: int = 10
    ) -> Tuple[List[float], List[float]]:
        """
        Get calibration curve data for visualization.

        Args:
            confidences: List of confidence scores to evaluate
            n_bins: Number of bins for the curve

        Returns:
            Tuple[List[float], List[float]]: (bin_centers, calibrated_values)
        """
        if not self.is_trained or self.calibrator is None:
            # Return identity line if not trained
            bin_centers = [i / (n_bins - 1) for i in range(n_bins)]
            return bin_centers, bin_centers

        try:
            bin_centers = [i / (n_bins - 1) for i in range(n_bins)]
            calibrated_values = [self.calibrate(conf) for conf in bin_centers]

            return bin_centers, calibrated_values

        except Exception as e:
            logger.error("Failed to generate calibration curve: %s", str(e))
            bin_centers = [i / (n_bins - 1) for i in range(n_bins)]
            return bin_centers, bin_centers


def evaluate_model_confidence(
    model_output: Dict[str, Any],
    calibrator: Optional[ConfidenceCalibrator] = None,
    additional_features: Optional[Dict[str, float]] = None,
) -> ConfidenceMetrics:
    """
    Convenience function to evaluate model confidence.

    Args:
        model_output: Raw model output
        calibrator: Optional trained calibrator
        additional_features: Additional features for evaluation

    Returns:
        ConfidenceMetrics: Comprehensive confidence metrics
    """
    evaluator = ConfidenceEvaluator()
    metrics = evaluator.evaluate_confidence(model_output, additional_features)

    # Apply calibration if available
    if calibrator and calibrator.is_trained:
        metrics.calibrated_confidence = calibrator.calibrate(metrics.raw_confidence)

    return metrics


# Example usage and testing
if __name__ == "__main__":
    # Test confidence evaluation
    model_output = {
        "taxonomy": ["PII.Contact.Email"],
        "scores": {"PII.Contact.Email": 0.95, "PII.Contact.Phone": 0.1},
        "confidence": 0.85,
    }

    metrics = evaluate_model_confidence(model_output)

    print(f"Raw confidence: {metrics.raw_confidence}")
    print(f"Calibrated confidence: {metrics.calibrated_confidence}")
    print(f"Uncertainty score: {metrics.uncertainty_score}")
    print(f"Reliability score: {metrics.reliability_score}")
    print(f"Prediction entropy: {metrics.prediction_entropy}")

    # Test calibrator training
    calibrator = ConfidenceCalibrator()

    # Simulate training data (confidence scores and actual accuracies)
    confidences = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    accuracies = [1, 1, 1, 0, 0, 0, 0, 0, 0]  # High confidence = accurate

    calibrator.train(confidences, accuracies)

    # Test calibration
    test_confidence = 0.75
    calibrated = calibrator.calibrate(test_confidence)
    print(f"Original: {test_confidence}, Calibrated: {calibrated}")

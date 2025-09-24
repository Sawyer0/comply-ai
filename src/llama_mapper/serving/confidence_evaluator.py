"""
Confidence evaluation system for model outputs with calibrated thresholds.

This module provides confidence scoring using model logit softmax probabilities
with support for confidence calibration and threshold tuning.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

try:
    import torch  # type: ignore
except Exception:  # torch is an optional dependency at runtime
    torch = None  # type: ignore[assignment]

from ..config.manager import ConfigManager

logger = logging.getLogger(__name__)


try:
    from sklearn.isotonic import (
        IsotonicRegression as _IsoType,  # pylint: disable=import-error; type: ignore[import-not-found]
    )
except Exception:
    _IsoType = Any  # type: ignore[assignment]


class ConfidenceCalibrator:
    """
    Calibrates model confidence scores using temperature scaling or Platt scaling.

    This helps ensure that confidence scores reflect true probabilities of correctness.
    """

    def __init__(self, method: str = "temperature"):
        """
        Initialize confidence calibrator.

        Args:
            method: Calibration method ("temperature", "platt", or "isotonic")
        """
        self.method = method
        self.temperature = 1.0
        self.platt_a = 1.0
        self.platt_b = 0.0
        self.isotonic_regressor: Optional[Any] = None
        self.is_calibrated = False

    def calibrate(self, logits: np.ndarray, labels: np.ndarray) -> None:
        """
        Calibrate confidence scores using validation data.

        Args:
            logits: Model logits for validation examples [N, num_classes]
            labels: True labels for validation examples [N]
        """
        if self.method == "temperature":
            self._calibrate_temperature(logits, labels)
        elif self.method == "platt":
            self._calibrate_platt(logits, labels)
        elif self.method == "isotonic":
            self._calibrate_isotonic(logits, labels)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.is_calibrated = True
        logger.info("Confidence calibration completed using %s method", self.method)

    def _calibrate_temperature(self, logits: np.ndarray, labels: np.ndarray) -> None:
        """Calibrate using temperature scaling."""
        try:
            from scipy.optimize import (
                minimize,  # type: ignore # pylint: disable=import-error
            )
        except Exception:
            minimize = None

        def temperature_loss(temp: float) -> float:
            scaled_logits = logits / temp
            probs = self._softmax(scaled_logits)
            max_probs = np.max(probs, axis=1)
            correct = (np.argmax(probs, axis=1) == labels).astype(float)
            # Negative log-likelihood for calibration
            return float(
                -np.mean(
                    correct * np.log(max_probs + 1e-8)
                    + (1 - correct) * np.log(1 - max_probs + 1e-8)
                )
            )

        if minimize is not None:
            result = minimize(
                temperature_loss, x0=1.0, bounds=[(0.1, 10.0)], method="L-BFGS-B"
            )
            self.temperature = float(result.x[0])
        else:
            # Fallback: simple grid search if SciPy is unavailable
            candidates = np.linspace(0.1, 10.0, 50)
            losses = [temperature_loss(float(t)) for t in candidates]
            best_idx = int(np.argmin(losses))
            self.temperature = float(candidates[best_idx])
        logger.info("Optimal temperature: %.3f", self.temperature)

    def _calibrate_platt(self, logits: np.ndarray, labels: np.ndarray) -> None:
        """Calibrate using Platt scaling."""
        try:
            from scipy.optimize import (
                minimize,  # type: ignore # pylint: disable=import-error
            )
        except Exception:
            minimize = None

        max_probs = np.max(self._softmax(logits), axis=1)
        correct = (np.argmax(logits, axis=1) == labels).astype(float)

        def platt_loss(params: np.ndarray) -> float:
            a, b = params
            calibrated_probs = 1 / (1 + np.exp(a * max_probs + b))
            return float(
                -np.mean(
                    correct * np.log(calibrated_probs + 1e-8)
                    + (1 - correct) * np.log(1 - calibrated_probs + 1e-8)
                )
            )

        result = minimize(platt_loss, x0=[1.0, 0.0], method="L-BFGS-B")
        self.platt_a, self.platt_b = result.x
        logger.info("Platt parameters: a=%.3f, b=%.3f", self.platt_a, self.platt_b)

    def _calibrate_isotonic(self, logits: np.ndarray, labels: np.ndarray) -> None:
        """Calibrate using isotonic regression."""
        max_probs = np.max(self._softmax(logits), axis=1)
        correct = (np.argmax(logits, axis=1) == labels).astype(float)

        reg = _IsoType(out_of_bounds="clip")  # type: ignore[call-arg]
        try:
            reg.fit(max_probs, correct)  # type: ignore[attr-defined]
        except Exception:
            pass
        self.isotonic_regressor = reg
        logger.info("Isotonic regression calibration completed")

    def apply_calibration(self, logits: np.ndarray) -> NDArray[np.float64]:
        """
        Apply calibration to model logits.

        Args:
            logits: Raw model logits [N, num_classes]

        Returns:
            np.ndarray: Calibrated probabilities [N, num_classes]
        """
        if not self.is_calibrated:
            logger.warning(
                "Calibrator not trained, returning uncalibrated probabilities"
            )
            return self._softmax(logits)

        if self.method == "temperature":
            return self._softmax(logits / self.temperature)
        elif self.method == "platt":
            probs = self._softmax(logits)
            max_probs = np.max(probs, axis=1)
            calibrated_max = 1 / (1 + np.exp(self.platt_a * max_probs + self.platt_b))
            # Scale all probabilities proportionally
            scale_factor = calibrated_max / (max_probs + 1e-8)
            return cast(
                NDArray[np.float64], np.asarray(probs * scale_factor[:, np.newaxis])
            )
        elif self.method == "isotonic":
            probs = self._softmax(logits)
            max_probs = np.max(probs, axis=1)
            assert self.isotonic_regressor is not None
            calibrated_max = self.isotonic_regressor.predict(max_probs)
            scale_factor = calibrated_max / (max_probs + 1e-8)
            return cast(
                NDArray[np.float64], np.asarray(probs * scale_factor[:, np.newaxis])
            )
        else:
            return self._softmax(logits)

    def _softmax(self, logits: np.ndarray) -> NDArray[np.float64]:
        """Compute softmax probabilities."""
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        result = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return cast(NDArray[np.float64], np.asarray(result))

    def save(self, path: Union[str, Path]) -> None:
        """Save calibrator state."""
        state = {
            "method": self.method,
            "temperature": self.temperature,
            "platt_a": self.platt_a,
            "platt_b": self.platt_b,
            "isotonic_regressor": self.isotonic_regressor,
            "is_calibrated": self.is_calibrated,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("Calibrator saved to %s", path)

    def load(self, path: Union[str, Path]) -> None:
        """Load calibrator state."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.method = state["method"]
        self.temperature = state["temperature"]
        self.platt_a = state["platt_a"]
        self.platt_b = state["platt_b"]
        self.isotonic_regressor = state["isotonic_regressor"]
        self.is_calibrated = state["is_calibrated"]
        logger.info("Calibrator loaded from %s", path)


class ConfidenceEvaluator:
    """
    Evaluates confidence scores for model outputs with calibrated thresholds.

    This class provides confidence scoring using model logit softmax probabilities
    with support for confidence calibration and threshold tuning.
    """

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize confidence evaluator.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.calibrator: Optional[ConfidenceCalibrator] = None
        self.threshold = self.config_manager.confidence.threshold
        self.calibration_enabled = self.config_manager.confidence.calibration_enabled

        # Initialize calibrator if enabled
        if self.calibration_enabled:
            self.calibrator = ConfidenceCalibrator(method="temperature")

        logger.info("ConfidenceEvaluator initialized with threshold=%s", self.threshold)

    def evaluate_confidence(
        self,
        logits: Union[np.ndarray, List[float]],
        predicted_label: Optional[str] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate confidence score from model logits.

        Args:
            logits: Model logits for the prediction
            predicted_label: The predicted label (for logging)

        Returns:
            Tuple[float, Dict[str, float]]: (confidence_score, probability_distribution)
        """
        # Convert to numpy array
        if (torch is not None) and isinstance(logits, torch.Tensor):  # type: ignore[attr-defined]
            logits_np = logits.detach().cpu().numpy()
        elif isinstance(logits, list):
            logits_np = np.array(logits, dtype=float)
        else:
            logits_np = np.asarray(logits, dtype=float)

        # Ensure 2D array
        if logits_np.ndim == 1:
            logits_np = logits_np.reshape(1, -1)

        # Apply calibration if available
        probs: NDArray[np.float64]
        if self.calibrator and self.calibrator.is_calibrated:
            probs = self.calibrator.apply_calibration(logits_np)
        else:
            probs = self._softmax(logits_np)

        # Get confidence as max probability
        confidence = float(np.max(probs[0]))

        # Create probability distribution dict
        prob_dist = {f"class_{i}": float(prob) for i, prob in enumerate(probs[0])}

        logger.debug(
            f"Confidence evaluation: {confidence:.3f} for label {predicted_label}"
        )

        return confidence, prob_dist

    def should_use_fallback(self, confidence: float) -> bool:
        """
        Determine if fallback mapping should be used based on confidence.

        Args:
            confidence: Confidence score from model

        Returns:
            bool: True if fallback should be used
        """
        use_fallback = confidence < self.threshold
        if use_fallback:
            logger.info(
                f"Using fallback: confidence {confidence:.3f} < threshold {self.threshold}"
            )
        return use_fallback

    def calibrate_confidence(
        self,
        validation_logits: List[np.ndarray],
        validation_labels: List[int],
        save_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, float]:
        """
        Calibrate confidence scores using validation data.

        Args:
            validation_logits: List of logit arrays from validation set
            validation_labels: List of true labels for validation set
            save_path: Optional path to save calibrator

        Returns:
            Dict[str, float]: Calibration metrics
        """
        if not self.calibration_enabled:
            logger.warning("Confidence calibration is disabled")
            return {}

        if not validation_logits or not validation_labels:
            logger.error("No validation data provided for calibration")
            return {}

        # Stack validation data
        logits_array = np.vstack(validation_logits)
        labels_array = np.array(validation_labels)

        # Initialize and train calibrator
        self.calibrator = ConfidenceCalibrator(method="temperature")
        self.calibrator.calibrate(logits_array, labels_array)

        # Evaluate calibration quality
        metrics = self._evaluate_calibration_quality(logits_array, labels_array)

        # Save calibrator if path provided
        if save_path:
            self.calibrator.save(save_path)

        logger.info("Confidence calibration completed. Metrics: %s", metrics)
        return metrics

    def load_calibrator(self, path: Union[str, Path]) -> None:
        """
        Load pre-trained calibrator.

        Args:
            path: Path to saved calibrator
        """
        if not self.calibration_enabled:
            logger.warning("Confidence calibration is disabled")
            return

        self.calibrator = ConfidenceCalibrator()
        self.calibrator.load(path)
        logger.info("Calibrator loaded from %s", path)

    def update_threshold(self, new_threshold: float) -> None:
        """
        Update confidence threshold.

        Args:
            new_threshold: New threshold value [0.0, 1.0]
        """
        if not (0.0 <= new_threshold <= 1.0):
            raise ValueError("Threshold must be between 0.0 and 1.0")

        old_threshold = self.threshold
        self.threshold = new_threshold
        logger.info(
            f"Confidence threshold updated: {old_threshold:.3f} -> {new_threshold:.3f}"
        )

    def get_threshold_recommendations(
        self, validation_confidences: List[float], validation_correct: List[bool]
    ) -> Dict[str, float]:
        """
        Recommend optimal thresholds based on validation data.

        Args:
            validation_confidences: Confidence scores from validation set
            validation_correct: Whether each prediction was correct

        Returns:
            Dict[str, float]: Recommended thresholds for different criteria
        """
        confidences = np.asarray(validation_confidences, dtype=float)
        correct = np.asarray(validation_correct, dtype=bool)

        recommendations: Dict[str, float] = {}

        # Find threshold that maximizes F1 score
        thresholds = np.linspace(0.1, 0.9, 81)
        best_f1 = 0.0
        best_f1_threshold = 0.5

        for threshold in thresholds:
            above_threshold = confidences >= threshold
            if np.sum(above_threshold) == 0:
                continue

            precision = float(np.mean(correct[above_threshold]))
            recall = float(np.sum(above_threshold) / len(above_threshold))

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_f1_threshold = threshold

        recommendations["max_f1"] = best_f1_threshold

        # Find threshold for 95% precision
        for threshold in sorted(thresholds, reverse=True):
            above_threshold = confidences >= threshold
            if np.sum(above_threshold) > 0:
                precision = float(np.mean(correct[above_threshold]))
                if precision >= 0.95:
                    recommendations["precision_95"] = threshold
                    break

        # Find threshold for 90% recall
        for threshold in thresholds:
            above_threshold = confidences >= threshold
            recall = float(np.sum(above_threshold) / len(above_threshold))
            if recall >= 0.90:
                recommendations["recall_90"] = threshold
                break

        logger.info("Threshold recommendations: %s", recommendations)
        return recommendations

    def _evaluate_calibration_quality(
        self, logits: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate calibration quality metrics."""
        # Before calibration
        uncalibrated_probs: NDArray[np.float64] = self._softmax(logits)
        uncalibrated_conf = np.max(uncalibrated_probs, axis=1)
        uncalibrated_pred = np.argmax(uncalibrated_probs, axis=1)
        uncalibrated_correct = (uncalibrated_pred == labels).astype(float)

        # After calibration
        if self.calibrator is not None and self.calibrator.is_calibrated:
            calibrated_probs = self.calibrator.apply_calibration(logits)
        else:
            calibrated_probs = uncalibrated_probs
        calibrated_conf = np.max(calibrated_probs, axis=1)
        calibrated_pred = np.argmax(calibrated_probs, axis=1)
        calibrated_correct = (calibrated_pred == labels).astype(float)

        # Expected Calibration Error (ECE)
        def compute_ece(
            confidences: NDArray[np.float64],
            correct: NDArray[np.float64],
            n_bins: int = 10,
        ) -> float:
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            ece = 0.0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
                prop_in_bin = in_bin.mean()

                if prop_in_bin > 0:
                    accuracy_in_bin = correct[in_bin].mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            return ece

        uncalibrated_ece = compute_ece(uncalibrated_conf, uncalibrated_correct)
        calibrated_ece = compute_ece(calibrated_conf, calibrated_correct)

        return {
            "uncalibrated_ece": float(uncalibrated_ece),
            "calibrated_ece": float(calibrated_ece),
            "ece_improvement": float(uncalibrated_ece - calibrated_ece),
            "uncalibrated_accuracy": float(np.mean(uncalibrated_correct)),
            "calibrated_accuracy": float(np.mean(calibrated_correct)),
        }

    def _softmax(self, logits: np.ndarray) -> NDArray[np.float64]:
        """Compute softmax probabilities."""
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        result = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return cast(NDArray[np.float64], np.asarray(result))

    def get_stats(self) -> Dict[str, Any]:
        """
        Get confidence evaluator statistics.

        Returns:
            Dict[str, Any]: Statistics about the evaluator
        """
        return {
            "threshold": self.threshold,
            "calibration_enabled": self.calibration_enabled,
            "calibrator_trained": (
                self.calibrator.is_calibrated if self.calibrator else False
            ),
            "calibration_method": self.calibrator.method if self.calibrator else None,
        }

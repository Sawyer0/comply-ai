"""Performance prediction ML component following SRP.

This module provides ONLY performance prediction capabilities:
- Detector performance forecasting
- Response time prediction
- Success rate estimation
- Load-based performance modeling
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

try:
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import joblib

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Fallback imports for when sklearn is not available
    np = None

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a detector."""

    detector_id: str
    response_time_ms: float
    success_rate: float
    confidence_score: float
    error_rate: float
    throughput: float
    timestamp: datetime


@dataclass
class PredictionFeatures:
    """Features used for performance prediction."""

    content_length: int
    content_type: str
    time_of_day: int
    day_of_week: int
    current_load: float
    historical_avg_response: float
    recent_error_rate: float


class PerformancePredictor:
    """ML-based detector performance predictor following SRP."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize performance predictor.

        Args:
            model_path: Path to pre-trained model file
        """
        if not SKLEARN_AVAILABLE:
            logger.warning(
                "scikit-learn not available, using fallback predictions only"
            )
            self.model = None
            self.scaler = None
        else:
            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
            self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            "content_length",
            "content_type_encoded",
            "time_of_day",
            "day_of_week",
            "current_load",
            "historical_avg_response",
            "recent_error_rate",
        ]
        self.performance_history: List[PerformanceMetrics] = []

        if model_path:
            self.load_model(model_path)

    def add_performance_data(self, metrics: PerformanceMetrics) -> None:
        """Add performance data for training.

        Args:
            metrics: Performance metrics to add
        """
        self.performance_history.append(metrics)

        # Keep only recent data (last 30 days)
        cutoff_date = datetime.now() - timedelta(days=30)
        self.performance_history = [
            m for m in self.performance_history if m.timestamp > cutoff_date
        ]

    def _encode_content_type(self, content_type: str) -> int:
        """Encode content type as integer.

        Args:
            content_type: Content type string

        Returns:
            Encoded content type
        """
        content_type_mapping = {
            "text": 0,
            "json": 1,
            "xml": 2,
            "html": 3,
            "binary": 4,
            "unknown": 5,
        }
        return content_type_mapping.get(content_type.lower(), 5)

    def _extract_features(self, features: PredictionFeatures) -> np.ndarray:
        """Extract feature vector from prediction features.

        Args:
            features: Prediction features

        Returns:
            Feature vector
        """
        if not SKLEARN_AVAILABLE:
            return [
                features.content_length,
                self._encode_content_type(features.content_type),
                features.time_of_day,
                features.day_of_week,
                features.current_load,
                features.historical_avg_response,
                features.recent_error_rate,
            ]

        return np.array(
            [
                features.content_length,
                self._encode_content_type(features.content_type),
                features.time_of_day,
                features.day_of_week,
                features.current_load,
                features.historical_avg_response,
                features.recent_error_rate,
            ]
        ).reshape(1, -1)

    def _prepare_training_data(self, detector_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for a specific detector.

        Args:
            detector_id: Detector to prepare data for

        Returns:
            Feature matrix and target vector
        """
        detector_metrics = [
            m for m in self.performance_history if m.detector_id == detector_id
        ]

        if len(detector_metrics) < 10:
            raise ValueError(f"Insufficient data for detector {detector_id}")

        features = []
        targets = []

        for i, metrics in enumerate(detector_metrics):
            # Calculate historical average (excluding current)
            historical_data = detector_metrics[:i] if i > 0 else [metrics]
            if SKLEARN_AVAILABLE:
                historical_avg = np.mean([m.response_time_ms for m in historical_data])
            else:
                historical_avg = sum(m.response_time_ms for m in historical_data) / len(
                    historical_data
                )

            # Calculate recent error rate (last 10 samples)
            recent_data = detector_metrics[max(0, i - 10) : i] if i > 0 else [metrics]
            if SKLEARN_AVAILABLE:
                recent_error_rate = np.mean([m.error_rate for m in recent_data])
            else:
                recent_error_rate = sum(m.error_rate for m in recent_data) / len(
                    recent_data
                )

            # Create features
            feature_obj = PredictionFeatures(
                content_length=1000,  # Default, would come from request
                content_type="text",  # Default, would come from request
                time_of_day=metrics.timestamp.hour,
                day_of_week=metrics.timestamp.weekday(),
                current_load=0.5,  # Would come from load monitor
                historical_avg_response=historical_avg,
                recent_error_rate=recent_error_rate,
            )

            feature_vector = self._extract_features(feature_obj).flatten()
            features.append(feature_vector)
            targets.append(metrics.response_time_ms)

        if not SKLEARN_AVAILABLE:
            return features, targets
        return np.array(features), np.array(targets)

    def train_model(self, detector_id: str) -> Dict[str, float]:
        """Train performance prediction model for a detector.

        Args:
            detector_id: Detector to train model for

        Returns:
            Training metrics
        """
        try:
            if not SKLEARN_AVAILABLE or not self.model or not self.scaler:
                raise ValueError("scikit-learn not available for training")

            features, targets = self._prepare_training_data(detector_id)

            # Split data
            features_train, features_test, targets_train, targets_test = (
                train_test_split(features, targets, test_size=0.2, random_state=42)
            )

            # Scale features
            features_train_scaled = self.scaler.fit_transform(features_train)
            features_test_scaled = self.scaler.transform(features_test)

            # Train model
            self.model.fit(features_train_scaled, targets_train)
            self.is_trained = True

            # Evaluate model
            train_score = self.model.score(features_train_scaled, targets_train)
            test_score = self.model.score(features_test_scaled, targets_test)

            logger.info(
                "Trained performance model for detector %s",
                detector_id,
                extra={
                    "train_score": train_score,
                    "test_score": test_score,
                    "training_samples": len(features_train),
                },
            )

            return {
                "train_score": train_score,
                "test_score": test_score,
                "training_samples": len(features_train),
                "test_samples": len(features_test),
            }

        except (ValueError, ImportError) as e:
            logger.error("Failed to train performance model: %s", str(e))
            raise

    def predict_performance(
        self, detector_id: str, features: PredictionFeatures
    ) -> Dict[str, float]:
        """Predict detector performance.

        Args:
            detector_id: Detector to predict for
            features: Prediction features

        Returns:
            Performance predictions
        """
        if not self.is_trained:
            logger.warning("Model not trained, using fallback predictions")
            return self._fallback_prediction(detector_id, features)

        try:
            feature_vector = self._extract_features(features)
            feature_vector_scaled = self.scaler.transform(feature_vector)

            # Predict response time
            predicted_response_time = self.model.predict(feature_vector_scaled)[0]

            # Calculate confidence based on feature similarity to training data
            confidence = self._calculate_prediction_confidence(feature_vector_scaled)

            # Estimate success rate based on historical data
            success_rate = self._estimate_success_rate(detector_id, features)

            return {
                "predicted_response_time_ms": max(0, predicted_response_time),
                "confidence": confidence,
                "estimated_success_rate": success_rate,
                "prediction_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error("Prediction failed: %s", str(e))
            return self._fallback_prediction(detector_id, features)

    def _calculate_prediction_confidence(self, feature_vector: np.ndarray) -> float:
        """Calculate confidence in prediction based on feature similarity.

        Args:
            feature_vector: Scaled feature vector

        Returns:
            Confidence score between 0 and 1
        """
        # Simple confidence based on feature ranges
        # In production, could use more sophisticated methods
        if not SKLEARN_AVAILABLE:
            return 0.7  # Default confidence when sklearn not available
        return min(1.0, max(0.1, 0.8 - np.mean(np.abs(feature_vector))))

    def _estimate_success_rate(
        self,
        detector_id: str,
        features: PredictionFeatures,  # pylint: disable=unused-argument
    ) -> float:
        """Estimate success rate based on historical data.

        Args:
            detector_id: Detector ID
            features: Prediction features (unused but kept for future enhancement)

        Returns:
            Estimated success rate
        """
        detector_metrics = [
            m for m in self.performance_history if m.detector_id == detector_id
        ]

        if not detector_metrics:
            return 0.95  # Default optimistic success rate

        # Use recent success rates
        recent_metrics = detector_metrics[-20:]  # Last 20 samples
        if not SKLEARN_AVAILABLE:
            return sum(m.success_rate for m in recent_metrics) / len(recent_metrics)
        return np.mean([m.success_rate for m in recent_metrics])

    def _fallback_prediction(
        self,
        detector_id: str,
        features: PredictionFeatures,  # pylint: disable=unused-argument
    ) -> Dict[str, float]:
        """Fallback prediction when ML model is not available.

        Args:
            detector_id: Detector ID (unused but kept for consistency)
            features: Prediction features

        Returns:
            Fallback predictions
        """
        # Simple heuristic-based prediction
        base_response_time = 100.0  # Base 100ms

        # Adjust based on content length
        content_factor = min(2.0, features.content_length / 1000.0)

        # Adjust based on current load
        load_factor = 1.0 + features.current_load

        predicted_time = base_response_time * content_factor * load_factor

        return {
            "predicted_response_time_ms": predicted_time,
            "confidence": 0.5,  # Low confidence for fallback
            "estimated_success_rate": 0.9,
            "prediction_timestamp": datetime.now().isoformat(),
            "fallback": True,
        }

    def save_model(self, path: str) -> None:
        """Save trained model to disk.

        Args:
            path: Path to save model
        """
        if not SKLEARN_AVAILABLE:
            raise ValueError("Cannot save model: scikit-learn not available")

        if not self.is_trained:
            raise ValueError("No trained model to save")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
        }

        joblib.dump(model_data, path)
        logger.info("Saved performance model to %s", path)

    def load_model(self, path: str) -> None:
        """Load trained model from disk.

        Args:
            path: Path to load model from
        """
        if not SKLEARN_AVAILABLE:
            raise ValueError("Cannot load model: scikit-learn not available")

        try:
            model_data = joblib.load(path)
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.feature_names = model_data["feature_names"]
            self.is_trained = model_data["is_trained"]

            logger.info("Loaded performance model from %s", path)

        except (ImportError, ValueError) as e:
            logger.error("Failed to load model: %s", str(e))
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model.

        Returns:
            Model information
        """
        return {
            "is_trained": self.is_trained,
            "feature_names": self.feature_names,
            "training_data_size": len(self.performance_history),
            "model_type": "RandomForestRegressor",
            "last_updated": datetime.now().isoformat(),
        }

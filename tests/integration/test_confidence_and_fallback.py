"""
Tests for confidence evaluation and fallback system integration.
"""
import pytest
import numpy as np
import tempfile
import yaml
from pathlib import Path

from src.llama_mapper.serving.confidence_evaluator import ConfidenceEvaluator, ConfidenceCalibrator
from src.llama_mapper.serving.fallback_mapper import FallbackMapper
from src.llama_mapper.config.manager import ConfigManager


class TestConfidenceEvaluator:
    """Test confidence evaluation functionality."""
    
    def test_confidence_evaluator_initialization(self):
        """Test confidence evaluator initialization."""
        evaluator = ConfidenceEvaluator()
        assert evaluator.threshold == 0.6  # Default threshold
        assert evaluator.calibration_enabled is True
    
    def test_evaluate_confidence_with_logits(self):
        """Test confidence evaluation from logits."""
        evaluator = ConfidenceEvaluator()
        
        # Test with high confidence logits
        high_conf_logits = [10.0, 1.0, 0.5]  # Clear winner
        confidence, prob_dist = evaluator.evaluate_confidence(high_conf_logits)
        
        assert confidence > 0.9
        assert len(prob_dist) == 3
        assert sum(prob_dist.values()) == pytest.approx(1.0, rel=1e-3)
    
    def test_should_use_fallback(self):
        """Test fallback decision logic."""
        evaluator = ConfidenceEvaluator()
        
        # High confidence - should not use fallback
        assert not evaluator.should_use_fallback(0.8)
        
        # Low confidence - should use fallback
        assert evaluator.should_use_fallback(0.4)
        
        # Threshold boundary
        assert evaluator.should_use_fallback(0.59)
        assert not evaluator.should_use_fallback(0.61)
    
    def test_update_threshold(self):
        """Test threshold updating."""
        evaluator = ConfidenceEvaluator()
        
        evaluator.update_threshold(0.7)
        assert evaluator.threshold == 0.7
        
        # Test invalid thresholds
        with pytest.raises(ValueError):
            evaluator.update_threshold(-0.1)
        
        with pytest.raises(ValueError):
            evaluator.update_threshold(1.1)
    
    def test_threshold_recommendations(self):
        """Test threshold recommendation functionality."""
        evaluator = ConfidenceEvaluator()
        
        # Mock validation data
        confidences = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        correct = [True, True, True, False, False, False, False, False]
        
        recommendations = evaluator.get_threshold_recommendations(confidences, correct)
        
        assert 'max_f1' in recommendations
        assert 'precision_95' in recommendations or 'recall_90' in recommendations
        assert all(0.0 <= v <= 1.0 for v in recommendations.values())


class TestConfidenceCalibrator:
    """Test confidence calibration functionality."""
    
    def test_calibrator_initialization(self):
        """Test calibrator initialization."""
        calibrator = ConfidenceCalibrator("temperature")
        assert calibrator.method == "temperature"
        assert not calibrator.is_calibrated
    
    def test_temperature_calibration(self):
        """Test temperature scaling calibration."""
        calibrator = ConfidenceCalibrator("temperature")
        
        # Mock logits and labels
        logits = np.array([[2.0, 1.0, 0.5], [1.5, 2.5, 0.8], [0.5, 1.0, 2.0]])
        labels = np.array([0, 1, 2])
        
        calibrator.calibrate(logits, labels)
        assert calibrator.is_calibrated
        assert calibrator.temperature > 0
        
        # Test calibration application
        calibrated_probs = calibrator.apply_calibration(logits)
        assert calibrated_probs.shape == logits.shape
        assert np.allclose(np.sum(calibrated_probs, axis=1), 1.0)
    
    def test_calibrator_save_load(self):
        """Test calibrator save/load functionality."""
        calibrator = ConfidenceCalibrator("temperature")
        
        # Mock calibration
        logits = np.array([[2.0, 1.0], [1.0, 2.0]])
        labels = np.array([0, 1])
        calibrator.calibrate(logits, labels)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save and load
            calibrator.save(temp_path)
            
            new_calibrator = ConfidenceCalibrator()
            new_calibrator.load(temp_path)
            
            assert new_calibrator.is_calibrated
            assert new_calibrator.temperature == calibrator.temperature
            assert new_calibrator.method == calibrator.method
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestFallbackMapper:
    """Test fallback mapper functionality."""
    
    def setup_method(self):
        """Set up test detector configurations."""
        self.temp_dir = tempfile.mkdtemp()
        self.detector_configs_path = Path(self.temp_dir)
        
        # Create test detector config
        test_config = {
            "detector": "test-detector",
            "version": "v1",
            "maps": {
                "toxic": "HARM.SPEECH.Toxicity",
                "hate": "HARM.SPEECH.Hate.Other",
                "spam": "OTHER.Spam"
            }
        }
        
        config_file = self.detector_configs_path / "test-detector.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(test_config, f)
    
    def test_fallback_mapper_initialization(self):
        """Test fallback mapper initialization."""
        mapper = FallbackMapper(str(self.detector_configs_path))
        assert "test-detector" in mapper.detector_mappings
        assert len(mapper.get_supported_detectors()) == 1
    
    def test_exact_mapping(self):
        """Test exact rule-based mapping."""
        mapper = FallbackMapper(str(self.detector_configs_path))
        
        response = mapper.map("test-detector", "toxic", "test_reason")
        
        assert response.taxonomy == ["HARM.SPEECH.Toxicity"]
        assert response.confidence == 0.8
        assert "exact" in response.notes.lower()
        assert response.provenance.detector == "test-detector"
    
    def test_case_insensitive_mapping(self):
        """Test case-insensitive mapping."""
        mapper = FallbackMapper(str(self.detector_configs_path))
        
        response = mapper.map("test-detector", "TOXIC", "test_reason")
        
        assert response.taxonomy == ["HARM.SPEECH.Toxicity"]
        assert response.confidence == 0.7
        assert "case-insensitive" in response.notes.lower()
    
    def test_partial_mapping(self):
        """Test partial matching."""
        mapper = FallbackMapper(str(self.detector_configs_path))
        
        response = mapper.map("test-detector", "very toxic content", "test_reason")
        
        assert response.taxonomy == ["HARM.SPEECH.Toxicity"]
        assert response.confidence == 0.5
        assert "partial" in response.notes.lower()
    
    def test_no_mapping_found(self):
        """Test handling of unmapped outputs."""
        mapper = FallbackMapper(str(self.detector_configs_path))
        
        response = mapper.map("test-detector", "unknown_output", "test_reason")
        
        assert response.taxonomy == ["OTHER.Unknown"]
        assert response.confidence == 0.0
        assert "no mapping found" in response.notes.lower()
    
    def test_unknown_detector(self):
        """Test handling of unknown detectors."""
        mapper = FallbackMapper(str(self.detector_configs_path))
        
        response = mapper.map("unknown-detector", "some_output", "test_reason")
        
        assert response.taxonomy == ["OTHER.Unknown"]
        assert response.confidence == 0.0
    
    def test_fallback_usage_tracking(self):
        """Test fallback usage metrics tracking."""
        mapper = FallbackMapper(str(self.detector_configs_path))
        
        # Generate some fallback usage
        mapper.map("test-detector", "toxic", "low_confidence")
        mapper.map("test-detector", "unknown", "validation_failed")
        mapper.map("unknown-detector", "test", "low_confidence")
        
        stats = mapper.get_fallback_usage_stats()
        
        assert stats["total_fallback_usage"] >= 3
        assert "test-detector" in stats["by_detector"]
        assert "low_confidence" in stats["by_reason"]
        assert stats["match_types"]["exact_matches"] >= 1


class TestIntegration:
    """Test integration between confidence evaluator and fallback mapper."""
    
    def setup_method(self):
        """Set up integration test environment."""
        # Set up temporary detector configs
        self.temp_dir = tempfile.mkdtemp()
        self.detector_configs_path = Path(self.temp_dir)
        
        test_config = {
            "detector": "integration-test",
            "version": "v1",
            "maps": {
                "positive": "SENTIMENT.Positive",
                "negative": "SENTIMENT.Negative"
            }
        }
        
        config_file = self.detector_configs_path / "integration-test.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        # Initialize components
        self.evaluator = ConfidenceEvaluator()
        self.fallback_mapper = FallbackMapper(str(self.detector_configs_path))
    
    def test_confidence_based_fallback_decision(self):
        """Test complete confidence evaluation and fallback decision flow."""
        # High confidence case - should not use fallback
        high_conf_logits = [5.0, 1.0]
        confidence, _ = self.evaluator.evaluate_confidence(high_conf_logits)
        
        should_fallback = self.evaluator.should_use_fallback(confidence)
        assert not should_fallback
        
        # Low confidence case - should use fallback
        low_conf_logits = [1.1, 1.0]  # Very close probabilities
        confidence, _ = self.evaluator.evaluate_confidence(low_conf_logits)
        
        should_fallback = self.evaluator.should_use_fallback(confidence)
        assert should_fallback
        
        # If fallback is needed, use fallback mapper
        if should_fallback:
            response = self.fallback_mapper.map("integration-test", "positive", "low_confidence")
            assert response.taxonomy == ["SENTIMENT.Positive"]
            assert response.confidence > 0  # Fallback provides some confidence
    
    def test_end_to_end_mapping_flow(self):
        """Test complete end-to-end mapping flow."""
        detector = "integration-test"
        output = "positive"
        
        # Simulate model prediction with low confidence
        model_logits = [1.05, 1.0]  # Very low confidence
        confidence, prob_dist = self.evaluator.evaluate_confidence(model_logits, output)
        
        if self.evaluator.should_use_fallback(confidence):
            # Use fallback mapping
            response = self.fallback_mapper.map(detector, output, "low_confidence")
            
            # Verify fallback response
            assert response.taxonomy == ["SENTIMENT.Positive"]
            assert response.confidence > confidence  # Fallback should be more confident
            assert "rule-based" in response.notes.lower()
        else:
            # Would use model prediction (not implemented in this test)
            pass
        
        # Check that metrics were recorded
        stats = self.fallback_mapper.get_fallback_usage_stats()
        assert stats["total_fallback_usage"] > 0


if __name__ == "__main__":
    pytest.main([__file__])
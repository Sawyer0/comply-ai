"""Tests for configuration management."""

import pytest
import tempfile
import yaml
from pathlib import Path
from src.llama_mapper.config import ConfigManager, Settings


def test_config_manager_default_settings():
    """Test ConfigManager with default settings."""
    config_manager = ConfigManager()
    settings = config_manager.get_settings()
    
    assert isinstance(settings, Settings)
    assert settings.app_name == "llama-mapper"
    assert settings.confidence.default_threshold == 0.6
    assert settings.model.temperature == 0.1


def test_config_manager_yaml_override():
    """Test ConfigManager with YAML configuration override."""
    # Create temporary YAML config
    config_data = {
        'app_name': 'test-mapper',
        'confidence': {
            'default_threshold': 0.8
        },
        'model': {
            'temperature': 0.2
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name
    
    try:
        config_manager = ConfigManager(config_path)
        settings = config_manager.get_settings()
        
        assert settings.app_name == "test-mapper"
        assert settings.confidence.default_threshold == 0.8
        assert settings.model.temperature == 0.2
        
    finally:
        Path(config_path).unlink()


def test_confidence_threshold_management():
    """Test confidence threshold management."""
    config_manager = ConfigManager()
    
    # Test default threshold
    default_threshold = config_manager.get_confidence_threshold("unknown-detector")
    assert default_threshold == 0.6
    
    # Test setting detector-specific threshold
    config_manager.update_confidence_threshold("test-detector", 0.75)
    threshold = config_manager.get_confidence_threshold("test-detector")
    assert threshold == 0.75
    
    # Test invalid threshold
    with pytest.raises(ValueError):
        config_manager.update_confidence_threshold("test-detector", 1.5)


def test_config_dictionaries():
    """Test configuration dictionary getters."""
    config_manager = ConfigManager()
    
    model_config = config_manager.get_model_config()
    assert "name" in model_config
    assert "temperature" in model_config
    assert model_config["temperature"] == 0.1
    
    lora_config = config_manager.get_lora_config()
    assert "r" in lora_config
    assert "alpha" in lora_config
    assert lora_config["r"] == 16
    
    training_config = config_manager.get_training_config()
    assert "learning_rate" in training_config
    assert "num_epochs" in training_config
    assert training_config["learning_rate"] == 2e-4


if __name__ == "__main__":
    pytest.main([__file__])
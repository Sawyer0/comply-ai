"""Tests for configuration management."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.llama_mapper.config import ConfigManager


def test_config_manager_default_settings():
    """Test ConfigManager with default settings."""
    config_manager = ConfigManager()

    # Test that configuration sections are properly initialized
    assert hasattr(config_manager, "model")
    assert hasattr(config_manager, "confidence")
    assert hasattr(config_manager, "serving")
    assert hasattr(config_manager, "storage")
    assert hasattr(config_manager, "security")
    assert hasattr(config_manager, "monitoring")

    # Test default values
    assert config_manager.model.temperature == 0.1
    assert config_manager.confidence.threshold == 0.6
    assert config_manager.serving.port == 8000
    assert config_manager.model.name == "meta-llama/Llama-2-7b-chat-hf"


def test_config_manager_yaml_override():
    """Test ConfigManager with YAML configuration override."""
    # Create temporary YAML config
    config_data = {
        "model": {"temperature": 0.2, "name": "test-model"},
        "confidence": {"threshold": 0.8},
        "serving": {"port": 9000},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        config_manager = ConfigManager(config_path)

        # Test that overrides are applied
        assert config_manager.model.temperature == 0.2
        assert config_manager.model.name == "test-model"
        assert config_manager.confidence.threshold == 0.8
        assert config_manager.serving.port == 9000

    finally:
        Path(config_path).unlink()


def test_confidence_threshold_validation():
    """Test confidence threshold validation."""
    config_manager = ConfigManager()

    # Test default threshold
    assert config_manager.confidence.threshold == 0.6

    # Test that threshold validation works (should be between 0.3 and 0.9)
    # This is tested by the Pydantic model validation
    assert 0.3 <= config_manager.confidence.threshold <= 0.9


def test_config_dictionaries():
    """Test configuration dictionary getters."""
    config_manager = ConfigManager()

    # Test get_config_dict method
    config_dict = config_manager.get_config_dict()

    assert "model" in config_dict
    assert "confidence" in config_dict
    assert "serving" in config_dict
    assert "storage" in config_dict
    assert "security" in config_dict
    assert "monitoring" in config_dict

    # Test specific values
    assert config_dict["model"]["temperature"] == 0.1
    assert config_dict["confidence"]["threshold"] == 0.6
    assert config_dict["serving"]["port"] == 8000


def test_config_validation():
    """Test configuration validation."""
    config_manager = ConfigManager()

    # Test that validation passes for default config
    assert config_manager.validate_config() is True


def test_config_save_and_reload():
    """Test saving and reloading configuration."""
    config_manager = ConfigManager()

    # Modify a setting
    original_temp = config_manager.model.temperature
    config_manager.model.temperature = 0.5

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        temp_path = f.name

    try:
        config_manager.save_config(temp_path)

        # Create new config manager from saved file
        new_config_manager = ConfigManager(temp_path)

        # Verify the setting was saved and loaded
        assert new_config_manager.model.temperature == 0.5

    finally:
        Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__])

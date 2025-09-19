"""Tests for ConfigManager implementation."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.llama_mapper.config import ConfigManager


class TestConfigManager:
    """Test cases for ConfigManager."""
    
    def test_default_config_creation(self):
        """Test that default configuration is created when no config file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            
            # Initialize ConfigManager with non-existent config
            config_manager = ConfigManager(config_path=config_path)
            
            # Verify config file was created
            assert config_path.exists()
            
            # Verify default values
            assert config_manager.model.name == "meta-llama/Llama-2-7b-chat-hf"
            assert config_manager.confidence.threshold == 0.6
            assert config_manager.serving.backend == "vllm"
            assert config_manager.monitoring.log_level == "INFO"
    
    def test_config_loading_from_file(self):
        """Test loading configuration from existing YAML file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            
            # Create custom config
            custom_config = {
                "model": {
                    "name": "custom-model",
                    "temperature": 0.2,
                    "lora_r": 32
                },
                "confidence": {
                    "threshold": 0.8
                },
                "serving": {
                    "backend": "tgi",
                    "port": 9000
                }
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(custom_config, f)
            
            # Load config
            config_manager = ConfigManager(config_path=config_path)
            
            # Verify custom values
            assert config_manager.model.name == "custom-model"
            assert config_manager.model.temperature == 0.2
            assert config_manager.model.lora_r == 32
            assert config_manager.confidence.threshold == 0.8
            assert config_manager.serving.backend == "tgi"
            assert config_manager.serving.port == 9000
    
    def test_environment_variable_overrides(self):
        """Test that environment variables override config values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            
            # Set environment variables
            os.environ["MAPPER_CONFIDENCE_THRESHOLD"] = "0.7"
            os.environ["MAPPER_SERVING_PORT"] = "8080"
            os.environ["MAPPER_LOG_LEVEL"] = "DEBUG"
            
            try:
                config_manager = ConfigManager(config_path=config_path)
                
                # Verify environment overrides
                assert config_manager.confidence.threshold == 0.7
                assert config_manager.serving.port == 8080
                assert config_manager.monitoring.log_level == "DEBUG"
                
            finally:
                # Clean up environment variables
                for var in ["MAPPER_CONFIDENCE_THRESHOLD", "MAPPER_SERVING_PORT", "MAPPER_LOG_LEVEL"]:
                    os.environ.pop(var, None)
    
    def test_config_validation(self):
        """Test configuration validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            
            config_manager = ConfigManager(config_path=config_path)
            
            # Valid config should pass validation
            assert config_manager.validate_config() is True
    
    def test_invalid_confidence_threshold(self):
        """Test that invalid confidence thresholds are rejected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            
            # Create config with invalid threshold
            invalid_config = {
                "confidence": {
                    "threshold": 1.5  # Invalid: > 1.0
                }
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(invalid_config, f)
            
            # Should raise validation error
            with pytest.raises(ValueError):
                ConfigManager(config_path=config_path)
    
    def test_config_save_and_reload(self):
        """Test saving and reloading configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            
            # Create and modify config
            config_manager = ConfigManager(config_path=config_path)
            
            # Modify some values
            config_manager.model.temperature = 0.3
            config_manager.confidence.threshold = 0.75
            
            # Save config
            config_manager.save_config()
            
            # Create new instance and verify values persisted
            new_config_manager = ConfigManager(config_path=config_path)
            assert new_config_manager.model.temperature == 0.3
            assert new_config_manager.confidence.threshold == 0.75
    
    def test_get_config_dict(self):
        """Test getting configuration as dictionary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            
            config_manager = ConfigManager(config_path=config_path)
            config_dict = config_manager.get_config_dict()
            
            # Verify structure
            assert "model" in config_dict
            assert "serving" in config_dict
            assert "confidence" in config_dict
            assert "storage" in config_dict
            assert "security" in config_dict
            assert "monitoring" in config_dict
            
            # Verify some values
            assert config_dict["model"]["name"] == "meta-llama/Llama-2-7b-chat-hf"
            assert config_dict["confidence"]["threshold"] == 0.6
            assert config_dict["serving"]["backend"] == "vllm"
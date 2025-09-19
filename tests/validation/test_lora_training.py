"""
Tests for LoRA fine-tuning pipeline components.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from llama_mapper.training import (
    ModelLoader,
    LoRATrainer,
    LoRATrainingConfig,
    CheckpointManager,
    ModelVersion,
    create_training_config,
)


class TestModelLoader:
    """Test ModelLoader functionality."""
    
    def test_init_default_config(self):
        """Test ModelLoader initialization with default config."""
        loader = ModelLoader()
        assert loader.model_name == ModelLoader.DEFAULT_MODEL_NAME
        assert not loader.use_quantization
        assert loader.use_fp16
        assert loader.device_map == "auto"
    
    def test_init_custom_config(self):
        """Test ModelLoader initialization with custom config."""
        loader = ModelLoader(
            model_name="custom/model",
            use_quantization=True,
            quantization_bits=4,
            use_fp16=False,
        )
        assert loader.model_name == "custom/model"
        assert loader.use_quantization
        assert loader.quantization_bits == 4
        assert not loader.use_fp16
    
    def test_invalid_quantization_bits(self):
        """Test that invalid quantization bits raise ValueError."""
        with pytest.raises(ValueError, match="quantization_bits must be 4 or 8"):
            ModelLoader(use_quantization=True, quantization_bits=16)
    
    def test_create_lora_config(self):
        """Test LoRA configuration creation."""
        config = ModelLoader.create_lora_config(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
        )
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.target_modules == ["q_proj", "v_proj"]
        assert config.task_type == "CAUSAL_LM"


class TestLoRATrainingConfig:
    """Test LoRATrainingConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LoRATrainingConfig()
        assert config.lora_r == 16
        assert config.lora_alpha == 32
        assert config.learning_rate == 2e-4
        assert config.num_train_epochs == 2
        assert config.max_sequence_length == 2048
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = LoRATrainingConfig(
            lora_r=8,
            learning_rate=1e-4,
            num_train_epochs=1,
        )
        assert config.lora_r == 8
        assert config.learning_rate == 1e-4
        assert config.num_train_epochs == 1
    
    def test_to_training_arguments(self):
        """Test conversion to TrainingArguments."""
        config = LoRATrainingConfig(output_dir="./test")
        training_args = config.to_training_arguments()
        assert training_args.output_dir == "./test"
        assert training_args.learning_rate == 2e-4
        assert training_args.num_train_epochs == 2


class TestCheckpointManager:
    """Test CheckpointManager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(
            base_dir=self.temp_dir,
            version_prefix="test-model",
            max_versions=3,
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test CheckpointManager initialization."""
        assert self.checkpoint_manager.base_dir == Path(self.temp_dir)
        assert self.checkpoint_manager.version_prefix == "test-model"
        assert self.checkpoint_manager.max_versions == 3
        assert len(self.checkpoint_manager.versions) == 0
    
    def test_generate_version_id(self):
        """Test version ID generation."""
        version_id = self.checkpoint_manager._generate_version_id()
        assert version_id == "test-model@v1.0.0"
        
        # Add a version and test increment
        self.checkpoint_manager.versions["test-model@v1.0.0"] = Mock()
        version_id = self.checkpoint_manager._generate_version_id()
        assert version_id == "test-model@v1.0.1"
    
    @patch('llama_mapper.training.checkpoint_manager.datetime')
    def test_model_version_serialization(self, mock_datetime):
        """Test ModelVersion serialization/deserialization."""
        from datetime import datetime
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
        
        version = ModelVersion(
            version="test@v1.0.0",
            checkpoint_path="/path/to/checkpoint",
            metadata={"key": "value"},
            created_at=datetime(2024, 1, 1, 12, 0, 0),
        )
        
        # Test serialization
        data = version.to_dict()
        assert data["version"] == "test@v1.0.0"
        assert data["checkpoint_path"] == "/path/to/checkpoint"
        assert data["metadata"] == {"key": "value"}
        
        # Test deserialization
        restored_version = ModelVersion.from_dict(data)
        assert restored_version.version == version.version
        assert restored_version.checkpoint_path == version.checkpoint_path
        assert restored_version.metadata == version.metadata
    
    def test_list_versions_empty(self):
        """Test listing versions when none exist."""
        versions = self.checkpoint_manager.list_versions()
        assert len(versions) == 0
    
    def test_get_latest_version_empty(self):
        """Test getting latest version when none exist."""
        latest = self.checkpoint_manager.get_latest_version()
        assert latest is None


class TestCreateTrainingConfig:
    """Test create_training_config helper function."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = create_training_config()
        assert config.output_dir == "./checkpoints"
        assert config.learning_rate == 2e-4
        assert config.num_train_epochs == 2
        assert config.per_device_train_batch_size == 4
    
    def test_custom_config(self):
        """Test custom configuration creation."""
        config = create_training_config(
            output_dir="./custom",
            learning_rate=1e-4,
            num_epochs=1,
            batch_size=8,
            lora_r=8,
        )
        assert config.output_dir == "./custom"
        assert config.learning_rate == 1e-4
        assert config.num_train_epochs == 1
        assert config.per_device_train_batch_size == 8
        assert config.lora_r == 8


class TestLoRATrainer:
    """Test LoRATrainer functionality."""
    
    def test_init(self):
        """Test LoRATrainer initialization."""
        config = LoRATrainingConfig()
        trainer = LoRATrainer(config)
        assert trainer.config == config
        assert trainer.model is None
        assert trainer.tokenizer is None
        assert trainer.trainer is None
    
    def test_init_with_model_loader(self):
        """Test LoRATrainer initialization with custom model loader."""
        config = LoRATrainingConfig()
        model_loader = ModelLoader(model_name="custom/model")
        trainer = LoRATrainer(config, model_loader)
        assert trainer.model_loader.model_name == "custom/model"
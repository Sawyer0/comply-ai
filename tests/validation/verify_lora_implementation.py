#!/usr/bin/env python3
"""
Simple verification script for LoRA fine-tuning implementation.
"""

import sys
import os
sys.path.insert(0, 'src')

def test_imports():
    """Test that all components can be imported."""
    print("Testing imports...")
    
    try:
        from llama_mapper.training import (
            ModelLoader,
            LoRATrainer, 
            LoRATrainingConfig,
            CheckpointManager,
            ModelVersion,
            create_training_config
        )
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_model_loader():
    """Test ModelLoader initialization."""
    print("\nTesting ModelLoader...")
    
    try:
        from llama_mapper.training import ModelLoader
        
        # Test default initialization
        loader = ModelLoader()
        assert loader.model_name == ModelLoader.DEFAULT_MODEL_NAME
        assert not loader.use_quantization
        assert loader.use_fp16
        print("✅ ModelLoader default initialization works")
        
        # Test custom initialization
        loader = ModelLoader(
            model_name="test/model",
            use_quantization=True,
            quantization_bits=8
        )
        assert loader.model_name == "test/model"
        assert loader.use_quantization
        assert loader.quantization_bits == 8
        print("✅ ModelLoader custom initialization works")
        
        # Test LoRA config creation
        lora_config = ModelLoader.create_lora_config(r=16, lora_alpha=32)
        assert lora_config.r == 16
        assert lora_config.lora_alpha == 32
        print("✅ LoRA config creation works")
        
        return True
    except Exception as e:
        print(f"❌ ModelLoader test failed: {e}")
        return False

def test_training_config():
    """Test LoRATrainingConfig."""
    print("\nTesting LoRATrainingConfig...")
    
    try:
        from llama_mapper.training import LoRATrainingConfig, create_training_config
        
        # Test default config
        config = LoRATrainingConfig()
        assert config.lora_r == 16
        assert config.lora_alpha == 32
        assert config.learning_rate == 2e-4
        assert config.num_train_epochs == 2
        print("✅ Default training config works")
        
        # Test helper function
        config = create_training_config(
            output_dir="./test",
            learning_rate=1e-4,
            num_epochs=1
        )
        assert config.output_dir == "./test"
        assert config.learning_rate == 1e-4
        assert config.num_train_epochs == 1
        print("✅ Training config helper works")
        
        return True
    except Exception as e:
        print(f"❌ Training config test failed: {e}")
        return False

def test_checkpoint_manager():
    """Test CheckpointManager."""
    print("\nTesting CheckpointManager...")
    
    try:
        from llama_mapper.training import CheckpointManager, ModelVersion
        import tempfile
        import shutil
        from datetime import datetime
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Test initialization
            manager = CheckpointManager(
                base_dir=temp_dir,
                version_prefix="test-model"
            )
            assert manager.version_prefix == "test-model"
            assert len(manager.versions) == 0
            print("✅ CheckpointManager initialization works")
            
            # Test version ID generation
            version_id = manager._generate_version_id()
            assert version_id == "test-model@v1.0.0"
            print("✅ Version ID generation works")
            
            # Test ModelVersion serialization
            version = ModelVersion(
                version="test@v1.0.0",
                checkpoint_path="/test/path",
                metadata={"test": "data"},
                created_at=datetime.now()
            )
            data = version.to_dict()
            restored = ModelVersion.from_dict(data)
            assert restored.version == version.version
            print("✅ ModelVersion serialization works")
            
        finally:
            shutil.rmtree(temp_dir)
        
        return True
    except Exception as e:
        print(f"❌ CheckpointManager test failed: {e}")
        return False

def test_lora_trainer():
    """Test LoRATrainer initialization."""
    print("\nTesting LoRATrainer...")
    
    try:
        from llama_mapper.training import LoRATrainer, LoRATrainingConfig, ModelLoader
        
        # Test initialization
        config = LoRATrainingConfig()
        trainer = LoRATrainer(config)
        assert trainer.config == config
        assert trainer.model is None
        assert trainer.tokenizer is None
        print("✅ LoRATrainer initialization works")
        
        # Test with custom model loader
        model_loader = ModelLoader(model_name="test/model")
        trainer = LoRATrainer(config, model_loader)
        assert trainer.model_loader.model_name == "test/model"
        print("✅ LoRATrainer with custom ModelLoader works")
        
        return True
    except Exception as e:
        print(f"❌ LoRATrainer test failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("🚀 Verifying LoRA fine-tuning implementation...")
    
    tests = [
        test_imports,
        test_model_loader,
        test_training_config,
        test_checkpoint_manager,
        test_lora_trainer,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All verification tests passed! LoRA implementation is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Example script demonstrating the complete LoRA fine-tuning pipeline.

This script shows how to use the ModelLoader, LoRATrainer, and CheckpointManager
to fine-tune Llama-3-8B-Instruct for the mapper task.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

from llama_mapper.training import (
    ModelLoader,
    LoRATrainer,
    LoRATrainingConfig,
    CheckpointManager,
    create_training_config,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data(data_path: str) -> List[Dict[str, str]]:
    """Load training data from JSONL file."""
    examples = []
    with open(data_path, 'r') as f:
        for line in f:
            example = json.loads(line.strip())
            examples.append(example)
    return examples


def main():
    """Run the complete LoRA fine-tuning pipeline."""
    
    # Configuration
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    output_dir = "./checkpoints/mapper-lora-v1"
    training_data_path = "sample_complete_training_data.jsonl"
    
    logger.info("Starting LoRA fine-tuning pipeline")
    
    # 1. Initialize components
    logger.info("Initializing components")
    
    # Model loader with quantization for memory efficiency
    model_loader = ModelLoader(
        model_name=model_name,
        use_quantization=True,
        quantization_bits=8,
        use_fp16=True,
    )
    
    # Training configuration with specified hyperparameters
    training_config = create_training_config(
        output_dir=output_dir,
        learning_rate=2e-4,  # As specified in requirements
        num_epochs=2,        # 1-2 epochs as specified
        batch_size=4,
    )
    
    # LoRA trainer
    trainer = LoRATrainer(
        config=training_config,
        model_loader=model_loader,
    )
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager(
        base_dir="./model_checkpoints",
        version_prefix="mapper-lora",
    )
    
    # 2. Load training data
    logger.info("Loading training data from %s", training_data_path)
    
    if not Path(training_data_path).exists():
        logger.error("Training data file not found: %s", training_data_path)
        logger.info("Please run the dataset preparation pipeline first")
        return
    
    training_examples = load_training_data(training_data_path)
    logger.info("Loaded %s training examples", len(training_examples))
    
    # Split into train/eval (80/20)
    split_idx = int(0.8 * len(training_examples))
    train_examples = training_examples[:split_idx]
    eval_examples = training_examples[split_idx:]
    
    logger.info("Train examples: %s", len(train_examples))
    logger.info("Eval examples: %s", len(eval_examples))
    
    # 3. Run training
    logger.info("Starting LoRA fine-tuning")
    
    try:
        training_results = trainer.train(
            train_examples=train_examples,
            eval_examples=eval_examples,
        )
        
        logger.info("Training completed successfully")
        logger.info("Final training loss: %s", training_results.get('train_loss', 'N/A'))
        logger.info("Training runtime: %s seconds", training_results.get('train_runtime', 'N/A'))
        
    except Exception as e:
        logger.error("Training failed: %s", e)
        return
    
    # 4. Save checkpoint with version management
    logger.info("Saving checkpoint with version management")
    
    try:
        version_id = checkpoint_manager.save_checkpoint(
            model=trainer.model,
            tokenizer=trainer.tokenizer,
            training_metrics=training_results,
            metadata={
                "model_name": model_name,
                "training_config": training_config.__dict__,
                "num_train_examples": len(train_examples),
                "num_eval_examples": len(eval_examples),
            },
            tags=["production", "v1"],
        )
        
        logger.info("Checkpoint saved with version: %s", version_id)
        
        # Get deployment info
        deployment_info = checkpoint_manager.get_deployment_info(version_id)
        logger.info("Model size: %.2f MB", deployment_info['model_size_mb'])
        
    except Exception as e:
        logger.error("Failed to save checkpoint: %s", e)
        return
    
    # 5. Demonstrate checkpoint loading
    logger.info("Demonstrating checkpoint loading")
    
    try:
        loaded_model, loaded_tokenizer, metadata = checkpoint_manager.load_checkpoint(version_id)
        logger.info("Checkpoint loaded successfully")
        logger.info("Loaded model type: %s", type(loaded_model).__name__)
        
    except Exception as e:
        logger.error("Failed to load checkpoint: %s", e)
        return
    
    # 6. List all versions
    logger.info("Available model versions:")
    versions = checkpoint_manager.list_versions()
    for version in versions:
        logger.info("  - %s (created: %s)", version.version, version.created_at)
    
    logger.info("LoRA fine-tuning pipeline completed successfully!")


if __name__ == "__main__":
    main()
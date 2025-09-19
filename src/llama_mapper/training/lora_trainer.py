"""
LoRATrainer for fine-tuning Llama-3-8B-Instruct with specified hyperparameters.

This module provides the training loop implementation with LoRA configuration,
metrics collection, and validation for the mapper fine-tuning task.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    PreTrainedTokenizer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, PeftModel
import structlog
from datasets import Dataset as HFDataset

from .model_loader import ModelLoader, create_instruction_prompt

logger = structlog.get_logger(__name__)


@dataclass
class LoRATrainingConfig:
    """Configuration for LoRA fine-tuning with specified hyperparameters."""
    
    # LoRA hyperparameters (as specified in requirements)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    # Training hyperparameters (as specified in requirements)
    learning_rate: float = 2e-4
    num_train_epochs: int = 2  # 1-2 epochs as specified
    max_sequence_length: int = 2048  # 1-2k tokens as specified
    
    # Training configuration
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    
    # Memory optimization
    fp16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    
    # Output configuration
    output_dir: str = "./checkpoints"
    run_name: str = "llama-mapper-lora"
    
    def to_training_arguments(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments."""
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            logging_steps=self.logging_steps,
            eval_steps=self.eval_steps,
            save_steps=self.save_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.fp16,
            gradient_checkpointing=self.gradient_checkpointing,
            dataloader_num_workers=self.dataloader_num_workers,
            remove_unused_columns=False,
            run_name=self.run_name,
            report_to=["tensorboard"],
            logging_dir=f"{self.output_dir}/logs",
        )


class MapperDataset(Dataset):
    """Dataset class for instruction-following mapper training data."""
    
    def __init__(
        self,
        examples: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
    ):
        """
        Initialize dataset with training examples.
        
        Args:
            examples: List of training examples with 'instruction' and 'response' keys
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(
            "Dataset initialized",
            num_examples=len(examples),
            max_length=max_length,
        )
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized example for training."""
        example = self.examples[idx]
        
        # Create instruction-following prompt
        prompt = create_instruction_prompt(
            instruction=example["instruction"],
            response=example["response"]
        )
        
        # Tokenize the full prompt
        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Create labels (same as input_ids for causal LM)
        labels = tokenized["input_ids"].clone()
        
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
        }


class LoRATrainer:
    """
    LoRA trainer for fine-tuning Llama-3-8B-Instruct on mapper task.
    
    Implements training loop with specified hyperparameters, metrics collection,
    and validation according to requirements 2.1 and 2.2.
    """
    
    def __init__(
        self,
        config: LoRATrainingConfig,
        model_loader: Optional[ModelLoader] = None,
    ):
        """
        Initialize LoRA trainer with configuration.
        
        Args:
            config: Training configuration with hyperparameters
            model_loader: Optional model loader (creates default if None)
        """
        self.config = config
        self.model_loader = model_loader or ModelLoader()
        self.model: Optional[PeftModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.trainer: Optional[Trainer] = None
        self.training_metrics: Dict[str, List[float]] = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
            "epoch": [],
        }
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        logger.info(
            "LoRATrainer initialized",
            config=config.__dict__,
        )
    
    def setup_model_and_tokenizer(self) -> Tuple[PeftModel, PreTrainedTokenizer]:
        """
        Set up model and tokenizer for training.
        
        Returns:
            Tuple of (peft_model, tokenizer)
        """
        logger.info("Setting up model and tokenizer for LoRA training")
        
        # Load base model and tokenizer
        base_model, tokenizer = self.model_loader.load_model_and_tokenizer()
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA to model
        peft_model = self.model_loader.prepare_model_for_lora(base_model, lora_config)
        
        self.model = peft_model
        self.tokenizer = tokenizer
        
        return peft_model, tokenizer
    
    def prepare_datasets(
        self,
        train_examples: List[Dict[str, str]],
        eval_examples: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[MapperDataset, Optional[MapperDataset]]:
        """
        Prepare training and evaluation datasets.
        
        Args:
            train_examples: Training examples with 'instruction' and 'response'
            eval_examples: Optional evaluation examples
            
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call setup_model_and_tokenizer first.")
        
        logger.info(
            "Preparing datasets",
            num_train=len(train_examples),
            num_eval=len(eval_examples) if eval_examples else 0,
        )
        
        train_dataset = MapperDataset(
            examples=train_examples,
            tokenizer=self.tokenizer,
            max_length=self.config.max_sequence_length,
        )
        
        eval_dataset = None
        if eval_examples:
            eval_dataset = MapperDataset(
                examples=eval_examples,
                tokenizer=self.tokenizer,
                max_length=self.config.max_sequence_length,
            )
        
        return train_dataset, eval_dataset
    
    def create_trainer(
        self,
        train_dataset: MapperDataset,
        eval_dataset: Optional[MapperDataset] = None,
    ) -> Trainer:
        """
        Create HuggingFace Trainer with custom metrics collection.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            
        Returns:
            Configured Trainer instance
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer not initialized.")
        
        # Create training arguments
        training_args = self.config.to_training_arguments()
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Custom trainer with metrics collection
        trainer = MapperTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            metrics_collector=self,
        )
        
        self.trainer = trainer
        return trainer
    
    def train(
        self,
        train_examples: List[Dict[str, str]],
        eval_examples: Optional[List[Dict[str, str]]] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            train_examples: Training examples
            eval_examples: Optional evaluation examples
            resume_from_checkpoint: Optional checkpoint path to resume from
            
        Returns:
            Training metrics and results
        """
        logger.info("Starting LoRA training pipeline")
        
        # Setup model and tokenizer
        if self.model is None or self.tokenizer is None:
            self.setup_model_and_tokenizer()
        
        # Prepare datasets
        train_dataset, eval_dataset = self.prepare_datasets(train_examples, eval_examples)
        
        # Create trainer
        trainer = self.create_trainer(train_dataset, eval_dataset)
        
        # Start training
        logger.info("Beginning training")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        trainer.save_model()
        trainer.save_state()
        
        # Collect final metrics
        final_metrics = {
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "train_steps_per_second": train_result.metrics.get("train_steps_per_second", 0),
            "total_flos": train_result.metrics.get("total_flos", 0),
            "train_loss": train_result.metrics.get("train_loss", 0),
            "training_metrics": self.training_metrics,
        }
        
        # Save metrics to file
        metrics_path = os.path.join(self.config.output_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(final_metrics, f, indent=2)
        
        logger.info(
            "Training completed",
            final_loss=final_metrics["train_loss"],
            runtime=final_metrics["train_runtime"],
            samples_per_second=final_metrics["train_samples_per_second"],
        )
        
        return final_metrics
    
    def collect_metrics(self, logs: Dict[str, float]) -> None:
        """Collect training metrics for monitoring."""
        for key, value in logs.items():
            if key in self.training_metrics:
                self.training_metrics[key].append(value)
    
    def evaluate(self, eval_dataset: Optional[MapperDataset] = None) -> Dict[str, float]:
        """
        Run evaluation on the model.
        
        Args:
            eval_dataset: Optional evaluation dataset
            
        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call train first.")
        
        logger.info("Running evaluation")
        eval_results = self.trainer.evaluate(eval_dataset=eval_dataset)
        
        logger.info("Evaluation completed", metrics=eval_results)
        return eval_results


class MapperTrainer(Trainer):
    """Custom Trainer with metrics collection for mapper fine-tuning."""
    
    def __init__(self, metrics_collector: LoRATrainer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_collector = metrics_collector
    
    def log(self, logs: Dict[str, float]) -> None:
        """Override log method to collect custom metrics."""
        super().log(logs)
        self.metrics_collector.collect_metrics(logs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss for causal language modeling.
        
        Args:
            model: The model
            inputs: Input batch
            return_outputs: Whether to return model outputs
            
        Returns:
            Loss tensor and optionally model outputs
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        # Compute causal LM loss
        if labels is not None:
            # Shift labels for causal LM
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for loss computation
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss


def create_training_config(
    output_dir: str = "./checkpoints",
    learning_rate: float = 2e-4,
    num_epochs: int = 2,
    batch_size: int = 4,
    **kwargs
) -> LoRATrainingConfig:
    """
    Create training configuration with sensible defaults.
    
    Args:
        output_dir: Directory to save checkpoints
        learning_rate: Learning rate (default: 2e-4 as specified)
        num_epochs: Number of training epochs (default: 2 as specified)
        batch_size: Per-device batch size
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured LoRATrainingConfig
    """
    config = LoRATrainingConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        **kwargs
    )
    
    return config
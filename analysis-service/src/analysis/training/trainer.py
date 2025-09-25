"""
Phi-3-Mini trainer for risk assessment and compliance analysis.

Handles fine-tuning of Phi-3-Mini models for context-aware risk assessments
and remediation guidance.

Note: This module requires additional ML dependencies. Install with:
pip install -r requirements-training.txt
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Initialize variables before try block
TRAINING_DEPENDENCIES_AVAILABLE = False
_MISSING_DEPENDENCY_ERROR = "Unknown import error"

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    TRAINING_DEPENDENCIES_AVAILABLE = True
    _MISSING_DEPENDENCY_ERROR = None
except ImportError as e:
    TRAINING_DEPENDENCIES_AVAILABLE = False
    _MISSING_DEPENDENCY_ERROR = str(e)

from ..shared_integration import get_shared_logger

logger = get_shared_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for Phi-3-Mini training."""

    # Model configuration
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    max_seq_length: int = 4096

    # Training parameters
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 5

    # Output
    output_dir: str = "checkpoints/phi3-mini-analysis"

    def get_lora_config(self) -> Dict[str, Any]:
        """Get LoRA configuration parameters."""
        return {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        }

    def get_training_args(self) -> Dict[str, Any]:
        """Get additional training arguments."""
        return {
            "gradient_accumulation_steps": 4,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "gradient_checkpointing": True,
            "fp16": True,
            "dataloader_num_workers": 4,
            "eval_steps": 500,
            "save_steps": 1000,
            "logging_steps": 100,
        }


class Phi3Trainer:
    """Trainer for Phi-3-Mini models focused on risk assessment and compliance analysis."""

    def __init__(self, config: TrainingConfig):
        if not TRAINING_DEPENDENCIES_AVAILABLE:
            raise ImportError(
                f"Training dependencies not available: {_MISSING_DEPENDENCY_ERROR}. "
                "Please install with: pip install -r requirements-training.txt"
            )

        self.config = config
        self.logger = logger.bind(component="phi3_trainer")
        self.model = None
        self.tokenizer = None

    def setup_model(self) -> None:
        """Setup Phi-3-Mini model with LoRA configuration."""
        try:
            self.logger.info(
                "Loading Phi-3-Mini model", model_name=self.config.model_name
            )

            # Track model setup with metrics
            # Note: track_request_metrics is a decorator, not a context manager
            # We'll use direct metrics tracking here

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name, trust_remote_code=True
            )

            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            extra_args = self.config.get_training_args()
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=(
                    torch.float16 if extra_args.get("fp16", False) else torch.float32
                ),
                device_map="auto",
                trust_remote_code=True,
            )

            # Setup LoRA
            lora_config = self.config.get_lora_config()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_config["r"],
                lora_alpha=lora_config["lora_alpha"],
                lora_dropout=lora_config["lora_dropout"],
                target_modules=lora_config["target_modules"],
                bias="none",
            )

            self.model = get_peft_model(self.model, peft_config)

            # Print trainable parameters
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in self.model.parameters())

            self.logger.info(
                "Model setup complete",
                trainable_params=trainable_params,
                total_params=total_params,
                trainable_percentage=f"{100 * trainable_params / total_params:.2f}%",
            )

        except Exception as e:
            self.logger.error("Failed to setup model", error=str(e))
            raise

    def train(
        self,
        train_dataset,
        eval_dataset=None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the Phi-3-Mini model.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            resume_from_checkpoint: Optional checkpoint path to resume from

        Returns:
            Training results and metrics
        """
        if self.model is None:
            self.setup_model()

        # Setup training arguments
        extra_args = self.config.get_training_args()
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            # Additional arguments from config
            **extra_args,
            # Evaluation settings
            eval_steps=extra_args["eval_steps"] if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=3,
            # Other settings
            remove_unused_columns=False,
            load_best_model_at_end=bool(eval_dataset),
            metric_for_best_model="eval_loss" if eval_dataset else None,
            report_to=None,  # Disable wandb by default
        )

        # Setup data collator
        extra_args = self.config.get_training_args()
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8 if extra_args.get("fp16", False) else None,
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        try:
            self.logger.info("Starting training", config=self.config.__dict__)

            # Track training start
            start_time = time.time()
            try:
                # Train the model
                train_result = trainer.train(
                    resume_from_checkpoint=resume_from_checkpoint
                )
            finally:
                training_duration = time.time() - start_time
                self.logger.info(
                    "Training duration", duration_seconds=training_duration
                )

            # Save the final model
            trainer.save_model()
            trainer.save_state()

            # Save training metrics
            metrics = train_result.metrics
            metrics["train_samples"] = len(train_dataset)
            if eval_dataset:
                metrics["eval_samples"] = len(eval_dataset)

            # Save metrics to file
            with open(
                os.path.join(self.config.output_dir, "training_metrics.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(metrics, f, indent=2)

            self.logger.info("Training completed successfully", metrics=metrics)

            return {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "metrics": metrics,
                "trainer": trainer,
            }

        except Exception as e:
            self.logger.error("Training failed", error=str(e))
            raise

    def evaluate(self, eval_dataset) -> Dict[str, float]:
        """
        Evaluate the trained model.

        Args:
            eval_dataset: Evaluation dataset

        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model() first.")

        # Create a temporary trainer for evaluation
        extra_args = self.config.get_training_args()
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_eval_batch_size=self.config.batch_size,
            dataloader_num_workers=extra_args["dataloader_num_workers"],
            remove_unused_columns=False,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        # Run evaluation with metrics tracking
        start_time = time.time()
        try:
            eval_results = trainer.evaluate()
        finally:
            eval_duration = time.time() - start_time
            self.logger.info("Evaluation duration", duration_seconds=eval_duration)

        self.logger.info("Evaluation completed", metrics=eval_results)

        return eval_results

    def save_model(
        self, output_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save the trained model with metadata.

        Args:
            output_path: Path to save the model
            metadata: Optional metadata to save with the model
        """
        if self.model is None:
            raise ValueError("Model not initialized")

        os.makedirs(output_path, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        # Save metadata
        if metadata:
            with open(
                os.path.join(output_path, "metadata.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(metadata, f, indent=2)

        # Save training config
        with open(
            os.path.join(output_path, "training_config.json"), "w", encoding="utf-8"
        ) as f:
            lora_config = self.config.get_lora_config()
            config_dict = {
                "model_name": self.config.model_name,
                "max_seq_length": self.config.max_seq_length,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "num_epochs": self.config.num_epochs,
                "output_dir": self.config.output_dir,
                "lora_config": lora_config,
            }
            json.dump(config_dict, f, indent=2)

        self.logger.info("Model saved successfully", output_path=output_path)

    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from path.

        Args:
            model_path: Path to the saved model
        """
        try:
            self.logger.info("Loading trained model", model_path=model_path)

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )

            self.logger.info("Model loaded successfully")

        except Exception as e:
            self.logger.error("Failed to load model", error=str(e))
            raise

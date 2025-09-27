"""
LoRA trainer for fine-tuning models.

Single responsibility: Model training orchestration and execution.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .model_loader import ModelLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for LoRA fine-tuning."""

    # LoRA hyperparameters (aligned with guidelines for Llama-3-8B)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Training hyperparameters (aligned with guidelines)
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    max_sequence_length: int = 2048
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
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


class LoRATrainer:
    """
    LoRA trainer for fine-tuning models.

    Single responsibility: Execute training with LoRA configuration.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize LoRA trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.training_metrics: Dict[str, List[float]] = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
            "epoch": [],
        }

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        logger.info("LoRATrainer initialized", config=config.__dict__)

    def setup_model_and_tokenizer(self, model_loader: "ModelLoader") -> Tuple[Any, Any]:
        """
        Set up model and tokenizer for training.

        Args:
            model_loader: Model loader instance

        Returns:
            Tuple of (peft_model, tokenizer)
        """
        logger.info("Setting up model and tokenizer for LoRA training")

        # Load base model and tokenizer
        base_model, tokenizer = model_loader.load_model_and_tokenizer()

        # Create LoRA configuration
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )

            # Apply LoRA to model
            peft_model = get_peft_model(base_model, lora_config)

        except ImportError:
            logger.warning("PEFT not available, using base model")
            peft_model = base_model

        self.model = peft_model
        self.tokenizer = tokenizer

        return peft_model, tokenizer

    def prepare_datasets(
        self,
        train_examples: List[Dict[str, str]],
        eval_examples: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[Any, Optional[Any]]:
        """
        Prepare training and evaluation datasets.

        Args:
            train_examples: Training examples
            eval_examples: Optional evaluation examples

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")

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

    def train(
        self,
        train_examples: List[Dict[str, str]],
        eval_examples: Optional[List[Dict[str, str]]] = None,
        model_loader: Optional["ModelLoader"] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete training pipeline.

        Args:
            train_examples: Training examples
            eval_examples: Optional evaluation examples
            model_loader: Model loader instance
            resume_from_checkpoint: Optional checkpoint path

        Returns:
            Training metrics and results
        """
        logger.info("Starting LoRA training pipeline")

        if model_loader is None:
            from .model_loader import ModelLoader

            model_loader = ModelLoader()

        # Setup model and tokenizer
        if self.model is None or self.tokenizer is None:
            self.setup_model_and_tokenizer(model_loader)

        # Prepare datasets
        train_dataset, eval_dataset = self.prepare_datasets(
            train_examples, eval_examples
        )

        # Create trainer
        trainer = self._create_trainer(train_dataset, eval_dataset)

        # Start training
        logger.info("Beginning training")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save final model
        trainer.save_model()
        trainer.save_state()

        # Collect final metrics
        final_metrics = {
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get(
                "train_samples_per_second", 0
            ),
            "train_steps_per_second": train_result.metrics.get(
                "train_steps_per_second", 0
            ),
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
        )

        return final_metrics

    def _create_trainer(
        self, train_dataset: Any, eval_dataset: Optional[Any] = None
    ) -> Any:
        """Create HuggingFace Trainer."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer not initialized")

        try:
            from transformers import (
                Trainer,
                TrainingArguments,
                DataCollatorForLanguageModeling,
            )

            # Create training arguments
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_train_epochs,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.per_device_eval_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                warmup_steps=self.config.warmup_steps,
                logging_steps=self.config.logging_steps,
                eval_steps=self.config.eval_steps,
                save_steps=self.config.save_steps,
                eval_strategy="steps" if eval_dataset else "no",
                save_strategy="steps",
                load_best_model_at_end=True if eval_dataset else False,
                metric_for_best_model="eval_loss" if eval_dataset else None,
                greater_is_better=False,
                fp16=self.config.fp16,
                gradient_checkpointing=self.config.gradient_checkpointing,
                dataloader_num_workers=self.config.dataloader_num_workers,
                remove_unused_columns=False,
                run_name=self.config.run_name,
                report_to=["tensorboard"],
                logging_dir=f"{self.config.output_dir}/logs",
            )

            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
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

            self.trainer = trainer
            return trainer

        except ImportError:
            raise RuntimeError("transformers library not available")

    def collect_metrics(self, logs: Dict[str, float]) -> None:
        """Collect training metrics for monitoring."""
        for key, value in logs.items():
            if key in self.training_metrics:
                self.training_metrics[key].append(value)


class MapperDataset:
    """Dataset class for instruction-following mapper training data."""

    def __init__(
        self, examples: List[Dict[str, str]], tokenizer: Any, max_length: int = 2048
    ):
        """
        Initialize dataset.

        Args:
            examples: Training examples with 'instruction' and 'response' keys
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

        logger.info(
            "Dataset initialized", num_examples=len(examples), max_length=max_length
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get tokenized example for training."""
        example = self.examples[idx]

        # Create instruction-following prompt
        prompt = self._create_instruction_prompt(
            instruction=example["instruction"], response=example["response"]
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

    def _create_instruction_prompt(self, instruction: str, response: str = "") -> str:
        """Create instruction-following prompt format."""
        if response:
            return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
        else:
            return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

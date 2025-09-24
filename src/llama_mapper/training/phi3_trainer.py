"""
Phi-3 Trainer for fine-tuning Microsoft Phi-3-Mini models for compliance analysis.

This module provides specialized training configuration and pipeline for Phi-3 models
optimized for compliance analysis tasks with lower resource requirements.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import structlog

try:
    import torch  # type: ignore
except Exception:
    torch = None

try:
    from peft import LoraConfig, PeftModel  # type: ignore
except Exception:

    class LoraConfig:  # type: ignore
        def __init__(self, **kwargs: Any) -> None:
            pass

    class PeftModel:  # type: ignore
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass


try:
    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        PreTrainedTokenizer,
        Trainer,
        TrainingArguments,
    )
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False
    # Minimal stubs for testing
    class TrainingArguments:  # type: ignore
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    class Trainer:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class PreTrainedTokenizer:  # type: ignore
        pad_token_id: int = 0

        def __len__(self) -> int:
            return 0

    class DataCollatorForLanguageModeling:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class AutoTokenizer:  # type: ignore
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> Any:
            return PreTrainedTokenizer()

    class AutoModelForCausalLM:  # type: ignore
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> Any:
            return None

    class BitsAndBytesConfig:  # type: ignore
        def __init__(self, **kwargs: Any) -> None:
            pass


from .model_loader import create_instruction_prompt

logger = structlog.get_logger(__name__)


@dataclass
class Phi3TrainingConfig:
    """Configuration for Phi-3 fine-tuning optimized for compliance analysis."""

    # Model configuration
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    model_type: str = "phi-3-mini"

    # LoRA hyperparameters (optimized for Phi-3 efficiency)
    lora_r: int = 128  # Lower rank for efficiency
    lora_alpha: int = 256  # 2x rank for stability
    lora_dropout: float = 0.1
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]  # Attention layers only
    )

    # Training hyperparameters (optimized for Phi-3)
    learning_rate: float = 1e-4  # Higher learning rate for smaller model
    num_train_epochs: int = 2  # Fewer epochs for efficiency
    max_sequence_length: int = 4096  # Phi-3 supports longer sequences

    # Training configuration (optimized for Phi-3 memory usage)
    per_device_train_batch_size: int = 8  # Larger batch size possible
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4  # Effective batch size = 32
    warmup_steps: int = 50  # Fewer warmup steps
    weight_decay: float = 0.01
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 250

    # Memory optimization (Phi-3 specific)
    fp16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    use_4bit_quantization: bool = True

    # Output configuration
    output_dir: str = "./checkpoints/phi3-analyst"
    run_name: str = "phi3-analyst-lora"

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
            eval_strategy="steps",
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


class Phi3Dataset:
    """Dataset class for Phi-3 compliance analysis training data."""

    def __init__(
        self,
        examples: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 4096,
    ):
        """
        Initialize dataset with training examples.

        Args:
            examples: List of training examples with 'instruction' and 'response' keys
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length (Phi-3 supports up to 4k)
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

        logger.info(
            "Phi3Dataset initialized",
            num_examples=len(examples),
            max_length=max_length,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get tokenized example for training."""
        example = self.examples[idx]

        # Create instruction-following prompt for Phi-3 format
        prompt = self._create_phi3_prompt(
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

    def _create_phi3_prompt(self, instruction: str, response: str) -> str:
        """Create Phi-3 specific prompt format."""
        return f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n{response}<|end|>"


class Phi3Trainer:
    """
    Phi-3 trainer for fine-tuning on compliance analysis tasks.

    Optimized for Microsoft Phi-3-Mini models with efficient LoRA configuration
    and memory optimization for compliance analysis scenarios.
    """

    def __init__(self, config: Phi3TrainingConfig):
        """
        Initialize Phi-3 trainer with configuration.

        Args:
            config: Training configuration optimized for Phi-3
        """
        self.config = config
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.trainer: Optional[Any] = None
        self.training_metrics: Dict[str, List[float]] = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
            "epoch": [],
        }

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        logger.info(
            "Phi3Trainer initialized",
            config=config.__dict__,
        )

    def setup_model_and_tokenizer(self) -> Tuple[PeftModel, PreTrainedTokenizer]:
        """
        Set up Phi-3 model and tokenizer for training.

        Returns:
            Tuple of (peft_model, tokenizer)
        """
        logger.info("Setting up Phi-3 model and tokenizer for LoRA training")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Quantization configuration for memory efficiency
        quantization_config = None
        if self.config.use_4bit_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch else None,
            )

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch else None,
        )

        # Create LoRA configuration optimized for Phi-3
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply LoRA to model
        from peft import get_peft_model  # type: ignore

        peft_model = get_peft_model(base_model, lora_config)  # type: ignore

        # Print trainable parameters
        peft_model.print_trainable_parameters()

        self.model = peft_model
        self.tokenizer = tokenizer

        return peft_model, tokenizer

    def prepare_datasets(
        self,
        train_examples: List[Dict[str, str]],
        eval_examples: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[Phi3Dataset, Optional[Phi3Dataset]]:
        """
        Prepare training and evaluation datasets.

        Args:
            train_examples: Training examples with 'instruction' and 'response'
            eval_examples: Optional evaluation examples

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer not initialized. Call setup_model_and_tokenizer first."
            )

        logger.info(
            "Preparing Phi-3 datasets",
            num_train=len(train_examples),
            num_eval=len(eval_examples) if eval_examples else 0,
        )

        train_dataset = Phi3Dataset(
            examples=train_examples,
            tokenizer=self.tokenizer,
            max_length=self.config.max_sequence_length,
        )

        eval_dataset = None
        if eval_examples:
            eval_dataset = Phi3Dataset(
                examples=eval_examples,
                tokenizer=self.tokenizer,
                max_length=self.config.max_sequence_length,
            )

        return train_dataset, eval_dataset

    def create_trainer(
        self,
        train_dataset: Phi3Dataset,
        eval_dataset: Optional[Phi3Dataset] = None,
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
        trainer = Phi3CustomTrainer(
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
        Run the complete Phi-3 training pipeline.

        Args:
            train_examples: Training examples
            eval_examples: Optional evaluation examples
            resume_from_checkpoint: Optional checkpoint path to resume from

        Returns:
            Training metrics and results
        """
        logger.info("Starting Phi-3 training pipeline")

        # Setup model and tokenizer
        if self.model is None or self.tokenizer is None:
            self.setup_model_and_tokenizer()

        # Prepare datasets
        train_dataset, eval_dataset = self.prepare_datasets(
            train_examples, eval_examples
        )

        # Create trainer
        trainer = self.create_trainer(train_dataset, eval_dataset)

        # Start training
        logger.info("Beginning Phi-3 training")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)  # type: ignore

        # Save final model
        trainer.save_model()  # type: ignore
        trainer.save_state()  # type: ignore

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
            "model_type": "phi-3-mini",
            "lora_config": {
                "r": self.config.lora_r,
                "alpha": self.config.lora_alpha,
                "target_modules": self.config.target_modules,
            },
        }

        # Save metrics to file
        metrics_path = os.path.join(
            self.config.output_dir, "phi3_training_metrics.json"
        )
        with open(metrics_path, "w") as f:
            json.dump(final_metrics, f, indent=2)

        logger.info(
            "Phi-3 training completed",
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

    def evaluate(self, eval_dataset: Optional[Phi3Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation on the Phi-3 model.

        Args:
            eval_dataset: Optional evaluation dataset

        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call train first.")

        logger.info("Running Phi-3 evaluation")
        eval_results = self.trainer.evaluate(eval_dataset=eval_dataset)

        logger.info("Phi-3 evaluation completed", metrics=eval_results)
        from typing import cast

        return cast(Dict[str, float], eval_results)


class Phi3CustomTrainer(Trainer):
    """Custom Trainer with metrics collection for Phi-3 fine-tuning."""

    def __init__(self, metrics_collector: Phi3Trainer, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.metrics_collector = metrics_collector

    def log(self, logs: Dict[str, float]) -> None:
        """Override log method to collect custom metrics."""
        super().log(logs)
        self.metrics_collector.collect_metrics(logs)

    def compute_loss(
        self,
        model: Any,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
    ) -> Any:
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
            if torch:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
            else:
                loss = outputs.loss
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


def create_phi3_training_config(
    output_dir: str = "./checkpoints/phi3-analyst",
    learning_rate: float = 1e-4,
    num_epochs: int = 2,
    batch_size: int = 8,
    **kwargs: Any,
) -> Phi3TrainingConfig:
    """
    Create Phi-3 training configuration with sensible defaults.

    Args:
        output_dir: Directory to save checkpoints
        learning_rate: Learning rate (default: 1e-4 for Phi-3)
        num_epochs: Number of training epochs (default: 2 for efficiency)
        batch_size: Per-device batch size (default: 8 for Phi-3)
        **kwargs: Additional configuration parameters

    Returns:
        Configured Phi3TrainingConfig
    """
    config = Phi3TrainingConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        **kwargs,
    )

    return config

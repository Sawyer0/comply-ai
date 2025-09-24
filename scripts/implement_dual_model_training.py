#!/usr/bin/env python3
"""
Dual-Model Training Implementation Script
Implements the Llama-3-8B Mapper + Phi-3-Mini Analyst architecture
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DualModelTrainer:
    """Orchestrates training of both Llama-3-8B Mapper and Phi-3-Mini Analyst models."""

    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        self.config = self._load_config(config_path)
        self.setup_wandb()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration."""
        with open(config_path, "r") as f:
            return json.load(f)

    def setup_wandb(self):
        """Initialize Weights & Biases for experiment tracking."""
        wandb.init(
            project="comply-ai-dual-model",
            config=self.config,
            name=f"dual-model-{self.config['experiment_name']}",
        )

    def load_mapper_datasets(self) -> Dict[str, Any]:
        """Load datasets for Llama-3-8B Mapper training."""
        logger.info("Loading mapper training datasets...")

        datasets = {}

        # PII Detection Dataset
        if self.config["datasets"]["pii_detection"]["enabled"]:
            datasets["pii"] = load_dataset("ai4privacy/pii-masking-43k", split="train")
            logger.info("Loaded PII dataset: %s examples", len(datasets["pii"]))

        # Security Attack Dataset
        if self.config["datasets"]["security_attacks"]["enabled"]:
            datasets["security"] = load_dataset("ibm-research/AttaQ", split="train")
            logger.info(
                "Loaded security dataset: %s examples", len(datasets["security"])
            )

        # Content Moderation Dataset
        if self.config["datasets"]["content_moderation"]["enabled"]:
            datasets["content"] = load_dataset("allenai/wildguardmix", split="train")
            logger.info("Loaded content dataset: %s examples", len(datasets["content"]))

        # Custom compliance mapping dataset
        if self.config["datasets"]["custom_mapping"]["enabled"]:
            custom_path = self.config["datasets"]["custom_mapping"]["path"]
            datasets["custom"] = self._load_custom_dataset(custom_path)
            logger.info("Loaded custom dataset: %s examples", len(datasets["custom"]))

        return datasets

    def load_analyst_datasets(self) -> Dict[str, Any]:
        """Load datasets for Phi-3-Mini Analyst training."""
        logger.info("Loading analyst training datasets...")

        datasets = {}

        # Policy Compliance Q&A
        if self.config["datasets"]["policy_qa"]["enabled"]:
            datasets["policy_qa"] = load_dataset("qa4pc/QA4PC", split="train")
            logger.info(
                "Loaded policy QA dataset: %s examples", len(datasets["policy_qa"])
            )

        # GDPR Dataset
        if self.config["datasets"]["gdpr"]["enabled"]:
            datasets["gdpr"] = load_dataset("AndreaSimeri/GDPR", split="train")
            logger.info("Loaded GDPR dataset: %s examples", len(datasets["gdpr"]))

        # Legal Reasoning Dataset
        if self.config["datasets"]["legal_reasoning"]["enabled"]:
            datasets["legal"] = load_dataset("nguha/legalbench", split="train")
            logger.info("Loaded legal dataset: %s examples", len(datasets["legal"]))

        return datasets

    def _load_custom_dataset(self, path: str):
        """Load custom dataset from JSONL file."""
        import pandas as pd

        data = []
        with open(path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return pd.DataFrame(data)

    def setup_mapper_model(self):
        """Setup Llama-3-8B model with LoRA for mapping tasks."""
        logger.info("Setting up Llama-3-8B Mapper model...")

        model_name = self.config["models"]["mapper"]["base_model"]

        # Quantization configuration for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Load model and tokenizer
        self.mapper_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mapper_model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quantization_config, device_map="auto"
        )

        # Add padding token if not present
        if self.mapper_tokenizer.pad_token is None:
            self.mapper_tokenizer.pad_token = self.mapper_tokenizer.eos_token

        # LoRA configuration for mapper
        lora_config = LoraConfig(
            r=self.config["models"]["mapper"]["lora"]["r"],
            lora_alpha=self.config["models"]["mapper"]["lora"]["alpha"],
            target_modules=self.config["models"]["mapper"]["lora"]["target_modules"],
            lora_dropout=self.config["models"]["mapper"]["lora"]["dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA to model
        self.mapper_model = get_peft_model(self.mapper_model, lora_config)
        self.mapper_model.print_trainable_parameters()

        logger.info("Mapper model setup complete")

    def setup_analyst_model(self):
        """Setup Phi-3-Mini model with LoRA for analysis tasks."""
        logger.info("Setting up Phi-3-Mini Analyst model...")

        model_name = self.config["models"]["analyst"]["base_model"]

        # Quantization configuration
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Load model and tokenizer
        self.analyst_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.analyst_model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quantization_config, device_map="auto"
        )

        # Add padding token if not present
        if self.analyst_tokenizer.pad_token is None:
            self.analyst_tokenizer.pad_token = self.analyst_tokenizer.eos_token

        # LoRA configuration for analyst
        lora_config = LoraConfig(
            r=self.config["models"]["analyst"]["lora"]["r"],
            lora_alpha=self.config["models"]["analyst"]["lora"]["alpha"],
            target_modules=self.config["models"]["analyst"]["lora"]["target_modules"],
            lora_dropout=self.config["models"]["analyst"]["lora"]["dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA to model
        self.analyst_model = get_peft_model(self.analyst_model, lora_config)
        self.analyst_model.print_trainable_parameters()

        logger.info("Analyst model setup complete")

    def format_mapper_data(self, datasets: Dict[str, Any]) -> List[Dict[str, str]]:
        """Format datasets for mapper training."""
        logger.info("Formatting mapper training data...")

        formatted_data = []

        # Format PII detection data
        if "pii" in datasets:
            for example in datasets["pii"]:
                formatted_data.append(
                    {
                        "instruction": f"Map this PII detection to compliance taxonomy: {example.get('text', '')}",
                        "input": example.get("text", ""),
                        "output": self._generate_pii_mapping(example),
                        "metadata": {"source": "pii_detection", "type": "mapping"},
                    }
                )

        # Format security attack data
        if "security" in datasets:
            for example in datasets["security"]:
                formatted_data.append(
                    {
                        "instruction": f"Map this security attack to compliance taxonomy: {example.get('text', '')}",
                        "input": example.get("text", ""),
                        "output": self._generate_security_mapping(example),
                        "metadata": {"source": "security_attack", "type": "mapping"},
                    }
                )

        # Format content moderation data
        if "content" in datasets:
            for example in datasets["content"]:
                formatted_data.append(
                    {
                        "instruction": f"Map this content to compliance taxonomy: {example.get('text', '')}",
                        "input": example.get("text", ""),
                        "output": self._generate_content_mapping(example),
                        "metadata": {"source": "content_moderation", "type": "mapping"},
                    }
                )

        logger.info("Formatted %s mapper training examples", len(formatted_data))
        return formatted_data

    def format_analyst_data(self, datasets: Dict[str, Any]) -> List[Dict[str, str]]:
        """Format datasets for analyst training."""
        logger.info("Formatting analyst training data...")

        formatted_data = []

        # Format policy Q&A data
        if "policy_qa" in datasets:
            for example in datasets["policy_qa"]:
                formatted_data.append(
                    {
                        "instruction": f"Analyze this compliance scenario: {example.get('question', '')}",
                        "input": example.get("question", ""),
                        "output": self._generate_compliance_analysis(example),
                        "metadata": {"source": "policy_qa", "type": "analysis"},
                    }
                )

        # Format GDPR data
        if "gdpr" in datasets:
            for example in datasets["gdpr"]:
                formatted_data.append(
                    {
                        "instruction": f"Provide compliance analysis for: {example.get('text', '')[:200]}...",
                        "input": example.get("text", ""),
                        "output": self._generate_gdpr_analysis(example),
                        "metadata": {"source": "gdpr", "type": "analysis"},
                    }
                )

        logger.info("Formatted %s analyst training examples", len(formatted_data))
        return formatted_data

    def _generate_pii_mapping(self, example: Dict[str, Any]) -> str:
        """Generate PII mapping output."""
        return json.dumps(
            {
                "taxonomy": ["PII.Identifier.Email", "PII.Contact.Phone"],
                "scores": {"PII.Identifier.Email": 0.95, "PII.Contact.Phone": 0.87},
                "confidence": 0.91,
                "provenance": {
                    "detector": "regex-pii",
                    "timestamp": "2025-01-27T10:30:00Z",
                },
                "notes": "High-confidence PII detection in user profile data",
            }
        )

    def _generate_security_mapping(self, example: Dict[str, Any]) -> str:
        """Generate security attack mapping output."""
        return json.dumps(
            {
                "taxonomy": ["HARM.VIOLENCE.Physical", "PROMPT_INJECTION.Other"],
                "scores": {
                    "HARM.VIOLENCE.Physical": 0.88,
                    "PROMPT_INJECTION.Other": 0.92,
                },
                "confidence": 0.90,
                "provenance": {
                    "detector": "llama-guard",
                    "timestamp": "2025-01-27T10:30:00Z",
                },
                "notes": "Security threat detected with high confidence",
            }
        )

    def _generate_content_mapping(self, example: Dict[str, Any]) -> str:
        """Generate content moderation mapping output."""
        return json.dumps(
            {
                "taxonomy": ["HARM.SPEECH.Toxicity", "HARM.SPEECH.Hate.Other"],
                "scores": {
                    "HARM.SPEECH.Toxicity": 0.85,
                    "HARM.SPEECH.Hate.Other": 0.78,
                },
                "confidence": 0.82,
                "provenance": {
                    "detector": "deberta-toxicity",
                    "timestamp": "2025-01-27T10:30:00Z",
                },
                "notes": "Content moderation alert with moderate confidence",
            }
        )

    def _generate_compliance_analysis(self, example: Dict[str, Any]) -> str:
        """Generate compliance analysis output."""
        return json.dumps(
            {
                "risk_assessment": {
                    "severity": "HIGH",
                    "compliance_frameworks": ["GDPR", "CCPA"],
                    "affected_articles": ["GDPR-6", "GDPR-32"],
                },
                "remediation_steps": [
                    "1. Implement data pseudonymization",
                    "2. Establish consent management system",
                    "3. Create data subject rights portal",
                ],
                "audit_evidence": {
                    "required_documentation": ["DPIA", "Consent Records"],
                    "compliance_gaps": ["Missing consent mechanism"],
                },
                "confidence": 0.89,
            }
        )

    def _generate_gdpr_analysis(self, example: Dict[str, Any]) -> str:
        """Generate GDPR analysis output."""
        return json.dumps(
            {
                "risk_assessment": {
                    "severity": "MEDIUM",
                    "compliance_frameworks": ["GDPR"],
                    "affected_articles": ["GDPR-5", "GDPR-25"],
                },
                "remediation_steps": [
                    "1. Review data minimization practices",
                    "2. Implement privacy by design principles",
                ],
                "confidence": 0.85,
            }
        )

    def train_mapper_model(self, training_data: List[Dict[str, str]]):
        """Train the Llama-3-8B Mapper model."""
        logger.info("Starting mapper model training...")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config["training"]["mapper"]["output_dir"],
            per_device_train_batch_size=self.config["training"]["mapper"]["batch_size"],
            gradient_accumulation_steps=self.config["training"]["mapper"][
                "gradient_accumulation"
            ],
            num_train_epochs=self.config["training"]["mapper"]["epochs"],
            learning_rate=self.config["training"]["mapper"]["learning_rate"],
            warmup_steps=self.config["training"]["mapper"]["warmup_steps"],
            logging_steps=self.config["training"]["mapper"]["logging_steps"],
            save_steps=self.config["training"]["mapper"]["save_steps"],
            fp16=True,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to="wandb",
        )

        # Create trainer
        trainer = Trainer(
            model=self.mapper_model,
            args=training_args,
            train_dataset=training_data,
            tokenizer=self.mapper_tokenizer,
            data_collator=self._create_data_collator(self.mapper_tokenizer),
        )

        # Train model
        trainer.train()

        # Save model
        trainer.save_model()
        self.mapper_tokenizer.save_pretrained(
            self.config["training"]["mapper"]["output_dir"]
        )

        logger.info("Mapper model training complete")

    def train_analyst_model(self, training_data: List[Dict[str, str]]):
        """Train the Phi-3-Mini Analyst model."""
        logger.info("Starting analyst model training...")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config["training"]["analyst"]["output_dir"],
            per_device_train_batch_size=self.config["training"]["analyst"][
                "batch_size"
            ],
            gradient_accumulation_steps=self.config["training"]["analyst"][
                "gradient_accumulation"
            ],
            num_train_epochs=self.config["training"]["analyst"]["epochs"],
            learning_rate=self.config["training"]["analyst"]["learning_rate"],
            warmup_steps=self.config["training"]["analyst"]["warmup_steps"],
            logging_steps=self.config["training"]["analyst"]["logging_steps"],
            save_steps=self.config["training"]["analyst"]["save_steps"],
            fp16=True,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to="wandb",
        )

        # Create trainer
        trainer = Trainer(
            model=self.analyst_model,
            args=training_args,
            train_dataset=training_data,
            tokenizer=self.analyst_tokenizer,
            data_collator=self._create_data_collator(self.analyst_tokenizer),
        )

        # Train model
        trainer.train()

        # Save model
        trainer.save_model()
        self.analyst_tokenizer.save_pretrained(
            self.config["training"]["analyst"]["output_dir"]
        )

        logger.info("Analyst model training complete")

    def _create_data_collator(self, tokenizer):
        """Create data collator for training."""
        from transformers import DataCollatorForSeq2Seq

        return DataCollatorForSeq2Seq(
            tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt"
        )

    def run_training(self):
        """Run complete dual-model training pipeline."""
        logger.info("Starting dual-model training pipeline...")

        try:
            # Load datasets
            mapper_datasets = self.load_mapper_datasets()
            analyst_datasets = self.load_analyst_datasets()

            # Setup models
            self.setup_mapper_model()
            self.setup_analyst_model()

            # Format training data
            mapper_data = self.format_mapper_data(mapper_datasets)
            analyst_data = self.format_analyst_data(analyst_datasets)

            # Train models
            self.train_mapper_model(mapper_data)
            self.train_analyst_model(analyst_data)

            logger.info("Dual-model training pipeline complete!")

        except Exception as e:
            logger.error("Training failed: %s", e)
            raise
        finally:
            wandb.finish()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Dual-Model Training Implementation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration JSON file",
    )
    parser.add_argument(
        "--mapper-only", action="store_true", help="Train only the mapper model"
    )
    parser.add_argument(
        "--analyst-only", action="store_true", help="Train only the analyst model"
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = DualModelTrainer(args.config)

    # Run training
    trainer.run_training()


if __name__ == "__main__":
    main()

"""
Training Pipeline for Analysis Service

Implements training pipelines for Phi-3 models used in compliance analysis.
Follows SRP by focusing only on training orchestration within the analysis domain.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from ..config.settings import AnalysisSettings
from ..ml.model_server import ModelServer
from ..quality.monitoring import QualityMonitor
from ..schemas.analysis_schemas import TrainingResult

logger = structlog.get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""

    # Model configuration
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    model_type: str = "phi-3-mini"

    # LoRA configuration
    lora_r: int = 128
    lora_alpha: int = 256

    # Training parameters
    learning_rate: float = 1e-4
    num_epochs: int = 2

    # Output configuration
    output_dir: str = "./checkpoints/analysis-phi3"


class TrainingPipeline:
    """
    Training pipeline for analysis models.

    Handles the complete training workflow including:
    - Data preparation
    - Model training
    - Quality validation
    - Checkpoint management
    """

    def __init__(
        self,
        config: TrainingConfig,
        model_server: ModelServer,
        quality_monitor: QualityMonitor,
        settings: AnalysisSettings,
    ):
        self.config = config
        self.model_server = model_server
        self.quality_monitor = quality_monitor
        self.settings = settings
        self.logger = logger.bind(component="training_pipeline")

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

    async def execute_training(
        self,
        training_data: List[Dict[str, Any]],
        validation_data: Optional[List[Dict[str, Any]]] = None,
    ) -> TrainingResult:
        """
        Execute the complete training pipeline.

        Args:
            training_data: Training examples
            validation_data: Optional validation examples

        Returns:
            Training results with metrics and model path
        """
        training_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.logger.info(
            "Starting training pipeline",
            training_id=training_id,
            num_training_samples=len(training_data),
            num_validation_samples=len(validation_data) if validation_data else 0,
        )

        try:
            # Prepare training data
            prepared_data = await self._prepare_training_data(
                training_data, validation_data
            )

            # Initialize training environment
            await self._setup_training_environment()

            # Execute training
            training_metrics = await self._run_training(
                prepared_data["train"], prepared_data.get("validation")
            )

            # Validate training quality
            quality_results = await self._validate_training_quality(training_metrics)

            # Save model and artifacts
            model_path = await self._save_training_artifacts(
                training_id, training_metrics
            )

            result = TrainingResult(
                training_id=training_id,
                model_path=model_path,
                metrics=training_metrics,
                quality_score=quality_results["overall_score"],
                config=self.config.__dict__,
                completed_at=datetime.now(),
            )

            self.logger.info(
                "Training pipeline completed successfully",
                training_id=training_id,
                final_loss=training_metrics.get("final_loss"),
                quality_score=quality_results["overall_score"],
            )

            return result

        except Exception as e:
            self.logger.error(
                "Training pipeline failed", training_id=training_id, error=str(e)
            )
            raise

    async def get_training_status(self, training_id: str) -> Dict[str, Any]:
        """Get status of a training job."""

        return {
            "training_id": training_id,
            "status": "completed",  # Simulated
            "progress": 1.0,
            "current_epoch": self.config.num_epochs,
            "total_epochs": self.config.num_epochs,
        }

    async def _prepare_training_data(
        self,
        training_data: List[Dict[str, Any]],
        validation_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Prepare and validate training data."""

        self.logger.info("Preparing training data")

        # Validate data format
        for i, example in enumerate(training_data):
            if "instruction" not in example or "response" not in example:
                raise ValueError(f"Training example {i} missing required fields")

        # Create instruction-response pairs for Phi-3 format
        prepared_train = []
        for example in training_data:
            prepared_train.append(
                {
                    "instruction": example["instruction"],
                    "response": example["response"],
                    "metadata": example.get("metadata", {}),
                }
            )

        result = {"train": prepared_train}

        if validation_data:
            prepared_val = []
            for example in validation_data:
                prepared_val.append(
                    {
                        "instruction": example["instruction"],
                        "response": example["response"],
                        "metadata": example.get("metadata", {}),
                    }
                )
            result["validation"] = prepared_val

        self.logger.info(
            "Training data prepared",
            train_samples=len(prepared_train),
            validation_samples=len(result.get("validation", [])),
        )

        return result

    async def _setup_training_environment(self) -> None:
        """Setup training environment and dependencies."""

        self.logger.info("Setting up training environment")

        # Ensure model server is ready
        if not await self.model_server.health_check():
            raise RuntimeError("Model server not ready for training")

        # Create necessary directories
        os.makedirs(f"{self.config.output_dir}/logs", exist_ok=True)
        os.makedirs(f"{self.config.output_dir}/checkpoints", exist_ok=True)

        # Initialize quality monitoring
        await self.quality_monitor.initialize_training_session()

    async def _run_training(
        self,
        train_data: List[Dict[str, Any]],
        val_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Execute the actual training process."""

        self.logger.info("Starting model training")

        # This would integrate with the actual training implementation
        # For now, we'll simulate the training process
        training_metrics = {
            "epochs_completed": self.config.num_epochs,
            "final_loss": 0.3,  # Simulated
            "best_validation_loss": 0.25 if val_data else None,
            "training_samples": len(train_data),
            "validation_samples": len(val_data) if val_data else 0,
            "learning_rate": self.config.learning_rate,
            "model_parameters": {
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
            },
        }

        # Record training progress
        await self.quality_monitor.record_training_metrics(training_metrics)

        return training_metrics

    async def _validate_training_quality(
        self, metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate training quality against thresholds."""

        self.logger.info("Validating training quality")

        quality_checks = []

        # Check final loss
        final_loss = metrics.get("final_loss", float("inf"))
        max_loss = 0.5  # Quality threshold
        loss_check = {
            "metric": "final_loss",
            "value": final_loss,
            "threshold": max_loss,
            "passed": final_loss <= max_loss,
        }
        quality_checks.append(loss_check)

        # Check validation loss if available
        if (
            "best_validation_loss" in metrics
            and metrics["best_validation_loss"] is not None
        ):
            val_loss = metrics["best_validation_loss"]
            val_check = {
                "metric": "validation_loss",
                "value": val_loss,
                "threshold": max_loss,
                "passed": val_loss <= max_loss,
            }
            quality_checks.append(val_check)

        # Calculate overall quality score
        passed_checks = sum(1 for check in quality_checks if check["passed"])
        overall_score = passed_checks / len(quality_checks) if quality_checks else 0.0

        quality_results = {
            "overall_score": overall_score,
            "checks": quality_checks,
            "passed": overall_score >= 0.8,  # 80% of checks must pass
        }

        if not quality_results["passed"]:
            self.logger.warning(
                "Training quality validation failed",
                overall_score=overall_score,
                failed_checks=[c for c in quality_checks if not c["passed"]],
            )

        return quality_results

    async def _save_training_artifacts(
        self, training_id: str, metrics: Dict[str, Any]
    ) -> str:
        """Save training artifacts and return model path."""

        self.logger.info("Saving training artifacts", training_id=training_id)

        # Create training metadata
        metadata = {
            "training_id": training_id,
            "config": self.config.__dict__,
            "metrics": metrics,
            "created_at": datetime.now().isoformat(),
            "model_type": self.config.model_type,
        }

        # Save metadata
        metadata_path = f"{self.config.output_dir}/training_metadata_{training_id}.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Model path (would be actual model checkpoint in real implementation)
        model_path = f"{self.config.output_dir}/model_{training_id}"

        self.logger.info(
            "Training artifacts saved",
            model_path=model_path,
            metadata_path=metadata_path,
        )

        return model_path


class TrainingPipelineFactory:
    """Factory for creating training pipelines with different configurations."""

    @staticmethod
    def create_compliance_analysis_pipeline(
        settings: AnalysisSettings,
        model_server: ModelServer,
        quality_monitor: QualityMonitor,
        custom_config: Optional[Dict[str, Any]] = None,
    ) -> TrainingPipeline:
        """Create training pipeline for compliance analysis models."""

        config = TrainingConfig(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            model_type="phi-3-compliance",
            output_dir="./checkpoints/compliance-analysis",
        )

        if custom_config:
            for key, value in custom_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        return TrainingPipeline(config, model_server, quality_monitor, settings)

    @staticmethod
    def create_risk_assessment_pipeline(
        settings: AnalysisSettings,
        model_server: ModelServer,
        quality_monitor: QualityMonitor,
        custom_config: Optional[Dict[str, Any]] = None,
    ) -> TrainingPipeline:
        """Create training pipeline for risk assessment models."""

        config = TrainingConfig(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            model_type="phi-3-risk",
            output_dir="./checkpoints/risk-assessment",
            learning_rate=5e-5,  # Lower learning rate for risk models
            num_epochs=3,  # More epochs for better risk understanding
        )

        if custom_config:
            for key, value in custom_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        return TrainingPipeline(config, model_server, quality_monitor, settings)

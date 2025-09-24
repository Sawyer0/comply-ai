"""
Training Pipeline Factory

Factory for creating training pipelines with proper stage configuration.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..core.pipeline import Pipeline
from ..stages.data_preparation import DataPreparationStage
from ..stages.training import TrainingStage
from ..stages.evaluation import EvaluationStage
from ..stages.deployment import DeploymentStage
from ..config.pipeline_config import create_training_config


class TrainingPipelineFactory:
    """Factory for creating training pipelines."""
    
    async def create_training_pipeline(self, 
                                      model_type: str = "dual",
                                      config: Optional[Dict[str, Any]] = None) -> Pipeline:
        """
        Create a complete training pipeline.
        
        Args:
            model_type: Type of model to train ("dual", "mapper", "analyst")
            config: Optional configuration overrides
            
        Returns:
            Configured training pipeline
        """
        config = config or {}
        
        # Create pipeline stages
        stages = [
            DataPreparationStage(config=config),
            TrainingStage(config=config),
            EvaluationStage(config=config),
            DeploymentStage(config=config)
        ]
        
        # Create pipeline configuration
        pipeline_config = create_training_config(model_type)
        
        # Create pipeline
        pipeline = Pipeline(
            name=f"{model_type}-training-pipeline",
            stages=stages,
            config=config
        )
        
        return pipeline
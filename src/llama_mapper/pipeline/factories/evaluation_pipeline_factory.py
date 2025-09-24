"""
Evaluation Pipeline Factory

Factory for creating evaluation-only pipelines.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..core.pipeline import Pipeline
from ..stages.evaluation import EvaluationStage
from ..config.pipeline_config import create_evaluation_config


class EvaluationPipelineFactory:
    """Factory for creating evaluation pipelines."""
    
    async def create_evaluation_pipeline(self, 
                                        model_name: str,
                                        config: Optional[Dict[str, Any]] = None) -> Pipeline:
        """
        Create an evaluation-only pipeline.
        
        Args:
            model_name: Name of model to evaluate
            config: Optional configuration overrides
            
        Returns:
            Configured evaluation pipeline
        """
        config = config or {}
        
        # Create pipeline stages
        stages = [
            EvaluationStage(config=config)
        ]
        
        # Create pipeline configuration
        pipeline_config = create_evaluation_config(model_name)
        
        # Create pipeline
        pipeline = Pipeline(
            name=f"{model_name}-evaluation-pipeline",
            stages=stages,
            config=config
        )
        
        return pipeline
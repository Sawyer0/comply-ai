"""
Deployment Pipeline Factory

Factory for creating deployment-only pipelines.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..core.pipeline import Pipeline
from ..stages.deployment import DeploymentStage
from ..config.pipeline_config import create_deployment_config


class DeploymentPipelineFactory:
    """Factory for creating deployment pipelines."""
    
    async def create_deployment_pipeline(self, 
                                        model_name: str,
                                        version: str,
                                        config: Optional[Dict[str, Any]] = None) -> Pipeline:
        """
        Create a deployment-only pipeline.
        
        Args:
            model_name: Name of model to deploy
            version: Version of model to deploy
            config: Optional configuration overrides
            
        Returns:
            Configured deployment pipeline
        """
        config = config or {}
        
        # Create pipeline stages
        stages = [
            DeploymentStage(config=config)
        ]
        
        # Create pipeline configuration
        pipeline_config = create_deployment_config(model_name, version)
        
        # Create pipeline
        pipeline = Pipeline(
            name=f"{model_name}-{version}-deployment-pipeline",
            stages=stages,
            config=config
        )
        
        return pipeline
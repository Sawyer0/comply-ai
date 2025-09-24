"""
Pipeline Registry

Maintains registry of available pipelines and their configurations.
Provides discovery and management capabilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from .pipeline import Pipeline
from ..config.pipeline_config import PipelineConfig
from ..exceptions import PipelineError

logger = structlog.get_logger(__name__)


class PipelineRegistry:
    """
    Registry for managing pipeline definitions and configurations.
    
    Provides:
    - Pipeline registration and discovery
    - Configuration management
    - Persistence and loading
    """
    
    def __init__(self, registry_path: Optional[str] = None):
        self.registry_path = Path(registry_path or "pipeline_registry.json")
        self.logger = logger.bind(component="registry")
        
        # In-memory registry
        self._pipelines: Dict[str, Pipeline] = {}
        self._configs: Dict[str, PipelineConfig] = {}
        
        # Load existing registry
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load registry from persistent storage."""
        if not self.registry_path.exists():
            self.logger.info("No existing registry found, starting fresh")
            return
        
        try:
            with open(self.registry_path, 'r') as f:
                registry_data = json.load(f)
            
            # Load configurations (pipelines are registered at runtime)
            for pipeline_name, config_data in registry_data.get("configs", {}).items():
                try:
                    config = PipelineConfig.from_dict(config_data)
                    self._configs[pipeline_name] = config
                except Exception as e:
                    self.logger.warning("Failed to load config for pipeline",
                                      pipeline_name=pipeline_name,
                                      error=str(e))
            
            self.logger.info("Registry loaded successfully",
                           pipelines_count=len(self._configs))
            
        except Exception as e:
            self.logger.error("Failed to load registry", error=str(e))
            # Continue with empty registry
    
    def _save_registry(self) -> None:
        """Save registry to persistent storage."""
        try:
            registry_data = {
                "configs": {
                    name: config.to_dict() 
                    for name, config in self._configs.items()
                },
                "metadata": {
                    "version": "1.0",
                    "last_updated": "2024-01-01T00:00:00Z"  # Would use actual timestamp
                }
            }
            
            # Ensure directory exists
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
            
            self.logger.debug("Registry saved successfully")
            
        except Exception as e:
            self.logger.error("Failed to save registry", error=str(e))
    
    async def register_pipeline(self, 
                               pipeline: Pipeline,
                               config: Optional[PipelineConfig] = None) -> None:
        """Register a pipeline with optional configuration."""
        if pipeline.name in self._pipelines:
            self.logger.warning("Pipeline already registered, updating",
                              pipeline_name=pipeline.name)
        
        self._pipelines[pipeline.name] = pipeline
        
        if config:
            self._configs[pipeline.name] = config
        
        # Save to persistent storage
        self._save_registry()
        
        self.logger.info("Pipeline registered successfully",
                        pipeline_name=pipeline.name,
                        has_config=config is not None)
    
    async def unregister_pipeline(self, pipeline_name: str) -> bool:
        """Unregister a pipeline."""
        removed = False
        
        if pipeline_name in self._pipelines:
            del self._pipelines[pipeline_name]
            removed = True
        
        if pipeline_name in self._configs:
            del self._configs[pipeline_name]
            removed = True
        
        if removed:
            self._save_registry()
            self.logger.info("Pipeline unregistered", pipeline_name=pipeline_name)
        
        return removed
    
    async def get_pipeline(self, pipeline_name: str) -> Optional[Pipeline]:
        """Get registered pipeline by name."""
        return self._pipelines.get(pipeline_name)
    
    async def get_pipeline_config(self, pipeline_name: str) -> Optional[PipelineConfig]:
        """Get pipeline configuration by name."""
        return self._configs.get(pipeline_name)
    
    async def list_pipelines(self) -> List[str]:
        """List all registered pipeline names."""
        return list(self._pipelines.keys())
    
    async def list_pipeline_configs(self) -> List[str]:
        """List all pipeline names with configurations."""
        return list(self._configs.keys())
    
    async def update_pipeline_config(self, 
                                    pipeline_name: str,
                                    config: PipelineConfig) -> None:
        """Update configuration for a registered pipeline."""
        if pipeline_name not in self._pipelines:
            raise PipelineError(f"Pipeline {pipeline_name} not registered")
        
        self._configs[pipeline_name] = config
        self._save_registry()
        
        self.logger.info("Pipeline configuration updated",
                        pipeline_name=pipeline_name)
    
    async def get_pipeline_info(self, pipeline_name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a pipeline."""
        pipeline = self._pipelines.get(pipeline_name)
        if not pipeline:
            return None
        
        config = self._configs.get(pipeline_name)
        
        return {
            "name": pipeline.name,
            "stages": pipeline.list_stages(),
            "has_config": config is not None,
            "config": config.to_dict() if config else None,
            "stage_count": len(pipeline.stages),
            "dependencies": self._get_pipeline_dependencies(pipeline)
        }
    
    def _get_pipeline_dependencies(self, pipeline: Pipeline) -> Dict[str, List[str]]:
        """Get dependency information for all stages in pipeline."""
        dependencies = {}
        for stage_name, stage in pipeline.stages.items():
            dependencies[stage_name] = stage.get_dependencies()
        return dependencies
    
    async def validate_pipeline_registry(self) -> List[str]:
        """
        Validate the entire pipeline registry.
        
        Returns list of validation errors.
        """
        errors = []
        
        # Check for orphaned configurations
        for config_name in self._configs:
            if config_name not in self._pipelines:
                errors.append(f"Configuration exists for unregistered pipeline: {config_name}")
        
        # Validate each pipeline
        for pipeline_name, pipeline in self._pipelines.items():
            try:
                # Validate stage dependencies
                for stage in pipeline.stages.values():
                    for dep in stage.get_dependencies():
                        if dep not in pipeline.stages:
                            errors.append(
                                f"Pipeline {pipeline_name}: Stage {stage.name} "
                                f"depends on non-existent stage {dep}"
                            )
                
                # Validate configuration if present
                config = self._configs.get(pipeline_name)
                if config:
                    config_errors = config.validate({})
                    for error in config_errors:
                        errors.append(f"Pipeline {pipeline_name}: {error}")
                        
            except Exception as e:
                errors.append(f"Pipeline {pipeline_name}: Validation failed - {str(e)}")
        
        return errors
    
    async def export_registry(self, export_path: str) -> None:
        """Export registry to specified path."""
        export_file = Path(export_path)
        
        registry_data = {
            "pipelines": {
                name: {
                    "stages": pipeline.list_stages(),
                    "dependencies": self._get_pipeline_dependencies(pipeline)
                }
                for name, pipeline in self._pipelines.items()
            },
            "configs": {
                name: config.to_dict()
                for name, config in self._configs.items()
            }
        }
        
        export_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_file, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        self.logger.info("Registry exported successfully", export_path=export_path)
    
    async def import_registry(self, import_path: str, merge: bool = True) -> None:
        """Import registry from specified path."""
        import_file = Path(import_path)
        
        if not import_file.exists():
            raise PipelineError(f"Import file not found: {import_path}")
        
        with open(import_file, 'r') as f:
            registry_data = json.load(f)
        
        if not merge:
            # Clear existing registry
            self._configs.clear()
        
        # Import configurations
        for pipeline_name, config_data in registry_data.get("configs", {}).items():
            try:
                config = PipelineConfig.from_dict(config_data)
                self._configs[pipeline_name] = config
            except Exception as e:
                self.logger.warning("Failed to import config",
                                  pipeline_name=pipeline_name,
                                  error=str(e))
        
        self._save_registry()
        
        self.logger.info("Registry imported successfully",
                        import_path=import_path,
                        merge=merge)
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_pipelines": len(self._pipelines),
            "pipelines_with_config": len(self._configs),
            "registry_path": str(self.registry_path),
            "registry_exists": self.registry_path.exists()
        }
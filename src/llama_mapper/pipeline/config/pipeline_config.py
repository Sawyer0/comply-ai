"""
Pipeline Configuration Management

Type-safe configuration system for pipelines and stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..exceptions import ConfigurationError


@dataclass
class StageConfig:
    """Configuration for a pipeline stage."""
    name: str
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: Optional[int] = None
    retry_count: int = 0
    skip_on_failure: bool = False
    dependencies: List[str] = field(default_factory=list)
    
    def validate(self) -> List[str]:
        """Validate stage configuration."""
        errors = []
        
        if not self.name:
            errors.append("Stage name cannot be empty")
        
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            errors.append("Timeout must be positive")
        
        if self.retry_count < 0:
            errors.append("Retry count cannot be negative")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "config": self.config,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "skip_on_failure": self.skip_on_failure,
            "dependencies": self.dependencies
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StageConfig:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            enabled=data.get("enabled", True),
            config=data.get("config", {}),
            timeout_seconds=data.get("timeout_seconds"),
            retry_count=data.get("retry_count", 0),
            skip_on_failure=data.get("skip_on_failure", False),
            dependencies=data.get("dependencies", [])
        )


@dataclass
class PipelineConfig:
    """Configuration for a complete pipeline."""
    name: str
    description: str = ""
    version: str = "1.0.0"
    stages: List[StageConfig] = field(default_factory=list)
    global_config: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: Optional[int] = None
    max_parallel_stages: int = 4
    failure_strategy: str = "fail_fast"  # fail_fast, continue_on_error, best_effort
    notification_config: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self, runtime_config: Optional[Dict[str, Any]] = None) -> List[str]:
        """Validate pipeline configuration."""
        errors = []
        
        if not self.name:
            errors.append("Pipeline name cannot be empty")
        
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            errors.append("Pipeline timeout must be positive")
        
        if self.max_parallel_stages <= 0:
            errors.append("Max parallel stages must be positive")
        
        if self.failure_strategy not in ["fail_fast", "continue_on_error", "best_effort"]:
            errors.append("Invalid failure strategy")
        
        # Validate stages
        stage_names = set()
        for stage in self.stages:
            stage_errors = stage.validate()
            errors.extend([f"Stage {stage.name}: {error}" for error in stage_errors])
            
            if stage.name in stage_names:
                errors.append(f"Duplicate stage name: {stage.name}")
            stage_names.add(stage.name)
        
        # Validate stage dependencies
        for stage in self.stages:
            for dep in stage.dependencies:
                if dep not in stage_names:
                    errors.append(f"Stage {stage.name} depends on non-existent stage {dep}")
        
        # Validate runtime configuration if provided
        if runtime_config:
            errors.extend(self._validate_runtime_config(runtime_config))
        
        return errors
    
    def _validate_runtime_config(self, runtime_config: Dict[str, Any]) -> List[str]:
        """Validate runtime configuration against pipeline requirements."""
        errors = []
        
        # Check for required configuration keys
        required_keys = self.global_config.get("required_keys", [])
        for key in required_keys:
            if key not in runtime_config:
                errors.append(f"Required configuration key missing: {key}")
        
        # Validate configuration types
        type_constraints = self.global_config.get("type_constraints", {})
        for key, expected_type in type_constraints.items():
            if key in runtime_config:
                value = runtime_config[key]
                if expected_type == "string" and not isinstance(value, str):
                    errors.append(f"Configuration key {key} must be string")
                elif expected_type == "integer" and not isinstance(value, int):
                    errors.append(f"Configuration key {key} must be integer")
                elif expected_type == "boolean" and not isinstance(value, bool):
                    errors.append(f"Configuration key {key} must be boolean")
                elif expected_type == "list" and not isinstance(value, list):
                    errors.append(f"Configuration key {key} must be list")
                elif expected_type == "dict" and not isinstance(value, dict):
                    errors.append(f"Configuration key {key} must be dictionary")
        
        return errors
    
    def get_stage_config(self, stage_name: str) -> Optional[StageConfig]:
        """Get configuration for a specific stage."""
        for stage in self.stages:
            if stage.name == stage_name:
                return stage
        return None
    
    def add_stage_config(self, stage_config: StageConfig) -> None:
        """Add stage configuration."""
        # Remove existing config for same stage
        self.stages = [s for s in self.stages if s.name != stage_config.name]
        self.stages.append(stage_config)
    
    def remove_stage_config(self, stage_name: str) -> bool:
        """Remove stage configuration."""
        original_count = len(self.stages)
        self.stages = [s for s in self.stages if s.name != stage_name]
        return len(self.stages) < original_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "stages": [stage.to_dict() for stage in self.stages],
            "global_config": self.global_config,
            "timeout_seconds": self.timeout_seconds,
            "max_parallel_stages": self.max_parallel_stages,
            "failure_strategy": self.failure_strategy,
            "notification_config": self.notification_config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineConfig:
        """Create from dictionary."""
        stages = [StageConfig.from_dict(stage_data) for stage_data in data.get("stages", [])]
        
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            stages=stages,
            global_config=data.get("global_config", {}),
            timeout_seconds=data.get("timeout_seconds"),
            max_parallel_stages=data.get("max_parallel_stages", 4),
            failure_strategy=data.get("failure_strategy", "fail_fast"),
            notification_config=data.get("notification_config", {})
        )
    
    @classmethod
    def from_yaml_file(cls, file_path: str) -> PipelineConfig:
        """Load configuration from YAML file."""
        import yaml
        
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    def to_yaml_file(self, file_path: str) -> None:
        """Save configuration to YAML file."""
        import yaml
        
        with open(file_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)


class ConfigurationBuilder:
    """Builder for creating pipeline configurations."""
    
    def __init__(self, name: str):
        self.config = PipelineConfig(name=name)
    
    def description(self, description: str) -> ConfigurationBuilder:
        """Set pipeline description."""
        self.config.description = description
        return self
    
    def version(self, version: str) -> ConfigurationBuilder:
        """Set pipeline version."""
        self.config.version = version
        return self
    
    def timeout(self, seconds: int) -> ConfigurationBuilder:
        """Set pipeline timeout."""
        self.config.timeout_seconds = seconds
        return self
    
    def max_parallel(self, count: int) -> ConfigurationBuilder:
        """Set maximum parallel stages."""
        self.config.max_parallel_stages = count
        return self
    
    def failure_strategy(self, strategy: str) -> ConfigurationBuilder:
        """Set failure handling strategy."""
        if strategy not in ["fail_fast", "continue_on_error", "best_effort"]:
            raise ConfigurationError(f"Invalid failure strategy: {strategy}")
        self.config.failure_strategy = strategy
        return self
    
    def global_config(self, config: Dict[str, Any]) -> ConfigurationBuilder:
        """Set global configuration."""
        self.config.global_config.update(config)
        return self
    
    def add_stage(self, 
                  name: str,
                  enabled: bool = True,
                  config: Optional[Dict[str, Any]] = None,
                  timeout: Optional[int] = None,
                  dependencies: Optional[List[str]] = None) -> ConfigurationBuilder:
        """Add stage configuration."""
        stage_config = StageConfig(
            name=name,
            enabled=enabled,
            config=config or {},
            timeout_seconds=timeout,
            dependencies=dependencies or []
        )
        self.config.add_stage_config(stage_config)
        return self
    
    def build(self) -> PipelineConfig:
        """Build final configuration."""
        errors = self.config.validate()
        if errors:
            raise ConfigurationError(f"Configuration validation failed: {errors}")
        return self.config


# Factory functions for common configurations
def create_training_config(model_type: str = "dual") -> PipelineConfig:
    """Create configuration for training pipeline."""
    builder = ConfigurationBuilder(f"{model_type}-model-training")
    
    return (builder
            .description(f"Training pipeline for {model_type} model")
            .version("1.0.0")
            .timeout(7200)  # 2 hours
            .max_parallel(2)
            .failure_strategy("fail_fast")
            .global_config({
                "model_type": model_type,
                "required_keys": ["output_dir", "training_data_path"],
                "type_constraints": {
                    "output_dir": "string",
                    "training_data_path": "string",
                    "epochs": "integer",
                    "batch_size": "integer"
                }
            })
            .add_stage("data_preparation", dependencies=[])
            .add_stage("training", dependencies=["data_preparation"])
            .add_stage("evaluation", dependencies=["training"])
            .add_stage("model_registration", dependencies=["evaluation"])
            .build())


def create_evaluation_config(model_name: str) -> PipelineConfig:
    """Create configuration for evaluation pipeline."""
    builder = ConfigurationBuilder(f"{model_name}-evaluation")
    
    return (builder
            .description(f"Evaluation pipeline for {model_name}")
            .version("1.0.0")
            .timeout(1800)  # 30 minutes
            .max_parallel(3)
            .failure_strategy("continue_on_error")
            .global_config({
                "model_name": model_name,
                "required_keys": ["model_path", "test_data_path"],
                "type_constraints": {
                    "model_path": "string",
                    "test_data_path": "string"
                }
            })
            .add_stage("model_loading", dependencies=[])
            .add_stage("golden_dataset_evaluation", dependencies=["model_loading"])
            .add_stage("performance_evaluation", dependencies=["model_loading"])
            .add_stage("bias_evaluation", dependencies=["model_loading"])
            .add_stage("report_generation", dependencies=["golden_dataset_evaluation", "performance_evaluation", "bias_evaluation"])
            .build())


def create_deployment_config(model_name: str, version: str) -> PipelineConfig:
    """Create configuration for deployment pipeline."""
    builder = ConfigurationBuilder(f"{model_name}-{version}-deployment")
    
    return (builder
            .description(f"Deployment pipeline for {model_name}@{version}")
            .version("1.0.0")
            .timeout(3600)  # 1 hour
            .max_parallel(1)  # Sequential deployment
            .failure_strategy("fail_fast")
            .global_config({
                "model_name": model_name,
                "model_version": version,
                "required_keys": ["deployment_environment", "canary_percentage"],
                "type_constraints": {
                    "deployment_environment": "string",
                    "canary_percentage": "integer"
                }
            })
            .add_stage("pre_deployment_validation", dependencies=[])
            .add_stage("canary_deployment", dependencies=["pre_deployment_validation"])
            .add_stage("canary_evaluation", dependencies=["canary_deployment"])
            .add_stage("production_promotion", dependencies=["canary_evaluation"])
            .build())
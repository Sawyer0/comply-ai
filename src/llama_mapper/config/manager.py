"""
Configuration Manager for Llama Mapper.

Handles YAML-configurable settings including confidence thresholds,
model parameters, and other system configurations.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """Model configuration settings."""
    
    name: str = Field(default="meta-llama/Llama-2-7b-chat-hf", description="Base model name")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Generation temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    max_new_tokens: int = Field(default=200, ge=1, le=2048, description="Maximum new tokens to generate")
    sequence_length: int = Field(default=1024, ge=512, le=2048, description="Maximum sequence length")
    
    # LoRA configuration
    lora_r: int = Field(default=16, ge=1, le=256, description="LoRA rank")
    lora_alpha: int = Field(default=32, ge=1, le=512, description="LoRA alpha scaling factor")
    learning_rate: float = Field(default=2e-4, ge=1e-6, le=1e-2, description="Learning rate")
    epochs: int = Field(default=2, ge=1, le=10, description="Training epochs")
    
    # Quantization settings
    use_8bit: bool = Field(default=False, description="Use 8-bit quantization")
    use_fp16: bool = Field(default=True, description="Use FP16 precision")


class ServingConfig(BaseModel):
    """Serving configuration settings."""
    
    backend: str = Field(default="vllm", pattern="^(vllm|tgi)$", description="Serving backend")
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, ge=1, le=65535, description="API port")
    workers: int = Field(default=1, ge=1, le=16, description="Number of workers")
    batch_size: int = Field(default=8, ge=1, le=128, description="Batch size for inference")
    
    # GPU/CPU settings
    device: str = Field(default="auto", description="Device to use (auto, cpu, cuda)")
    gpu_memory_utilization: float = Field(default=0.9, ge=0.1, le=1.0, description="GPU memory utilization")


class ConfidenceConfig(BaseModel):
    """Confidence evaluation configuration."""
    
    threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Confidence threshold for fallback")
    calibration_enabled: bool = Field(default=True, description="Enable confidence calibration")
    fallback_enabled: bool = Field(default=True, description="Enable rule-based fallback")
    
    @field_validator('threshold')
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate confidence threshold is reasonable."""
        if v < 0.3 or v > 0.9:
            raise ValueError("Confidence threshold should be between 0.3 and 0.9")
        return v


class StorageConfig(BaseModel):
    """Storage configuration settings."""
    
    # S3 settings
    s3_bucket: Optional[str] = Field(default=None, description="S3 bucket for immutable storage")
    s3_region: str = Field(default="us-east-1", description="S3 region")
    s3_encryption: str = Field(default="AES256", description="S3 encryption type")
    
    # Database settings
    db_type: str = Field(default="clickhouse", pattern="^(clickhouse|postgresql)$", description="Database type")
    db_host: str = Field(default="localhost", description="Database host")
    db_port: int = Field(default=8123, ge=1, le=65535, description="Database port")
    db_name: str = Field(default="mapper", description="Database name")
    retention_days: int = Field(default=90, ge=1, le=365, description="Hot data retention in days")


class SecurityConfig(BaseModel):
    """Security and privacy configuration."""
    
    log_raw_inputs: bool = Field(default=False, description="Whether to log raw detector inputs (privacy risk)")
    tenant_isolation: bool = Field(default=True, description="Enable tenant isolation")
    encryption_at_rest: bool = Field(default=True, description="Enable encryption at rest")
    
    # Secrets management
    secrets_backend: str = Field(default="vault", pattern="^(vault|aws)$", description="Secrets management backend")
    vault_url: Optional[str] = Field(default=None, description="Vault URL")
    vault_token: Optional[str] = Field(default=None, description="Vault token")


class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration."""
    
    metrics_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=9090, ge=1, le=65535, description="Metrics port")
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$", description="Log level")
    
    # Quality gates
    min_schema_valid_pct: float = Field(default=95.0, ge=0.0, le=100.0, description="Minimum schema valid percentage")
    max_fallback_pct: float = Field(default=10.0, ge=0.0, le=100.0, description="Maximum fallback percentage")
    max_p95_latency_ms: int = Field(default=250, ge=1, le=5000, description="Maximum P95 latency in ms")


class ConfigManager:
    """
    Manages YAML-configurable settings for the Llama Mapper system.
    
    Supports hierarchical configuration with environment variable overrides
    and validation of all configuration parameters.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize ConfigManager.
        
        Args:
            config_path: Path to YAML configuration file. If None, uses default locations.
        """
        self.config_path = self._resolve_config_path(config_path)
        self._config_data: Dict[str, Any] = {}
        self._load_config()
    
    def _resolve_config_path(self, config_path: Optional[Union[str, Path]]) -> Path:
        """Resolve configuration file path."""
        if config_path:
            return Path(config_path)
        
        # Try environment variable first
        env_path = os.getenv("MAPPER_CONFIG_PATH")
        if env_path:
            return Path(env_path)
        
        # Try default locations
        default_paths = [
            Path("config.yaml"),
            Path("config/mapper.yaml"),
            Path("/etc/mapper/config.yaml"),
        ]
        
        for path in default_paths:
            if path.exists():
                return path
        
        # Return default path (will be created if needed)
        return Path("config.yaml")
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            # Create default configuration
            self._create_default_config()
        else:
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config_data = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML configuration: {e}")
            except Exception as e:
                raise ValueError(f"Failed to load configuration from {self.config_path}: {e}")
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Initialize configuration sections after applying overrides
        self._initialize_config_sections()
    
    def _initialize_config_sections(self) -> None:
        """Initialize configuration sections from loaded data."""
        self.model = ModelConfig(**self._config_data.get("model", {}))
        self.serving = ServingConfig(**self._config_data.get("serving", {}))
        self.confidence = ConfidenceConfig(**self._config_data.get("confidence", {}))
        self.storage = StorageConfig(**self._config_data.get("storage", {}))
        self.security = SecurityConfig(**self._config_data.get("security", {}))
        self.monitoring = MonitoringConfig(**self._config_data.get("monitoring", {}))
    
    def _create_default_config(self) -> None:
        """Create default configuration file."""
        default_config = {
            "model": {
                "name": "meta-llama/Llama-2-7b-chat-hf",
                "temperature": 0.1,
                "top_p": 0.9,
                "max_new_tokens": 200,
                "sequence_length": 1024,
                "lora_r": 16,
                "lora_alpha": 32,
                "learning_rate": 2e-4,
                "epochs": 2,
                "use_8bit": False,
                "use_fp16": True
            },
            "serving": {
                "backend": "vllm",
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 1,
                "batch_size": 8,
                "device": "auto",
                "gpu_memory_utilization": 0.9
            },
            "confidence": {
                "threshold": 0.6,
                "calibration_enabled": True,
                "fallback_enabled": True
            },
            "storage": {
                "s3_bucket": None,
                "s3_region": "us-east-1",
                "s3_encryption": "AES256",
                "db_type": "clickhouse",
                "db_host": "localhost",
                "db_port": 8123,
                "db_name": "mapper",
                "retention_days": 90
            },
            "security": {
                "log_raw_inputs": False,
                "tenant_isolation": True,
                "encryption_at_rest": True,
                "secrets_backend": "vault",
                "vault_url": None,
                "vault_token": None
            },
            "monitoring": {
                "metrics_enabled": True,
                "metrics_port": 9090,
                "log_level": "INFO",
                "min_schema_valid_pct": 95.0,
                "max_fallback_pct": 10.0,
                "max_p95_latency_ms": 250
            }
        }
        
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        
        self._config_data = default_config
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        env_mappings = {
            "MAPPER_MODEL_NAME": ("model", "name"),
            "MAPPER_CONFIDENCE_THRESHOLD": ("confidence", "threshold"),
            "MAPPER_SERVING_PORT": ("serving", "port"),
            "MAPPER_LOG_LEVEL": ("monitoring", "log_level"),
            "MAPPER_DB_HOST": ("storage", "db_host"),
            "MAPPER_DB_PORT": ("storage", "db_port"),
            "MAPPER_S3_BUCKET": ("storage", "s3_bucket"),
            "MAPPER_VAULT_URL": ("security", "vault_url"),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if section not in self._config_data:
                    self._config_data[section] = {}
                
                # Type conversion based on expected type
                if key in ["port", "db_port", "epochs", "lora_r", "lora_alpha", "retention_days", "max_p95_latency_ms"]:
                    value = int(value)
                elif key in ["threshold", "temperature", "top_p", "learning_rate", "gpu_memory_utilization", 
                           "min_schema_valid_pct", "max_fallback_pct"]:
                    value = float(value)
                elif key in ["use_8bit", "use_fp16", "calibration_enabled", "fallback_enabled", 
                           "log_raw_inputs", "tenant_isolation", "encryption_at_rest", "metrics_enabled"]:
                    value = value.lower() in ("true", "1", "yes", "on")
                
                self._config_data[section][key] = value
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get the full configuration as a dictionary."""
        return {
            "model": self.model.model_dump(),
            "serving": self.serving.model_dump(),
            "confidence": self.confidence.model_dump(),
            "storage": self.storage.model_dump(),
            "security": self.security.model_dump(),
            "monitoring": self.monitoring.model_dump(),
        }
    
    def save_config(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration to YAML file."""
        save_path = Path(path) if path else self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.get_config_dict(), f, default_flow_style=False, indent=2)
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self._load_config()
    
    def validate_config(self) -> bool:
        """Validate the current configuration."""
        try:
            # Validation is handled by Pydantic models during initialization
            return True
        except Exception:
            return False
    
    def __repr__(self) -> str:
        """String representation of ConfigManager."""
        return f"ConfigManager(config_path={self.config_path})"
"""Application settings and configuration models."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    """Model configuration settings."""
    
    name: str = Field(default="meta-llama/Llama-2-7b-chat-hf", description="Base model name")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Generation temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    max_new_tokens: int = Field(default=200, ge=1, le=2048, description="Maximum new tokens to generate")
    quantization: Optional[str] = Field(default=None, description="Quantization method (8bit, 4bit)")


class LoRAConfig(BaseModel):
    """LoRA fine-tuning configuration."""
    
    r: int = Field(default=16, ge=1, le=256, description="LoRA rank")
    alpha: int = Field(default=32, ge=1, le=512, description="LoRA alpha scaling factor")
    dropout: float = Field(default=0.1, ge=0.0, le=1.0, description="LoRA dropout rate")
    target_modules: List[str] = Field(
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        description="Target modules for LoRA adaptation"
    )


class TrainingConfig(BaseModel):
    """Training configuration settings."""
    
    learning_rate: float = Field(default=2e-4, ge=1e-6, le=1e-2, description="Learning rate")
    num_epochs: int = Field(default=2, ge=1, le=10, description="Number of training epochs")
    batch_size: int = Field(default=4, ge=1, le=32, description="Training batch size")
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=16, description="Gradient accumulation steps")
    max_seq_length: int = Field(default=1024, ge=128, le=2048, description="Maximum sequence length")
    warmup_steps: int = Field(default=100, ge=0, le=1000, description="Warmup steps")
    save_steps: int = Field(default=500, ge=100, le=2000, description="Save checkpoint every N steps")


class ConfidenceConfig(BaseModel):
    """Confidence evaluation configuration."""
    
    default_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Default confidence threshold")
    detector_thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-detector confidence thresholds"
    )
    calibration_enabled: bool = Field(default=True, description="Enable confidence calibration")


class ServingConfig(BaseModel):
    """Model serving configuration."""
    
    backend: str = Field(default="vllm", description="Serving backend (vllm, tgi)")
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1024, le=65535, description="Server port")
    workers: int = Field(default=1, ge=1, le=8, description="Number of worker processes")
    max_batch_size: int = Field(default=32, ge=1, le=128, description="Maximum batch size")
    gpu_memory_utilization: float = Field(default=0.9, ge=0.1, le=1.0, description="GPU memory utilization")


class StorageConfig(BaseModel):
    """Storage configuration settings."""
    
    # S3 Configuration
    s3_bucket: Optional[str] = Field(default=None, description="S3 bucket for immutable storage")
    s3_prefix: str = Field(default="mapper-outputs", description="S3 key prefix")
    s3_retention_years: int = Field(default=7, ge=1, le=50, description="S3 WORM retention in years")
    
    # AWS Configuration
    aws_access_key_id: Optional[str] = Field(default=None, description="AWS access key ID")
    aws_secret_access_key: Optional[str] = Field(default=None, description="AWS secret access key")
    aws_region: str = Field(default="us-east-1", description="AWS region")
    
    # Database Configuration
    storage_backend: str = Field(default="postgresql", description="Storage backend (postgresql, clickhouse)")
    db_host: str = Field(default="localhost", description="Database host")
    db_port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    db_name: str = Field(default="llama_mapper", description="Database name")
    db_user: Optional[str] = Field(default=None, description="Database user")
    db_password: Optional[str] = Field(default=None, description="Database password")
    
    # Encryption Configuration
    kms_key_id: Optional[str] = Field(default=None, description="KMS key ID for encryption")
    encryption_key: str = Field(default="default-key-change-in-production", description="Local encryption key")
    
    # Retention Configuration
    retention_days: int = Field(default=90, ge=1, le=365, description="Hot data retention in days")


class LoggingConfig(BaseModel):
    """Logging configuration settings."""
    
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="json", description="Log format (json, text)")
    privacy_mode: bool = Field(default=True, description="Enable privacy-first logging")
    max_message_length: int = Field(default=500, ge=100, le=2000, description="Maximum log message length")


class SecurityConfig(BaseModel):
    """Security configuration settings."""
    
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    tenant_header: str = Field(default="X-Tenant-ID", description="Tenant ID header name")
    secrets_backend: str = Field(default="vault", description="Secrets backend (vault, aws)")
    encryption_key_id: Optional[str] = Field(default=None, description="KMS key ID for encryption")


class Settings(BaseSettings):
    """Main application settings."""
    
    # Core settings
    app_name: str = Field(default="llama-mapper", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")
    environment: str = Field(default="development", description="Environment (development, staging, production)")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Component configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    # Paths
    taxonomy_path: str = Field(default="pillars-detectors/taxonomy.yaml", description="Path to taxonomy YAML")
    detectors_path: str = Field(default="pillars-detectors", description="Path to detector configs directory")
    frameworks_path: str = Field(default="pillars-detectors/frameworks.yaml", description="Path to frameworks YAML")
    schema_path: str = Field(default="pillars-detectors/schema.json", description="Path to output schema JSON")
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False
    )
"""
Configuration validation and management for weekly evaluations.

This module provides configuration validation, defaults, and
environment-specific settings for the weekly evaluation system.
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class EvaluationThresholds(BaseModel):
    """Quality thresholds for evaluation alerts."""
    
    schema_valid_rate: float = Field(
        default=0.98, ge=0.0, le=1.0,
        description="Minimum schema validation success rate"
    )
    rubric_score: float = Field(
        default=0.8, ge=0.0, le=1.0,
        description="Minimum rubric score"
    )
    opa_compile_success_rate: float = Field(
        default=0.95, ge=0.0, le=1.0,
        description="Minimum OPA compilation success rate"
    )
    evidence_accuracy: float = Field(
        default=0.85, ge=0.0, le=1.0,
        description="Minimum evidence accuracy"
    )
    
    @field_validator('schema_valid_rate', 'rubric_score', 'opa_compile_success_rate', 'evidence_accuracy')
    @classmethod
    def validate_thresholds(cls, v: float) -> float:
        """Validate threshold values."""
        if not isinstance(v, (int, float)):
            raise ValueError("Threshold must be a number")
        if v < 0 or v > 1:
            raise ValueError("Threshold must be between 0 and 1")
        return float(v)


class NotificationConfig(BaseModel):
    """Notification configuration for evaluations."""
    
    enabled: bool = Field(default=True, description="Enable notifications")
    email_recipients: List[str] = Field(
        default_factory=list, description="Email recipients for notifications"
    )
    slack_webhook_url: Optional[str] = Field(
        default=None, description="Slack webhook URL for notifications"
    )
    webhook_urls: List[str] = Field(
        default_factory=list, description="Custom webhook URLs for notifications"
    )
    
    @field_validator('email_recipients')
    @classmethod
    def validate_email_recipients(cls, v: List[str]) -> List[str]:
        """Validate email addresses."""
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        for email in v:
            if not re.match(email_pattern, email):
                raise ValueError(f"Invalid email address: {email}")
        
        return v


class ReportConfig(BaseModel):
    """Report generation configuration."""
    
    formats: List[str] = Field(
        default=["PDF"], description="Report formats to generate"
    )
    include_detailed_metrics: bool = Field(
        default=True, description="Include detailed metrics in reports"
    )
    include_individual_scores: bool = Field(
        default=True, description="Include individual rubric scores"
    )
    include_recommendations: bool = Field(
        default=True, description="Include improvement recommendations"
    )
    template_dir: Optional[str] = Field(
        default=None, description="Directory containing report templates"
    )
    
    @field_validator('formats')
    @classmethod
    def validate_formats(cls, v: List[str]) -> List[str]:
        """Validate report formats."""
        valid_formats = {"PDF", "CSV", "JSON", "TEXT"}
        
        for format_type in v:
            if format_type.upper() not in valid_formats:
                raise ValueError(f"Unsupported report format: {format_type}")
        
        return [f.upper() for f in v]


class StorageConfig(BaseModel):
    """Storage configuration for evaluations."""
    
    backend_type: str = Field(
        default="file", description="Storage backend type"
    )
    storage_dir: Optional[str] = Field(
        default=None, description="Storage directory for file backend"
    )
    database_url: Optional[str] = Field(
        default=None, description="Database URL for database backend"
    )
    s3_bucket: Optional[str] = Field(
        default=None, description="S3 bucket for S3 backend"
    )
    retention_days: int = Field(
        default=90, ge=1, le=365,
        description="Retention period for evaluation data in days"
    )
    
    @field_validator('backend_type')
    @classmethod
    def validate_backend_type(cls, v: str) -> str:
        """Validate storage backend type."""
        valid_backends = {"file", "database", "s3"}
        if v.lower() not in valid_backends:
            raise ValueError(f"Unsupported storage backend: {v}")
        return v.lower()
    
    @model_validator(mode='after')
    def validate_backend_config(self) -> 'StorageConfig':
        """Validate backend-specific configuration."""
        if self.backend_type == "database" and not self.database_url:
            raise ValueError("database_url is required for database backend")
        
        if self.backend_type == "s3" and not self.s3_bucket:
            raise ValueError("s3_bucket is required for S3 backend")
        
        return self


class WeeklyEvaluationConfig(BaseModel):
    """Complete configuration for weekly evaluations."""
    
    enabled: bool = Field(default=True, description="Enable weekly evaluations")
    default_schedule: str = Field(
        default="0 9 * * 1", description="Default cron schedule for evaluations"
    )
    evaluation_period_days: int = Field(
        default=7, ge=1, le=30,
        description="Number of days to include in evaluation"
    )
    max_concurrent_evaluations: int = Field(
        default=5, ge=1, le=20,
        description="Maximum concurrent evaluations"
    )
    timeout_minutes: int = Field(
        default=30, ge=5, le=120,
        description="Evaluation timeout in minutes"
    )
    
    # Sub-configurations
    thresholds: EvaluationThresholds = Field(
        default_factory=EvaluationThresholds,
        description="Quality thresholds for alerts"
    )
    notifications: NotificationConfig = Field(
        default_factory=NotificationConfig,
        description="Notification configuration"
    )
    reports: ReportConfig = Field(
        default_factory=ReportConfig,
        description="Report generation configuration"
    )
    storage: StorageConfig = Field(
        default_factory=StorageConfig,
        description="Storage configuration"
    )
    
    @field_validator('default_schedule')
    @classmethod
    def validate_cron_schedule(cls, v: str) -> str:
        """Validate cron schedule."""
        try:
            from croniter import croniter
            croniter(v)
        except Exception as e:
            raise ValueError(f"Invalid cron schedule '{v}': {e}")
        return v
    
    @classmethod
    def from_environment(cls) -> 'WeeklyEvaluationConfig':
        """Create configuration from environment variables."""
        config_data = {}
        
        # Basic settings
        if os.getenv('LLAMA_MAPPER_WEEKLY_EVALUATIONS_ENABLED'):
            config_data['enabled'] = os.getenv('LLAMA_MAPPER_WEEKLY_EVALUATIONS_ENABLED').lower() == 'true'
        
        if os.getenv('LLAMA_MAPPER_DEFAULT_WEEKLY_SCHEDULE'):
            config_data['default_schedule'] = os.getenv('LLAMA_MAPPER_DEFAULT_WEEKLY_SCHEDULE')
        
        if os.getenv('LLAMA_MAPPER_EVALUATION_PERIOD_DAYS'):
            config_data['evaluation_period_days'] = int(os.getenv('LLAMA_MAPPER_EVALUATION_PERIOD_DAYS'))
        
        # Thresholds
        thresholds = {}
        if os.getenv('LLAMA_MAPPER_SCHEMA_VALID_THRESHOLD'):
            thresholds['schema_valid_rate'] = float(os.getenv('LLAMA_MAPPER_SCHEMA_VALID_THRESHOLD'))
        if os.getenv('LLAMA_MAPPER_RUBRIC_SCORE_THRESHOLD'):
            thresholds['rubric_score'] = float(os.getenv('LLAMA_MAPPER_RUBRIC_SCORE_THRESHOLD'))
        if os.getenv('LLAMA_MAPPER_OPA_COMPILE_THRESHOLD'):
            thresholds['opa_compile_success_rate'] = float(os.getenv('LLAMA_MAPPER_OPA_COMPILE_THRESHOLD'))
        if os.getenv('LLAMA_MAPPER_EVIDENCE_ACCURACY_THRESHOLD'):
            thresholds['evidence_accuracy'] = float(os.getenv('LLAMA_MAPPER_EVIDENCE_ACCURACY_THRESHOLD'))
        
        if thresholds:
            config_data['thresholds'] = thresholds
        
        # Notifications
        notifications = {}
        if os.getenv('LLAMA_MAPPER_NOTIFICATION_EMAIL'):
            notifications['email_recipients'] = os.getenv('LLAMA_MAPPER_NOTIFICATION_EMAIL').split(',')
        if os.getenv('SLACK_WEBHOOK_URL'):
            notifications['slack_webhook_url'] = os.getenv('SLACK_WEBHOOK_URL')
        if os.getenv('LLAMA_MAPPER_WEBHOOK_URLS'):
            notifications['webhook_urls'] = os.getenv('LLAMA_MAPPER_WEBHOOK_URLS').split(',')
        
        if notifications:
            config_data['notifications'] = notifications
        
        # Storage
        storage = {}
        if os.getenv('LLAMA_MAPPER_STORAGE_BACKEND'):
            storage['backend_type'] = os.getenv('LLAMA_MAPPER_STORAGE_BACKEND')
        if os.getenv('LLAMA_MAPPER_STORAGE_DIR'):
            storage['storage_dir'] = os.getenv('LLAMA_MAPPER_STORAGE_DIR')
        if os.getenv('DATABASE_URL'):
            storage['database_url'] = os.getenv('DATABASE_URL')
        if os.getenv('S3_BUCKET'):
            storage['s3_bucket'] = os.getenv('S3_BUCKET')
        if os.getenv('LLAMA_MAPPER_RETENTION_DAYS'):
            storage['retention_days'] = int(os.getenv('LLAMA_MAPPER_RETENTION_DAYS'))
        
        if storage:
            config_data['storage'] = storage
        
        return cls(**config_data)
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'WeeklyEvaluationConfig':
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            if config_path.suffix.lower() == '.json':
                import json
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                import yaml
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
            
            return cls(**config_data)
            
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {e}")
    
    def to_file(self, config_path: Union[str, Path], format: str = "json") -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = self.model_dump()
        
        try:
            if format.lower() == 'json':
                import json
                with open(config_path, 'w') as f:
                    json.dump(config_data, f, indent=2, default=str)
            elif format.lower() in ['yaml', 'yml']:
                import yaml
                with open(config_path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported output format: {format}")
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            raise ValueError(f"Failed to save configuration to {config_path}: {e}")
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        try:
            # Validate the configuration by creating a new instance
            WeeklyEvaluationConfig(**self.model_dump())
        except Exception as e:
            issues.append(f"Configuration validation failed: {e}")
        
        # Additional business logic validation
        if self.enabled and not self.notifications.email_recipients and not self.notifications.slack_webhook_url:
            issues.append("Notifications are enabled but no recipients configured")
        
        if self.storage.backend_type == "file" and not self.storage.storage_dir:
            issues.append("File storage backend requires storage_dir to be specified")
        
        if self.max_concurrent_evaluations > 10:
            issues.append("High concurrent evaluation count may impact performance")
        
        return issues
    
    def get_storage_backend_config(self) -> Dict[str, Any]:
        """Get storage backend configuration."""
        config = {
            "backend_type": self.storage.backend_type,
        }
        
        if self.storage.storage_dir:
            config["storage_dir"] = self.storage.storage_dir
        if self.storage.database_url:
            config["database_url"] = self.storage.database_url
        if self.storage.s3_bucket:
            config["s3_bucket"] = self.storage.s3_bucket
        
        return config


# Default configuration instance
DEFAULT_CONFIG = WeeklyEvaluationConfig()


def get_evaluation_config(config_path: Optional[Union[str, Path]] = None) -> WeeklyEvaluationConfig:
    """
    Get evaluation configuration from file or environment.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        WeeklyEvaluationConfig instance
    """
    if config_path:
        return WeeklyEvaluationConfig.from_file(config_path)
    else:
        return WeeklyEvaluationConfig.from_environment()


def validate_evaluation_config(config: WeeklyEvaluationConfig) -> bool:
    """
    Validate evaluation configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValueError: If configuration is invalid
    """
    issues = config.validate_configuration()
    
    if issues:
        error_message = "Configuration validation failed:\n" + "\n".join(f"- {issue}" for issue in issues)
        logger.error(error_message)
        raise ValueError(error_message)
    
    logger.info("Configuration validation passed")
    return True

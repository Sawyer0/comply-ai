"""
Extended configuration manager for analysis module.

This module extends the base ConfigManager with analysis-specific
configuration management capabilities.
"""

import logging
import os
from typing import Any, Dict, Optional, Union
from pathlib import Path

from ...config.manager import ConfigManager
from .settings import AnalysisSettings, AnalysisConfig

logger = logging.getLogger(__name__)


class AnalysisConfigManager(ConfigManager):
    """
    Extended configuration manager for analysis module.
    
    Provides analysis-specific configuration management including
    environment-specific loading, validation, and hot-reloading.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the analysis configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        super().__init__(config_path)
        self.analysis_settings: Optional[AnalysisSettings] = None
        self.analysis_config: Optional[AnalysisConfig] = None
        self._load_analysis_config()
    
    def _load_analysis_config(self):
        """Load analysis-specific configuration."""
        try:
            # Load analysis settings from environment
            self.analysis_settings = AnalysisSettings()
            
            # Create analysis config
            self.analysis_config = AnalysisConfig(self.analysis_settings)
            
            logger.info("Analysis configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load analysis configuration: {e}")
            raise
    
    def get_analysis_settings(self) -> AnalysisSettings:
        """
        Get analysis settings.
        
        Returns:
            Analysis settings instance
        """
        if self.analysis_settings is None:
            self._load_analysis_config()
        
        return self.analysis_settings
    
    def get_analysis_config(self) -> AnalysisConfig:
        """
        Get analysis configuration.
        
        Returns:
            Analysis configuration instance
        """
        if self.analysis_config is None:
            self._load_analysis_config()
        
        return self.analysis_config
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Returns:
            Model configuration dictionary
        """
        settings = self.get_analysis_settings()
        
        return {
            "model_path": settings.analysis_model_path,
            "temperature": settings.analysis_temperature,
            "confidence_cutoff": settings.analysis_confidence_cutoff,
            "max_concurrent_requests": settings.max_concurrent_requests,
            "request_timeout_seconds": settings.request_timeout_seconds
        }
    
    def get_api_config(self) -> Dict[str, Any]:
        """
        Get API configuration.
        
        Returns:
            API configuration dictionary
        """
        settings = self.get_analysis_settings()
        
        return {
            "rate_limit_requests_per_minute": settings.rate_limit_requests_per_minute,
            "cors_origins": settings.cors_origins,
            "pii_redaction_enabled": settings.pii_redaction_enabled
        }
    
    def get_quality_config(self) -> Dict[str, Any]:
        """
        Get quality configuration.
        
        Returns:
            Quality configuration dictionary
        """
        settings = self.get_analysis_settings()
        
        return {
            "golden_dataset_path": settings.golden_dataset_path,
            "evaluation_enabled": settings.quality_evaluation_enabled,
            "schema_validation_rate_threshold": settings.schema_validation_rate_threshold,
            "rubric_score_threshold": settings.rubric_score_threshold
        }
    
    def get_cache_config(self) -> Dict[str, Any]:
        """
        Get cache configuration.
        
        Returns:
            Cache configuration dictionary
        """
        settings = self.get_analysis_settings()
        
        return {
            "ttl_seconds": settings.cache_ttl_seconds,
            "max_size": settings.cache_max_size,
            "cleanup_interval_seconds": settings.cache_cleanup_interval_seconds
        }
    
    def get_opa_config(self) -> Dict[str, Any]:
        """
        Get OPA configuration.
        
        Returns:
            OPA configuration dictionary
        """
        settings = self.get_analysis_settings()
        
        return {
            "validation_enabled": settings.opa_validation_enabled,
            "timeout_seconds": settings.opa_timeout_seconds
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """
        Get monitoring configuration.
        
        Returns:
            Monitoring configuration dictionary
        """
        settings = self.get_analysis_settings()
        
        return {
            "metrics_enabled": settings.metrics_enabled,
            "logging_level": settings.logging_level,
            "log_format": settings.log_format
        }
    
    def validate_config(self) -> tuple[bool, list[str]]:
        """
        Validate analysis configuration.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            settings = self.get_analysis_settings()
            
            # Validate model configuration
            if not os.path.exists(settings.analysis_model_path):
                errors.append(f"Model path does not exist: {settings.analysis_model_path}")
            
            if not 0.0 <= settings.analysis_temperature <= 2.0:
                errors.append(f"Temperature must be between 0.0 and 2.0, got: {settings.analysis_temperature}")
            
            if not 0.0 <= settings.analysis_confidence_cutoff <= 1.0:
                errors.append(f"Confidence cutoff must be between 0.0 and 1.0, got: {settings.analysis_confidence_cutoff}")
            
            # Validate processing configuration
            if settings.max_concurrent_requests <= 0:
                errors.append(f"Max concurrent requests must be positive, got: {settings.max_concurrent_requests}")
            
            if settings.request_timeout_seconds <= 0:
                errors.append(f"Request timeout must be positive, got: {settings.request_timeout_seconds}")
            
            # Validate quality configuration
            if settings.golden_dataset_path and not os.path.exists(settings.golden_dataset_path):
                errors.append(f"Golden dataset path does not exist: {settings.golden_dataset_path}")
            
            if not 0.0 <= settings.schema_validation_rate_threshold <= 1.0:
                errors.append(f"Schema validation rate threshold must be between 0.0 and 1.0, got: {settings.schema_validation_rate_threshold}")
            
            if not 0.0 <= settings.rubric_score_threshold <= 5.0:
                errors.append(f"Rubric score threshold must be between 0.0 and 5.0, got: {settings.rubric_score_threshold}")
            
            # Validate cache configuration
            if settings.cache_ttl_seconds <= 0:
                errors.append(f"Cache TTL must be positive, got: {settings.cache_ttl_seconds}")
            
            if settings.cache_max_size <= 0:
                errors.append(f"Cache max size must be positive, got: {settings.cache_max_size}")
            
            # Validate API configuration
            if settings.rate_limit_requests_per_minute <= 0:
                errors.append(f"Rate limit must be positive, got: {settings.rate_limit_requests_per_minute}")
            
            # Validate OPA configuration
            if settings.opa_timeout_seconds <= 0:
                errors.append(f"OPA timeout must be positive, got: {settings.opa_timeout_seconds}")
            
        except Exception as e:
            errors.append(f"Configuration validation error: {str(e)}")
        
        return len(errors) == 0, errors
    
    def reload_config(self):
        """Reload configuration from environment."""
        try:
            self._load_analysis_config()
            logger.info("Analysis configuration reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload analysis configuration: {e}")
            raise
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get configuration summary.
        
        Returns:
            Configuration summary dictionary
        """
        settings = self.get_analysis_settings()
        
        return {
            "model": {
                "path": settings.analysis_model_path,
                "temperature": settings.analysis_temperature,
                "confidence_cutoff": settings.analysis_confidence_cutoff
            },
            "processing": {
                "max_concurrent_requests": settings.max_concurrent_requests,
                "request_timeout_seconds": settings.request_timeout_seconds
            },
            "api": {
                "rate_limit_requests_per_minute": settings.rate_limit_requests_per_minute,
                "cors_origins": settings.cors_origins,
                "pii_redaction_enabled": settings.pii_redaction_enabled
            },
            "quality": {
                "golden_dataset_path": settings.golden_dataset_path,
                "evaluation_enabled": settings.quality_evaluation_enabled,
                "schema_validation_rate_threshold": settings.schema_validation_rate_threshold,
                "rubric_score_threshold": settings.rubric_score_threshold
            },
            "cache": {
                "ttl_seconds": settings.cache_ttl_seconds,
                "max_size": settings.cache_max_size,
                "cleanup_interval_seconds": settings.cache_cleanup_interval_seconds
            },
            "opa": {
                "validation_enabled": settings.opa_validation_enabled,
                "timeout_seconds": settings.opa_timeout_seconds
            },
            "monitoring": {
                "metrics_enabled": settings.metrics_enabled,
                "logging_level": settings.logging_level,
                "log_format": settings.log_format
            }
        }
    
    def export_config(self, output_path: str):
        """
        Export configuration to file.
        
        Args:
            output_path: Path to export configuration
        """
        try:
            config_summary = self.get_config_summary()
            
            with open(output_path, 'w') as f:
                import json
                json.dump(config_summary, f, indent=2)
            
            logger.info(f"Configuration exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            raise
    
    def import_config(self, config_path: str):
        """
        Import configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                import json
                config_data = json.load(f)
            
            # Update settings based on imported config
            settings = self.get_analysis_settings()
            
            # Update model settings
            if "model" in config_data:
                model_config = config_data["model"]
                if "path" in model_config:
                    settings.analysis_model_path = model_config["path"]
                if "temperature" in model_config:
                    settings.analysis_temperature = model_config["temperature"]
                if "confidence_cutoff" in model_config:
                    settings.analysis_confidence_cutoff = model_config["confidence_cutoff"]
            
            # Update processing settings
            if "processing" in config_data:
                processing_config = config_data["processing"]
                if "max_concurrent_requests" in processing_config:
                    settings.max_concurrent_requests = processing_config["max_concurrent_requests"]
                if "request_timeout_seconds" in processing_config:
                    settings.request_timeout_seconds = processing_config["request_timeout_seconds"]
            
            # Reload configuration
            self._load_analysis_config()
            
            logger.info(f"Configuration imported from: {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")
            raise
    
    def get_environment_config(self, environment: str) -> Dict[str, Any]:
        """
        Get environment-specific configuration.
        
        Args:
            environment: Environment name (dev, staging, prod)
            
        Returns:
            Environment-specific configuration
        """
        base_config = self.get_config_summary()
        
        # Environment-specific overrides
        if environment == "dev":
            base_config["model"]["temperature"] = 0.2  # Higher temperature for dev
            base_config["processing"]["max_concurrent_requests"] = 5
            base_config["api"]["rate_limit_requests_per_minute"] = 100
            base_config["monitoring"]["logging_level"] = "DEBUG"
        
        elif environment == "staging":
            base_config["model"]["temperature"] = 0.1
            base_config["processing"]["max_concurrent_requests"] = 8
            base_config["api"]["rate_limit_requests_per_minute"] = 200
            base_config["monitoring"]["logging_level"] = "INFO"
        
        elif environment == "prod":
            base_config["model"]["temperature"] = 0.05  # Lower temperature for prod
            base_config["processing"]["max_concurrent_requests"] = 10
            base_config["api"]["rate_limit_requests_per_minute"] = 60
            base_config["monitoring"]["logging_level"] = "WARNING"
        
        return base_config
    
    def setup_environment(self, environment: str):
        """
        Setup environment-specific configuration.
        
        Args:
            environment: Environment name (dev, staging, prod)
        """
        try:
            env_config = self.get_environment_config(environment)
            
            # Set environment variables
            os.environ["ANALYSIS_TEMPERATURE"] = str(env_config["model"]["temperature"])
            os.environ["ANALYSIS_MAX_CONCURRENT_REQUESTS"] = str(env_config["processing"]["max_concurrent_requests"])
            os.environ["ANALYSIS_RATE_LIMIT_REQUESTS_PER_MINUTE"] = str(env_config["api"]["rate_limit_requests_per_minute"])
            os.environ["ANALYSIS_LOGGING_LEVEL"] = env_config["monitoring"]["logging_level"]
            
            # Reload configuration
            self.reload_config()
            
            logger.info(f"Environment setup complete for: {environment}")
            
        except Exception as e:
            logger.error(f"Failed to setup environment {environment}: {e}")
            raise

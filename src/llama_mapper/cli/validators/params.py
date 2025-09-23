"""Parameter validation utilities for CLI commands."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..core.base import CLIError


class ParameterValidator:
    """Utility class for validating CLI parameters."""
    
    @staticmethod
    def validate_required(params: Dict[str, Any], required: List[str]) -> None:
        """Validate that required parameters are provided."""
        missing = [param for param in required if not params.get(param)]
        if missing:
            raise CLIError(f"Missing required parameters: {', '.join(missing)}")
    
    @staticmethod
    def validate_file_exists(file_path: Union[str, Path]) -> Path:
        """Validate that a file exists and return the Path object."""
        path = Path(file_path)
        if not path.exists():
            raise CLIError(f"File not found: {path}")
        if not path.is_file():
            raise CLIError(f"Path is not a file: {path}")
        return path
    
    @staticmethod
    def validate_directory_exists(dir_path: Union[str, Path]) -> Path:
        """Validate that a directory exists and return the Path object."""
        path = Path(dir_path)
        if not path.exists():
            raise CLIError(f"Directory not found: {path}")
        if not path.is_dir():
            raise CLIError(f"Path is not a directory: {path}")
        return path
    
    @staticmethod
    def validate_positive_integer(value: Any, param_name: str) -> int:
        """Validate that a value is a positive integer."""
        try:
            int_value = int(value)
            if int_value <= 0:
                raise CLIError(f"{param_name} must be a positive integer, got: {value}")
            return int_value
        except (ValueError, TypeError):
            raise CLIError(f"{param_name} must be an integer, got: {value}")
    
    @staticmethod
    def validate_non_negative_integer(value: Any, param_name: str) -> int:
        """Validate that a value is a non-negative integer."""
        try:
            int_value = int(value)
            if int_value < 0:
                raise CLIError(f"{param_name} must be a non-negative integer, got: {value}")
            return int_value
        except (ValueError, TypeError):
            raise CLIError(f"{param_name} must be an integer, got: {value}")
    
    @staticmethod
    def validate_float_range(value: Any, param_name: str, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Validate that a value is a float within the specified range."""
        try:
            float_value = float(value)
            if not (min_val <= float_value <= max_val):
                raise CLIError(f"{param_name} must be between {min_val} and {max_val}, got: {value}")
            return float_value
        except (ValueError, TypeError):
            raise CLIError(f"{param_name} must be a number, got: {value}")
    
    @staticmethod
    def validate_choice(value: Any, param_name: str, choices: List[str]) -> str:
        """Validate that a value is one of the allowed choices."""
        if value not in choices:
            raise CLIError(f"{param_name} must be one of {choices}, got: {value}")
        return str(value)
    
    @staticmethod
    def validate_regex(value: Any, param_name: str, pattern: str, description: str = "") -> str:
        """Validate that a value matches a regex pattern."""
        if not re.match(pattern, str(value)):
            desc = f" ({description})" if description else ""
            raise CLIError(f"{param_name} does not match required pattern{desc}, got: {value}")
        return str(value)
    
    @staticmethod
    def validate_tenant_id(tenant_id: str) -> str:
        """Validate tenant ID format."""
        if not tenant_id:
            raise CLIError("Tenant ID cannot be empty")
        
        # Basic validation: alphanumeric, hyphens, underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', tenant_id):
            raise CLIError("Tenant ID must contain only alphanumeric characters, hyphens, and underscores")
        
        if len(tenant_id) < 3 or len(tenant_id) > 50:
            raise CLIError("Tenant ID must be between 3 and 50 characters")
        
        return tenant_id
    
    @staticmethod
    def validate_request_id(request_id: str) -> str:
        """Validate request ID format."""
        if not request_id:
            raise CLIError("Request ID cannot be empty")
        
        # UUID-like format validation
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        if not re.match(uuid_pattern, request_id, re.IGNORECASE):
            raise CLIError("Request ID must be a valid UUID format")
        
        return request_id
    
    @staticmethod
    def validate_environment(environment: str) -> str:
        """Validate environment name."""
        valid_environments = ["development", "staging", "production", "test"]
        return ParameterValidator.validate_choice(environment, "Environment", valid_environments)
    
    @staticmethod
    def validate_log_level(log_level: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        return ParameterValidator.validate_choice(log_level, "Log level", valid_levels)
    
    @staticmethod
    def validate_output_format(format_type: str) -> str:
        """Validate output format."""
        valid_formats = ["json", "yaml", "text", "table"]
        return ParameterValidator.validate_choice(format_type, "Output format", valid_formats)
    
    @staticmethod
    def validate_port(port: Any) -> int:
        """Validate port number."""
        try:
            port_int = int(port)
            if not (1 <= port_int <= 65535):
                raise CLIError(f"Port must be between 1 and 65535, got: {port}")
            return port_int
        except (ValueError, TypeError):
            raise CLIError(f"Port must be an integer, got: {port}")
    
    @staticmethod
    def validate_host(host: str) -> str:
        """Validate host address."""
        if not host:
            raise CLIError("Host cannot be empty")
        
        # Basic host validation (IP or hostname)
        ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        hostname_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
        
        if not (re.match(ip_pattern, host) or re.match(hostname_pattern, host) or host == "localhost"):
            raise CLIError(f"Invalid host format: {host}")
        
        return host

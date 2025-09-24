"""
Configuration Manager for dynamic service configuration and hot-reloading.

This manager handles configuration loading, validation, hot-reloading,
and distribution across analysis services.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class ConfigurationChangeEvent:
    """Event representing a configuration change."""
    
    def __init__(self, config_key: str, old_value: Any, new_value: Any):
        self.config_key = config_key
        self.old_value = old_value
        self.new_value = new_value
        self.timestamp = datetime.now(timezone.utc)


class ConfigurationManager:
    """
    Manages dynamic configuration with hot-reloading capabilities.
    
    Supports file-based configuration, environment variable overrides,
    and real-time configuration updates with validation.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(config_file) if config_file else None
        self.config: Dict[str, Any] = {}
        self.config_version = 1
        self.last_loaded = None
        self.observers: List[Observer] = []
        self.change_handlers: List[Callable[[ConfigurationChangeEvent], None]] = []
        self.validation_rules: Dict[str, Callable[[Any], bool]] = {}
        self._lock = asyncio.Lock()
        
        # Load initial configuration
        if self.config_file and self.config_file.exists():
            self.load_configuration()
    
    def load_configuration(self) -> None:
        """Load configuration from file."""
        if not self.config_file or not self.config_file.exists():
            logger.warning("Configuration file not found, using defaults")
            self._load_default_configuration()
            return
        
        try:
            with open(self.config_file, 'r') as f:
                new_config = json.load(f)
            
            # Validate configuration
            if self._validate_configuration(new_config):
                old_config = self.config.copy()
                self.config = new_config
                self.config_version += 1
                self.last_loaded = datetime.now(timezone.utc)
                
                # Notify handlers of changes
                self._notify_configuration_changes(old_config, new_config)
                
                logger.info(f"Configuration loaded successfully (version {self.config_version})")
            else:
                logger.error("Configuration validation failed")
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            if not self.config:  # If no config loaded yet, use defaults
                self._load_default_configuration()
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_config(self, key: str, value: Any) -> bool:
        """
        Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
            
        Returns:
            True if set successfully, False otherwise
        """
        try:
            # Validate the new value
            if key in self.validation_rules:
                if not self.validation_rules[key](value):
                    logger.error(f"Validation failed for config key: {key}")
                    return False
            
            # Get old value for change notification
            old_value = self.get_config(key)
            
            # Set the value
            keys = key.split('.')
            config_ref = self.config
            
            # Navigate to parent
            for k in keys[:-1]:
                if k not in config_ref:
                    config_ref[k] = {}
                config_ref = config_ref[k]
            
            # Set final value
            config_ref[keys[-1]] = value
            self.config_version += 1
            
            # Notify handlers
            event = ConfigurationChangeEvent(key, old_value, value)
            self._notify_change_handlers(event)
            
            logger.info(f"Configuration updated: {key} = {value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set configuration {key}: {e}")
            return False
    
    def register_change_handler(self, handler: Callable[[ConfigurationChangeEvent], None]) -> None:
        """
        Register a handler for configuration changes.
        
        Args:
            handler: Function to call when configuration changes
        """
        self.change_handlers.append(handler)
        logger.debug(f"Registered configuration change handler: {handler.__name__}")
    
    def register_validation_rule(self, key: str, validator: Callable[[Any], bool]) -> None:
        """
        Register a validation rule for a configuration key.
        
        Args:
            key: Configuration key
            validator: Function that returns True if value is valid
        """
        self.validation_rules[key] = validator
        logger.debug(f"Registered validation rule for: {key}")
    
    def start_hot_reloading(self) -> None:
        """Start hot-reloading of configuration file."""
        if not self.config_file:
            logger.warning("No configuration file specified, hot-reloading disabled")
            return
        
        try:
            event_handler = ConfigFileHandler(self)
            observer = Observer()
            observer.schedule(event_handler, str(self.config_file.parent), recursive=False)
            observer.start()
            self.observers.append(observer)
            
            logger.info(f"Started hot-reloading for: {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to start hot-reloading: {e}")
    
    def stop_hot_reloading(self) -> None:
        """Stop hot-reloading."""
        for observer in self.observers:
            try:
                observer.stop()
                observer.join()
            except Exception as e:
                logger.error(f"Error stopping observer: {e}")
        
        self.observers.clear()
        logger.info("Stopped configuration hot-reloading")
    
    def save_configuration(self, file_path: Optional[str] = None) -> bool:
        """
        Save current configuration to file.
        
        Args:
            file_path: Optional path to save to (defaults to current config file)
            
        Returns:
            True if saved successfully, False otherwise
        """
        target_file = Path(file_path) if file_path else self.config_file
        
        if not target_file:
            logger.error("No file path specified for saving configuration")
            return False
        
        try:
            # Create directory if it doesn't exist
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(target_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Configuration saved to: {target_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def get_configuration_info(self) -> Dict[str, Any]:
        """
        Get information about current configuration.
        
        Returns:
            Dictionary with configuration metadata
        """
        return {
            'config_file': str(self.config_file) if self.config_file else None,
            'config_version': self.config_version,
            'last_loaded': self.last_loaded.isoformat() if self.last_loaded else None,
            'hot_reloading_active': len(self.observers) > 0,
            'change_handlers_count': len(self.change_handlers),
            'validation_rules_count': len(self.validation_rules),
            'config_keys': list(self._get_all_keys(self.config))
        }
    
    def validate_current_configuration(self) -> Dict[str, Any]:
        """
        Validate current configuration against all rules.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        for key, validator in self.validation_rules.items():
            try:
                value = self.get_config(key)
                if value is not None and not validator(value):
                    results['valid'] = False
                    results['errors'].append(f"Validation failed for {key}: {value}")
            except Exception as e:
                results['warnings'].append(f"Validation error for {key}: {e}")
        
        return results
    
    def _load_default_configuration(self) -> None:
        """Load default configuration."""
        self.config = {
            'analysis': {
                'engines': {
                    'pattern_recognition': {
                        'enabled': True,
                        'confidence_threshold': 0.7
                    },
                    'risk_scoring': {
                        'enabled': True,
                        'confidence_threshold': 0.7
                    },
                    'compliance_intelligence': {
                        'enabled': True,
                        'confidence_threshold': 0.7
                    }
                }
            },
            'logging': {
                'level': 'INFO'
            }
        }
        self.config_version = 1
        self.last_loaded = datetime.now(timezone.utc)
        logger.info("Loaded default configuration")
    
    def _validate_configuration(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration structure and values.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic structure validation
            if not isinstance(config, dict):
                logger.error("Configuration must be a dictionary")
                return False
            
            # Validate against registered rules
            for key, validator in self.validation_rules.items():
                keys = key.split('.')
                value = config
                
                try:
                    for k in keys:
                        value = value[k]
                    
                    if not validator(value):
                        logger.error(f"Validation failed for {key}")
                        return False
                        
                except KeyError:
                    # Key not present - this might be OK depending on the rule
                    pass
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    def _notify_configuration_changes(self, old_config: Dict[str, Any], 
                                    new_config: Dict[str, Any]) -> None:
        """Notify handlers of configuration changes."""
        # Find changed keys
        changed_keys = self._find_changed_keys(old_config, new_config)
        
        for key in changed_keys:
            old_value = self._get_nested_value(old_config, key)
            new_value = self._get_nested_value(new_config, key)
            
            event = ConfigurationChangeEvent(key, old_value, new_value)
            self._notify_change_handlers(event)
    
    def _notify_change_handlers(self, event: ConfigurationChangeEvent) -> None:
        """Notify all change handlers of an event."""
        for handler in self.change_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in configuration change handler: {e}")
    
    def _find_changed_keys(self, old_config: Dict[str, Any], 
                          new_config: Dict[str, Any]) -> List[str]:
        """Find keys that changed between configurations."""
        changed = []
        
        # Get all keys from both configs
        all_keys = set(self._get_all_keys(old_config)) | set(self._get_all_keys(new_config))
        
        for key in all_keys:
            old_value = self._get_nested_value(old_config, key)
            new_value = self._get_nested_value(new_config, key)
            
            if old_value != new_value:
                changed.append(key)
        
        return changed
    
    def _get_all_keys(self, config: Dict[str, Any], prefix: str = '') -> List[str]:
        """Get all keys from nested configuration."""
        keys = []
        
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            keys.append(full_key)
            
            if isinstance(value, dict):
                keys.extend(self._get_all_keys(value, full_key))
        
        return keys
    
    def _get_nested_value(self, config: Dict[str, Any], key: str) -> Any:
        """Get nested value by dot-notation key."""
        keys = key.split('.')
        value = config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return None


class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for configuration hot-reloading."""
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        super().__init__()
    
    def on_modified(self, event):
        """Handle file modification events."""
        if (not event.is_directory and 
            event.src_path == str(self.config_manager.config_file)):
            
            logger.info("Configuration file changed, reloading...")
            
            # Add small delay to ensure file write is complete
            import time
            time.sleep(0.1)
            
            self.config_manager.load_configuration()


# Validation functions
def validate_confidence_threshold(value: Any) -> bool:
    """Validate confidence threshold value."""
    return isinstance(value, (int, float)) and 0.0 <= value <= 1.0


def validate_engine_enabled(value: Any) -> bool:
    """Validate engine enabled flag."""
    return isinstance(value, bool)


def validate_log_level(value: Any) -> bool:
    """Validate log level."""
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    return isinstance(value, str) and value.upper() in valid_levels


def create_default_configuration_manager(config_file: Optional[str] = None) -> ConfigurationManager:
    """
    Create a configuration manager with default validation rules.
    
    Args:
        config_file: Optional path to configuration file
        
    Returns:
        Configured configuration manager
    """
    manager = ConfigurationManager(config_file)
    
    # Register validation rules
    manager.register_validation_rule('analysis.engines.pattern_recognition.confidence_threshold', 
                                   validate_confidence_threshold)
    manager.register_validation_rule('analysis.engines.risk_scoring.confidence_threshold', 
                                   validate_confidence_threshold)
    manager.register_validation_rule('analysis.engines.compliance_intelligence.confidence_threshold', 
                                   validate_confidence_threshold)
    manager.register_validation_rule('analysis.engines.pattern_recognition.enabled', 
                                   validate_engine_enabled)
    manager.register_validation_rule('analysis.engines.risk_scoring.enabled', 
                                   validate_engine_enabled)
    manager.register_validation_rule('analysis.engines.compliance_intelligence.enabled', 
                                   validate_engine_enabled)
    manager.register_validation_rule('logging.level', validate_log_level)
    
    return manager
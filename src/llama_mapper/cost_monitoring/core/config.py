"""Configuration classes for cost monitoring components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class CostMonitoringConfig:
    """Configuration for cost monitoring system."""
    
    # General settings
    enabled: bool = True
    collection_interval: int = 60  # seconds
    retention_days: int = 30
    
    # Cost calculation settings
    cpu_cost_per_hour: float = 0.05  # USD per CPU hour
    memory_cost_per_gb_hour: float = 0.01  # USD per GB-hour
    storage_cost_per_gb_hour: float = 0.001  # USD per GB-hour
    network_cost_per_gb: float = 0.01  # USD per GB
    
    # Alerting settings
    alert_threshold_percent: float = 80.0
    alert_cooldown_minutes: int = 15
    
    # Scaling settings
    enable_autoscaling: bool = True
    min_instances: int = 1
    max_instances: int = 10
    scale_up_threshold: float = 70.0
    scale_down_threshold: float = 30.0
    
    # Resource monitoring settings
    monitor_cpu: bool = True
    monitor_memory: bool = True
    monitor_disk: bool = True
    monitor_network: bool = True
    monitor_gpu: bool = False
    
    # Storage settings
    storage_type: str = "in_memory"  # in_memory, redis, database
    storage_config: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "enabled": self.enabled,
            "collection_interval": self.collection_interval,
            "retention_days": self.retention_days,
            "cpu_cost_per_hour": self.cpu_cost_per_hour,
            "memory_cost_per_gb_hour": self.memory_cost_per_gb_hour,
            "storage_cost_per_gb_hour": self.storage_cost_per_gb_hour,
            "network_cost_per_gb": self.network_cost_per_gb,
            "alert_threshold_percent": self.alert_threshold_percent,
            "alert_cooldown_minutes": self.alert_cooldown_minutes,
            "enable_autoscaling": self.enable_autoscaling,
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "scale_up_threshold": self.scale_up_threshold,
            "scale_down_threshold": self.scale_down_threshold,
            "monitor_cpu": self.monitor_cpu,
            "monitor_memory": self.monitor_memory,
            "monitor_disk": self.monitor_disk,
            "monitor_network": self.monitor_network,
            "monitor_gpu": self.monitor_gpu,
            "storage_type": self.storage_type,
            "storage_config": self.storage_config or {}
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> CostMonitoringConfig:
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def create_default_config(cls) -> CostMonitoringConfig:
        """Create default configuration."""
        return cls()

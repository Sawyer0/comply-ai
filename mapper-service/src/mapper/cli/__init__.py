"""
CLI commands for the Mapper Service.

This package provides comprehensive CLI commands following SRP principles:
- Mapping operations (map, batch, validate)
- Model management (load, unload, list, health)
- Service operations (health, config, metrics, start)
- Tenant management (create, list, usage, enable/disable)
"""

from .commands import cli
from .training_cli import training_cli

__all__ = [
    "cli",
    "training_cli",
]

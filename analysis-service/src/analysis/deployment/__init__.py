"""
Deployment management for the Analysis Service.

This module handles:
- Blue-green deployments
- Canary releases
- Feature flags
- Model deployment strategies
"""

from .manager import DeploymentManager, DeploymentRecord, DeploymentStatus

__all__ = [
    "DeploymentManager",
    "DeploymentRecord",
    "DeploymentStatus",
]

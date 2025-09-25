"""
Deployment management for the mapper service.

This module provides comprehensive deployment capabilities including
canary deployments, blue-green deployments, and feature flag management.
"""

from .canary import (
    CanaryController,
    CanaryConfig,
    CanaryDeployment,
    CanaryStatus,
    TrafficSplitter,
    HealthValidator,
    HealthCheckResult,
    get_canary_controller,
)

from .blue_green import (
    BlueGreenController,
    BlueGreenConfig,
    BlueGreenDeployment,
    EnvironmentColor,
    DeploymentStatus,
    EnvironmentManager,
    ValidationSuite,
    ValidationResult,
    DNSManager,
    get_blue_green_controller,
)

from .feature_flags import (
    FeatureFlagManager,
    FeatureFlag,
    FlagType,
    RolloutStrategy,
    FlagStatus,
    RolloutConfig,
    FlagRule,
    EvaluationContext,
    FlagEvaluationResult,
    get_feature_flag_manager,
    initialize_feature_flag_manager,
)

from .integration import (
    DeploymentManager,
    load_deployment_config,
    initialize_deployment_manager,
    get_deployment_manager,
    shutdown_deployment_manager,
)

__all__ = [
    # Canary deployment
    "CanaryController",
    "CanaryConfig",
    "CanaryDeployment",
    "CanaryStatus",
    "TrafficSplitter",
    "HealthValidator",
    "HealthCheckResult",
    "get_canary_controller",
    # Blue-green deployment
    "BlueGreenController",
    "BlueGreenConfig",
    "BlueGreenDeployment",
    "EnvironmentColor",
    "DeploymentStatus",
    "EnvironmentManager",
    "ValidationSuite",
    "ValidationResult",
    "DNSManager",
    "get_blue_green_controller",
    # Feature flags
    "FeatureFlagManager",
    "FeatureFlag",
    "FlagType",
    "RolloutStrategy",
    "FlagStatus",
    "RolloutConfig",
    "FlagRule",
    "EvaluationContext",
    "FlagEvaluationResult",
    "get_feature_flag_manager",
    "initialize_feature_flag_manager",
    # Integration
    "DeploymentManager",
    "load_deployment_config",
    "initialize_deployment_manager",
    "get_deployment_manager",
    "shutdown_deployment_manager",
]

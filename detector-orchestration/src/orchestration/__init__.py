"""Detector Orchestration Service - SRP-organized functionality.

This module provides the main orchestration service with properly separated responsibilities:
- Core: DetectorCoordinator, ContentRouter, ResponseAggregator
- Monitoring: HealthMonitor
- Discovery: ServiceDiscoveryManager
- Policy: PolicyManager
- Resilience: CircuitBreaker
"""

# Core functionality
from .core import (
    DetectorCoordinator,
    ContentRouter,
    ResponseAggregator,
    RoutingPlan,
    RoutingDecision,
    DetectorConfig,
    AggregatedOutput,
)

# Monitoring functionality
from .monitoring import HealthMonitor, HealthCheck, HealthStatus

# Discovery functionality
from .discovery import ServiceDiscoveryManager, ServiceEndpoint

# Policy functionality
from .policy import PolicyManager, PolicyDecision

# Resilience functionality
from .resilience import CircuitBreaker, RateLimiter, RateLimitResult, RateLimitStrategy

# Cache functionality
from .cache import RedisCache, IdempotencyManager, IdempotencyResult

# Pipeline functionality
from .pipelines import AsyncJobProcessor, AsyncJob, JobResult, JobStatus, JobPriority

# Plugin functionality
from .plugins import PluginManager, DetectorPluginInterface, PolicyPluginInterface

# Security functionality
from .security import (
    ApiKeyManager,
    RBACManager,
    AttackDetector,
    InputSanitizer,
    Permission,
    Role,
)

# Multi-tenancy functionality
from .tenancy import TenantManager, TenantIsolationManager, TenantContext

__all__ = [
    # Core
    "DetectorCoordinator",
    "ContentRouter",
    "ResponseAggregator",
    "RoutingPlan",
    "RoutingDecision",
    "DetectorConfig",
    "AggregatedOutput",
    # Monitoring
    "HealthMonitor",
    "HealthCheck",
    "HealthStatus",
    # Discovery
    "ServiceDiscoveryManager",
    "ServiceEndpoint",
    # Policy
    "PolicyManager",
    "PolicyDecision",
    # Resilience
    "CircuitBreaker",
    "RateLimiter",
    "RateLimitResult",
    "RateLimitStrategy",
    # Cache
    "RedisCache",
    "IdempotencyManager",
    "IdempotencyResult",
    # Pipelines
    "AsyncJobProcessor",
    "AsyncJob",
    "JobResult",
    "JobStatus",
    "JobPriority",
    # Plugins
    "PluginManager",
    "DetectorPluginInterface",
    "PolicyPluginInterface",
    # Security
    "ApiKeyManager",
    "RBACManager",
    "AttackDetector",
    "InputSanitizer",
    "Permission",
    "Role",
    # Multi-tenancy
    "TenantManager",
    "TenantIsolationManager",
    "TenantContext",
]

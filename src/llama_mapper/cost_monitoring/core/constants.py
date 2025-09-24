"""Constants and configuration for cost monitoring."""

from __future__ import annotations

from typing import Dict, List

# Default cost per unit (in USD)
DEFAULT_COST_PER_UNIT: Dict[str, float] = {
    "cpu_per_hour": 0.05,
    "memory_per_gb_hour": 0.01,
    "gpu_per_hour": 2.00,
    "storage_per_gb_month": 0.10,
    "network_per_gb": 0.05,
    "api_call": 0.001,
}

# Default cost thresholds
DEFAULT_COST_THRESHOLDS: Dict[str, float] = {
    "daily_budget": 100.0,
    "hourly_budget": 10.0,
    "api_call_cost": 0.001,
    "compute_cost_per_hour": 0.50,
}

# Time conversion constants
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR
SECONDS_PER_MONTH = 30 * SECONDS_PER_DAY

# Metric collection constants
DEFAULT_COLLECTION_INTERVAL_SECONDS = 60
DEFAULT_RETENTION_DAYS = 90
DEFAULT_RECENT_METRICS_COUNT = 10

# Guardrail constants
DEFAULT_MAX_VIOLATIONS_PER_HOUR = 10
DEFAULT_ESCALATION_DELAY_MINUTES = 30
DEFAULT_COOLDOWN_MINUTES = 60

# Autoscaling constants
DEFAULT_EVALUATION_INTERVAL_SECONDS = 60
DEFAULT_COST_THRESHOLD_PERCENT = 80.0
DEFAULT_PERFORMANCE_THRESHOLD_PERCENT = 70.0
DEFAULT_MAX_COST_INCREASE_PERCENT = 50.0
DEFAULT_MIN_PERFORMANCE_IMPROVEMENT_PERCENT = 20.0

# Analytics constants
DEFAULT_ANALYSIS_INTERVAL_HOURS = 24
DEFAULT_ANOMALY_THRESHOLD = 2.0
DEFAULT_FORECAST_HORIZON_DAYS = 30
DEFAULT_MIN_DATA_POINTS = 7

# Resource types
RESOURCE_TYPES: List[str] = [
    "cpu",
    "memory",
    "gpu",
    "storage",
    "network",
    "api_calls",
]

# Metric types for guardrails
GUARDRAIL_METRIC_TYPES: List[str] = [
    "daily_cost",
    "hourly_cost",
    "api_calls",
    "compute_cost",
    "memory_cost",
    "storage_cost",
    "network_cost",
]

# Alert severity levels
ALERT_SEVERITY_LEVELS: List[str] = [
    "low",
    "medium",
    "high",
    "critical",
]

# Scaling actions
SCALING_ACTIONS: List[str] = [
    "scale_up",
    "scale_down",
    "scale_out",
    "scale_in",
    "no_action",
]

# Guardrail actions
GUARDRAIL_ACTIONS: List[str] = [
    "alert",
    "throttle",
    "scale_down",
    "pause_service",
    "block_requests",
    "notify_admin",
]

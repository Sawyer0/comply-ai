"""
Quality alerting system configuration.

This module provides configuration classes and settings for the
quality alerting system.
"""

from .quality_config import (
    EmailConfig,
    QualityAlertingConfig,
    QualityAlertingSettings,
    QualityThresholdConfig,
    SlackConfig,
    WebhookConfig,
)

__all__ = [
    "QualityAlertingConfig",
    "QualityAlertingSettings",
    "EmailConfig",
    "SlackConfig",
    "WebhookConfig",
    "QualityThresholdConfig",
]

"""
Alerting components.

This module provides alert management and notification functionality
including alert creation, routing, and multiple notification channels.
"""

from .alert_handlers import (
    CompositeAlertHandler,
    EmailAlertHandler,
    LoggingAlertHandler,
    SlackAlertHandler,
    WebhookAlertHandler,
)
from .alert_manager import AlertManager

__all__ = [
    "AlertManager",
    "LoggingAlertHandler",
    "EmailAlertHandler",
    "SlackAlertHandler",
    "WebhookAlertHandler",
    "CompositeAlertHandler",
]

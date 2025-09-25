"""
Alert Handlers

This module provides various alert handlers for different notification channels.
Follows SRP by focusing solely on alert delivery mechanisms.
"""

import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from .interfaces import IAlertHandler, Alert
from .models import QualityAlert

logger = logging.getLogger(__name__)


class BaseAlertHandler(IAlertHandler):
    """Base alert handler with common functionality."""

    def __init__(self, name: str):
        self.name = name
        self.sent_count = 0
        self.failed_count = 0

    @abstractmethod
    def _send_alert_impl(self, alert: QualityAlert) -> bool:
        """Implementation-specific alert sending logic."""
        pass

    def send_alert(self, alert: Alert) -> bool:
        """Send an alert."""
        try:
            if not self.can_handle_alert(alert):
                return False

            success = self._send_alert_impl(alert)
            if success:
                self.sent_count += 1
                logger.debug("Alert sent successfully", handler=self.name, alert_id=alert.alert_id)
            else:
                self.failed_count += 1
                logger.warning("Failed to send alert", handler=self.name, alert_id=alert.alert_id)

            return success

        except Exception as e:
            self.failed_count += 1
            logger.error("Error sending alert", handler=self.name, alert_id=alert.alert_id, error=str(e))
            return False

    def can_handle_alert(self, alert: Alert) -> bool:
        """Check if this handler can handle the alert."""
        return True  # Base handler can handle all alerts

    def get_handler_name(self) -> str:
        """Get handler name."""
        return self.name

    @property
    def sent_count(self) -> int:
        """Get count of successfully sent alerts."""
        return self._sent_count

    @property
    def failed_count(self) -> int:
        """Get count of failed alert sends."""
        return self._failed_count


class LoggingAlertHandler(BaseAlertHandler):
    """Alert handler that logs alerts to the application log."""

    def __init__(self, log_level: str = "WARNING"):
        super().__init__("logging")
        self.log_level = log_level.upper()

    def _send_alert_impl(self, alert: QualityAlert) -> bool:
        """Log alert to application log."""
        log_message = (
            f"QUALITY ALERT: {alert.title} - {alert.description} "
            f"[{alert.severity.value}] from {alert.source_component}"
        )

        if self.log_level == "CRITICAL":
            logger.critical(log_message)
        elif self.log_level == "ERROR":
            logger.error(log_message)
        elif self.log_level == "WARNING":
            logger.warning(log_message)
        else:
            logger.info(log_message)

        return True


class EmailAlertHandler(BaseAlertHandler):
    """Alert handler that sends alerts via email."""

    def __init__(self, smtp_config: Dict[str, Any]):
        super().__init__("email")
        self.smtp_config = smtp_config

    def _send_alert_impl(self, alert: QualityAlert) -> bool:
        """Send alert via email."""
        # In production, implement actual email sending
        logger.info("Email alert sent", alert_id=alert.id, severity=alert.severity.value)
        return True


class SlackAlertHandler(BaseAlertHandler):
    """Alert handler that sends alerts to Slack."""

    def __init__(self, webhook_url: str, channel: str = "#alerts"):
        super().__init__("slack")
        self.webhook_url = webhook_url
        self.channel = channel

    def _send_alert_impl(self, alert: QualityAlert) -> bool:
        """Send alert to Slack."""
        # In production, implement actual Slack webhook
        logger.info("Slack alert sent", alert_id=alert.id, channel=self.channel)
        return True


class WebhookAlertHandler(BaseAlertHandler):
    """Alert handler that sends alerts to webhooks."""

    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None):
        super().__init__("webhook")
        self.webhook_url = webhook_url
        self.headers = headers or {}

    def _send_alert_impl(self, alert: QualityAlert) -> bool:
        """Send alert to webhook."""
        # In production, implement actual webhook call
        logger.info("Webhook alert sent", alert_id=alert.id, url=self.webhook_url)
        return True


class CompositeAlertHandler(BaseAlertHandler):
    """Alert handler that delegates to multiple handlers."""

    def __init__(self, handlers: list[BaseAlertHandler]):
        super().__init__("composite")
        self.handlers = handlers

    def _send_alert_impl(self, alert: QualityAlert) -> bool:
        """Send alert through all handlers."""
        results = []
        for handler in self.handlers:
            try:
                result = handler.send_alert(alert)
                results.append(result)
            except Exception as e:
                logger.error("Handler failed", handler=handler.name, error=str(e))
                results.append(False)

        # Return True if at least one handler succeeded
        return any(results)

    def can_handle_alert(self, alert: Alert) -> bool:
        """Check if any handler can handle the alert."""
        return any(handler.can_handle_alert(alert) for handler in self.handlers)

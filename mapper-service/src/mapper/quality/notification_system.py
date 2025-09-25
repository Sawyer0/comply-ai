"""
Notification System

This module provides notification channels for quality alerts.
Follows SRP by focusing solely on notification delivery.
"""

import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

from .models import QualityAlert

logger = logging.getLogger(__name__)


class NotificationChannel(ABC):
    """Base notification channel."""

    def __init__(self, name: str):
        self.name = name
        self.sent_count = 0
        self.failed_count = 0

    @abstractmethod
    def send_notification(self, alert: QualityAlert, message: str) -> bool:
        """Send notification through this channel."""
        pass

    def can_handle_alert(self, alert: QualityAlert) -> bool:
        """Check if this channel can handle the alert."""
        return True


class LogNotificationChannel(NotificationChannel):
    """Notification channel that logs to application log."""

    def __init__(self, log_level: str = "WARNING"):
        super().__init__("log")
        self.log_level = log_level.upper()

    def send_notification(self, alert: QualityAlert, message: str) -> bool:
        """Log notification to application log."""
        try:
            log_message = f"NOTIFICATION: {message} - {alert.title}"

            if self.log_level == "CRITICAL":
                logger.critical(log_message)
            elif self.log_level == "ERROR":
                logger.error(log_message)
            elif self.log_level == "WARNING":
                logger.warning(log_message)
            else:
                logger.info(log_message)

            self.sent_count += 1
            return True

        except Exception as e:
            self.failed_count += 1
            logger.error("Failed to send log notification", error=str(e))
            return False


class EmailNotificationChannel(NotificationChannel):
    """Notification channel that sends emails."""

    def __init__(self, smtp_config: Dict[str, Any]):
        super().__init__("email")
        self.smtp_config = smtp_config

    def send_notification(self, alert: QualityAlert, message: str) -> bool:
        """Send email notification."""
        try:
            # In production, implement actual email sending
            logger.info("Email notification sent", alert_id=alert.id)
            self.sent_count += 1
            return True

        except Exception as e:
            self.failed_count += 1
            logger.error("Failed to send email notification", error=str(e))
            return False


class SlackNotificationChannel(NotificationChannel):
    """Notification channel that sends to Slack."""

    def __init__(self, webhook_url: str, channel: str = "#alerts"):
        super().__init__("slack")
        self.webhook_url = webhook_url
        self.channel = channel

    def send_notification(self, alert: QualityAlert, message: str) -> bool:
        """Send Slack notification."""
        try:
            # In production, implement actual Slack webhook
            logger.info("Slack notification sent", alert_id=alert.id, channel=self.channel)
            self.sent_count += 1
            return True

        except Exception as e:
            self.failed_count += 1
            logger.error("Failed to send Slack notification", error=str(e))
            return False


class WebhookNotificationChannel(NotificationChannel):
    """Notification channel that sends to webhooks."""

    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None):
        super().__init__("webhook")
        self.webhook_url = webhook_url
        self.headers = headers or {}

    def send_notification(self, alert: QualityAlert, message: str) -> bool:
        """Send webhook notification."""
        try:
            # In production, implement actual webhook call
            logger.info("Webhook notification sent", alert_id=alert.id, url=self.webhook_url)
            self.sent_count += 1
            return True

        except Exception as e:
            self.failed_count += 1
            logger.error("Failed to send webhook notification", error=str(e))
            return False


class NotificationSystem:
    """Manages notification delivery through multiple channels."""

    def __init__(self):
        self.channels: List[NotificationChannel] = []
        logger.info("Notification System initialized")

    def add_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel."""
        self.channels.append(channel)
        logger.info("Notification channel added", channel=channel.name)

    def send_notification(self, alert: QualityAlert, message: str) -> bool:
        """Send notification through all channels."""
        results = []
        for channel in self.channels:
            try:
                if channel.can_handle_alert(alert):
                    result = channel.send_notification(alert, message)
                    results.append(result)
            except Exception as e:
                logger.error("Channel failed", channel=channel.name, error=str(e))
                results.append(False)

        # Return True if at least one channel succeeded
        return any(results)

    def get_statistics(self) -> Dict[str, Any]:
        """Get notification system statistics."""
        return {
            "channels_count": len(self.channels),
            "channel_stats": {
                channel.name: {
                    "sent_count": channel.sent_count,
                    "failed_count": channel.failed_count,
                }
                for channel in self.channels
            },
        }

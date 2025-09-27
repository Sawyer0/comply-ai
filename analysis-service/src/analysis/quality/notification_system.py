"""
Notification System

This module handles sending notifications for quality alerts.
Follows SRP by focusing solely on notification delivery.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from .models import QualityAlert, AlertSeverity

logger = logging.getLogger(__name__)


class NotificationChannel:
    """Base class for notification channels."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", "unknown")
        self.min_severity = AlertSeverity(config.get("min_severity", "low"))
        self.enabled = config.get("enabled", True)

    async def send_notification(self, alert: QualityAlert) -> bool:
        """Send notification for alert. To be implemented by subclasses."""
        # Default implementation logs the alert
        logger.info(
            "Quality alert notification",
            alert_id=alert.id,
            alert_type=alert.alert_type,
            severity=alert.severity.value,
            message=alert.message,
            channel=self.__class__.__name__
        )
        return True

    def should_notify(self, alert: QualityAlert) -> bool:
        """Check if this channel should notify for the alert."""
        if not self.enabled:
            return False

        # Check severity filter
        severity_levels = ["low", "medium", "high", "critical"]
        alert_severity_index = severity_levels.index(alert.severity.value)
        min_severity_index = severity_levels.index(self.min_severity.value)

        return alert_severity_index >= min_severity_index


class LogNotificationChannel(NotificationChannel):
    """Log-based notification channel."""

    async def send_notification(self, alert: QualityAlert) -> bool:
        """Send notification via logging."""
        try:
            log_level = {
                "low": logging.INFO,
                "medium": logging.WARNING,
                "high": logging.ERROR,
                "critical": logging.CRITICAL,
            }.get(alert.severity.value, logging.WARNING)

            logger.log(
                log_level,
                "Quality Alert: %s - %s (%s)",
                alert.title,
                alert.description,
                alert.source_component,
                extra={
                    "alert_id": alert.id,
                    "severity": alert.severity.value,
                    "component": alert.source_component,
                    "alert_type": alert.alert_type,
                },
            )

            return True

        except Exception as e:
            logger.error("Log notification failed", alert_id=alert.id, error=str(e))
            return False


class WebhookNotificationChannel(NotificationChannel):
    """Webhook-based notification channel."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.webhook_url = config.get("webhook_url")
        self.timeout = config.get("timeout", 10)

    async def send_notification(self, alert: QualityAlert) -> bool:
        """Send notification via webhook."""
        try:
            if not self.webhook_url:
                logger.error("Webhook URL not configured")
                return False

            # In production, would send actual HTTP request
            # For now, simulate webhook call
            await asyncio.sleep(0.1)  # Simulate network delay

            logger.info(
                "Webhook notification sent",
                alert_id=alert.id,
                webhook_url=self.webhook_url,
                severity=alert.severity.value,
            )

            return True

        except Exception as e:
            logger.error(
                "Webhook notification failed",
                alert_id=alert.id,
                webhook_url=self.webhook_url,
                error=str(e),
            )
            return False


class EmailNotificationChannel(NotificationChannel):
    """Email-based notification channel."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.recipients = config.get("recipients", [])
        self.smtp_config = config.get("smtp", {})

    async def send_notification(self, alert: QualityAlert) -> bool:
        """Send notification via email."""
        try:
            if not self.recipients:
                logger.error("No email recipients configured")
                return False

            # In production, would send actual email
            # For now, simulate email sending
            await asyncio.sleep(0.2)  # Simulate email sending delay

            logger.info(
                "Email notification sent",
                alert_id=alert.id,
                recipients=self.recipients,
                severity=alert.severity.value,
            )

            return True

        except Exception as e:
            logger.error(
                "Email notification failed",
                alert_id=alert.id,
                recipients=self.recipients,
                error=str(e),
            )
            return False


class SlackNotificationChannel(NotificationChannel):
    """Slack-based notification channel."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.webhook_url = config.get("webhook_url")
        self.channel = config.get("channel")

    async def send_notification(self, alert: QualityAlert) -> bool:
        """Send notification via Slack."""
        try:
            if not self.webhook_url:
                logger.error("Slack webhook URL not configured")
                return False

            # In production, would send actual Slack message
            # For now, simulate Slack notification
            await asyncio.sleep(0.1)  # Simulate API call delay

            logger.info(
                "Slack notification sent",
                alert_id=alert.id,
                channel=self.channel,
                severity=alert.severity.value,
            )

            return True

        except Exception as e:
            logger.error(
                "Slack notification failed",
                alert_id=alert.id,
                channel=self.channel,
                error=str(e),
            )
            return False


class NotificationSystem:
    """
    Manages notification delivery for quality alerts.

    Single responsibility: Route and deliver notifications through configured channels.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.channels: List[NotificationChannel] = []
        self.notification_stats = {
            "total_notifications": 0,
            "successful_notifications": 0,
            "failed_notifications": 0,
            "notifications_by_channel": {},
            "notifications_by_severity": {
                "low": 0,
                "medium": 0,
                "high": 0,
                "critical": 0,
            },
        }

        self._initialize_channels()

        logger.info("Notification System initialized", channels=len(self.channels))

    def _initialize_channels(self):
        """Initialize notification channels from configuration."""
        channels_config = self.config.get("notification_channels", {})

        for channel_name, channel_config in channels_config.items():
            try:
                channel_type = channel_config.get("type", "log")
                channel_config["name"] = channel_name

                if channel_type == "log":
                    channel = LogNotificationChannel(channel_config)
                elif channel_type == "webhook":
                    channel = WebhookNotificationChannel(channel_config)
                elif channel_type == "email":
                    channel = EmailNotificationChannel(channel_config)
                elif channel_type == "slack":
                    channel = SlackNotificationChannel(channel_config)
                else:
                    logger.warning(
                        "Unknown notification channel type",
                        channel=channel_name,
                        type=channel_type,
                    )
                    continue

                self.channels.append(channel)
                self.notification_stats["notifications_by_channel"][channel_name] = {
                    "sent": 0,
                    "failed": 0,
                }

                logger.info(
                    "Notification channel initialized",
                    channel=channel_name,
                    type=channel_type,
                )

            except Exception as e:
                logger.error(
                    "Failed to initialize notification channel",
                    channel=channel_name,
                    error=str(e),
                )

    async def send_alert_notification(self, alert: QualityAlert) -> Dict[str, bool]:
        """
        Send notification for alert through all applicable channels.

        Args:
            alert: Alert to send notification for

        Returns:
            Dictionary mapping channel names to success status
        """
        results = {}

        try:
            self.notification_stats["total_notifications"] += 1
            self.notification_stats["notifications_by_severity"][
                alert.severity.value
            ] += 1

            # Send through all applicable channels
            for channel in self.channels:
                if channel.should_notify(alert):
                    try:
                        success = await channel.send_notification(alert)
                        results[channel.name] = success

                        # Update statistics
                        if success:
                            self.notification_stats["successful_notifications"] += 1
                            self.notification_stats["notifications_by_channel"][
                                channel.name
                            ]["sent"] += 1
                        else:
                            self.notification_stats["failed_notifications"] += 1
                            self.notification_stats["notifications_by_channel"][
                                channel.name
                            ]["failed"] += 1

                    except Exception as e:
                        logger.error(
                            "Channel notification failed",
                            channel=channel.name,
                            alert_id=alert.id,
                            error=str(e),
                        )
                        results[channel.name] = False
                        self.notification_stats["failed_notifications"] += 1
                        self.notification_stats["notifications_by_channel"][
                            channel.name
                        ]["failed"] += 1
                else:
                    logger.debug(
                        "Channel skipped notification",
                        channel=channel.name,
                        alert_id=alert.id,
                        reason="severity_filter",
                    )

            logger.info(
                "Alert notifications processed",
                alert_id=alert.id,
                channels_notified=len([r for r in results.values() if r]),
                channels_failed=len([r for r in results.values() if not r]),
            )

            return results

        except Exception as e:
            logger.error(
                "Alert notification processing failed", alert_id=alert.id, error=str(e)
            )
            return {}

    async def send_escalation_notification(
        self, alert: QualityAlert
    ) -> Dict[str, bool]:
        """
        Send escalation notification for alert.

        Args:
            alert: Alert that was escalated

        Returns:
            Dictionary mapping channel names to success status
        """
        # For escalations, notify all channels regardless of normal filters
        results = {}

        try:
            for channel in self.channels:
                if channel.enabled:
                    try:
                        success = await channel.send_notification(alert)
                        results[channel.name] = success

                        if success:
                            logger.warning(
                                "Escalation notification sent",
                                channel=channel.name,
                                alert_id=alert.id,
                                escalation_level=alert.escalation_level,
                            )

                    except Exception as e:
                        logger.error(
                            "Escalation notification failed",
                            channel=channel.name,
                            alert_id=alert.id,
                            error=str(e),
                        )
                        results[channel.name] = False

            return results

        except Exception as e:
            logger.error(
                "Escalation notification processing failed",
                alert_id=alert.id,
                error=str(e),
            )
            return {}

    def add_notification_channel(self, channel: NotificationChannel):
        """Add a new notification channel."""
        self.channels.append(channel)
        self.notification_stats["notifications_by_channel"][channel.name] = {
            "sent": 0,
            "failed": 0,
        }

        logger.info("Notification channel added", channel=channel.name)

    def remove_notification_channel(self, channel_name: str) -> bool:
        """Remove a notification channel."""
        try:
            self.channels = [c for c in self.channels if c.name != channel_name]

            if channel_name in self.notification_stats["notifications_by_channel"]:
                del self.notification_stats["notifications_by_channel"][channel_name]

            logger.info("Notification channel removed", channel=channel_name)
            return True

        except Exception as e:
            logger.error(
                "Failed to remove notification channel",
                channel=channel_name,
                error=str(e),
            )
            return False

    def get_notification_statistics(self) -> Dict[str, Any]:
        """Get notification system statistics."""
        success_rate = self.notification_stats["successful_notifications"] / max(
            1, self.notification_stats["total_notifications"]
        )

        return {
            "total_notifications": self.notification_stats["total_notifications"],
            "successful_notifications": self.notification_stats[
                "successful_notifications"
            ],
            "failed_notifications": self.notification_stats["failed_notifications"],
            "success_rate": success_rate,
            "notifications_by_severity": self.notification_stats[
                "notifications_by_severity"
            ].copy(),
            "notifications_by_channel": self.notification_stats[
                "notifications_by_channel"
            ].copy(),
            "active_channels": len(self.channels),
            "enabled_channels": len([c for c in self.channels if c.enabled]),
        }

    async def test_notification_channels(self) -> Dict[str, bool]:
        """Test all notification channels with a test alert."""
        test_alert = QualityAlert(
            id="test_alert",
            alert_type="test",
            severity=AlertSeverity.LOW,
            title="Test Alert",
            description="This is a test notification",
            source_component="notification_system",
            metrics={},
            threshold_violated={},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        results = {}

        for channel in self.channels:
            if channel.enabled:
                try:
                    success = await channel.send_notification(test_alert)
                    results[channel.name] = success

                    logger.info(
                        "Test notification result",
                        channel=channel.name,
                        success=success,
                    )

                except Exception as e:
                    logger.error(
                        "Test notification failed", channel=channel.name, error=str(e)
                    )
                    results[channel.name] = False
            else:
                results[channel.name] = False

        return results

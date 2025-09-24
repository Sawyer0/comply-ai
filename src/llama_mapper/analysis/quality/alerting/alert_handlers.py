"""
Alert handlers for different notification channels.

This module provides various alert handlers for sending alerts
through different channels like email, Slack, webhooks, etc.
"""

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

import requests

from ..interfaces import Alert, AlertSeverity, IAlertHandler

logger = logging.getLogger(__name__)


class BaseAlertHandler(IAlertHandler):
    """Base alert handler with common functionality."""

    def __init__(self, name: str, enabled: bool = True):
        """
        Initialize base alert handler.

        Args:
            name: Handler name
            enabled: Whether handler is enabled
        """
        self.name = name
        self.enabled = enabled
        self._sent_count = 0
        self._failed_count = 0

    @property
    def sent_count(self) -> int:
        """Get count of successfully sent alerts."""
        return self._sent_count

    @property
    def failed_count(self) -> int:
        """Get count of failed alert sends."""
        return self._failed_count

    def get_handler_name(self) -> str:
        """Get handler name."""
        return self.name

    def can_handle_alert(self, alert: Alert) -> bool:
        """Check if this handler can handle the alert."""
        return self.enabled

    def _format_alert_message(self, alert: Alert) -> str:
        """Format alert message for display."""
        severity_emoji = {
            AlertSeverity.LOW: "🟡",
            AlertSeverity.MEDIUM: "🟠",
            AlertSeverity.HIGH: "🔴",
            AlertSeverity.CRITICAL: "🚨",
        }

        emoji = severity_emoji.get(alert.severity, "⚠️")

        message = f"{emoji} **{alert.title}**\n\n"
        message += f"**Severity:** {alert.severity.value.upper()}\n"
        message += f"**Metric:** {alert.metric_type.value}\n"
        message += f"**Description:** {alert.description}\n"
        message += f"**Alert ID:** {alert.alert_id}\n"
        message += f"**Created:** {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"

        if alert.degradation_detection:
            detection = alert.degradation_detection
            message += "\n**Degradation Details:**\n"
            message += f"- Type: {detection.degradation_type.value}\n"
            message += f"- Current Value: {detection.current_value:.3f}\n"
            message += f"- Expected Value: {detection.expected_value:.3f}\n"
            message += f"- Deviation: {detection.deviation_percentage:.1f}%\n"
            message += f"- Confidence: {detection.confidence:.2f}\n"

        return message


class LoggingAlertHandler(BaseAlertHandler):
    """Alert handler that logs alerts to the application log."""

    def __init__(self, log_level: str = "WARNING"):
        """
        Initialize logging alert handler.

        Args:
            log_level: Log level for alerts (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        super().__init__("logging")
        self.log_level = getattr(logging, log_level.upper(), logging.WARNING)

    def send_alert(self, alert: Alert) -> bool:
        """Send alert to application log."""
        try:
            message = self._format_alert_message(alert)

            # Log with appropriate level based on severity
            if alert.severity == AlertSeverity.CRITICAL:
                logger.critical("CRITICAL ALERT: %s", message)
            elif alert.severity == AlertSeverity.HIGH:
                logger.error("HIGH ALERT: %s", message)
            elif alert.severity == AlertSeverity.MEDIUM:
                logger.warning("MEDIUM ALERT: %s", message)
            else:
                logger.info("LOW ALERT: %s", message)

            self._sent_count += 1
            return True

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to log alert: %s", e)
            self._failed_count += 1
            return False


class EmailAlertHandler(BaseAlertHandler):
    """Alert handler that sends alerts via email."""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        to_emails: List[str],
        use_tls: bool = True,
    ):
        """
        Initialize email alert handler.

        Args:
            smtp_server: SMTP server hostname
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_email: From email address
            to_emails: List of recipient email addresses
            use_tls: Whether to use TLS
        """
        super().__init__("email")
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
        self.use_tls = use_tls

    def send_alert(self, alert: Alert) -> bool:
        """Send alert via email."""
        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.from_email
            msg["To"] = ", ".join(self.to_emails)
            msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.title}"

            # Add body
            body = self._format_alert_message(alert)
            msg.attach(MIMEText(body, "plain"))

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            self._sent_count += 1
            logger.info("Email alert sent for %s", alert.alert_id)
            return True

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to send email alert: %s", e)
            self._failed_count += 1
            return False


class SlackAlertHandler(BaseAlertHandler):
    """Alert handler that sends alerts to Slack."""

    def __init__(
        self,
        webhook_url: str,
        channel: Optional[str] = None,
        username: str = "Quality Monitor",
        icon_emoji: str = ":warning:",
    ):
        """
        Initialize Slack alert handler.

        Args:
            webhook_url: Slack webhook URL
            channel: Slack channel (optional)
            username: Bot username
            icon_emoji: Bot icon emoji
        """
        super().__init__("slack")
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username
        self.icon_emoji = icon_emoji

    def send_alert(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        try:
            # Create Slack message
            color = {
                AlertSeverity.LOW: "good",
                AlertSeverity.MEDIUM: "warning",
                AlertSeverity.HIGH: "danger",
                AlertSeverity.CRITICAL: "danger",
            }.get(alert.severity, "warning")

            # Create attachment
            attachment = {
                "color": color,
                "title": alert.title,
                "text": alert.description,
                "fields": [
                    {
                        "title": "Severity",
                        "value": alert.severity.value.upper(),
                        "short": True,
                    },
                    {
                        "title": "Metric",
                        "value": alert.metric_type.value,
                        "short": True,
                    },
                    {"title": "Alert ID", "value": alert.alert_id, "short": True},
                    {
                        "title": "Created",
                        "value": alert.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "short": True,
                    },
                ],
                "footer": "Quality Monitor",
                "ts": int(alert.created_at.timestamp()),
            }

            # Add degradation details if available
            if alert.degradation_detection:
                detection = alert.degradation_detection
                attachment["fields"].extend(
                    [
                        {
                            "title": "Degradation Type",
                            "value": detection.degradation_type.value,
                            "short": True,
                        },
                        {
                            "title": "Current Value",
                            "value": f"{detection.current_value:.3f}",
                            "short": True,
                        },
                        {
                            "title": "Expected Value",
                            "value": f"{detection.expected_value:.3f}",
                            "short": True,
                        },
                        {
                            "title": "Deviation",
                            "value": f"{detection.deviation_percentage:.1f}%",
                            "short": True,
                        },
                    ]
                )

            # Create payload
            payload = {
                "username": self.username,
                "icon_emoji": self.icon_emoji,
                "attachments": [attachment],
            }

            if self.channel:
                payload["channel"] = self.channel

            # Send to Slack
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            self._sent_count += 1
            logger.info("Slack alert sent for %s", alert.alert_id)
            return True

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to send Slack alert: %s", e)
            self._failed_count += 1
            return False


class WebhookAlertHandler(BaseAlertHandler):
    """Alert handler that sends alerts to webhooks."""

    def __init__(
        self,
        webhook_url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 10,
    ):
        """
        Initialize webhook alert handler.

        Args:
            webhook_url: Webhook URL
            headers: Additional headers
            timeout: Request timeout in seconds
        """
        super().__init__("webhook")
        self.webhook_url = webhook_url
        self.headers = headers or {}
        self.timeout = timeout

    def send_alert(self, alert: Alert) -> bool:
        """Send alert to webhook."""
        try:
            # Create payload
            payload: Dict[str, Any] = {
                "alert_id": alert.alert_id,
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "metric_type": alert.metric_type.value,
                "created_at": alert.created_at.isoformat(),
                "updated_at": alert.updated_at.isoformat(),
            }

            # Add degradation detection if available
            if alert.degradation_detection:
                detection = alert.degradation_detection
                payload["degradation_detection"] = {
                    "degradation_type": detection.degradation_type.value,
                    "current_value": detection.current_value,
                    "expected_value": detection.expected_value,
                    "deviation_percentage": detection.deviation_percentage,
                    "confidence": detection.confidence,
                    "timestamp": detection.timestamp.isoformat(),
                    "description": detection.description,
                    "metadata": detection.metadata,
                }

            # Add labels and metadata
            if alert.labels:
                payload["labels"] = alert.labels
            if alert.metadata:
                payload["metadata"] = alert.metadata

            # Send webhook
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()

            self._sent_count += 1
            logger.info("Webhook alert sent for %s", alert.alert_id)
            return True

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to send webhook alert: %s", e)
            self._failed_count += 1
            return False


class CompositeAlertHandler(BaseAlertHandler):
    """Composite alert handler that routes alerts to multiple handlers."""

    def __init__(self, handlers: List[IAlertHandler]):
        """
        Initialize composite alert handler.

        Args:
            handlers: List of alert handlers
        """
        super().__init__("composite")
        self.handlers = handlers

    def send_alert(self, alert: Alert) -> bool:
        """Send alert through all applicable handlers."""
        success_count = 0
        total_handlers = 0

        for handler in self.handlers:
            if handler.can_handle_alert(alert):
                total_handlers += 1
                if handler.send_alert(alert):
                    success_count += 1

        # Consider successful if at least one handler succeeded
        success = success_count > 0

        if success:
            self._sent_count += 1
        else:
            self._failed_count += 1

        logger.info(
            "Composite alert sent: %d/%d handlers succeeded",
            success_count,
            total_handlers,
        )

        return success

    def can_handle_alert(self, alert: Alert) -> bool:
        """Check if any handler can handle the alert."""
        return any(handler.can_handle_alert(alert) for handler in self.handlers)

    def get_handler_statistics(self) -> Dict[str, Dict[str, int]]:
        """Get statistics for all handlers."""
        stats = {}
        for handler in self.handlers:
            stats[handler.get_handler_name()] = {
                "sent": handler.sent_count,
                "failed": handler.failed_count,
            }
        return stats

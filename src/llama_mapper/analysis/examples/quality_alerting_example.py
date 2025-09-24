"""
Quality alerting system usage example.

This example demonstrates how to set up and use the quality alerting system
for monitoring analysis module quality metrics.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict

from ..quality import (
    AlertSeverity,
    QualityAlertingSettings,
    QualityAlertingSystem,
    QualityMetric,
    QualityMetricType,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityAlertingExample:
    """Example demonstrating quality alerting system usage."""

    def __init__(self):
        """Initialize the example."""
        self.alerting_system = None
        self.setup_alerting_system()

    def setup_alerting_system(self):
        """Setup the quality alerting system with configuration."""
        # Create settings
        settings = QualityAlertingSettings(
            monitoring_interval_seconds=30,
            max_metrics_per_type=1000,
            alert_retention_days=7,
            deduplication_window_minutes=10,
            # Email configuration (optional)
            email_enabled=False,  # Set to True and configure for email alerts
            email_smtp_server="smtp.gmail.com",
            email_smtp_port=587,
            email_username="your-email@gmail.com",
            email_password="your-app-password",
            email_from="your-email@gmail.com",
            email_to=["admin@company.com", "devops@company.com"],
            # Slack configuration (optional)
            slack_enabled=False,  # Set to True and configure for Slack alerts
            slack_webhook_url="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
            slack_channel="#alerts",
            # Webhook configuration (optional)
            webhook_enabled=False,  # Set to True and configure for webhook alerts
            webhook_url="https://your-monitoring-system.com/webhook",
            webhook_headers={"Authorization": "Bearer your-token"},
            # Threshold configuration
            schema_validation_warning=0.95,
            schema_validation_critical=0.90,
            template_fallback_warning=0.20,
            template_fallback_critical=0.30,
            confidence_score_warning=0.70,
            confidence_score_critical=0.60,
            response_time_warning=2.0,
            response_time_critical=5.0,
            error_rate_warning=0.05,
            error_rate_critical=0.10,
        )

        # Create alerting system
        self.alerting_system = QualityAlertingSystem(
            monitoring_interval_seconds=settings.monitoring_interval_seconds,
            max_metrics_per_type=settings.max_metrics_per_type,
            alert_retention_days=settings.alert_retention_days,
            deduplication_window_minutes=settings.deduplication_window_minutes,
        )

        # Add alert handlers based on configuration
        if settings.email_enabled:
            self.alerting_system.add_email_handler(
                smtp_server=settings.email_smtp_server,
                smtp_port=settings.email_smtp_port,
                username=settings.email_username,
                password=settings.email_password,
                from_email=settings.email_from,
                to_emails=settings.email_to,
            )

        if settings.slack_enabled:
            self.alerting_system.add_slack_handler(
                webhook_url=settings.slack_webhook_url, channel=settings.slack_channel
            )

        if settings.webhook_enabled:
            self.alerting_system.add_webhook_handler(
                webhook_url=settings.webhook_url, headers=settings.webhook_headers
            )

        # Add custom thresholds
        self.add_custom_thresholds()

        logger.info("Quality alerting system configured")

    def add_custom_thresholds(self):
        """Add custom quality thresholds."""
        from ..quality import QualityThreshold

        # Add custom threshold for throughput
        self.alerting_system.add_threshold(
            QualityThreshold(
                metric_type=QualityMetricType.THROUGHPUT,
                warning_threshold=10.0,  # requests per second
                critical_threshold=5.0,
                min_samples=5,
                time_window_minutes=30,
            )
        )

        logger.info("Custom thresholds added")

    async def simulate_quality_metrics(self):
        """Simulate quality metrics to demonstrate the system."""
        logger.info("Starting quality metrics simulation...")

        # Start monitoring
        self.alerting_system.start_monitoring()

        try:
            # Simulate normal operation
            await self.simulate_normal_operation()

            # Simulate quality degradation
            await self.simulate_quality_degradation()

            # Simulate recovery
            await self.simulate_recovery()

            # Show system status
            self.show_system_status()

        finally:
            # Stop monitoring
            self.alerting_system.stop_monitoring()

    async def simulate_normal_operation(self):
        """Simulate normal quality metrics."""
        logger.info("Simulating normal operation...")

        base_time = datetime.now()

        # Simulate good quality metrics
        for i in range(20):
            timestamp = base_time + timedelta(minutes=i)

            # High schema validation rate
            self.alerting_system.process_metric(
                QualityMetric(
                    metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                    value=0.98 + (i % 3) * 0.01,  # 0.98-1.00
                    timestamp=timestamp,
                    labels={"service": "analysis", "version": "1.0"},
                )
            )

            # Low template fallback rate
            self.alerting_system.process_metric(
                QualityMetric(
                    metric_type=QualityMetricType.TEMPLATE_FALLBACK_RATE,
                    value=0.05 + (i % 2) * 0.02,  # 0.05-0.07
                    timestamp=timestamp,
                    labels={"service": "analysis", "version": "1.0"},
                )
            )

            # High confidence scores
            self.alerting_system.process_metric(
                QualityMetric(
                    metric_type=QualityMetricType.CONFIDENCE_SCORE,
                    value=0.85 + (i % 4) * 0.03,  # 0.85-0.97
                    timestamp=timestamp,
                    labels={"service": "analysis", "version": "1.0"},
                )
            )

            # Good response times
            self.alerting_system.process_metric(
                QualityMetric(
                    metric_type=QualityMetricType.RESPONSE_TIME,
                    value=0.5 + (i % 3) * 0.2,  # 0.5-0.9 seconds
                    timestamp=timestamp,
                    labels={"service": "analysis", "version": "1.0"},
                )
            )

            # Low error rates
            self.alerting_system.process_metric(
                QualityMetric(
                    metric_type=QualityMetricType.ERROR_RATE,
                    value=0.01 + (i % 2) * 0.005,  # 0.01-0.015
                    timestamp=timestamp,
                    labels={"service": "analysis", "version": "1.0"},
                )
            )

            await asyncio.sleep(0.1)  # Small delay

    async def simulate_quality_degradation(self):
        """Simulate quality degradation scenarios."""
        logger.info("Simulating quality degradation...")

        base_time = datetime.now()

        # Simulate schema validation rate drop
        for i in range(10):
            timestamp = base_time + timedelta(minutes=i)

            # Gradually decreasing schema validation rate
            validation_rate = 0.95 - (i * 0.02)  # 0.95 -> 0.77
            self.alerting_system.process_metric(
                QualityMetric(
                    metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                    value=validation_rate,
                    timestamp=timestamp,
                    labels={"service": "analysis", "version": "1.0"},
                )
            )

            # Increasing template fallback rate
            fallback_rate = 0.10 + (i * 0.03)  # 0.10 -> 0.37
            self.alerting_system.process_metric(
                QualityMetric(
                    metric_type=QualityMetricType.TEMPLATE_FALLBACK_RATE,
                    value=fallback_rate,
                    timestamp=timestamp,
                    labels={"service": "analysis", "version": "1.0"},
                )
            )

            # Decreasing confidence scores
            confidence = 0.75 - (i * 0.02)  # 0.75 -> 0.55
            self.alerting_system.process_metric(
                QualityMetric(
                    metric_type=QualityMetricType.CONFIDENCE_SCORE,
                    value=confidence,
                    timestamp=timestamp,
                    labels={"service": "analysis", "version": "1.0"},
                )
            )

            # Increasing response times
            response_time = 1.0 + (i * 0.5)  # 1.0 -> 5.5 seconds
            self.alerting_system.process_metric(
                QualityMetric(
                    metric_type=QualityMetricType.RESPONSE_TIME,
                    value=response_time,
                    timestamp=timestamp,
                    labels={"service": "analysis", "version": "1.0"},
                )
            )

            # Increasing error rates
            error_rate = 0.02 + (i * 0.01)  # 0.02 -> 0.11
            self.alerting_system.process_metric(
                QualityMetric(
                    metric_type=QualityMetricType.ERROR_RATE,
                    value=error_rate,
                    timestamp=timestamp,
                    labels={"service": "analysis", "version": "1.0"},
                )
            )

            await asyncio.sleep(0.2)  # Longer delay to see alerts

    async def simulate_recovery(self):
        """Simulate quality recovery."""
        logger.info("Simulating quality recovery...")

        base_time = datetime.now()

        # Simulate recovery to good metrics
        for i in range(15):
            timestamp = base_time + timedelta(minutes=i)

            # Recovering schema validation rate
            validation_rate = 0.80 + (i * 0.01)  # 0.80 -> 0.94
            self.alerting_system.process_metric(
                QualityMetric(
                    metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                    value=validation_rate,
                    timestamp=timestamp,
                    labels={"service": "analysis", "version": "1.0"},
                )
            )

            # Decreasing template fallback rate
            fallback_rate = 0.35 - (i * 0.02)  # 0.35 -> 0.05
            self.alerting_system.process_metric(
                QualityMetric(
                    metric_type=QualityMetricType.TEMPLATE_FALLBACK_RATE,
                    value=fallback_rate,
                    timestamp=timestamp,
                    labels={"service": "analysis", "version": "1.0"},
                )
            )

            # Recovering confidence scores
            confidence = 0.60 + (i * 0.02)  # 0.60 -> 0.90
            self.alerting_system.process_metric(
                QualityMetric(
                    metric_type=QualityMetricType.CONFIDENCE_SCORE,
                    value=confidence,
                    timestamp=timestamp,
                    labels={"service": "analysis", "version": "1.0"},
                )
            )

            # Decreasing response times
            response_time = 5.0 - (i * 0.3)  # 5.0 -> 0.5 seconds
            self.alerting_system.process_metric(
                QualityMetric(
                    metric_type=QualityMetricType.RESPONSE_TIME,
                    value=response_time,
                    timestamp=timestamp,
                    labels={"service": "analysis", "version": "1.0"},
                )
            )

            # Decreasing error rates
            error_rate = 0.10 - (i * 0.006)  # 0.10 -> 0.01
            self.alerting_system.process_metric(
                QualityMetric(
                    metric_type=QualityMetricType.ERROR_RATE,
                    value=error_rate,
                    timestamp=timestamp,
                    labels={"service": "analysis", "version": "1.0"},
                )
            )

            await asyncio.sleep(0.1)

    def show_system_status(self):
        """Show system status and statistics."""
        logger.info("=== Quality Alerting System Status ===")

        status = self.alerting_system.get_system_status()

        logger.info("Monitoring Active: %s", status["monitoring_active"])
        logger.info("Thresholds Configured: %s", status["thresholds_configured"])
        logger.info("Alert Handlers: %s", status["alert_handlers"])
        logger.info("Metrics Processed: %s", status["statistics"]["metrics_processed"])
        logger.info(
            f"Degradations Detected: {status['statistics']['degradations_detected']}"
        )
        logger.info("Alerts Created: %s", status["statistics"]["alerts_created"])

        # Show active alerts
        active_alerts = self.alerting_system.alert_manager.get_active_alerts()
        logger.info("Active Alerts: %s", len(active_alerts))

        for alert in active_alerts[:5]:  # Show first 5
            logger.info("  - %s: %s", alert.severity.value, alert.title)

        # Show dashboard data
        dashboard_data = self.alerting_system.get_quality_dashboard_data()
        logger.info("=== Current Quality Metrics ===")

        for metric_type, value in dashboard_data["current_metrics"].items():
            logger.info("%s: %.3f", metric_type, value)

        logger.info("=== Quality Trends ===")
        for metric_type, trend in dashboard_data["trends"].items():
            logger.info(
                f"{metric_type}: {trend['trend']} (slope: {trend['slope']:.4f})"
            )


async def main():
    """Main example function."""
    example = QualityAlertingExample()
    await example.simulate_quality_metrics()


if __name__ == "__main__":
    asyncio.run(main())

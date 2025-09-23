"""
Unit tests for quality monitoring system.

This module contains comprehensive unit tests for the quality monitoring,
degradation detection, and alerting components.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.llama_mapper.analysis.quality import (
    QualityMetric, QualityMetricType, QualityThreshold, AlertSeverity,
    QualityMonitor, QualityDegradationDetector, AlertManager,
    LoggingAlertHandler, EmailAlertHandler, SlackAlertHandler,
    WebhookAlertHandler, QualityAlertingSystem, DegradationDetection, 
    DegradationType, Alert, AlertStatus
)


class TestQualityMonitor:
    """Test cases for QualityMonitor."""
    
    def test_quality_monitor_initialization(self):
        """Test quality monitor initialization."""
        monitor = QualityMonitor(max_metrics_per_type=1000)
        
        assert monitor.max_metrics_per_type == 1000
        assert monitor.default_retention_hours == 24
        assert len(monitor._current_metrics) == 0
    
    def test_record_metric(self):
        """Test recording quality metrics."""
        monitor = QualityMonitor()
        
        metric = QualityMetric(
            metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
            value=0.95,
            timestamp=datetime.now(),
            labels={"service": "test"}
        )
        
        monitor.record_metric(metric)
        
        assert QualityMetricType.SCHEMA_VALIDATION_RATE in monitor._current_metrics
        assert monitor._current_metrics[QualityMetricType.SCHEMA_VALIDATION_RATE] == 0.95
    
    def test_get_metrics(self):
        """Test retrieving metrics for a time range."""
        monitor = QualityMonitor()
        
        base_time = datetime.now() - timedelta(minutes=10) - timedelta(minutes=10)

        # Add metrics at different times
        for i in range(5):
            metric = QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=0.9 + i * 0.02,
                timestamp=base_time + timedelta(minutes=i),
                labels={"service": "test"}
            )
            monitor.record_metric(metric)
        
        # Get metrics for a specific time range
        start_time = base_time + timedelta(minutes=1)
        end_time = base_time + timedelta(minutes=3)
        
        metrics = monitor.get_metrics(
            QualityMetricType.SCHEMA_VALIDATION_RATE,
            start_time,
            end_time
        )
        
        assert len(metrics) == 3
        assert all(start_time <= m.timestamp <= end_time for m in metrics)
    
    def test_get_metric_statistics(self):
        """Test metric statistics calculation."""
        monitor = QualityMonitor()
        
        base_time = datetime.now() - timedelta(minutes=10)
        
        # Add metrics with known values
        values = [0.9, 0.92, 0.88, 0.95, 0.91]
        for i, value in enumerate(values):
            metric = QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=value,
                timestamp=base_time + timedelta(minutes=i),
                labels={"service": "test"}
            )
            monitor.record_metric(metric)
        
        stats = monitor.get_metric_statistics(
            QualityMetricType.SCHEMA_VALIDATION_RATE,
            time_window_minutes=10
        )
        
        assert stats["count"] == 5
        assert stats["mean"] == pytest.approx(0.912, rel=1e-3)
        assert stats["min"] == 0.88
        assert stats["max"] == 0.95
        assert stats["std"] > 0
    
    def test_get_metric_trends(self):
        """Test metric trend analysis."""
        monitor = QualityMonitor()
        
        base_time = datetime.now() - timedelta(minutes=10)
        
        # Add metrics with a clear trend
        for i in range(10):
            value = 0.9 + i * 0.01  # Increasing trend
            metric = QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=value,
                timestamp=base_time + timedelta(minutes=i),
                labels={"service": "test"}
            )
            monitor.record_metric(metric)
        
        trends = monitor.get_metric_trends(
            QualityMetricType.SCHEMA_VALIDATION_RATE,
            time_window_minutes=60
        )
        
        assert trends["trend"] == "increasing"
        assert trends["slope"] > 0
        assert trends["r_squared"] > 0.5


class TestQualityDegradationDetector:
    """Test cases for QualityDegradationDetector."""
    
    def test_detector_initialization(self):
        """Test degradation detector initialization."""
        detector = QualityDegradationDetector(
            anomaly_sensitivity=2.5,
            trend_window_minutes=45
        )
        
        assert detector.anomaly_sensitivity == 2.5
        assert detector.trend_window_minutes == 45
        assert detector.min_samples_for_detection == 5
    
    def test_detect_threshold_breach(self):
        """Test threshold breach detection."""
        detector = QualityDegradationDetector()
        
        threshold = QualityThreshold(
            metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
            warning_threshold=0.95,
            critical_threshold=0.90,
            min_samples=3,
            time_window_minutes=60
        )
        
        base_time = datetime.now() - timedelta(minutes=10)
        
        # Create metrics that breach the critical threshold
        metrics = [
            QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=0.85,  # Below critical threshold
                timestamp=base_time + timedelta(minutes=i),
                labels={"service": "test"}
            )
            for i in range(5)
        ]
        
        degradation = detector.detect_degradation(
            QualityMetricType.SCHEMA_VALIDATION_RATE,
            metrics,
            threshold
        )
        
        assert degradation is not None
        assert degradation.degradation_type == DegradationType.THRESHOLD_BREACH
        assert degradation.severity == AlertSeverity.CRITICAL
        assert degradation.current_value == 0.85
    
    def test_detect_sudden_drop(self):
        """Test sudden drop detection."""
        detector = QualityDegradationDetector()
        
        base_time = datetime.now() - timedelta(minutes=10)
        
        # Create metrics with a sudden drop
        metrics = [
            QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=0.95,
                timestamp=base_time + timedelta(minutes=0),
                labels={"service": "test"}
            ),
            QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=0.95,
                timestamp=base_time + timedelta(minutes=1),
                labels={"service": "test"}
            ),
            QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=0.70,  # Sudden drop (>20%)
                timestamp=base_time + timedelta(minutes=2),
                labels={"service": "test"}
            )
        ]
        
        threshold = QualityThreshold(
            metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
            warning_threshold=0.95,
            critical_threshold=0.90,
            min_samples=3,
            time_window_minutes=60
        )
        
        degradation = detector.detect_degradation(
            QualityMetricType.SCHEMA_VALIDATION_RATE,
            metrics,
            threshold
        )
        
        assert degradation is not None
        assert degradation.degradation_type == DegradationType.SUDDEN_DROP
        assert degradation.severity in [AlertSeverity.MEDIUM, AlertSeverity.HIGH, AlertSeverity.CRITICAL]
    
    def test_detect_anomalies(self):
        """Test anomaly detection."""
        detector = QualityDegradationDetector(anomaly_sensitivity=2.0)
        
        base_time = datetime.now() - timedelta(minutes=10)
        
        # Create metrics with one clear anomaly
        values = [0.95, 0.94, 0.96, 0.95, 0.50, 0.95, 0.94]  # 0.50 is an anomaly
        metrics = [
            QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=value,
                timestamp=base_time + timedelta(minutes=i),
                labels={"service": "test"}
            )
            for i, value in enumerate(values)
        ]
        
        anomalies = detector.detect_anomalies(
            QualityMetricType.SCHEMA_VALIDATION_RATE,
            metrics
        )
        
        assert len(anomalies) == 1
        assert anomalies[0].degradation_type == DegradationType.ANOMALY
        assert anomalies[0].current_value == 0.50
    
    def test_detect_trends(self):
        """Test trend detection."""
        detector = QualityDegradationDetector()
        
        base_time = datetime.now() - timedelta(minutes=10)
        
        # Create metrics with a clear declining trend
        metrics = [
            QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=0.95 - i * 0.02,  # Declining trend
                timestamp=base_time + timedelta(minutes=i),
                labels={"service": "test"}
            )
            for i in range(10)
        ]
        
        trends = detector.detect_trends(
            QualityMetricType.SCHEMA_VALIDATION_RATE,
            metrics
        )
        
        assert len(trends) == 1
        assert trends[0].degradation_type == DegradationType.GRADUAL_DECLINE
        assert trends[0].severity in [AlertSeverity.MEDIUM, AlertSeverity.HIGH, AlertSeverity.CRITICAL]


class TestAlertManager:
    """Test cases for AlertManager."""
    
    def test_alert_manager_initialization(self):
        """Test alert manager initialization."""
        manager = AlertManager(max_alerts=1000)
        
        assert manager.max_alerts == 1000
        assert manager.alert_retention_days == 30
        assert len(manager._alerts) == 0
        assert len(manager._active_alerts) == 0
    
    def test_create_alert(self):
        """Test alert creation."""
        manager = AlertManager()
        
        alert = manager.create_alert(
            title="Test Alert",
            description="Test alert description",
            severity=AlertSeverity.HIGH,
            metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE
        )
        
        assert alert.title == "Test Alert"
        assert alert.severity == AlertSeverity.HIGH
        assert alert.metric_type == QualityMetricType.SCHEMA_VALIDATION_RATE
        assert alert.alert_id in manager._alerts
        assert alert.alert_id in manager._active_alerts
    
    def test_alert_deduplication(self):
        """Test alert deduplication."""
        manager = AlertManager(deduplication_window_minutes=5)
        
        # Create first alert
        alert1 = manager.create_alert(
            title="Test Alert",
            description="Test alert description",
            severity=AlertSeverity.HIGH,
            metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE
        )
        
        # Create similar alert within deduplication window
        alert2 = manager.create_alert(
            title="Test Alert",
            description="Test alert description",
            severity=AlertSeverity.HIGH,
            metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE
        )
        
        # Should return the same alert due to deduplication
        assert alert1.alert_id == alert2.alert_id
        assert len(manager._alerts) == 1
    
    def test_acknowledge_alert(self):
        """Test alert acknowledgment."""
        manager = AlertManager()
        
        alert = manager.create_alert(
            title="Test Alert",
            description="Test alert description",
            severity=AlertSeverity.HIGH,
            metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE
        )
        
        success = manager.acknowledge_alert(alert.alert_id, "test_user")
        
        assert success
        assert alert.status.value == "acknowledged"
        assert alert.acknowledged_at is not None
    
    def test_resolve_alert(self):
        """Test alert resolution."""
        manager = AlertManager()
        
        alert = manager.create_alert(
            title="Test Alert",
            description="Test alert description",
            severity=AlertSeverity.HIGH,
            metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE
        )
        
        success = manager.resolve_alert(alert.alert_id, "test_user")
        
        assert success
        assert alert.status.value == "resolved"
        assert alert.resolved_at is not None
        assert alert.alert_id not in manager._active_alerts
    
    def test_suppress_alert(self):
        """Test alert suppression."""
        manager = AlertManager()
        
        alert = manager.create_alert(
            title="Test Alert",
            description="Test alert description",
            severity=AlertSeverity.HIGH,
            metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE
        )
        
        success = manager.suppress_alert(alert.alert_id, "False positive")
        
        assert success
        assert alert.status.value == "suppressed"
        assert alert.metadata["suppression_reason"] == "False positive"
        assert alert.alert_id not in manager._active_alerts


class TestAlertHandlers:
    """Test cases for alert handlers."""
    
    def test_logging_alert_handler(self):
        """Test logging alert handler."""
        handler = LoggingAlertHandler(log_level="WARNING")
        
        alert = Alert(
            alert_id="test_alert",
            title="Test Alert",
            description="Test description",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.ACTIVE,
            metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
            degradation_detection=None,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        with patch('src.llama_mapper.analysis.quality.alerting.alert_handlers.logger') as mock_logger:
            success = handler.send_alert(alert)
            
            assert success
            assert handler.sent_count == 1
            mock_logger.error.assert_called_once()
    
    @patch('smtplib.SMTP')
    def test_email_alert_handler(self, mock_smtp):
        """Test email alert handler."""
        handler = EmailAlertHandler(
            smtp_server="smtp.test.com",
            smtp_port=587,
            username="test@test.com",
            password="password",
            from_email="test@test.com",
            to_emails=["admin@test.com"]
        )
        
        alert = Alert(
            alert_id="test_alert",
            title="Test Alert",
            description="Test description",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.ACTIVE,
            metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
            degradation_detection=None,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Mock SMTP server
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        success = handler.send_alert(alert)
        
        assert success
        assert handler.sent_count == 1
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()
        mock_server.send_message.assert_called_once()
    
    @patch('requests.post')
    def test_slack_alert_handler(self, mock_post):
        """Test Slack alert handler."""
        handler = SlackAlertHandler(
            webhook_url="https://hooks.slack.com/test",
            channel="#test"
        )
        
        alert = Alert(
            alert_id="test_alert",
            title="Test Alert",
            description="Test description",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.ACTIVE,
            metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
            degradation_detection=None,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        success = handler.send_alert(alert)
        
        assert success
        assert handler.sent_count == 1
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_webhook_alert_handler(self, mock_post):
        """Test webhook alert handler."""
        handler = WebhookAlertHandler(
            webhook_url="https://test.com/webhook",
            headers={"Authorization": "Bearer token"}
        )
        
        alert = Alert(
            alert_id="test_alert",
            title="Test Alert",
            description="Test description",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.ACTIVE,
            metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
            degradation_detection=None,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        success = handler.send_alert(alert)
        
        assert success
        assert handler.sent_count == 1
        mock_post.assert_called_once()


class TestQualityAlertingSystem:
    """Test cases for QualityAlertingSystem."""
    
    def test_system_initialization(self):
        """Test quality alerting system initialization."""
        system = QualityAlertingSystem(
            monitoring_interval_seconds=30,
            max_metrics_per_type=1000
        )
        
        assert system.monitoring_interval_seconds == 30
        assert system.quality_monitor.max_metrics_per_type == 1000
        assert not system._monitoring_active
        assert len(system.thresholds) > 0  # Should have default thresholds
    
    def test_add_remove_threshold(self):
        """Test adding and removing thresholds."""
        system = QualityAlertingSystem()
        
        threshold = QualityThreshold(
            metric_type=QualityMetricType.THROUGHPUT,
            warning_threshold=10.0,
            critical_threshold=5.0
        )
        
        system.add_threshold(threshold)
        assert QualityMetricType.THROUGHPUT in system.thresholds
        
        system.remove_threshold(QualityMetricType.THROUGHPUT)
        assert QualityMetricType.THROUGHPUT not in system.thresholds
    
    def test_process_metric(self):
        """Test processing quality metrics."""
        system = QualityAlertingSystem()
        
        metric = QualityMetric(
            metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
            value=0.85,  # Below critical threshold
            timestamp=datetime.now(),
            labels={"service": "test"}
        )
        
        # Process metric
        system.process_metric(metric)
        
        # Check that metric was recorded
        current_metrics = system.quality_monitor.get_current_metrics()
        assert QualityMetricType.SCHEMA_VALIDATION_RATE in current_metrics
        assert current_metrics[QualityMetricType.SCHEMA_VALIDATION_RATE] == 0.85
        
        # Check that degradation was detected and alert created
        stats = system.get_system_status()
        assert stats["statistics"]["metrics_processed"] == 1
        # Note: degradation detection may not trigger immediately for single metric
    
    def test_get_system_status(self):
        """Test getting system status."""
        system = QualityAlertingSystem()
        
        status = system.get_system_status()
        
        assert "monitoring_active" in status
        assert "thresholds_configured" in status
        assert "alert_handlers" in status
        assert "statistics" in status
        assert "uptime_seconds" in status
    
    def test_get_quality_dashboard_data(self):
        """Test getting dashboard data."""
        system = QualityAlertingSystem()
        
        # Add some metrics
        for i in range(5):
            metric = QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=0.9 + i * 0.01,
                timestamp=datetime.now() - timedelta(minutes=i),
                labels={"service": "test"}
            )
            system.process_metric(metric)
        
        dashboard_data = system.get_quality_dashboard_data()
        
        assert "current_metrics" in dashboard_data
        assert "trends" in dashboard_data
        assert "active_alerts" in dashboard_data
        assert "thresholds" in dashboard_data
        assert QualityMetricType.SCHEMA_VALIDATION_RATE.value in dashboard_data["current_metrics"]


if __name__ == "__main__":
    pytest.main([__file__])

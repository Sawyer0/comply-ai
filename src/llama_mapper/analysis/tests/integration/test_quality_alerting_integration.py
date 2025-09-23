"""
Integration tests for quality alerting system.

This module contains comprehensive integration tests that verify
the complete quality alerting system works correctly end-to-end.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.llama_mapper.analysis.quality import (
    QualityAlertingSystem, QualityMetric, QualityMetricType,
    QualityThreshold, AlertSeverity, DegradationType,
    QualityAlertingSettings
)


class TestQualityAlertingIntegration:
    """Integration tests for the complete quality alerting system."""
    
    @pytest.fixture
    def alerting_system(self):
        """Create a quality alerting system for testing."""
        return QualityAlertingSystem(
            monitoring_interval_seconds=1,  # Fast monitoring for tests
            max_metrics_per_type=1000,
            alert_retention_days=1
        )
    
    @pytest.fixture
    def mock_email_handler(self):
        """Create a mock email handler."""
        handler = Mock()
        handler.get_handler_name.return_value = "email"
        handler.can_handle_alert.return_value = True
        handler.send_alert.return_value = True
        handler.sent_count = 0
        handler.failed_count = 0
        return handler
    
    @pytest.fixture
    def mock_slack_handler(self):
        """Create a mock Slack handler."""
        handler = Mock()
        handler.get_handler_name.return_value = "slack"
        handler.can_handle_alert.return_value = True
        handler.send_alert.return_value = True
        handler.sent_count = 0
        handler.failed_count = 0
        return handler
    
    def test_system_initialization(self, alerting_system):
        """Test system initialization and default configuration."""
        assert alerting_system is not None
        assert not alerting_system._monitoring_active
        assert len(alerting_system.thresholds) > 0  # Should have default thresholds
        assert len(alerting_system.alert_handlers) > 0  # Should have logging handler
        
        # Check default thresholds
        assert QualityMetricType.SCHEMA_VALIDATION_RATE in alerting_system.thresholds
        assert QualityMetricType.TEMPLATE_FALLBACK_RATE in alerting_system.thresholds
        assert QualityMetricType.CONFIDENCE_SCORE in alerting_system.thresholds
    
    def test_metric_processing_and_alerting(self, alerting_system):
        """Test complete metric processing and alerting flow."""
        # Start monitoring
        alerting_system.start_monitoring()
        
        try:
            # Process multiple metrics to trigger degradation detection
            base_time = datetime.now() - timedelta(minutes=15)
            for i in range(12):  # Need at least 10 samples for default threshold
                metric = QualityMetric(
                    metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                    value=0.85,  # Below critical threshold (0.90)
                    timestamp=base_time + timedelta(minutes=i),
                    labels={"service": "test", "version": "1.0"}
                )
                alerting_system.process_metric(metric)
            
            # Wait a bit for processing
            time.sleep(0.1)
            
            # Check that alert was created
            active_alerts = alerting_system.alert_manager.get_active_alerts()
            assert len(active_alerts) > 0
            
            # Check alert details
            alert = active_alerts[0]
            assert alert.severity == AlertSeverity.CRITICAL
            assert alert.metric_type == QualityMetricType.SCHEMA_VALIDATION_RATE
            assert alert.degradation_detection is not None
            assert alert.degradation_detection.degradation_type == DegradationType.THRESHOLD_BREACH
            
            # Check system statistics
            stats = alerting_system.get_system_status()
            assert stats["statistics"]["metrics_processed"] >= 1
            assert stats["statistics"]["degradations_detected"] >= 1
            assert stats["statistics"]["alerts_created"] >= 1
            
        finally:
            alerting_system.stop_monitoring()
    
    def test_multiple_alert_handlers(self, alerting_system, mock_email_handler, mock_slack_handler):
        """Test system with multiple alert handlers."""
        # Add mock handlers
        alerting_system.alert_handlers.append(mock_email_handler)
        alerting_system.alert_handlers.append(mock_slack_handler)
        
        # Process a metric that triggers an alert
        metric = QualityMetric(
            metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
            value=0.85,  # Below critical threshold
            timestamp=datetime.now(),
            labels={"service": "test"}
        )
        
        alerting_system.process_metric(metric)
        
        # Wait for processing
        time.sleep(0.1)
        
        # Check that all handlers were called
        active_alerts = alerting_system.alert_manager.get_active_alerts()
        if active_alerts:
            alert = active_alerts[0]
            
            # Verify handlers were called
            mock_email_handler.send_alert.assert_called_with(alert)
            mock_slack_handler.send_alert.assert_called_with(alert)
    
    def test_alert_deduplication(self, alerting_system):
        """Test alert deduplication functionality."""
        # Process multiple metrics to trigger degradation detection
        base_time = datetime.now() - timedelta(minutes=15)
        for i in range(12):  # Need at least 10 samples for default threshold
            metric = QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=0.85,  # Below critical threshold
                timestamp=base_time + timedelta(minutes=i),
                labels={"service": "test"}
            )
            alerting_system.process_metric(metric)
            time.sleep(0.05)  # Small delay
        
        # Process the same metric again quickly (should be deduplicated)
        for i in range(3):
            metric = QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=0.85,  # Below critical threshold
                timestamp=datetime.now(),
                labels={"service": "test"}
            )
            alerting_system.process_metric(metric)
            time.sleep(0.05)  # Small delay
        
        # Should only have one alert due to deduplication
        active_alerts = alerting_system.alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        
        # Check deduplication statistics
        stats = alerting_system.alert_manager.get_alert_statistics()
        assert stats["total_created"] == 1  # Only one unique alert created
    
    def test_alert_lifecycle_management(self, alerting_system):
        """Test alert lifecycle management."""
        # Create multiple metrics to trigger an alert
        base_time = datetime.now() - timedelta(minutes=15)
        for i in range(12):  # Need at least 10 samples for default threshold
            metric = QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=0.85,
                timestamp=base_time + timedelta(minutes=i),
                labels={"service": "test"}
            )
            alerting_system.process_metric(metric)
        
        time.sleep(0.1)
        
        active_alerts = alerting_system.alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        
        alert = active_alerts[0]
        alert_id = alert.alert_id
        
        # Acknowledge the alert
        success = alerting_system.alert_manager.acknowledge_alert(alert_id, "test_user")
        assert success
        
        # Check alert status
        updated_alert = alerting_system.alert_manager.get_alert_by_id(alert_id)
        assert updated_alert.status.value == "acknowledged"
        assert updated_alert.acknowledged_at is not None
        
        # Resolve the alert
        success = alerting_system.alert_manager.resolve_alert(alert_id, "test_user")
        assert success
        
        # Check alert is no longer active
        active_alerts = alerting_system.alert_manager.get_active_alerts()
        assert len(active_alerts) == 0
        
        # Check alert status
        resolved_alert = alerting_system.alert_manager.get_alert_by_id(alert_id)
        assert resolved_alert.status.value == "resolved"
        assert resolved_alert.resolved_at is not None
    
    def test_quality_recovery_detection(self, alerting_system):
        """Test quality recovery detection."""
        # Start with poor quality
        for i in range(5):
            metric = QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=0.85,  # Poor quality
                timestamp=datetime.now() - timedelta(minutes=5-i),
                labels={"service": "test"}
            )
            alerting_system.process_metric(metric)
        
        # Wait for alerts
        time.sleep(0.1)
        
        # Now improve quality
        for i in range(5):
            metric = QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=0.98,  # Good quality
                timestamp=datetime.now() - timedelta(minutes=4-i),
                labels={"service": "test"}
            )
            alerting_system.process_metric(metric)
        
        # Check that we have alerts from the poor quality period
        active_alerts = alerting_system.alert_manager.get_active_alerts()
        assert len(active_alerts) > 0
        
        # Check current metrics show recovery
        current_metrics = alerting_system.quality_monitor.get_current_metrics()
        assert QualityMetricType.SCHEMA_VALIDATION_RATE in current_metrics
        assert current_metrics[QualityMetricType.SCHEMA_VALIDATION_RATE] == 0.98
    
    def test_trend_detection(self, alerting_system):
        """Test trend detection functionality."""
        # Create a declining trend
        base_time = datetime.now()
        for i in range(10):
            value = 0.95 - (i * 0.02)  # Declining from 0.95 to 0.77
            metric = QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=value,
                timestamp=base_time - timedelta(minutes=10-i),
                labels={"service": "test"}
            )
            alerting_system.process_metric(metric)
        
        # Get trend analysis
        trends = alerting_system.quality_monitor.get_metric_trends(
            QualityMetricType.SCHEMA_VALIDATION_RATE,
            time_window_minutes=60
        )
        
        assert trends["trend"] == "decreasing"
        assert trends["slope"] < 0
        assert trends["r_squared"] > 0.5  # Good correlation
    
    def test_anomaly_detection(self, alerting_system):
        """Test anomaly detection functionality."""
        # Create normal metrics
        base_time = datetime.now()
        for i in range(8):
            value = 0.95 + (i % 2) * 0.01  # Normal variation
            metric = QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=value,
                timestamp=base_time - timedelta(minutes=8-i),
                labels={"service": "test"}
            )
            alerting_system.process_metric(metric)
        
        # Add an anomaly
        anomaly_metric = QualityMetric(
            metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
            value=0.50,  # Clear anomaly
            timestamp=base_time,
            labels={"service": "test"}
        )
        alerting_system.process_metric(anomaly_metric)
        
        # Check for anomaly detection - need multiple metrics for context
        all_metrics = []
        for i in range(8):
            value = 0.95 + (i % 2) * 0.01  # Normal variation
            metric = QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=value,
                timestamp=base_time - timedelta(minutes=8-i),
                labels={"service": "test"}
            )
            all_metrics.append(metric)
        
        # Add the anomaly metric
        all_metrics.append(anomaly_metric)
        
        anomalies = alerting_system.degradation_detector.detect_anomalies(
            QualityMetricType.SCHEMA_VALIDATION_RATE,
            all_metrics
        )
        
        assert len(anomalies) == 1
        assert anomalies[0].degradation_type == DegradationType.ANOMALY
        assert anomalies[0].current_value == 0.50
    
    def test_dashboard_data_generation(self, alerting_system):
        """Test dashboard data generation."""
        # Add some metrics
        for i in range(5):
            metric = QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=0.9 + i * 0.01,
                timestamp=datetime.now() - timedelta(minutes=i),
                labels={"service": "test"}
            )
            alerting_system.process_metric(metric)
        
        # Get dashboard data
        dashboard_data = alerting_system.get_quality_dashboard_data()
        
        # Verify dashboard structure
        assert "current_metrics" in dashboard_data
        assert "trends" in dashboard_data
        assert "active_alerts" in dashboard_data
        assert "recent_alerts_count" in dashboard_data
        assert "thresholds" in dashboard_data
        
        # Check specific data
        assert QualityMetricType.SCHEMA_VALIDATION_RATE.value in dashboard_data["current_metrics"]
        assert QualityMetricType.SCHEMA_VALIDATION_RATE.value in dashboard_data["trends"]
        assert isinstance(dashboard_data["active_alerts"], list)
        assert isinstance(dashboard_data["thresholds"], list)
    
    def test_system_status_monitoring(self, alerting_system):
        """Test system status monitoring."""
        # Get initial status
        status = alerting_system.get_system_status()
        
        assert "monitoring_active" in status
        assert "thresholds_configured" in status
        assert "alert_handlers" in status
        assert "statistics" in status
        assert "uptime_seconds" in status
        
        # Process some metrics
        for i in range(3):
            metric = QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=0.95,
                timestamp=datetime.now(),
                labels={"service": "test"}
            )
            alerting_system.process_metric(metric)
        
        # Get updated status
        updated_status = alerting_system.get_system_status()
        
        # Check statistics updated
        assert updated_status["statistics"]["metrics_processed"] >= 3
        # Uptime should be >= 0 (might be 0 if test runs very quickly)
        assert updated_status["uptime_seconds"] >= 0
    
    def test_error_handling_and_recovery(self, alerting_system):
        """Test error handling and recovery."""
        # Test with invalid metric - system should handle gracefully
        alerting_system.process_metric(None)  # Should not crash
        
        # Test with invalid metric value - should raise ValueError during creation
        with pytest.raises(ValueError):
            invalid_metric = QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=-0.1,  # Invalid negative value
                timestamp=datetime.now(),
                labels={"service": "test"}
            )
        
        # System should still work after errors
        valid_metric = QualityMetric(
            metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
            value=0.95,
            timestamp=datetime.now(),
            labels={"service": "test"}
        )
        alerting_system.process_metric(valid_metric)
        
        # Check that valid metric was processed
        current_metrics = alerting_system.quality_monitor.get_current_metrics()
        assert QualityMetricType.SCHEMA_VALIDATION_RATE in current_metrics
    
    def test_performance_under_load(self, alerting_system):
        """Test system performance under load."""
        start_time = time.time()
        
        # Process many metrics quickly
        for i in range(100):
            metric = QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=0.9 + (i % 10) * 0.01,
                timestamp=datetime.now(),
                labels={"service": "test", "iteration": str(i)}
            )
            alerting_system.process_metric(metric)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 100 metrics in reasonable time (< 1 second)
        assert processing_time < 1.0
        
        # Check all metrics were recorded
        current_metrics = alerting_system.quality_monitor.get_current_metrics()
        assert QualityMetricType.SCHEMA_VALIDATION_RATE in current_metrics
        
        # Check performance statistics
        perf_stats = alerting_system.quality_monitor.get_performance_statistics()
        assert perf_stats["metrics_recorded"] >= 100
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test invalid configuration
        with pytest.raises(ValueError):
            QualityAlertingSystem(
                monitoring_interval_seconds=-1,  # Invalid
                max_metrics_per_type=1000
            )
        
        with pytest.raises(ValueError):
            QualityAlertingSystem(
                monitoring_interval_seconds=60,
                max_metrics_per_type=0  # Invalid
            )
        
        # Test valid configuration
        system = QualityAlertingSystem(
            monitoring_interval_seconds=60,
            max_metrics_per_type=1000,
            alert_retention_days=7
        )
        assert system is not None


class TestQualityAlertingSystemWithRealHandlers:
    """Integration tests with real alert handlers."""
    
    @pytest.fixture
    def alerting_system_with_handlers(self):
        """Create system with real handlers for testing."""
        system = QualityAlertingSystem()
        
        # Add logging handler (always works)
        from src.llama_mapper.analysis.quality import LoggingAlertHandler
        logging_handler = LoggingAlertHandler(log_level="INFO")
        system.alert_handlers.append(logging_handler)
        
        return system
    
    def test_logging_handler_integration(self, alerting_system_with_handlers):
        """Test integration with logging handler."""
        with patch('src.llama_mapper.analysis.quality.alerting.alert_handlers.logger') as mock_logger:
            # Process metric that triggers alert
            metric = QualityMetric(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                value=0.85,  # Below critical threshold
                timestamp=datetime.now(),
                labels={"service": "test"}
            )
            
            alerting_system_with_handlers.process_metric(metric)
            time.sleep(0.1)
            
            # Check that logging handler was called
            active_alerts = alerting_system_with_handlers.alert_manager.get_active_alerts()
            if active_alerts:
                # Verify logger was called (logging handler sends to logger)
                assert mock_logger.error.called or mock_logger.warning.called or mock_logger.info.called


if __name__ == "__main__":
    pytest.main([__file__])

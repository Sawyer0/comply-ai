"""
Test configuration and fixtures for quality alerting system tests.

This module provides shared test fixtures and configuration for all
quality alerting system tests.
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Generator
from unittest.mock import Mock, MagicMock

from src.llama_mapper.analysis.quality import (
    QualityAlertingSystem, QualityMetric, QualityMetricType,
    QualityThreshold, AlertSeverity, QualityAlertingSettings
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def sample_quality_metric():
    """Create a sample quality metric for testing."""
    return QualityMetric(
        metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
        value=0.95,
        timestamp=datetime.now(),
        labels={"service": "test", "version": "1.0"},
        metadata={"test": True}
    )


@pytest.fixture
def sample_quality_threshold():
    """Create a sample quality threshold for testing."""
    return QualityThreshold(
        metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
        warning_threshold=0.95,
        critical_threshold=0.90,
        min_samples=10,
        time_window_minutes=60,
        enabled=True
    )


@pytest.fixture
def quality_alerting_system():
    """Create a quality alerting system for testing."""
    return QualityAlertingSystem(
        monitoring_interval_seconds=1,  # Fast for tests
        max_metrics_per_type=1000,
        alert_retention_days=1
    )


@pytest.fixture
def quality_alerting_system_with_monitoring(quality_alerting_system):
    """Create a quality alerting system with monitoring started."""
    quality_alerting_system.start_monitoring()
    yield quality_alerting_system
    quality_alerting_system.stop_monitoring()


@pytest.fixture
def mock_email_handler():
    """Create a mock email alert handler."""
    handler = Mock()
    handler.get_handler_name.return_value = "email"
    handler.can_handle_alert.return_value = True
    handler.send_alert.return_value = True
    handler.sent_count = 0
    handler.failed_count = 0
    return handler


@pytest.fixture
def mock_slack_handler():
    """Create a mock Slack alert handler."""
    handler = Mock()
    handler.get_handler_name.return_value = "slack"
    handler.can_handle_alert.return_value = True
    handler.send_alert.return_value = True
    handler.sent_count = 0
    handler.failed_count = 0
    return handler


@pytest.fixture
def mock_webhook_handler():
    """Create a mock webhook alert handler."""
    handler = Mock()
    handler.get_handler_name.return_value = "webhook"
    handler.can_handle_alert.return_value = True
    handler.send_alert.return_value = True
    handler.sent_count = 0
    handler.failed_count = 0
    return handler


@pytest.fixture
def sample_metrics_batch():
    """Create a batch of sample metrics for testing."""
    base_time = datetime.now()
    metrics = []
    
    # Create metrics with different values and timestamps
    for i in range(20):
        metric = QualityMetric(
            metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
            value=0.9 + (i % 10) * 0.01,  # Values from 0.9 to 0.99
            timestamp=base_time + timedelta(minutes=i),
            labels={"service": "test", "iteration": str(i)}
        )
        metrics.append(metric)
    
    return metrics


@pytest.fixture
def declining_metrics_batch():
    """Create a batch of metrics showing a declining trend."""
    base_time = datetime.now()
    metrics = []
    
    # Create metrics with declining values
    for i in range(15):
        value = 0.95 - (i * 0.02)  # Declining from 0.95 to 0.65
        metric = QualityMetric(
            metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
            value=value,
            timestamp=base_time + timedelta(minutes=i),
            labels={"service": "test", "trend": "declining"}
        )
        metrics.append(metric)
    
    return metrics


@pytest.fixture
def anomaly_metrics_batch():
    """Create a batch of metrics with an anomaly."""
    base_time = datetime.now()
    metrics = []
    
    # Create normal metrics
    for i in range(10):
        value = 0.95 + (i % 2) * 0.01  # Normal variation around 0.95-0.96
        metric = QualityMetric(
            metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
            value=value,
            timestamp=base_time + timedelta(minutes=i),
            labels={"service": "test", "type": "normal"}
        )
        metrics.append(metric)
    
    # Add an anomaly
    anomaly_metric = QualityMetric(
        metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
        value=0.50,  # Clear anomaly
        timestamp=base_time + timedelta(minutes=10),
        labels={"service": "test", "type": "anomaly"}
    )
    metrics.append(anomaly_metric)
    
    return metrics


@pytest.fixture
def quality_alerting_settings():
    """Create quality alerting settings for testing."""
    return QualityAlertingSettings(
        monitoring_interval_seconds=30,
        max_metrics_per_type=5000,
        alert_retention_days=7,
        deduplication_window_minutes=15,
        
        # Email settings
        email_enabled=False,
        email_smtp_server="smtp.test.com",
        email_smtp_port=587,
        email_username="test@test.com",
        email_password="test_password",
        email_from="test@test.com",
        email_to=["admin@test.com"],
        
        # Slack settings
        slack_enabled=False,
        slack_webhook_url="https://hooks.slack.com/test",
        slack_channel="#test",
        
        # Webhook settings
        webhook_enabled=False,
        webhook_url="https://test.com/webhook",
        
        # Threshold settings
        schema_validation_warning=0.95,
        schema_validation_critical=0.90,
        template_fallback_warning=0.20,
        template_fallback_critical=0.30,
        confidence_score_warning=0.70,
        confidence_score_critical=0.60,
        response_time_warning=2.0,
        response_time_critical=5.0,
        error_rate_warning=0.05,
        error_rate_critical=0.10
    )


@pytest.fixture
def performance_test_config():
    """Configuration for performance tests."""
    return {
        "num_metrics": 1000,
        "num_threads": 10,
        "max_processing_time": 2.0,
        "min_metrics_per_second": 500,
        "max_memory_increase_mb": 100
    }


@pytest.fixture
def test_environment_variables():
    """Set up test environment variables."""
    env_vars = {
        "LLAMA_MAPPER_QUALITY_MONITORING_ENABLED": "true",
        "LLAMA_MAPPER_QUALITY_MAX_METRICS": "10000",
        "LLAMA_MAPPER_QUALITY_RETENTION_HOURS": "24",
        "LLAMA_MAPPER_QUALITY_CLEANUP_INTERVAL": "60"
    }
    
    # Store original values
    original_values = {}
    for key, value in env_vars.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield env_vars
    
    # Restore original values
    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file paths."""
    for item in items:
        # Add markers based on file path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        
        # Add slow marker for tests that might take time
        if "performance" in item.name or "load" in item.name:
            item.add_marker(pytest.mark.slow)


# Test utilities
class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def generate_metrics(
        metric_type: QualityMetricType,
        count: int,
        start_time: datetime,
        value_range: tuple = (0.9, 1.0),
        time_interval_minutes: int = 1
    ) -> list:
        """Generate a list of test metrics."""
        metrics = []
        for i in range(count):
            value = value_range[0] + (i % (int((value_range[1] - value_range[0]) * 100))) * 0.01
            metric = QualityMetric(
                metric_type=metric_type,
                value=value,
                timestamp=start_time + timedelta(minutes=i * time_interval_minutes),
                labels={"service": "test", "generated": "true", "index": str(i)}
            )
            metrics.append(metric)
        return metrics
    
    @staticmethod
    def generate_thresholds() -> list:
        """Generate a list of test thresholds."""
        thresholds = []
        for metric_type in QualityMetricType:
            if metric_type != QualityMetricType.CUSTOM_METRIC:
                threshold = QualityThreshold(
                    metric_type=metric_type,
                    warning_threshold=0.95,
                    critical_threshold=0.90,
                    min_samples=10,
                    time_window_minutes=60
                )
                thresholds.append(threshold)
        return thresholds


@pytest.fixture
def test_data_generator():
    """Provide test data generator utility."""
    return TestDataGenerator


# Mock utilities
class MockAlertHandler:
    """Mock alert handler for testing."""
    
    def __init__(self, name: str = "mock", should_succeed: bool = True):
        self.name = name
        self.should_succeed = should_succeed
        self.sent_count = 0
        self.failed_count = 0
        self.sent_alerts = []
    
    def get_handler_name(self) -> str:
        return self.name
    
    def can_handle_alert(self, alert) -> bool:
        return True
    
    def send_alert(self, alert) -> bool:
        self.sent_alerts.append(alert)
        if self.should_succeed:
            self.sent_count += 1
            return True
        else:
            self.failed_count += 1
            return False


@pytest.fixture
def mock_alert_handler():
    """Create a mock alert handler."""
    return MockAlertHandler()


@pytest.fixture
def failing_alert_handler():
    """Create a failing alert handler."""
    return MockAlertHandler(name="failing", should_succeed=False)
"""
Unit tests for evaluation configuration system.

Tests configuration validation, loading from environment/files,
and configuration management functionality.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.llama_mapper.analysis.config.evaluation_config import (
    EvaluationThresholds,
    NotificationConfig,
    ReportConfig,
    StorageConfig,
    WeeklyEvaluationConfig,
    get_evaluation_config,
    validate_evaluation_config,
)


class TestEvaluationThresholds:
    """Test cases for EvaluationThresholds."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = EvaluationThresholds()

        assert thresholds.schema_valid_rate == 0.98
        assert thresholds.rubric_score == 0.8
        assert thresholds.opa_compile_success_rate == 0.95
        assert thresholds.evidence_accuracy == 0.85

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        thresholds = EvaluationThresholds(
            schema_valid_rate=0.99,
            rubric_score=0.85,
            opa_compile_success_rate=0.97,
            evidence_accuracy=0.90,
        )

        assert thresholds.schema_valid_rate == 0.99
        assert thresholds.rubric_score == 0.85
        assert thresholds.opa_compile_success_rate == 0.97
        assert thresholds.evidence_accuracy == 0.90

    def test_invalid_thresholds(self):
        """Test invalid threshold values."""
        with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
            EvaluationThresholds(schema_valid_rate=1.5)

        with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
            EvaluationThresholds(rubric_score=-0.1)

        with pytest.raises(ValueError, match="Threshold must be a number"):
            EvaluationThresholds(schema_valid_rate="invalid")


class TestNotificationConfig:
    """Test cases for NotificationConfig."""

    def test_default_notifications(self):
        """Test default notification settings."""
        config = NotificationConfig()

        assert config.enabled is True
        assert config.email_recipients == []
        assert config.slack_webhook_url is None
        assert config.webhook_urls == []

    def test_valid_email_recipients(self):
        """Test valid email recipients."""
        config = NotificationConfig(
            email_recipients=["admin@example.com", "team@example.com"]
        )

        assert config.email_recipients == ["admin@example.com", "team@example.com"]

    def test_invalid_email_recipients(self):
        """Test invalid email recipients."""
        with pytest.raises(ValueError, match="Invalid email address"):
            NotificationConfig(email_recipients=["invalid-email"])

        with pytest.raises(ValueError, match="Invalid email address"):
            NotificationConfig(email_recipients=["admin@example.com", "invalid-email"])


class TestReportConfig:
    """Test cases for ReportConfig."""

    def test_default_report_config(self):
        """Test default report configuration."""
        config = ReportConfig()

        assert config.formats == ["PDF"]
        assert config.include_detailed_metrics is True
        assert config.include_individual_scores is True
        assert config.include_recommendations is True
        assert config.template_dir is None

    def test_valid_formats(self):
        """Test valid report formats."""
        config = ReportConfig(formats=["PDF", "CSV", "JSON"])

        assert config.formats == ["PDF", "CSV", "JSON"]

    def test_invalid_formats(self):
        """Test invalid report formats."""
        with pytest.raises(ValueError, match="Unsupported report format"):
            ReportConfig(formats=["PDF", "INVALID"])


class TestStorageConfig:
    """Test cases for StorageConfig."""

    def test_default_storage_config(self):
        """Test default storage configuration."""
        config = StorageConfig()

        assert config.backend_type == "file"
        assert config.storage_dir is None
        assert config.database_url is None
        assert config.s3_bucket is None
        assert config.retention_days == 90

    def test_file_storage_config(self):
        """Test file storage configuration."""
        config = StorageConfig(backend_type="file", storage_dir="/tmp/evaluations")

        assert config.backend_type == "file"
        assert config.storage_dir == "/tmp/evaluations"

    def test_database_storage_config(self):
        """Test database storage configuration."""
        config = StorageConfig(
            backend_type="database", database_url="postgresql://user:pass@localhost/db"
        )

        assert config.backend_type == "database"
        assert config.database_url == "postgresql://user:pass@localhost/db"

    def test_s3_storage_config(self):
        """Test S3 storage configuration."""
        config = StorageConfig(backend_type="s3", s3_bucket="my-bucket")

        assert config.backend_type == "s3"
        assert config.s3_bucket == "my-bucket"

    def test_invalid_backend_type(self):
        """Test invalid backend type."""
        with pytest.raises(ValueError, match="Unsupported storage backend"):
            StorageConfig(backend_type="invalid")

    def test_missing_database_url(self):
        """Test missing database URL for database backend."""
        with pytest.raises(ValueError, match="database_url is required"):
            StorageConfig(backend_type="database")

    def test_missing_s3_bucket(self):
        """Test missing S3 bucket for S3 backend."""
        with pytest.raises(ValueError, match="s3_bucket is required"):
            StorageConfig(backend_type="s3")


class TestWeeklyEvaluationConfig:
    """Test cases for WeeklyEvaluationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = WeeklyEvaluationConfig()

        assert config.enabled is True
        assert config.default_schedule == "0 9 * * 1"
        assert config.evaluation_period_days == 7
        assert config.max_concurrent_evaluations == 5
        assert config.timeout_minutes == 30

        # Check sub-configurations
        assert isinstance(config.thresholds, EvaluationThresholds)
        assert isinstance(config.notifications, NotificationConfig)
        assert isinstance(config.reports, ReportConfig)
        assert isinstance(config.storage, StorageConfig)

    def test_custom_config(self):
        """Test custom configuration."""
        config = WeeklyEvaluationConfig(
            enabled=False,
            default_schedule="0 10 * * 2",
            evaluation_period_days=14,
            max_concurrent_evaluations=10,
            timeout_minutes=60,
            thresholds=EvaluationThresholds(schema_valid_rate=0.99),
            notifications=NotificationConfig(email_recipients=["admin@example.com"]),
            reports=ReportConfig(formats=["PDF", "JSON"]),
            storage=StorageConfig(
                backend_type="database", database_url="postgresql://localhost/db"
            ),
        )

        assert config.enabled is False
        assert config.default_schedule == "0 10 * * 2"
        assert config.evaluation_period_days == 14
        assert config.max_concurrent_evaluations == 10
        assert config.timeout_minutes == 60
        assert config.thresholds.schema_valid_rate == 0.99
        assert config.notifications.email_recipients == ["admin@example.com"]
        assert config.reports.formats == ["PDF", "JSON"]
        assert config.storage.backend_type == "database"

    def test_invalid_cron_schedule(self):
        """Test invalid cron schedule."""
        with pytest.raises(ValueError, match="Invalid cron schedule"):
            WeeklyEvaluationConfig(default_schedule="invalid-cron")

    def test_validation_issues(self):
        """Test configuration validation."""
        config = WeeklyEvaluationConfig(
            enabled=True,
            notifications=NotificationConfig(enabled=True),  # No recipients
        )

        issues = config.validate_configuration()
        assert len(issues) > 0
        assert any("no recipients configured" in issue for issue in issues)

    def test_get_storage_backend_config(self):
        """Test getting storage backend configuration."""
        config = WeeklyEvaluationConfig(
            storage=StorageConfig(
                backend_type="s3",
                s3_bucket="my-bucket",
                database_url="postgresql://localhost/db",
            )
        )

        backend_config = config.get_storage_backend_config()

        assert backend_config["backend_type"] == "s3"
        assert backend_config["s3_bucket"] == "my-bucket"
        assert backend_config["database_url"] == "postgresql://localhost/db"


class TestConfigurationLoading:
    """Test cases for configuration loading."""

    def test_from_environment(self):
        """Test loading configuration from environment variables."""
        env_vars = {
            "LLAMA_MAPPER_WEEKLY_EVALUATIONS_ENABLED": "true",
            "LLAMA_MAPPER_DEFAULT_WEEKLY_SCHEDULE": "0 10 * * 2",
            "LLAMA_MAPPER_SCHEMA_VALID_THRESHOLD": "0.99",
            "LLAMA_MAPPER_NOTIFICATION_EMAIL": "admin@example.com,team@example.com",
            "LLAMA_MAPPER_STORAGE_BACKEND": "file",
            "LLAMA_MAPPER_STORAGE_DIR": "/tmp/evaluations",
        }

        with patch.dict(os.environ, env_vars):
            config = WeeklyEvaluationConfig.from_environment()

            assert config.enabled is True
            assert config.default_schedule == "0 10 * * 2"
            assert config.thresholds.schema_valid_rate == 0.99
            assert config.notifications.email_recipients == [
                "admin@example.com",
                "team@example.com",
            ]
            assert config.storage.backend_type == "file"
            assert config.storage.storage_dir == "/tmp/evaluations"

    def test_from_json_file(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "enabled": True,
            "default_schedule": "0 10 * * 2",
            "evaluation_period_days": 14,
            "thresholds": {"schema_valid_rate": 0.99, "rubric_score": 0.85},
            "notifications": {
                "enabled": True,
                "email_recipients": ["admin@example.com"],
            },
            "storage": {"backend_type": "file", "storage_dir": "/tmp/evaluations"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            config = WeeklyEvaluationConfig.from_file(config_file)

            assert config.enabled is True
            assert config.default_schedule == "0 10 * * 2"
            assert config.evaluation_period_days == 14
            assert config.thresholds.schema_valid_rate == 0.99
            assert config.thresholds.rubric_score == 0.85
            assert config.notifications.email_recipients == ["admin@example.com"]
            assert config.storage.backend_type == "file"
            assert config.storage.storage_dir == "/tmp/evaluations"

        finally:
            os.unlink(config_file)

    def test_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
enabled: true
default_schedule: "0 10 * * 2"
evaluation_period_days: 14
thresholds:
  schema_valid_rate: 0.99
  rubric_score: 0.85
notifications:
  enabled: true
  email_recipients:
    - admin@example.com
storage:
  backend_type: file
  storage_dir: /tmp/evaluations
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            config_file = f.name

        try:
            config = WeeklyEvaluationConfig.from_file(config_file)

            assert config.enabled is True
            assert config.default_schedule == "0 10 * * 2"
            assert config.evaluation_period_days == 14
            assert config.thresholds.schema_valid_rate == 0.99
            assert config.thresholds.rubric_score == 0.85
            assert config.notifications.email_recipients == ["admin@example.com"]
            assert config.storage.backend_type == "file"
            assert config.storage.storage_dir == "/tmp/evaluations"

        finally:
            os.unlink(config_file)

    def test_file_not_found(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            WeeklyEvaluationConfig.from_file("nonexistent.json")

    def test_invalid_file_format(self):
        """Test loading from file with invalid format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("invalid content")
            config_file = f.name

        try:
            with pytest.raises(
                ValueError, match="Unsupported configuration file format"
            ):
                WeeklyEvaluationConfig.from_file(config_file)

        finally:
            os.unlink(config_file)


class TestConfigurationSaving:
    """Test cases for configuration saving."""

    def test_save_to_json_file(self):
        """Test saving configuration to JSON file."""
        config = WeeklyEvaluationConfig(
            enabled=True,
            default_schedule="0 10 * * 2",
            thresholds=EvaluationThresholds(schema_valid_rate=0.99),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.json"
            config.to_file(config_file, format="json")

            assert config_file.exists()

            # Verify content
            with open(config_file, "r") as f:
                saved_data = json.load(f)

            assert saved_data["enabled"] is True
            assert saved_data["default_schedule"] == "0 10 * * 2"
            assert saved_data["thresholds"]["schema_valid_rate"] == 0.99

    def test_save_to_yaml_file(self):
        """Test saving configuration to YAML file."""
        config = WeeklyEvaluationConfig(enabled=True, default_schedule="0 10 * * 2")

        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.yaml"
            config.to_file(config_file, format="yaml")

            assert config_file.exists()

    def test_invalid_save_format(self):
        """Test saving with invalid format."""
        config = WeeklyEvaluationConfig()

        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.txt"

            with pytest.raises(ValueError, match="Unsupported output format"):
                config.to_file(config_file, format="txt")


class TestConfigurationValidation:
    """Test cases for configuration validation."""

    def test_valid_configuration(self):
        """Test validation of valid configuration."""
        config = WeeklyEvaluationConfig(
            enabled=True,
            notifications=NotificationConfig(
                enabled=True, email_recipients=["admin@example.com"]
            ),
            storage=StorageConfig(backend_type="file", storage_dir="/tmp/evaluations"),
        )

        assert validate_evaluation_config(config) is True

    def test_invalid_configuration(self):
        """Test validation of invalid configuration."""
        config = WeeklyEvaluationConfig(
            enabled=True,
            notifications=NotificationConfig(enabled=True),  # No recipients
            storage=StorageConfig(backend_type="file"),  # No storage_dir
        )

        with pytest.raises(ValueError, match="Configuration validation failed"):
            validate_evaluation_config(config)

    def test_get_evaluation_config_from_file(self):
        """Test get_evaluation_config with file path."""
        config_data = {"enabled": True, "default_schedule": "0 10 * * 2"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            config = get_evaluation_config(config_file)
            assert config.enabled is True
            assert config.default_schedule == "0 10 * * 2"

        finally:
            os.unlink(config_file)

    def test_get_evaluation_config_from_environment(self):
        """Test get_evaluation_config from environment."""
        env_vars = {
            "LLAMA_MAPPER_WEEKLY_EVALUATIONS_ENABLED": "false",
            "LLAMA_MAPPER_DEFAULT_WEEKLY_SCHEDULE": "0 8 * * 1",
        }

        with patch.dict(os.environ, env_vars):
            config = get_evaluation_config()
            assert config.enabled is False
            assert config.default_schedule == "0 8 * * 1"

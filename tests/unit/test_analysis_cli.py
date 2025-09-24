"""Tests for analysis CLI commands."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from src.llama_mapper.analysis import AnalysisResponse, HealthStatus
from src.llama_mapper.cli.main import main


class TestAnalysisCLI:
    """Test cases for analysis CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.sample_metrics = {
            "tenant_id": "test-tenant",
            "timestamp": "2024-01-15T10:30:00Z",
            "detector_results": [
                {
                    "detector_name": "security-scan",
                    "detector_version": "1.2.3",
                    "scan_type": "vulnerability",
                    "results": [
                        {
                            "finding_id": "CVE-2024-1234",
                            "severity": "high",
                            "category": "vulnerability",
                            "description": "SQL injection vulnerability",
                            "affected_component": "auth-service",
                            "confidence": 0.95,
                        }
                    ],
                }
            ],
        }

    def test_analysis_help(self):
        """Test that analysis command shows help."""
        result = self.runner.invoke(main, ["analysis", "--help"])
        assert result.exit_code == 0
        assert "Analysis module commands" in result.output

    def test_analyze_command_help(self):
        """Test that analyze command shows help."""
        result = self.runner.invoke(main, ["analysis", "analyze", "--help"])
        assert result.exit_code == 0
        assert "Analyze security metrics" in result.output

    def test_batch_analyze_command_help(self):
        """Test that batch-analyze command shows help."""
        result = self.runner.invoke(main, ["analysis", "batch-analyze", "--help"])
        assert result.exit_code == 0
        assert "Perform batch analysis" in result.output

    def test_health_command_help(self):
        """Test that health command shows help."""
        result = self.runner.invoke(main, ["analysis", "health", "--help"])
        assert result.exit_code == 0
        assert "Check analysis module health" in result.output

    def test_quality_eval_command_help(self):
        """Test that quality-eval command shows help."""
        result = self.runner.invoke(main, ["analysis", "quality-eval", "--help"])
        assert result.exit_code == 0
        assert "Evaluate analysis quality" in result.output

    def test_cache_command_help(self):
        """Test that cache command shows help."""
        result = self.runner.invoke(main, ["analysis", "cache", "--help"])
        assert result.exit_code == 0
        assert "Manage analysis module cache" in result.output

    def test_validate_config_command_help(self):
        """Test that validate-config command shows help."""
        result = self.runner.invoke(main, ["analysis", "validate-config", "--help"])
        assert result.exit_code == 0
        assert "Validate analysis module configuration" in result.output

    @patch("src.llama_mapper.cli.commands.analysis.AnalysisModuleFactory")
    @patch("src.llama_mapper.cli.commands.analysis.AnalyzeMetricsUseCase")
    def test_analyze_command_success(self, mock_use_case, mock_factory):
        """Test successful analysis command execution."""
        # Create temporary metrics file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.sample_metrics, f)
            metrics_file = f.name

        try:
            # Mock the analysis response
            from src.llama_mapper.analysis.domain.entities import VersionInfo

            mock_response = AnalysisResponse(
                reason="Test explanation",
                remediation="Test remediation",
                opa_diff="",
                confidence=0.95,
                confidence_cutoff_used=0.3,
                evidence_refs=["detector-1", "detector-2"],
                notes="Test notes",
                version_info=VersionInfo(
                    taxonomy="1.0.0", frameworks="1.0.0", analyst_model="phi3-mini-3.8b"
                ),
                processing_time_ms=150,
            )

            # Mock the use case execution
            mock_use_case_instance = AsyncMock()
            mock_use_case_instance.execute.return_value = mock_response
            mock_use_case.return_value = mock_use_case_instance

            # Mock the factory
            mock_factory_instance = MagicMock()
            mock_factory_instance.get_component.return_value = mock_use_case_instance
            mock_factory.create_from_config.return_value = mock_factory_instance

            # Run the command
            result = self.runner.invoke(
                main,
                [
                    "analysis",
                    "analyze",
                    "--metrics-file",
                    metrics_file,
                    "--format",
                    "json",
                ],
            )

            assert result.exit_code == 0
            assert "Test explanation" in result.output
            assert "Test remediation" in result.output

        finally:
            # Clean up
            Path(metrics_file).unlink()

    @patch("src.llama_mapper.cli.commands.analysis.AnalysisModuleFactory")
    @patch("src.llama_mapper.cli.commands.analysis.HealthCheckUseCase")
    def test_health_command_success(self, mock_use_case, mock_factory):
        """Test successful health command execution."""
        # Mock the health status
        mock_health_status = HealthStatus(
            status="healthy",
            service="analysis-module",
            version="1.0.0",
            checks={"model_server": "healthy", "validator": "healthy"},
        )

        # Mock the use case execution
        mock_use_case_instance = AsyncMock()
        mock_use_case_instance.execute.return_value = mock_health_status
        mock_use_case.return_value = mock_use_case_instance

        # Mock the factory
        mock_factory_instance = MagicMock()
        mock_factory_instance.get_component.return_value = mock_use_case_instance
        mock_factory.create_from_config.return_value = mock_factory_instance

        # Run the command
        result = self.runner.invoke(main, ["analysis", "health"])

        assert result.exit_code == 0
        assert "Status: healthy" in result.output
        assert "Version: 1.0.0" in result.output

    @patch("src.llama_mapper.cli.commands.analysis.AnalysisModuleFactory")
    @patch("src.llama_mapper.cli.commands.analysis.QualityEvaluationUseCase")
    def test_quality_eval_command_success(self, mock_use_case, mock_factory):
        """Test successful quality evaluation command execution."""
        # Mock the quality evaluation result
        mock_quality_result = MagicMock()
        mock_quality_result.overall_score = 8.5
        mock_quality_result.confidence = 0.9
        mock_quality_result.metrics = {"accuracy": 0.9, "completeness": 0.85}
        mock_quality_result.recommendations = [
            "Improve accuracy",
            "Add more test cases",
        ]
        mock_quality_result.model_dump.return_value = {
            "overall_score": 8.5,
            "confidence": 0.9,
            "metrics": {"accuracy": 0.9, "completeness": 0.85},
            "recommendations": ["Improve accuracy", "Add more test cases"],
        }

        # Mock the use case execution
        mock_use_case_instance = AsyncMock()
        mock_use_case_instance.execute.return_value = mock_quality_result
        mock_use_case.return_value = mock_use_case_instance

        # Mock the factory
        mock_factory_instance = MagicMock()
        mock_factory_instance.get_component.return_value = mock_use_case_instance
        mock_factory.create_from_config.return_value = mock_factory_instance

        # Run the command
        result = self.runner.invoke(main, ["analysis", "quality-eval"])

        assert result.exit_code == 0
        assert "Overall Score: 8.50/10" in result.output
        assert "Confidence: 0.90" in result.output

    @patch("src.llama_mapper.cli.commands.analysis.AnalysisModuleFactory")
    @patch("src.llama_mapper.cli.commands.analysis.CacheManagementUseCase")
    def test_cache_command_stats(self, mock_use_case, mock_factory):
        """Test cache stats command execution."""
        # Mock the cache stats result
        mock_cache_stats = MagicMock()
        mock_cache_stats.total_entries = 100
        mock_cache_stats.memory_usage_mb = 25.5
        mock_cache_stats.hit_rate = 85.2

        # Mock the use case execution
        mock_use_case_instance = AsyncMock()
        mock_use_case_instance.get_cache_stats.return_value = mock_cache_stats
        mock_use_case.return_value = mock_use_case_instance

        # Mock the factory
        mock_factory_instance = MagicMock()
        mock_factory_instance.get_component.return_value = mock_use_case_instance
        mock_factory.create_from_config.return_value = mock_factory_instance

        # Run the command
        result = self.runner.invoke(main, ["analysis", "cache", "--action", "stats"])

        assert result.exit_code == 0
        assert "Total entries: 100" in result.output
        assert "Memory usage: 25.50 MB" in result.output
        assert "Hit rate: 85.20%" in result.output

    def test_validate_config_command_success(self):
        """Test successful configuration validation."""
        # Run the command
        result = self.runner.invoke(main, ["analysis", "validate-config"])

        # Should pass with default configuration
        assert result.exit_code == 0
        assert "Analysis Module Configuration Validation" in result.output

    def test_analyze_command_missing_file(self):
        """Test analyze command with missing metrics file."""
        result = self.runner.invoke(
            main, ["analysis", "analyze", "--metrics-file", "nonexistent.json"]
        )

        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_analyze_command_invalid_json(self):
        """Test analyze command with invalid JSON file."""
        # Create temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            invalid_file = f.name

        try:
            result = self.runner.invoke(
                main, ["analysis", "analyze", "--metrics-file", invalid_file]
            )

            assert result.exit_code != 0
            assert "Expecting value" in result.output

        finally:
            # Clean up
            Path(invalid_file).unlink()

    @patch("src.llama_mapper.cli.commands.analysis.AnalysisModuleFactory")
    def test_analyze_command_factory_error(self, mock_factory):
        """Test analyze command with factory initialization error."""
        # Create temporary metrics file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.sample_metrics, f)
            metrics_file = f.name

        try:
            # Mock factory to raise an exception
            mock_factory.create_from_config.side_effect = Exception(
                "Factory initialization failed"
            )

            # Run the command
            result = self.runner.invoke(
                main, ["analysis", "analyze", "--metrics-file", metrics_file]
            )

            assert result.exit_code != 0
            assert "Analysis failed" in result.output

        finally:
            # Clean up
            Path(metrics_file).unlink()

    def test_batch_analyze_command_missing_file(self):
        """Test batch analyze command with missing batch file."""
        result = self.runner.invoke(
            main, ["analysis", "batch-analyze", "--batch-file", "nonexistent.json"]
        )

        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_cache_command_invalid_action(self):
        """Test cache command with invalid action."""
        result = self.runner.invoke(
            main, ["analysis", "cache", "--action", "invalid_action"]
        )

        # Should show help or error message
        assert result.exit_code != 0 or "Usage:" in result.output

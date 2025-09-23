"""Simple tests for analysis CLI commands that don't require model loading."""

from click.testing import CliRunner
import pytest

from src.llama_mapper.cli.main import main


class TestAnalysisCLISimple:
    """Simple test cases for analysis CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

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

    def test_analyze_command_missing_file(self):
        """Test analyze command with missing metrics file."""
        result = self.runner.invoke(main, [
            "analysis", "analyze",
            "--metrics-file", "nonexistent.json"
        ])

        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_batch_analyze_command_missing_file(self):
        """Test batch analyze command with missing batch file."""
        result = self.runner.invoke(main, [
            "analysis", "batch-analyze",
            "--batch-file", "nonexistent.json"
        ])

        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_cache_command_invalid_action(self):
        """Test cache command with invalid action."""
        result = self.runner.invoke(main, [
            "analysis", "cache",
            "--action", "invalid_action"
        ])

        # Should show help or error message
        assert result.exit_code != 0 or "Usage:" in result.output

    def test_analysis_commands_registered(self):
        """Test that all analysis commands are properly registered."""
        result = self.runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "analysis" in result.output

        # Test that analysis subcommands are available
        result = self.runner.invoke(main, ["analysis", "--help"])
        assert result.exit_code == 0
        assert "analyze" in result.output
        assert "batch-analyze" in result.output
        assert "health" in result.output
        assert "quality-eval" in result.output
        assert "cache" in result.output
        assert "validate-config" in result.output

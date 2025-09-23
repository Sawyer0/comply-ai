"""Integration tests for analysis CLI commands."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from src.llama_mapper.cli.main import main


class TestAnalysisCLIIntegration:
    """Integration tests for analysis CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

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

    def test_analyze_command_with_sample_data(self):
        """Test analyze command with sample metrics data."""
        # Create sample metrics file
        sample_metrics = {
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
                            "confidence": 0.95
                        }
                    ]
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_metrics, f)
            metrics_file = f.name

        try:
            # Mock the analysis module to avoid actual model loading
            with patch('src.llama_mapper.cli.commands.analysis.AnalysisModuleFactory') as mock_factory:
                with patch('src.llama_mapper.cli.commands.analysis.AnalyzeMetricsUseCase') as mock_use_case:
                    # Mock successful analysis response
                    mock_response = type('MockResponse', (), {
                        'analysis_type': 'comprehensive',
                        'confidence': 0.95,
                        'processing_time_ms': 150,
                        'explanation': 'Test explanation',
                        'remediation': 'Test remediation',
                        'policy_recommendations': ['Test policy'],
                        'quality_metrics': {'accuracy': 0.9},
                        'model_dump': lambda: {
                            'analysis_type': 'comprehensive',
                            'confidence': 0.95,
                            'processing_time_ms': 150,
                            'explanation': 'Test explanation',
                            'remediation': 'Test remediation',
                            'policy_recommendations': ['Test policy'],
                            'quality_metrics': {'accuracy': 0.9}
                        }
                    })()

                    mock_use_case_instance = type('MockUseCase', (), {
                        'execute': lambda x: mock_response
                    })()
                    mock_use_case.return_value = mock_use_case_instance

                    mock_factory_instance = type('MockFactory', (), {
                        'get_component': lambda x: mock_use_case_instance
                    })()
                    mock_factory.create_from_config.return_value = mock_factory_instance

                    # Run the command
                    result = self.runner.invoke(main, [
                        "analysis", "analyze",
                        "--metrics-file", metrics_file,
                        "--format", "json"
                    ])

                    assert result.exit_code == 0
                    assert "Test explanation" in result.output

        finally:
            # Clean up
            Path(metrics_file).unlink()

    def test_batch_analyze_command_with_sample_data(self):
        """Test batch analyze command with sample batch data."""
        # Create sample batch file
        sample_batch = {
            "requests": [
                {
                    "metrics": {
                        "tenant_id": "tenant-1",
                        "detector_results": [
                            {
                                "detector_name": "security-scan",
                                "results": [
                                    {
                                        "finding_id": "CVE-2024-1234",
                                        "severity": "high",
                                        "description": "SQL injection vulnerability"
                                    }
                                ]
                            }
                        ]
                    },
                    "analysis_type": "remediation",
                    "tenant_id": "tenant-1"
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_batch, f)
            batch_file = f.name

        try:
            # Mock the analysis module
            with patch('src.llama_mapper.cli.commands.analysis.AnalysisModuleFactory') as mock_factory:
                with patch('src.llama_mapper.cli.commands.analysis.BatchAnalyzeMetricsUseCase') as mock_use_case:
                    # Mock successful batch analysis response
                    mock_response = type('MockResponse', (), {
                        'model_dump': lambda: {
                            'results': [
                                {
                                    'request_id': 'req-001',
                                    'status': 'success',
                                    'analysis': {
                                        'explanation': 'Batch test explanation',
                                        'remediation': 'Batch test remediation'
                                    }
                                }
                            ],
                            'summary': {
                                'total_requests': 1,
                                'successful': 1,
                                'failed': 0
                            }
                        }
                    })()

                    mock_use_case_instance = type('MockUseCase', (), {
                        'execute': lambda x: mock_response
                    })()
                    mock_use_case.return_value = mock_use_case_instance

                    mock_factory_instance = type('MockFactory', (), {
                        'get_component': lambda x: mock_use_case_instance
                    })()
                    mock_factory.create_from_config.return_value = mock_factory_instance

                    # Run the command
                    result = self.runner.invoke(main, [
                        "analysis", "batch-analyze",
                        "--batch-file", batch_file,
                        "--format", "json"
                    ])

                    assert result.exit_code == 0
                    assert "Batch test explanation" in result.output

        finally:
            # Clean up
            Path(batch_file).unlink()

    def test_health_command_integration(self):
        """Test health command integration."""
        # Mock the analysis module
        with patch('src.llama_mapper.cli.commands.analysis.AnalysisModuleFactory') as mock_factory:
            with patch('src.llama_mapper.cli.commands.analysis.HealthCheckUseCase') as mock_use_case:
                # Mock health status
                mock_health_status = type('MockHealthStatus', (), {
                    'status': type('MockStatus', (), {'value': 'healthy'})(),
                    'version': '1.0.0',
                    'uptime_seconds': 3600,
                    'components': {'model_server': 'healthy'},
                    'issues': []
                })()

                mock_use_case_instance = type('MockUseCase', (), {
                    'execute': lambda: mock_health_status
                })()
                mock_use_case.return_value = mock_use_case_instance

                mock_factory_instance = type('MockFactory', (), {
                    'get_component': lambda x: mock_use_case_instance
                })()
                mock_factory.create_from_config.return_value = mock_factory_instance

                # Run the command
                result = self.runner.invoke(main, ["analysis", "health"])

                assert result.exit_code == 0
                assert "Status: healthy" in result.output
                assert "Version: 1.0.0" in result.output

    def test_validate_config_command_integration(self):
        """Test configuration validation command integration."""
        # Run the command
        result = self.runner.invoke(main, ["analysis", "validate-config"])

        # Should pass with default configuration
        assert result.exit_code == 0
        assert "Analysis Module Configuration Validation" in result.output

    def test_cache_command_integration(self):
        """Test cache command integration."""
        # Mock the analysis module
        with patch('src.llama_mapper.cli.commands.analysis.AnalysisModuleFactory') as mock_factory:
            with patch('src.llama_mapper.cli.commands.analysis.CacheManagementUseCase') as mock_use_case:
                # Mock cache stats
                mock_cache_stats = type('MockCacheStats', (), {
                    'total_entries': 50,
                    'memory_usage_mb': 12.5,
                    'hit_rate': 78.5
                })()

                mock_use_case_instance = type('MockUseCase', (), {
                    'get_cache_stats': lambda: mock_cache_stats
                })()
                mock_use_case.return_value = mock_use_case_instance

                mock_factory_instance = type('MockFactory', (), {
                    'get_component': lambda x: mock_use_case_instance
                })()
                mock_factory.create_from_config.return_value = mock_factory_instance

                # Run the command
                result = self.runner.invoke(main, ["analysis", "cache", "--action", "stats"])

                assert result.exit_code == 0
                assert "Total entries: 50" in result.output
                assert "Memory usage: 12.50 MB" in result.output
                assert "Hit rate: 78.50%" in result.output

    def test_quality_eval_command_integration(self):
        """Test quality evaluation command integration."""
        # Mock the analysis module
        with patch('src.llama_mapper.cli.commands.analysis.AnalysisModuleFactory') as mock_factory:
            with patch('src.llama_mapper.cli.commands.analysis.QualityEvaluationUseCase') as mock_use_case:
                # Mock quality evaluation result
                mock_quality_result = type('MockQualityResult', (), {
                    'overall_score': 8.5,
                    'confidence': 0.9,
                    'metrics': {'accuracy': 0.9, 'completeness': 0.85},
                    'recommendations': ['Improve accuracy'],
                    'model_dump': lambda: {
                        'overall_score': 8.5,
                        'confidence': 0.9,
                        'metrics': {'accuracy': 0.9, 'completeness': 0.85},
                        'recommendations': ['Improve accuracy']
                    }
                })()

                mock_use_case_instance = type('MockUseCase', (), {
                    'execute': lambda x: mock_quality_result
                })()
                mock_use_case.return_value = mock_use_case_instance

                mock_factory_instance = type('MockFactory', (), {
                    'get_component': lambda x: mock_use_case_instance
                })()
                mock_factory.create_from_config.return_value = mock_factory_instance

                # Run the command
                result = self.runner.invoke(main, ["analysis", "quality-eval"])

                assert result.exit_code == 0
                assert "Overall Score: 8.50/10" in result.output
                assert "Confidence: 0.90" in result.output

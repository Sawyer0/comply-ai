"""
End-to-end integration tests for the weekly evaluation system.

Tests the complete system from CLI commands through to report generation
and notification delivery.
"""

import asyncio
import json
import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.llama_mapper.analysis.domain.services import WeeklyEvaluationService, QualityService
from src.llama_mapper.analysis.domain.entities import QualityMetrics
from src.llama_mapper.analysis.infrastructure.storage_backend import FileStorageBackend
from src.llama_mapper.analysis.infrastructure.report_generator import WeeklyEvaluationReportGenerator
from src.llama_mapper.analysis.quality.quality_alerting_system import QualityAlertingSystem
from src.llama_mapper.analysis.config.evaluation_config import WeeklyEvaluationConfig
from src.llama_mapper.cli.commands.weekly_evaluation import WeeklyEvaluationCommand


class TestWeeklyEvaluationEndToEnd:
    """End-to-end tests for the complete weekly evaluation system."""
    
    @pytest.fixture
    async def temp_storage_dir(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    async def mock_config_manager(self):
        """Create mock configuration manager."""
        return MagicMock()
    
    @pytest.fixture
    async def complete_system_setup(self, temp_storage_dir):
        """Set up complete system with all components."""
        # Create storage backend
        storage_backend = FileStorageBackend(temp_storage_dir)
        
        # Create quality service with mock evaluator
        quality_evaluator = AsyncMock()
        quality_evaluator.evaluate_batch.return_value = {
            "total_examples": 10,
            "schema_valid_rate": 0.98,
            "rubric_score": 0.85,
            "opa_compile_success_rate": 0.96,
            "evidence_accuracy": 0.82,
            "individual_rubric_scores": [0.8, 0.9, 0.85, 0.8, 0.9, 0.85, 0.8, 0.9, 0.85, 0.8]
        }
        
        quality_service = QualityService(quality_evaluator)
        
        # Create report generator
        report_generator = WeeklyEvaluationReportGenerator()
        
        # Create alerting system
        alerting_system = AsyncMock()
        alerting_system.send_evaluation_notification.return_value = True
        
        # Create weekly evaluation service
        weekly_eval_service = WeeklyEvaluationService(
            quality_service=quality_service,
            report_generator=report_generator,
            alerting_system=alerting_system,
            storage_backend=storage_backend,
        )
        
        return {
            "storage_backend": storage_backend,
            "quality_service": quality_service,
            "report_generator": report_generator,
            "alerting_system": alerting_system,
            "weekly_eval_service": weekly_eval_service,
        }
    
    @pytest.mark.asyncio
    async def test_complete_workflow_from_cli_to_notification(
        self, complete_system_setup, mock_config_manager
    ):
        """Test complete workflow from CLI command to notification delivery."""
        weekly_eval_service = complete_system_setup["weekly_eval_service"]
        alerting_system = complete_system_setup["alerting_system"]
        
        # Create CLI command with mocked service
        cli_command = WeeklyEvaluationCommand(mock_config_manager)
        cli_command.weekly_eval_service = weekly_eval_service
        
        tenant_id = "test-tenant-e2e"
        recipients = ["admin@example.com", "team@example.com"]
        
        # Step 1: Schedule evaluation via CLI
        schedule_id = cli_command.schedule(
            tenant_id=tenant_id,
            cron_schedule="0 9 * * 1",
            recipients=recipients,
            config_file=None
        )
        
        assert schedule_id["schedule_id"] is not None
        assert schedule_id["tenant_id"] == tenant_id
        assert schedule_id["status"] == "scheduled"
        
        # Step 2: Verify schedule was stored
        schedules = await weekly_eval_service.list_scheduled_evaluations(tenant_id)
        assert len(schedules) == 1
        assert schedules[0]["tenant_id"] == tenant_id
        
        # Step 3: Run evaluation via CLI
        with patch.object(weekly_eval_service, '_get_recent_analysis_data') as mock_get_data:
            mock_get_data.return_value = []  # Empty data for simplicity
            
            result = cli_command.run(
                schedule_id=schedule_id["schedule_id"],
                force=True
            )
            
            assert result["schedule_id"] == schedule_id["schedule_id"]
            assert result["status"] == "completed"
            assert "quality_metrics" in result
            assert "report_path" in result
        
        # Step 4: Verify notifications were sent
        assert alerting_system.send_evaluation_notification.call_count == len(recipients)
        
        # Step 5: Check service statistics
        stats = weekly_eval_service.get_service_statistics()
        assert stats["schedules_created"] >= 1
        assert stats["evaluations_run"] >= 1
        assert stats["reports_generated"] >= 1
        assert stats["notifications_sent"] >= len(recipients)
        assert stats["success_rate"] > 0
    
    @pytest.mark.asyncio
    async def test_cli_list_and_status_commands(
        self, complete_system_setup, mock_config_manager
    ):
        """Test CLI list and status commands."""
        weekly_eval_service = complete_system_setup["weekly_eval_service"]
        
        # Create CLI command
        cli_command = WeeklyEvaluationCommand(mock_config_manager)
        cli_command.weekly_eval_service = weekly_eval_service
        
        # Create multiple schedules
        schedule1 = cli_command.schedule(
            tenant_id="tenant-1",
            cron_schedule="0 9 * * 1"
        )
        schedule2 = cli_command.schedule(
            tenant_id="tenant-2",
            cron_schedule="0 10 * * 2"
        )
        
        # Test list command
        all_schedules = cli_command.list_schedules()
        assert len(all_schedules) == 2
        
        tenant1_schedules = cli_command.list_schedules(tenant_id="tenant-1")
        assert len(tenant1_schedules) == 1
        assert tenant1_schedules[0]["tenant_id"] == "tenant-1"
        
        # Test status command
        status = cli_command.status(schedule1["schedule_id"])
        assert status["schedule_id"] == schedule1["schedule_id"]
        assert status["tenant_id"] == "tenant-1"
        assert status["cron_schedule"] == "0 9 * * 1"
        assert status["active"] is True
    
    @pytest.mark.asyncio
    async def test_cli_cancel_command(
        self, complete_system_setup, mock_config_manager
    ):
        """Test CLI cancel command."""
        weekly_eval_service = complete_system_setup["weekly_eval_service"]
        
        # Create CLI command
        cli_command = WeeklyEvaluationCommand(mock_config_manager)
        cli_command.weekly_eval_service = weekly_eval_service
        
        # Create schedule
        schedule = cli_command.schedule(
            tenant_id="tenant-cancel",
            cron_schedule="0 9 * * 1"
        )
        
        # Verify schedule exists
        schedules = cli_command.list_schedules(tenant_id="tenant-cancel")
        assert len(schedules) == 1
        
        # Cancel schedule
        success = cli_command.cancel(
            schedule_id=schedule["schedule_id"],
            confirm=True
        )
        assert success is True
        
        # Verify schedule was cancelled
        remaining_schedules = cli_command.list_schedules(tenant_id="tenant-cancel")
        assert len(remaining_schedules) == 0
    
    @pytest.mark.asyncio
    async def test_configuration_loading_and_validation(
        self, temp_storage_dir
    ):
        """Test configuration loading and validation."""
        # Create test configuration
        config_data = {
            "enabled": True,
            "default_schedule": "0 9 * * 1",
            "evaluation_period_days": 7,
            "thresholds": {
                "schema_valid_rate": 0.98,
                "rubric_score": 0.8,
                "opa_compile_success_rate": 0.95,
                "evidence_accuracy": 0.85
            },
            "notifications": {
                "enabled": True,
                "email_recipients": ["admin@example.com"]
            },
            "storage": {
                "backend_type": "file",
                "storage_dir": temp_storage_dir,
                "retention_days": 90
            }
        }
        
        # Save configuration to file
        config_file = Path(temp_storage_dir) / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Load configuration
        from src.llama_mapper.analysis.config.evaluation_config import WeeklyEvaluationConfig
        config = WeeklyEvaluationConfig.from_file(config_file)
        
        # Verify configuration
        assert config.enabled is True
        assert config.default_schedule == "0 9 * * 1"
        assert config.evaluation_period_days == 7
        assert config.thresholds.schema_valid_rate == 0.98
        assert config.notifications.email_recipients == ["admin@example.com"]
        assert config.storage.backend_type == "file"
        assert config.storage.storage_dir == temp_storage_dir
        
        # Validate configuration
        from src.llama_mapper.analysis.config.evaluation_config import validate_evaluation_config
        assert validate_evaluation_config(config) is True
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(
        self, complete_system_setup, mock_config_manager
    ):
        """Test error handling and recovery scenarios."""
        weekly_eval_service = complete_system_setup["weekly_eval_service"]
        
        # Create CLI command
        cli_command = WeeklyEvaluationCommand(mock_config_manager)
        cli_command.weekly_eval_service = weekly_eval_service
        
        # Test invalid tenant ID
        with pytest.raises(Exception, match="tenant_id cannot be empty"):
            cli_command.schedule(tenant_id="")
        
        # Test invalid cron schedule
        with pytest.raises(Exception, match="Invalid cron schedule"):
            cli_command.schedule(
                tenant_id="test-tenant",
                cron_schedule="invalid-cron"
            )
        
        # Test invalid email addresses
        with pytest.raises(Exception, match="Invalid email address"):
            cli_command.schedule(
                tenant_id="test-tenant",
                recipients=["invalid-email"]
            )
        
        # Test running non-existent schedule
        with pytest.raises(Exception, match="Schedule nonexistent-id not found"):
            cli_command.run(schedule_id="nonexistent-id", force=True)
        
        # Test cancelling non-existent schedule
        success = cli_command.cancel(schedule_id="nonexistent-id", confirm=True)
        assert success is False
    
    @pytest.mark.asyncio
    async def test_report_generation_formats(
        self, complete_system_setup
    ):
        """Test report generation in different formats."""
        report_generator = complete_system_setup["report_generator"]
        
        quality_metrics = {
            "total_examples": 10,
            "schema_valid_rate": 0.98,
            "rubric_score": 0.85,
            "opa_compile_success_rate": 0.96,
            "evidence_accuracy": 0.82,
            "individual_rubric_scores": [0.8, 0.9, 0.85, 0.8, 0.9, 0.85, 0.8, 0.9, 0.85, 0.8]
        }
        
        report_data = MagicMock()
        report_data.custom_data = {"quality_metrics": quality_metrics}
        
        # Test JSON report
        json_report = report_generator.generate_report(
            report_data=report_data,
            format_type="JSON",
            tenant_id="test-tenant"
        )
        
        assert isinstance(json_report, dict)
        assert "metadata" in json_report
        assert "quality_metrics" in json_report
        assert "summary" in json_report
        assert "alerts" in json_report
        assert "recommendations" in json_report
        
        # Test CSV report
        csv_report = report_generator.generate_report(
            report_data=report_data,
            format_type="CSV",
            tenant_id="test-tenant"
        )
        
        assert isinstance(csv_report, str)
        assert "Metric,Value" in csv_report
        assert "total_examples" in csv_report
        
        # Test PDF report (fallback to text if reportlab not available)
        pdf_report = report_generator.generate_report(
            report_data=report_data,
            format_type="PDF",
            tenant_id="test-tenant"
        )
        
        assert isinstance(pdf_report, bytes)
        assert len(pdf_report) > 0
    
    @pytest.mark.asyncio
    async def test_storage_persistence_across_restarts(
        self, temp_storage_dir
    ):
        """Test that data persists across service restarts."""
        # Create first storage backend and service
        storage_backend1 = FileStorageBackend(temp_storage_dir)
        
        quality_evaluator = AsyncMock()
        quality_service = QualityService(quality_evaluator)
        report_generator = WeeklyEvaluationReportGenerator()
        alerting_system = AsyncMock()
        
        service1 = WeeklyEvaluationService(
            quality_service=quality_service,
            report_generator=report_generator,
            alerting_system=alerting_system,
            storage_backend=storage_backend1,
        )
        
        # Create schedule
        schedule_id = await service1.schedule_weekly_evaluation(
            tenant_id="persistence-test",
            cron_schedule="0 9 * * 1"
        )
        
        # Create second storage backend and service (simulating restart)
        storage_backend2 = FileStorageBackend(temp_storage_dir)
        
        service2 = WeeklyEvaluationService(
            quality_service=quality_service,
            report_generator=report_generator,
            alerting_system=alerting_system,
            storage_backend=storage_backend2,
        )
        
        # Verify schedule persists
        schedules = await service2.list_scheduled_evaluations("persistence-test")
        assert len(schedules) == 1
        assert schedules[0]["schedule_id"] == schedule_id
        assert schedules[0]["tenant_id"] == "persistence-test"
    
    @pytest.mark.asyncio
    async def test_concurrent_evaluation_handling(
        self, complete_system_setup
    ):
        """Test handling of concurrent evaluations."""
        weekly_eval_service = complete_system_setup["weekly_eval_service"]
        
        # Create multiple schedules
        schedule_ids = []
        for i in range(3):
            schedule_id = await weekly_eval_service.schedule_weekly_evaluation(
                tenant_id=f"concurrent-tenant-{i}",
                cron_schedule="0 9 * * 1"
            )
            schedule_ids.append(schedule_id)
        
        # Run evaluations concurrently
        with patch.object(weekly_eval_service, '_get_recent_analysis_data') as mock_get_data:
            mock_get_data.return_value = []
            
            # Run all evaluations concurrently
            tasks = [
                weekly_eval_service.run_scheduled_evaluation(schedule_id)
                for schedule_id in schedule_ids
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Verify all evaluations completed
            assert len(results) == 3
            for result in results:
                assert result["status"] == "completed"
                assert "quality_metrics" in result
        
        # Check statistics
        stats = weekly_eval_service.get_service_statistics()
        assert stats["evaluations_run"] >= 3
        assert stats["reports_generated"] >= 3

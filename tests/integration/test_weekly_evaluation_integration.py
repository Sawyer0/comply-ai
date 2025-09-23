"""
Integration tests for the weekly evaluation system.

Tests the complete workflow from scheduling to report generation
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


class TestWeeklyEvaluationIntegration:
    """Integration tests for the complete weekly evaluation workflow."""
    
    @pytest.fixture
    async def temp_storage_dir(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    async def storage_backend(self, temp_storage_dir):
        """Create file storage backend."""
        return FileStorageBackend(temp_storage_dir)
    
    @pytest.fixture
    async def report_generator(self):
        """Create report generator."""
        return WeeklyEvaluationReportGenerator()
    
    @pytest.fixture
    async def alerting_system(self):
        """Create mock alerting system."""
        system = AsyncMock()
        system.send_evaluation_notification.return_value = True
        return system
    
    @pytest.fixture
    async def quality_service(self):
        """Create mock quality service."""
        service = AsyncMock()
        service.evaluate_quality.return_value = QualityMetrics(
            total_examples=10,
            schema_valid_rate=0.98,
            rubric_score=0.85,
            opa_compile_success_rate=0.96,
            evidence_accuracy=0.82,
            individual_rubric_scores=[0.8, 0.9, 0.85, 0.8, 0.9, 0.85, 0.8, 0.9, 0.85, 0.8]
        )
        return service
    
    @pytest.fixture
    async def weekly_eval_service(
        self, quality_service, report_generator, alerting_system, storage_backend
    ):
        """Create WeeklyEvaluationService with all dependencies."""
        return WeeklyEvaluationService(
            quality_service=quality_service,
            report_generator=report_generator,
            alerting_system=alerting_system,
            storage_backend=storage_backend,
        )
    
    @pytest.mark.asyncio
    async def test_complete_evaluation_workflow(
        self, weekly_eval_service, storage_backend, quality_service, alerting_system
    ):
        """Test the complete evaluation workflow from scheduling to notification."""
        tenant_id = "test-tenant-123"
        recipients = ["admin@example.com", "team@example.com"]
        
        # Step 1: Schedule evaluation
        schedule_id = await weekly_eval_service.schedule_weekly_evaluation(
            tenant_id=tenant_id,
            cron_schedule="0 9 * * 1",
            report_recipients=recipients,
            evaluation_config={"threshold": 0.8}
        )
        
        assert schedule_id is not None
        assert len(schedule_id) > 0
        
        # Step 2: Verify schedule was stored
        schedules = await storage_backend.get_evaluation_schedules(tenant_id)
        assert len(schedules) == 1
        assert schedules[0]["tenant_id"] == tenant_id
        assert schedules[0]["schedule_id"] == schedule_id
        
        # Step 3: Run evaluation
        with patch.object(weekly_eval_service, '_get_recent_analysis_data') as mock_get_data:
            mock_get_data.return_value = []  # Empty data for simplicity
            
            result = await weekly_eval_service.run_scheduled_evaluation(schedule_id)
            
            # Verify result structure
            assert result["schedule_id"] == schedule_id
            assert result["tenant_id"] == tenant_id
            assert result["status"] == "completed"
            assert "quality_metrics" in result
            assert "report_path" in result
            assert "data_points" in result
            
            # Verify quality service was called
            quality_service.evaluate_quality.assert_called_once()
            
            # Verify notifications were sent
            assert alerting_system.send_evaluation_notification.call_count == len(recipients)
    
    @pytest.mark.asyncio
    async def test_evaluation_with_quality_issues(
        self, weekly_eval_service, storage_backend, quality_service, alerting_system
    ):
        """Test evaluation with quality issues that should trigger alerts."""
        tenant_id = "test-tenant-issues"
        
        # Create quality service that returns poor metrics
        poor_quality_service = AsyncMock()
        poor_quality_service.evaluate_quality.return_value = QualityMetrics(
            total_examples=5,
            schema_valid_rate=0.95,  # Below threshold
            rubric_score=0.75,       # Below threshold
            opa_compile_success_rate=0.90,  # Below threshold
            evidence_accuracy=0.80,  # Below threshold
            individual_rubric_scores=[0.7, 0.8, 0.75, 0.7, 0.8]
        )
        
        # Create service with poor quality metrics
        service_with_issues = WeeklyEvaluationService(
            quality_service=poor_quality_service,
            report_generator=weekly_eval_service.report_generator,
            alerting_system=weekly_eval_service.alerting_system,
            storage_backend=weekly_eval_service.storage_backend,
        )
        
        # Schedule and run evaluation
        schedule_id = await service_with_issues.schedule_weekly_evaluation(
            tenant_id=tenant_id,
            report_recipients=["admin@example.com"]
        )
        
        with patch.object(service_with_issues, '_get_recent_analysis_data') as mock_get_data:
            mock_get_data.return_value = []
            
            result = await service_with_issues.run_scheduled_evaluation(schedule_id)
            
            # Verify result
            assert result["status"] == "completed"
            
            # Verify notifications were sent (should include alerts)
            alerting_system.send_evaluation_notification.assert_called_once()
            call_args = alerting_system.send_evaluation_notification.call_args
            alerts = call_args[0][3]  # alerts parameter
            assert len(alerts) > 0  # Should have quality alerts
    
    @pytest.mark.asyncio
    async def test_evaluation_failure_handling(
        self, weekly_eval_service, storage_backend
    ):
        """Test handling of evaluation failures."""
        tenant_id = "test-tenant-failure"
        
        # Schedule evaluation
        schedule_id = await weekly_eval_service.schedule_weekly_evaluation(
            tenant_id=tenant_id
        )
        
        # Mock quality service to raise exception
        weekly_eval_service.quality_service.evaluate_quality.side_effect = Exception("Quality evaluation failed")
        
        # Run evaluation and expect it to fail
        with pytest.raises(RuntimeError, match="Failed to run scheduled evaluation"):
            await weekly_eval_service.run_scheduled_evaluation(schedule_id)
        
        # Verify failure was stored
        # Note: In a real implementation, we'd check the results storage
        # For now, we verify the exception was raised as expected
    
    @pytest.mark.asyncio
    async def test_schedule_management(
        self, weekly_eval_service, storage_backend
    ):
        """Test schedule management operations."""
        tenant_id = "test-tenant-management"
        
        # Create multiple schedules
        schedule1 = await weekly_eval_service.schedule_weekly_evaluation(
            tenant_id=tenant_id,
            cron_schedule="0 9 * * 1"
        )
        schedule2 = await weekly_eval_service.schedule_weekly_evaluation(
            tenant_id=tenant_id,
            cron_schedule="0 10 * * 2"
        )
        
        # List schedules
        all_schedules = await weekly_eval_service.list_scheduled_evaluations()
        assert len(all_schedules) == 2
        
        tenant_schedules = await weekly_eval_service.list_scheduled_evaluations(tenant_id)
        assert len(tenant_schedules) == 2
        
        # Cancel a schedule
        success = await weekly_eval_service.cancel_scheduled_evaluation(schedule1)
        assert success is True
        
        # Verify schedule was cancelled
        remaining_schedules = await weekly_eval_service.list_scheduled_evaluations(tenant_id)
        assert len(remaining_schedules) == 1
        assert remaining_schedules[0]["schedule_id"] == schedule2
    
    @pytest.mark.asyncio
    async def test_report_generation_formats(
        self, report_generator
    ):
        """Test report generation in different formats."""
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
            tenant_id="test-tenant",
            requested_by="test-user"
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
    async def test_storage_backend_persistence(
        self, storage_backend
    ):
        """Test storage backend persistence."""
        # Store a schedule
        schedule_data = {
            "schedule_id": "test-schedule-123",
            "tenant_id": "test-tenant",
            "cron_schedule": "0 9 * * 1",
            "report_recipients": ["admin@example.com"],
            "evaluation_config": {"threshold": 0.8},
            "created_at": datetime.now(timezone.utc),
            "active": True,
            "last_run": None,
            "next_run": None,
        }
        
        await storage_backend.store_evaluation_schedule(schedule_data)
        
        # Retrieve schedules
        schedules = await storage_backend.get_evaluation_schedules()
        assert len(schedules) == 1
        assert schedules[0]["schedule_id"] == "test-schedule-123"
        
        # Update schedule
        schedule_data["active"] = False
        await storage_backend.update_evaluation_schedule(schedule_data)
        
        # Verify update
        updated_schedules = await storage_backend.get_evaluation_schedules()
        assert len(updated_schedules) == 1
        assert updated_schedules[0]["active"] is False
        
        # Store evaluation result
        result_data = {
            "schedule_id": "test-schedule-123",
            "tenant_id": "test-tenant",
            "evaluation_date": datetime.now(timezone.utc),
            "status": "completed",
            "quality_metrics": {"total_examples": 10},
        }
        
        await storage_backend.store_evaluation_result(result_data)
        
        # Save report
        report_content = b"mock pdf content"
        report_path = await storage_backend.save_evaluation_report(
            "test-tenant", report_content, "weekly_evaluation"
        )
        
        assert report_path is not None
        assert Path(report_path).exists()
        assert Path(report_path).read_bytes() == report_content
    
    @pytest.mark.asyncio
    async def test_validation_and_error_handling(
        self, weekly_eval_service
    ):
        """Test input validation and error handling."""
        # Test invalid tenant ID
        with pytest.raises(ValueError, match="tenant_id cannot be empty"):
            await weekly_eval_service.schedule_weekly_evaluation(tenant_id="")
        
        # Test invalid cron schedule
        with pytest.raises(ValueError, match="Invalid cron schedule"):
            await weekly_eval_service.schedule_weekly_evaluation(
                tenant_id="test-tenant",
                cron_schedule="invalid-cron"
            )
        
        # Test invalid email addresses
        with pytest.raises(ValueError, match="Invalid email address"):
            await weekly_eval_service.schedule_weekly_evaluation(
                tenant_id="test-tenant",
                report_recipients=["invalid-email"]
            )
        
        # Test running non-existent schedule
        with pytest.raises(ValueError, match="Schedule nonexistent-id not found"):
            await weekly_eval_service.run_scheduled_evaluation("nonexistent-id")
        
        # Test cancelling non-existent schedule
        success = await weekly_eval_service.cancel_scheduled_evaluation("nonexistent-id")
        assert success is False

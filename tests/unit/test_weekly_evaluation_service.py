"""
Unit tests for the weekly evaluation service.

Tests the WeeklyEvaluationService functionality including scheduling,
running evaluations, and generating reports.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.llama_mapper.analysis.domain.services import WeeklyEvaluationService
from src.llama_mapper.analysis.domain.entities import QualityMetrics


class TestWeeklyEvaluationService:
    """Test cases for WeeklyEvaluationService."""
    
    @pytest.fixture
    def mock_quality_service(self):
        """Mock quality service."""
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
    def mock_report_generator(self):
        """Mock report generator."""
        generator = MagicMock()
        generator.generate_report.return_value = b"mock_pdf_content"
        return generator
    
    @pytest.fixture
    def mock_alerting_system(self):
        """Mock alerting system."""
        system = AsyncMock()
        system.send_evaluation_notification.return_value = True
        return system
    
    @pytest.fixture
    def mock_storage_backend(self):
        """Mock storage backend."""
        backend = AsyncMock()
        backend.store_evaluation_schedule.return_value = None
        backend.update_evaluation_schedule.return_value = None
        backend.store_evaluation_result.return_value = None
        backend.save_evaluation_report.return_value = "/tmp/mock_report.pdf"
        return backend
    
    @pytest.fixture
    def weekly_eval_service(
        self, mock_quality_service, mock_report_generator, 
        mock_alerting_system, mock_storage_backend
    ):
        """Create WeeklyEvaluationService with mocked dependencies."""
        return WeeklyEvaluationService(
            quality_service=mock_quality_service,
            report_generator=mock_report_generator,
            alerting_system=mock_alerting_system,
            storage_backend=mock_storage_backend,
        )
    
    @pytest.mark.asyncio
    async def test_schedule_weekly_evaluation(
        self, weekly_eval_service, mock_storage_backend
    ):
        """Test scheduling a weekly evaluation."""
        tenant_id = "test-tenant"
        cron_schedule = "0 9 * * 1"
        recipients = ["admin@example.com"]
        evaluation_config = {"threshold": 0.8}
        
        schedule_id = await weekly_eval_service.schedule_weekly_evaluation(
            tenant_id=tenant_id,
            cron_schedule=cron_schedule,
            report_recipients=recipients,
            evaluation_config=evaluation_config,
        )
        
        # Verify schedule was created
        assert schedule_id in weekly_eval_service.scheduled_evaluations
        
        schedule = weekly_eval_service.scheduled_evaluations[schedule_id]
        assert schedule["tenant_id"] == tenant_id
        assert schedule["cron_schedule"] == cron_schedule
        assert schedule["report_recipients"] == recipients
        assert schedule["evaluation_config"] == evaluation_config
        assert schedule["active"] is True
        
        # Verify storage backend was called
        mock_storage_backend.store_evaluation_schedule.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_scheduled_evaluation_success(
        self, weekly_eval_service, mock_quality_service, 
        mock_report_generator, mock_alerting_system, mock_storage_backend
    ):
        """Test running a scheduled evaluation successfully."""
        # Setup a scheduled evaluation
        tenant_id = "test-tenant"
        schedule_id = await weekly_eval_service.schedule_weekly_evaluation(
            tenant_id=tenant_id,
            report_recipients=["admin@example.com"]
        )
        
        # Mock the data retrieval
        with patch.object(weekly_eval_service, '_get_recent_analysis_data') as mock_get_data:
            mock_get_data.return_value = []  # Empty data for simplicity
            
            # Run the evaluation
            result = await weekly_eval_service.run_scheduled_evaluation(schedule_id)
            
            # Verify result
            assert result["schedule_id"] == schedule_id
            assert result["tenant_id"] == tenant_id
            assert result["status"] == "completed"
            assert "quality_metrics" in result
            assert "report_path" in result
            
            # Verify services were called
            mock_quality_service.evaluate_quality.assert_called_once()
            mock_report_generator.generate_report.assert_called_once()
            mock_alerting_system.send_evaluation_notification.assert_called_once()
            mock_storage_backend.store_evaluation_result.assert_called_once()
            mock_storage_backend.update_evaluation_schedule.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_scheduled_evaluation_failure(
        self, weekly_eval_service, mock_storage_backend
    ):
        """Test running a scheduled evaluation with failure."""
        # Setup a scheduled evaluation
        tenant_id = "test-tenant"
        schedule_id = await weekly_eval_service.schedule_weekly_evaluation(
            tenant_id=tenant_id
        )
        
        # Mock the data retrieval to raise an exception
        with patch.object(weekly_eval_service, '_get_recent_analysis_data') as mock_get_data:
            mock_get_data.side_effect = Exception("Data retrieval failed")
            
            # Run the evaluation and expect it to fail
            with pytest.raises(Exception, match="Data retrieval failed"):
                await weekly_eval_service.run_scheduled_evaluation(schedule_id)
            
            # Verify failure was stored
            mock_storage_backend.store_evaluation_result.assert_called_once()
            failure_result = mock_storage_backend.store_evaluation_result.call_args[0][0]
            assert failure_result["status"] == "failed"
            assert "Data retrieval failed" in failure_result["error"]
    
    @pytest.mark.asyncio
    async def test_list_scheduled_evaluations(
        self, weekly_eval_service
    ):
        """Test listing scheduled evaluations."""
        # Create multiple schedules
        schedule1 = await weekly_eval_service.schedule_weekly_evaluation(
            tenant_id="tenant1"
        )
        schedule2 = await weekly_eval_service.schedule_weekly_evaluation(
            tenant_id="tenant2"
        )
        
        # List all schedules
        all_schedules = await weekly_eval_service.list_scheduled_evaluations()
        assert len(all_schedules) == 2
        
        # List schedules for specific tenant
        tenant1_schedules = await weekly_eval_service.list_scheduled_evaluations("tenant1")
        assert len(tenant1_schedules) == 1
        assert tenant1_schedules[0]["tenant_id"] == "tenant1"
    
    @pytest.mark.asyncio
    async def test_cancel_scheduled_evaluation(
        self, weekly_eval_service, mock_storage_backend
    ):
        """Test cancelling a scheduled evaluation."""
        # Create a schedule
        schedule_id = await weekly_eval_service.schedule_weekly_evaluation(
            tenant_id="test-tenant"
        )
        
        # Cancel the schedule
        success = await weekly_eval_service.cancel_scheduled_evaluation(schedule_id)
        
        assert success is True
        assert schedule_id not in weekly_eval_service.scheduled_evaluations
        mock_storage_backend.update_evaluation_schedule.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cancel_nonexistent_schedule(
        self, weekly_eval_service
    ):
        """Test cancelling a non-existent schedule."""
        success = await weekly_eval_service.cancel_scheduled_evaluation("nonexistent-id")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_generate_evaluation_report(
        self, weekly_eval_service, mock_report_generator, mock_storage_backend
    ):
        """Test generating an evaluation report."""
        tenant_id = "test-tenant"
        quality_metrics = QualityMetrics(
            total_examples=5,
            schema_valid_rate=0.95,
            rubric_score=0.8,
            opa_compile_success_rate=0.9,
            evidence_accuracy=0.85,
            individual_rubric_scores=[0.8, 0.8, 0.8, 0.8, 0.8]
        )
        schedule = {"tenant_id": tenant_id}
        
        report = await weekly_eval_service._generate_evaluation_report(
            tenant_id, quality_metrics, schedule
        )
        
        assert "file_path" in report
        assert "content" in report
        assert "format" in report
        assert report["format"] == "PDF"
        
        mock_report_generator.generate_report.assert_called_once()
        mock_storage_backend.save_evaluation_report.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_evaluation_notifications(
        self, weekly_eval_service, mock_alerting_system
    ):
        """Test sending evaluation notifications."""
        recipients = ["admin@example.com", "user@example.com"]
        report = {"file_path": "/tmp/report.pdf"}
        quality_metrics = QualityMetrics(
            total_examples=5,
            schema_valid_rate=0.95,
            rubric_score=0.8,
            opa_compile_success_rate=0.9,
            evidence_accuracy=0.85,
            individual_rubric_scores=[0.8, 0.8, 0.8, 0.8, 0.8]
        )
        alerts = ["Schema validation rate below threshold"]
        
        await weekly_eval_service._send_evaluation_notifications(
            recipients, report, quality_metrics
        )
        
        # Verify notifications were sent to all recipients
        assert mock_alerting_system.send_evaluation_notification.call_count == len(recipients)
        
        # Verify each call had correct arguments
        for call in mock_alerting_system.send_evaluation_notification.call_args_list:
            args, kwargs = call
            assert args[0] in recipients  # recipient
            assert args[1] == report["file_path"]  # report_path
            assert args[2] == quality_metrics  # quality_metrics

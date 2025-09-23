"""
Unit tests for the weekly evaluation CLI commands.

Tests the CLI command functionality for managing weekly evaluations.
"""

import pytest
from unittest.mock import MagicMock, patch
from click.testing import CliRunner

from src.llama_mapper.cli.commands.weekly_evaluation import WeeklyEvaluationCommand


class TestWeeklyEvaluationCommand:
    """Test cases for WeeklyEvaluationCommand."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Mock configuration manager."""
        return MagicMock()
    
    @pytest.fixture
    def command(self, mock_config_manager):
        """Create WeeklyEvaluationCommand instance."""
        return WeeklyEvaluationCommand(mock_config_manager)
    
    def test_schedule_command_success(self, command):
        """Test successful scheduling of weekly evaluation."""
        with patch.object(command, '_get_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.schedule_weekly_evaluation.return_value = "schedule-123"
            mock_get_service.return_value = mock_service
            
            result = command.schedule(
                tenant_id="test-tenant",
                cron_schedule="0 9 * * 1",
                recipients=["admin@example.com"]
            )
            
            assert result["schedule_id"] == "schedule-123"
            assert result["tenant_id"] == "test-tenant"
            assert result["cron_schedule"] == "0 9 * * 1"
            assert result["report_recipients"] == ["admin@example.com"]
            assert result["status"] == "scheduled"
            
            mock_service.schedule_weekly_evaluation.assert_called_once_with(
                tenant_id="test-tenant",
                cron_schedule="0 9 * * 1",
                report_recipients=["admin@example.com"],
                evaluation_config=None,
            )
    
    def test_schedule_command_invalid_cron(self, command):
        """Test scheduling with invalid cron expression."""
        with pytest.raises(Exception, match="Invalid cron schedule"):
            command.schedule(
                tenant_id="test-tenant",
                cron_schedule="invalid-cron"
            )
    
    def test_run_command_success(self, command):
        """Test successful running of scheduled evaluation."""
        with patch.object(command, '_get_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.run_scheduled_evaluation.return_value = {
                "schedule_id": "schedule-123",
                "status": "completed",
                "report_path": "/tmp/report.pdf"
            }
            mock_get_service.return_value = mock_service
            
            result = command.run(
                schedule_id="schedule-123",
                force=True
            )
            
            assert result["schedule_id"] == "schedule-123"
            assert result["status"] == "completed"
            assert result["report_path"] == "/tmp/report.pdf"
            
            mock_service.run_scheduled_evaluation.assert_called_once_with("schedule-123")
    
    def test_run_command_not_forced(self, command):
        """Test running evaluation without force flag."""
        result = command.run(
            schedule_id="schedule-123",
            force=False
        )
        
        assert result["status"] == "skipped"
        assert result["reason"] == "not_due"
    
    def test_list_schedules_command(self, command):
        """Test listing scheduled evaluations."""
        with patch.object(command, '_get_service') as mock_get_service:
            mock_service = MagicMock()
            mock_schedules = [
                {
                    "schedule_id": "schedule-1",
                    "tenant_id": "tenant-1",
                    "cron_schedule": "0 9 * * 1",
                    "active": True,
                    "created_at": "2024-01-01T00:00:00Z",
                    "last_run": None,
                    "report_recipients": ["admin@example.com"]
                },
                {
                    "schedule_id": "schedule-2",
                    "tenant_id": "tenant-2",
                    "cron_schedule": "0 10 * * 2",
                    "active": False,
                    "created_at": "2024-01-02T00:00:00Z",
                    "last_run": "2024-01-08T10:00:00Z",
                    "report_recipients": []
                }
            ]
            mock_service.list_scheduled_evaluations.return_value = mock_schedules
            mock_get_service.return_value = mock_service
            
            # Test listing all schedules
            all_schedules = command.list_schedules()
            assert len(all_schedules) == 2
            
            # Test listing only active schedules
            active_schedules = command.list_schedules(active_only=True)
            assert len(active_schedules) == 1
            assert active_schedules[0]["active"] is True
            
            # Test listing schedules for specific tenant
            tenant_schedules = command.list_schedules(tenant_id="tenant-1")
            assert len(tenant_schedules) == 1
            assert tenant_schedules[0]["tenant_id"] == "tenant-1"
    
    def test_cancel_command_success(self, command):
        """Test successful cancellation of scheduled evaluation."""
        with patch.object(command, '_get_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.cancel_scheduled_evaluation.return_value = True
            mock_get_service.return_value = mock_service
            
            result = command.cancel(
                schedule_id="schedule-123",
                confirm=True
            )
            
            assert result is True
            mock_service.cancel_scheduled_evaluation.assert_called_once_with("schedule-123")
    
    def test_cancel_command_not_confirmed(self, command):
        """Test cancellation without confirmation."""
        with patch('click.confirm', return_value=False):
            result = command.cancel(
                schedule_id="schedule-123",
                confirm=False
            )
            
            assert result is False
    
    def test_status_command(self, command):
        """Test getting status of scheduled evaluation."""
        with patch.object(command, '_get_service') as mock_get_service:
            mock_service = MagicMock()
            mock_schedules = [
                {
                    "schedule_id": "schedule-123",
                    "tenant_id": "test-tenant",
                    "cron_schedule": "0 9 * * 1",
                    "active": True,
                    "created_at": "2024-01-01T00:00:00Z",
                    "last_run": "2024-01-08T09:00:00Z",
                    "report_recipients": ["admin@example.com"]
                }
            ]
            mock_service.list_scheduled_evaluations.return_value = mock_schedules
            mock_get_service.return_value = mock_service
            
            result = command.status("schedule-123")
            
            assert result["schedule_id"] == "schedule-123"
            assert result["tenant_id"] == "test-tenant"
            assert result["cron_schedule"] == "0 9 * * 1"
            assert result["active"] is True
            assert result["created_at"] == "2024-01-01T00:00:00Z"
            assert result["last_run"] == "2024-01-08T09:00:00Z"
            assert result["report_recipients"] == ["admin@example.com"]
    
    def test_status_command_schedule_not_found(self, command):
        """Test getting status of non-existent schedule."""
        with patch.object(command, '_get_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.list_scheduled_evaluations.return_value = []
            mock_get_service.return_value = mock_service
            
            with pytest.raises(Exception, match="Schedule nonexistent-id not found"):
                command.status("nonexistent-id")
    
    def test_schedule_with_config_file(self, command):
        """Test scheduling with configuration file."""
        import json
        import tempfile
        
        config_data = {
            "threshold": 0.8,
            "include_detailed_metrics": True,
            "notification_channels": ["email", "slack"]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            with patch.object(command, '_get_service') as mock_get_service:
                mock_service = MagicMock()
                mock_service.schedule_weekly_evaluation.return_value = "schedule-123"
                mock_get_service.return_value = mock_service
                
                result = command.schedule(
                    tenant_id="test-tenant",
                    config_file=config_file
                )
                
                assert result["schedule_id"] == "schedule-123"
                
                # Verify the service was called with the loaded config
                mock_service.schedule_weekly_evaluation.assert_called_once()
                call_args = mock_service.schedule_weekly_evaluation.call_args
                assert call_args[1]["evaluation_config"] == config_data
        
        finally:
            import os
            os.unlink(config_file)
    
    def test_schedule_with_invalid_config_file(self, command):
        """Test scheduling with invalid configuration file."""
        with pytest.raises(Exception, match="Failed to load config file"):
            command.schedule(
                tenant_id="test-tenant",
                config_file="nonexistent-config.json"
            )

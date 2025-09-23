"""
CLI commands for managing weekly evaluation scheduling and reporting.

This module provides commands for:
- Scheduling weekly evaluations
- Running scheduled evaluations
- Managing evaluation schedules
- Viewing evaluation reports
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import click
from croniter import croniter

from ..core import BaseCommand, CLIError
from ...analysis.domain.services import WeeklyEvaluationService
from ...analysis.domain.entities import QualityMetrics

logger = logging.getLogger(__name__)


class WeeklyEvaluationCommand(BaseCommand):
    """Command for managing weekly evaluations."""
    
    def __init__(self, config_manager):
        super().__init__(config_manager)
        self.weekly_eval_service = None
    
    def _get_service(self) -> WeeklyEvaluationService:
        """Get or create the weekly evaluation service."""
        if self.weekly_eval_service is None:
            try:
                # Import here to avoid circular imports
                from ...analysis.domain.services import QualityService
                from ...analysis.infrastructure.quality_evaluator import QualityEvaluator
                from ...analysis.infrastructure.report_generator import WeeklyEvaluationReportGenerator
                from ...analysis.infrastructure.storage_backend import create_storage_backend
                from ...analysis.quality.quality_alerting_system import QualityAlertingSystem
                from ...analysis.config.evaluation_config import get_evaluation_config
                
                # Get configuration
                config = get_evaluation_config()
                
                # Initialize dependencies
                quality_evaluator = QualityEvaluator()
                quality_service = QualityService(quality_evaluator)
                report_generator = WeeklyEvaluationReportGenerator()
                alerting_system = QualityAlertingSystem()
                storage_backend = create_storage_backend(**config.get_storage_backend_config())
                
                # Create service
                self.weekly_eval_service = WeeklyEvaluationService(
                    quality_service=quality_service,
                    report_generator=report_generator,
                    alerting_system=alerting_system,
                    storage_backend=storage_backend,
                )
                
            except Exception as e:
                logger.error(f"Failed to initialize weekly evaluation service: {e}")
                raise CLIError(f"Failed to initialize weekly evaluation service: {e}")
        
        return self.weekly_eval_service
    
    def schedule(
        self,
        tenant_id: str,
        cron_schedule: str = "0 9 * * 1",
        recipients: Optional[List[str]] = None,
        config_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Schedule a weekly evaluation for a tenant.
        
        Args:
            tenant_id: Tenant ID for the evaluation
            cron_schedule: Cron expression for scheduling (default: Monday 9 AM)
            recipients: List of email addresses for report distribution
            config_file: Path to evaluation configuration file
            
        Returns:
            Schedule information
        """
        try:
            # Validate cron schedule
            try:
                croniter(cron_schedule)
            except Exception as e:
                raise CLIError(f"Invalid cron schedule '{cron_schedule}': {e}")
            
            # Load evaluation config if provided
            evaluation_config = {}
            if config_file:
                try:
                    with open(config_file, 'r') as f:
                        evaluation_config = json.load(f)
                except Exception as e:
                    raise CLIError(f"Failed to load config file '{config_file}': {e}")
            
            # Schedule the evaluation
            service = self._get_service()
            schedule_id = service.schedule_weekly_evaluation(
                tenant_id=tenant_id,
                cron_schedule=cron_schedule,
                report_recipients=recipients,
                evaluation_config=evaluation_config,
            )
            
            result = {
                "schedule_id": schedule_id,
                "tenant_id": tenant_id,
                "cron_schedule": cron_schedule,
                "report_recipients": recipients or [],
                "evaluation_config": evaluation_config,
                "created_at": datetime.utcnow().isoformat(),
                "status": "scheduled",
            }
            
            click.echo(f"✅ Scheduled weekly evaluation for tenant '{tenant_id}'")
            click.echo(f"   Schedule ID: {schedule_id}")
            click.echo(f"   Cron Schedule: {cron_schedule}")
            click.echo(f"   Recipients: {', '.join(recipients or [])}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to schedule weekly evaluation: {e}")
            raise CLIError(f"Failed to schedule weekly evaluation: {e}")
    
    def run(
        self,
        schedule_id: str,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a scheduled evaluation.
        
        Args:
            schedule_id: ID of the scheduled evaluation
            force: Force run even if not due
            
        Returns:
            Evaluation results
        """
        try:
            service = self._get_service()
            
            if not force:
                # Check if evaluation is due (would be implemented with croniter)
                click.echo(f"⚠️  Use --force to run evaluation {schedule_id} immediately")
                return {"status": "skipped", "reason": "not_due"}
            
            click.echo(f"🔄 Running scheduled evaluation {schedule_id}...")
            
            result = service.run_scheduled_evaluation(schedule_id)
            
            click.echo(f"✅ Evaluation completed successfully")
            click.echo(f"   Status: {result.get('status', 'unknown')}")
            click.echo(f"   Report: {result.get('report_path', 'N/A')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to run scheduled evaluation: {e}")
            raise CLIError(f"Failed to run scheduled evaluation: {e}")
    
    def list_schedules(
        self,
        tenant_id: Optional[str] = None,
        active_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        List scheduled evaluations.
        
        Args:
            tenant_id: Filter by tenant ID
            active_only: Show only active schedules
            
        Returns:
            List of scheduled evaluations
        """
        try:
            service = self._get_service()
            schedules = service.list_scheduled_evaluations(tenant_id)
            
            if active_only:
                schedules = [s for s in schedules if s.get("active", True)]
            
            if not schedules:
                click.echo("No scheduled evaluations found")
                return []
            
            # Display schedules in a table format
            click.echo(f"\n📅 Scheduled Evaluations ({len(schedules)} found)")
            click.echo("=" * 80)
            
            for schedule in schedules:
                status = "🟢 Active" if schedule.get("active", True) else "🔴 Inactive"
                click.echo(f"Schedule ID: {schedule['schedule_id']}")
                click.echo(f"  Tenant: {schedule['tenant_id']}")
                click.echo(f"  Cron: {schedule['cron_schedule']}")
                click.echo(f"  Status: {status}")
                click.echo(f"  Created: {schedule.get('created_at', 'N/A')}")
                click.echo(f"  Last Run: {schedule.get('last_run', 'Never')}")
                click.echo(f"  Recipients: {', '.join(schedule.get('report_recipients', []))}")
                click.echo()
            
            return schedules
            
        except Exception as e:
            logger.error(f"Failed to list scheduled evaluations: {e}")
            raise CLIError(f"Failed to list scheduled evaluations: {e}")
    
    def cancel(
        self,
        schedule_id: str,
        confirm: bool = False,
    ) -> bool:
        """
        Cancel a scheduled evaluation.
        
        Args:
            schedule_id: ID of the scheduled evaluation
            confirm: Skip confirmation prompt
            
        Returns:
            True if cancelled successfully
        """
        try:
            if not confirm:
                if not click.confirm(f"Are you sure you want to cancel schedule {schedule_id}?"):
                    click.echo("Cancelled")
                    return False
            
            service = self._get_service()
            success = service.cancel_scheduled_evaluation(schedule_id)
            
            if success:
                click.echo(f"✅ Cancelled schedule {schedule_id}")
            else:
                click.echo(f"❌ Failed to cancel schedule {schedule_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to cancel scheduled evaluation: {e}")
            raise CLIError(f"Failed to cancel scheduled evaluation: {e}")
    
    def status(
        self,
        schedule_id: str,
    ) -> Dict[str, Any]:
        """
        Get status of a scheduled evaluation.
        
        Args:
            schedule_id: ID of the scheduled evaluation
            
        Returns:
            Schedule status information
        """
        try:
            service = self._get_service()
            schedules = service.list_scheduled_evaluations()
            
            schedule = next((s for s in schedules if s["schedule_id"] == schedule_id), None)
            if not schedule:
                raise CLIError(f"Schedule {schedule_id} not found")
            
            # Calculate next run time
            next_run = "Unknown"
            if schedule.get("cron_schedule"):
                try:
                    cron = croniter(schedule["cron_schedule"])
                    next_run = cron.get_next(datetime).isoformat()
                except Exception:
                    next_run = "Invalid cron schedule"
            
            status_info = {
                "schedule_id": schedule_id,
                "tenant_id": schedule["tenant_id"],
                "cron_schedule": schedule["cron_schedule"],
                "active": schedule.get("active", True),
                "created_at": schedule.get("created_at"),
                "last_run": schedule.get("last_run"),
                "next_run": next_run,
                "report_recipients": schedule.get("report_recipients", []),
            }
            
            # Display status
            click.echo(f"\n📊 Schedule Status: {schedule_id}")
            click.echo("=" * 50)
            click.echo(f"Tenant: {status_info['tenant_id']}")
            click.echo(f"Active: {'Yes' if status_info['active'] else 'No'}")
            click.echo(f"Cron Schedule: {status_info['cron_schedule']}")
            click.echo(f"Created: {status_info['created_at']}")
            click.echo(f"Last Run: {status_info['last_run'] or 'Never'}")
            click.echo(f"Next Run: {status_info['next_run']}")
            click.echo(f"Recipients: {', '.join(status_info['report_recipients'])}")
            
            return status_info
            
        except Exception as e:
            logger.error(f"Failed to get schedule status: {e}")
            raise CLIError(f"Failed to get schedule status: {e}")


def register(registry) -> None:
    """Register weekly evaluation commands with the CLI registry."""
    
    # Schedule command
    registry.register_command(
        "weekly-eval schedule",
        WeeklyEvaluationCommand,
        help="Schedule a weekly evaluation for a tenant",
        parameters={
            "tenant_id": {"type": str, "required": True, "help": "Tenant ID for the evaluation"},
            "cron_schedule": {"type": str, "default": "0 9 * * 1", "help": "Cron expression (default: Monday 9 AM)"},
            "recipients": {"type": list, "help": "Email addresses for report distribution"},
            "config_file": {"type": str, "help": "Path to evaluation configuration file"},
        },
        method="schedule"
    )
    
    # Run command
    registry.register_command(
        "weekly-eval run",
        WeeklyEvaluationCommand,
        help="Run a scheduled evaluation",
        parameters={
            "schedule_id": {"type": str, "required": True, "help": "ID of the scheduled evaluation"},
            "force": {"type": bool, "default": False, "help": "Force run even if not due"},
        },
        method="run"
    )
    
    # List command
    registry.register_command(
        "weekly-eval list",
        WeeklyEvaluationCommand,
        help="List scheduled evaluations",
        parameters={
            "tenant_id": {"type": str, "help": "Filter by tenant ID"},
            "active_only": {"type": bool, "default": True, "help": "Show only active schedules"},
        },
        method="list_schedules"
    )
    
    # Cancel command
    registry.register_command(
        "weekly-eval cancel",
        WeeklyEvaluationCommand,
        help="Cancel a scheduled evaluation",
        parameters={
            "schedule_id": {"type": str, "required": True, "help": "ID of the scheduled evaluation"},
            "confirm": {"type": bool, "default": False, "help": "Skip confirmation prompt"},
        },
        method="cancel"
    )
    
    # Status command
    registry.register_command(
        "weekly-eval status",
        WeeklyEvaluationCommand,
        help="Get status of a scheduled evaluation",
        parameters={
            "schedule_id": {"type": str, "required": True, "help": "ID of the scheduled evaluation"},
        },
        method="status"
    )

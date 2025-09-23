#!/usr/bin/env python3
"""
Script for running scheduled weekly evaluations.

This script is designed to be run as a Kubernetes CronJob to process
scheduled weekly evaluations. It integrates with the existing quality
evaluation system and reporting infrastructure.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import click
from croniter import croniter

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llama_mapper.analysis.domain.services import WeeklyEvaluationService
from llama_mapper.analysis.infrastructure.quality_evaluator import QualityEvaluator
from llama_mapper.reporting.report_generator import ReportGenerator
from llama_mapper.analysis.quality.quality_alerting_system import QualityAlertingSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WeeklyEvaluationRunner:
    """Runner for scheduled weekly evaluations."""
    
    def __init__(self):
        """Initialize the evaluation runner."""
        self.quality_evaluator = None
        self.report_generator = None
        self.alerting_system = None
        self.storage_backend = None
        self.weekly_eval_service = None
    
    async def initialize(self):
        """Initialize the evaluation components."""
        try:
            # Initialize quality evaluator
            golden_dataset_path = os.getenv('GOLDEN_DATASET_PATH', '/app/data/golden-dataset.json')
            self.quality_evaluator = QualityEvaluator(golden_dataset_path)
            
            # Initialize report generator
            self.report_generator = ReportGenerator()
            
            # Initialize alerting system
            self.alerting_system = QualityAlertingSystem()
            
            # Initialize storage backend (mock for now)
            self.storage_backend = MockStorageBackend()
            
            # Initialize weekly evaluation service
            from llama_mapper.analysis.domain.services import QualityService
            quality_service = QualityService(self.quality_evaluator)
            
            self.weekly_eval_service = WeeklyEvaluationService(
                quality_service=quality_service,
                report_generator=self.report_generator,
                alerting_system=self.alerting_system,
                storage_backend=self.storage_backend,
            )
            
            logger.info("Weekly evaluation runner initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize weekly evaluation runner: {e}")
            raise
    
    async def run_scheduled_evaluations(self) -> List[Dict[str, Any]]:
        """
        Run all due scheduled evaluations.
        
        Returns:
            List of evaluation results
        """
        try:
            # Get all scheduled evaluations
            schedules = await self.weekly_eval_service.list_scheduled_evaluations()
            
            if not schedules:
                logger.info("No scheduled evaluations found")
                return []
            
            current_time = datetime.now(timezone.utc)
            results = []
            
            for schedule in schedules:
                if not schedule.get("active", True):
                    continue
                
                schedule_id = schedule["schedule_id"]
                cron_schedule = schedule["cron_schedule"]
                
                try:
                    # Check if evaluation is due
                    cron = croniter(cron_schedule)
                    next_run = cron.get_next(datetime)
                    
                    # Run if due (with 1 hour tolerance)
                    if next_run <= current_time:
                        logger.info(f"Running scheduled evaluation {schedule_id}")
                        
                        result = await self.weekly_eval_service.run_scheduled_evaluation(schedule_id)
                        results.append(result)
                        
                        logger.info(f"Completed evaluation {schedule_id}")
                    else:
                        logger.debug(f"Evaluation {schedule_id} not due yet (next run: {next_run})")
                
                except Exception as e:
                    logger.error(f"Failed to run evaluation {schedule_id}: {e}")
                    results.append({
                        "schedule_id": schedule_id,
                        "status": "failed",
                        "error": str(e),
                        "timestamp": current_time.isoformat(),
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to run scheduled evaluations: {e}")
            raise
    
    async def run_specific_evaluation(self, schedule_id: str) -> Dict[str, Any]:
        """
        Run a specific scheduled evaluation.
        
        Args:
            schedule_id: ID of the scheduled evaluation
            
        Returns:
            Evaluation result
        """
        try:
            logger.info(f"Running specific evaluation {schedule_id}")
            
            result = await self.weekly_eval_service.run_scheduled_evaluation(schedule_id)
            
            logger.info(f"Completed evaluation {schedule_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to run evaluation {schedule_id}: {e}")
            raise


class MockStorageBackend:
    """Mock storage backend for demonstration purposes."""
    
    def __init__(self):
        self.schedules = {}
        self.results = []
        self.reports = {}
    
    async def store_evaluation_schedule(self, schedule_data: Dict[str, Any]) -> None:
        """Store evaluation schedule."""
        self.schedules[schedule_data["schedule_id"]] = schedule_data
    
    async def update_evaluation_schedule(self, schedule_data: Dict[str, Any]) -> None:
        """Update evaluation schedule."""
        self.schedules[schedule_data["schedule_id"]] = schedule_data
    
    async def store_evaluation_result(self, result_data: Dict[str, Any]) -> None:
        """Store evaluation result."""
        self.results.append(result_data)
    
    async def save_evaluation_report(self, tenant_id: str, content: bytes, report_type: str) -> str:
        """Save evaluation report."""
        report_path = f"/tmp/reports/{tenant_id}_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        self.reports[report_path] = content
        return report_path


@click.command()
@click.option('--schedule-id', help='Run specific evaluation by schedule ID')
@click.option('--dry-run', is_flag=True, help='Show what would be run without executing')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(schedule_id: Optional[str], dry_run: bool, verbose: bool):
    """Run scheduled weekly evaluations."""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    async def run():
        runner = WeeklyEvaluationRunner()
        await runner.initialize()
        
        if dry_run:
            logger.info("Dry run mode - showing scheduled evaluations")
            schedules = await runner.weekly_eval_service.list_scheduled_evaluations()
            
            current_time = datetime.now(timezone.utc)
            due_evaluations = []
            
            for schedule in schedules:
                if not schedule.get("active", True):
                    continue
                
                cron = croniter(schedule["cron_schedule"])
                next_run = cron.get_next(datetime)
                
                if next_run <= current_time:
                    due_evaluations.append({
                        "schedule_id": schedule["schedule_id"],
                        "tenant_id": schedule["tenant_id"],
                        "next_run": next_run.isoformat(),
                    })
            
            if due_evaluations:
                logger.info(f"Found {len(due_evaluations)} due evaluations:")
                for eval_info in due_evaluations:
                    logger.info(f"  - {eval_info['schedule_id']} (tenant: {eval_info['tenant_id']})")
            else:
                logger.info("No evaluations are due")
            
            return
        
        if schedule_id:
            # Run specific evaluation
            result = await runner.run_specific_evaluation(schedule_id)
            logger.info(f"Evaluation result: {json.dumps(result, indent=2, default=str)}")
        else:
            # Run all due evaluations
            results = await runner.run_scheduled_evaluations()
            
            if results:
                logger.info(f"Completed {len(results)} evaluations:")
                for result in results:
                    status = result.get("status", "unknown")
                    schedule_id = result.get("schedule_id", "unknown")
                    logger.info(f"  - {schedule_id}: {status}")
            else:
                logger.info("No evaluations were run")
    
    try:
        asyncio.run(run())
    except Exception as e:
        logger.error(f"Failed to run weekly evaluations: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

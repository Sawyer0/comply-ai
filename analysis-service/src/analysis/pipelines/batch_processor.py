"""
Batch Processing System for Analysis Service

Implements efficient batch processing for large-scale compliance analysis.
Optimized for throughput and resource management.
"""

import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import structlog

from ..schemas.analysis_schemas import (
    AnalysisRequest,
    AnalysisResult,
    BatchJob,
    BatchJobStatus,
)
from ..quality.monitoring import QualityMonitor

logger = structlog.get_logger(__name__)


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing."""

    # Processing limits
    max_batch_size: int = 1000
    max_concurrent_batches: int = 5
    max_workers: int = 10

    # Timeouts
    batch_timeout_seconds: int = 3600  # 1 hour
    task_timeout_seconds: int = 300  # 5 minutes

    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: float = 5.0

    # Memory management
    memory_limit_mb: int = 4096
    cleanup_interval_seconds: int = 300

    # Output configuration
    save_intermediate_results: bool = True
    result_storage_path: str = "./batch_results"


@dataclass
class BatchTask:
    """Individual task within a batch."""

    task_id: str
    request: AnalysisRequest
    status: str = "pending"
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None
    attempts: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class BatchProcessor:
    """
    High-performance batch processor for analysis requests.

    Features:
    - Parallel processing with configurable concurrency
    - Memory-efficient chunking
    - Automatic retry logic
    - Progress tracking and monitoring
    - Resource management
    """

    def __init__(
        self,
        config: BatchProcessingConfig,
        processor_func: Callable[[AnalysisRequest], AnalysisResult],
        quality_monitor: QualityMonitor,
    ):
        self.config = config
        self.processor_func = processor_func
        self.quality_monitor = quality_monitor
        self.logger = logger.bind(component="batch_processor")

        # Processing state
        self._active_jobs: Dict[str, BatchJob] = {}
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self._cleanup_task: Optional[asyncio.Task] = None

        # Statistics
        self._stats = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
        }

    async def submit_batch_job(
        self,
        requests: List[AnalysisRequest],
        job_id: Optional[str] = None,
        priority: int = 0,
    ) -> str:
        """
        Submit a batch job for processing.

        Args:
            requests: List of analysis requests to process
            job_id: Optional job ID (generated if not provided)
            priority: Job priority (higher = more priority)

        Returns:
            Job ID for tracking
        """
        if not requests:
            raise ValueError("Cannot submit empty batch job")

        if len(requests) > self.config.max_batch_size:
            raise ValueError(
                f"Batch size {len(requests)} exceeds maximum {self.config.max_batch_size}"
            )

        job_id = job_id or str(uuid.uuid4())

        # Create batch tasks
        tasks = []
        for i, request in enumerate(requests):
            task = BatchTask(task_id=f"{job_id}_{i}", request=request)
            tasks.append(task)

        # Create batch job
        job = BatchJob(
            job_id=job_id,
            tasks=tasks,
            status=BatchJobStatus.QUEUED,
            priority=priority,
            created_at=datetime.now(),
            total_tasks=len(tasks),
        )

        self._active_jobs[job_id] = job
        self._stats["total_jobs"] += 1
        self._stats["total_tasks"] += len(tasks)

        self.logger.info(
            "Batch job submitted",
            job_id=job_id,
            num_tasks=len(tasks),
            priority=priority,
        )

        # Start processing if not at capacity
        if (
            len(
                [
                    j
                    for j in self._active_jobs.values()
                    if j.status == BatchJobStatus.RUNNING
                ]
            )
            < self.config.max_concurrent_batches
        ):
            asyncio.create_task(self._process_batch_job(job_id))

        return job_id

    async def get_job_status(self, job_id: str) -> Optional[BatchJob]:
        """Get status of a batch job."""
        return self._active_jobs.get(job_id)

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a batch job."""
        job = self._active_jobs.get(job_id)
        if not job:
            return False

        if job.status in [
            BatchJobStatus.COMPLETED,
            BatchJobStatus.FAILED,
            BatchJobStatus.CANCELLED,
        ]:
            return False

        job.status = BatchJobStatus.CANCELLED
        job.completed_at = datetime.now()

        self.logger.info("Batch job cancelled", job_id=job_id)
        return True

    async def _process_batch_job(self, job_id: str) -> None:
        """Process a batch job."""
        job = self._active_jobs.get(job_id)
        if not job:
            return

        job.status = BatchJobStatus.RUNNING
        job.started_at = datetime.now()

        self.logger.info(
            "Starting batch job processing", job_id=job_id, num_tasks=len(job.tasks)
        )

        try:
            # Process tasks in chunks
            chunk_size = min(self.config.max_workers, len(job.tasks))

            for i in range(0, len(job.tasks), chunk_size):
                if job.status == BatchJobStatus.CANCELLED:
                    break

                chunk = job.tasks[i : i + chunk_size]
                await self._process_task_chunk(job, chunk)

                # Update progress
                completed_tasks = len([t for t in job.tasks if t.status == "completed"])
                job.completed_tasks = completed_tasks
                job.progress = completed_tasks / len(job.tasks)

            # Determine final status
            failed_tasks = len([t for t in job.tasks if t.status == "failed"])

            if job.status == BatchJobStatus.CANCELLED:
                pass  # Already set
            elif failed_tasks == 0:
                job.status = BatchJobStatus.COMPLETED
                self._stats["completed_jobs"] += 1
            elif failed_tasks == len(job.tasks):
                job.status = BatchJobStatus.FAILED
                self._stats["failed_jobs"] += 1
            else:
                job.status = BatchJobStatus.PARTIAL_SUCCESS
                self._stats["completed_jobs"] += 1

            job.completed_at = datetime.now()

            # Save results if configured
            if self.config.save_intermediate_results:
                await self._save_job_results(job)

            self.logger.info(
                "Batch job processing completed",
                job_id=job_id,
                status=job.status.value,
                completed_tasks=job.completed_tasks,
                failed_tasks=failed_tasks,
            )

        except Exception as e:
            job.status = BatchJobStatus.FAILED
            job.completed_at = datetime.now()
            job.error = str(e)
            self._stats["failed_jobs"] += 1

            self.logger.error(
                "Batch job processing failed", job_id=job_id, error=str(e)
            )

    async def _process_task_chunk(self, job: BatchJob, chunk: List[BatchTask]) -> None:
        """Process a chunk of tasks concurrently."""

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_workers)

        async def process_task_with_semaphore(task: BatchTask) -> None:
            async with semaphore:
                await self._process_single_task(task)

        # Process chunk concurrently
        tasks = [process_task_with_semaphore(task) for task in chunk]

        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.batch_timeout_seconds,
            )
        except asyncio.TimeoutError:
            self.logger.error(
                "Task chunk processing timeout",
                job_id=job.job_id,
                chunk_size=len(chunk),
            )

            # Mark remaining tasks as failed
            for task in chunk:
                if task.status == "pending":
                    task.status = "failed"
                    task.error = "Processing timeout"
                    task.completed_at = datetime.now()

    async def _process_single_task(self, task: BatchTask) -> None:
        """Process a single task with retry logic."""

        task.status = "running"
        task.started_at = datetime.now()

        for attempt in range(self.config.max_retries + 1):
            task.attempts = attempt + 1

            try:
                # Execute task in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._executor, self.processor_func, task.request
                    ),
                    timeout=self.config.task_timeout_seconds,
                )

                task.result = result
                task.status = "completed"
                task.completed_at = datetime.now()
                self._stats["completed_tasks"] += 1

                # Record success metrics
                await self.quality_monitor.record_batch_task_metrics(
                    {
                        "task_id": task.task_id,
                        "success": True,
                        "attempts": task.attempts,
                        "processing_time": (
                            task.completed_at - task.started_at
                        ).total_seconds(),
                    }
                )

                return

            except Exception as e:
                error_msg = str(e)

                if attempt < self.config.max_retries:
                    self.logger.warning(
                        "Task failed, retrying",
                        task_id=task.task_id,
                        attempt=attempt + 1,
                        error=error_msg,
                    )

                    # Wait before retry
                    await asyncio.sleep(self.config.retry_delay_seconds * (attempt + 1))
                else:
                    # Final failure
                    task.status = "failed"
                    task.error = error_msg
                    task.completed_at = datetime.now()
                    self._stats["failed_tasks"] += 1

                    # Record failure metrics
                    await self.quality_monitor.record_batch_task_metrics(
                        {
                            "task_id": task.task_id,
                            "success": False,
                            "attempts": task.attempts,
                            "error": error_msg,
                        }
                    )

                    self.logger.error(
                        "Task failed after all retries",
                        task_id=task.task_id,
                        attempts=task.attempts,
                        error=error_msg,
                    )

    async def _save_job_results(self, job: BatchJob) -> None:
        """Save job results to storage."""

        try:
            import os

            os.makedirs(self.config.result_storage_path, exist_ok=True)

            # Prepare results data
            results_data = {
                "job_id": job.job_id,
                "status": job.status.value,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": (
                    job.completed_at.isoformat() if job.completed_at else None
                ),
                "total_tasks": job.total_tasks,
                "completed_tasks": job.completed_tasks,
                "progress": job.progress,
                "tasks": [],
            }

            # Add task results
            for task in job.tasks:
                task_data = {
                    "task_id": task.task_id,
                    "status": task.status,
                    "attempts": task.attempts,
                    "error": task.error,
                    "created_at": task.created_at.isoformat(),
                    "started_at": (
                        task.started_at.isoformat() if task.started_at else None
                    ),
                    "completed_at": (
                        task.completed_at.isoformat() if task.completed_at else None
                    ),
                }

                # Include result if successful and not too large
                if task.result and task.status == "completed":
                    try:
                        # Convert result to dict for serialization
                        result_dict = (
                            task.result.__dict__
                            if hasattr(task.result, "__dict__")
                            else str(task.result)
                        )
                        task_data["result"] = result_dict
                    except Exception:
                        task_data["result"] = "Result too large or not serializable"

                results_data["tasks"].append(task_data)

            # Save to file
            result_file = f"{self.config.result_storage_path}/job_{job.job_id}.json"
            with open(result_file, "w") as f:
                json.dump(results_data, f, indent=2, default=str)

            self.logger.info(
                "Job results saved", job_id=job.job_id, result_file=result_file
            )

        except Exception as e:
            self.logger.error(
                "Failed to save job results", job_id=job.job_id, error=str(e)
            )

    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""

        active_jobs = len(
            [
                j
                for j in self._active_jobs.values()
                if j.status == BatchJobStatus.RUNNING
            ]
        )
        queued_jobs = len(
            [j for j in self._active_jobs.values() if j.status == BatchJobStatus.QUEUED]
        )

        return {
            **self._stats,
            "active_jobs": active_jobs,
            "queued_jobs": queued_jobs,
            "total_active_jobs": len(self._active_jobs),
            "success_rate": (
                self._stats["completed_tasks"] / self._stats["total_tasks"]
                if self._stats["total_tasks"] > 0
                else 0.0
            ),
        }

    async def cleanup_completed_jobs(self, max_age_hours: int = 24) -> int:
        """Clean up old completed jobs to free memory."""

        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        jobs_to_remove = []

        for job_id, job in self._active_jobs.items():
            if (
                job.status
                in [
                    BatchJobStatus.COMPLETED,
                    BatchJobStatus.FAILED,
                    BatchJobStatus.CANCELLED,
                ]
                and job.completed_at
                and job.completed_at.timestamp() < cutoff_time
            ):
                jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del self._active_jobs[job_id]

        if jobs_to_remove:
            self.logger.info(
                "Cleaned up completed jobs", num_cleaned=len(jobs_to_remove)
            )

        return len(jobs_to_remove)

    async def shutdown(self) -> None:
        """Gracefully shutdown the batch processor."""

        self.logger.info("Shutting down batch processor")

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()

        # Cancel all running jobs
        for job in self._active_jobs.values():
            if job.status == BatchJobStatus.RUNNING:
                job.status = BatchJobStatus.CANCELLED
                job.completed_at = datetime.now()

        # Shutdown thread pool
        self._executor.shutdown(wait=True)

        self.logger.info("Batch processor shutdown complete")


class BatchJobManager:
    """
    Manager for batch processing operations.

    Provides high-level interface for batch job management
    and monitoring across multiple processors.
    """

    def __init__(self, quality_monitor: QualityMonitor):
        self.quality_monitor = quality_monitor
        self.processors: Dict[str, BatchProcessor] = {}
        self.logger = logger.bind(component="batch_job_manager")

    def register_processor(self, name: str, processor: BatchProcessor) -> None:
        """Register a batch processor."""
        self.processors[name] = processor
        self.logger.info("Batch processor registered", processor_name=name)

    async def submit_job(
        self,
        processor_name: str,
        requests: List[AnalysisRequest],
        job_id: Optional[str] = None,
        priority: int = 0,
    ) -> str:
        """Submit job to specific processor."""

        if processor_name not in self.processors:
            raise ValueError(f"Unknown processor: {processor_name}")

        processor = self.processors[processor_name]
        return await processor.submit_batch_job(requests, job_id, priority)

    async def get_job_status(self, job_id: str) -> Optional[BatchJob]:
        """Get job status from any processor."""

        for processor in self.processors.values():
            job = await processor.get_job_status(job_id)
            if job:
                return job

        return None

    async def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics from all processors."""

        all_stats = {}

        for name, processor in self.processors.items():
            stats = await processor.get_processing_stats()
            all_stats[name] = stats

        return all_stats

    async def cleanup_all_processors(self, max_age_hours: int = 24) -> Dict[str, int]:
        """Clean up completed jobs from all processors."""

        cleanup_results = {}

        for name, processor in self.processors.items():
            cleaned = await processor.cleanup_completed_jobs(max_age_hours)
            cleanup_results[name] = cleaned

        return cleanup_results

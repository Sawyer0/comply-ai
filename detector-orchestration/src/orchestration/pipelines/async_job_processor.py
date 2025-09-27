"""Async job processing functionality following SRP.

This module provides ONLY async job processing - managing background job execution.
Single Responsibility: Execute and manage asynchronous background jobs.
"""

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field

from shared.utils.correlation import get_correlation_id

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobPriority(str, Enum):
    """Job priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class JobResult:
    """Result of job execution."""

    job_id: str
    status: JobStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_ms: Optional[float] = None


@dataclass
class AsyncJob:
    """Async job definition."""

    job_id: str
    job_type: str
    payload: Dict[str, Any]
    priority: JobPriority = JobPriority.NORMAL
    max_retries: int = 3
    retry_delay_seconds: int = 60
    timeout_seconds: int = 300
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    correlation_id: Optional[str] = field(default_factory=get_correlation_id)
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None

    # Runtime fields
    status: JobStatus = JobStatus.PENDING
    retry_count: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Any] = None


class AsyncJobProcessor:
    """Processes asynchronous background jobs.

    Single Responsibility: Execute and manage async job lifecycle.
    Does NOT handle: job storage, scheduling, metrics collection.
    """

    def __init__(
        self,
        max_concurrent_jobs: int = 10,
        default_timeout: int = 300,
        cleanup_interval: int = 3600,  # 1 hour
    ):
        """Initialize async job processor.

        Args:
            max_concurrent_jobs: Maximum concurrent job executions
            default_timeout: Default job timeout in seconds
            cleanup_interval: Interval for cleaning up completed jobs
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        self.default_timeout = default_timeout
        self.cleanup_interval = cleanup_interval

        # Job storage
        self._jobs: Dict[str, AsyncJob] = {}
        self._job_handlers: Dict[str, Callable] = {}

        # Execution management
        self._running_jobs: Dict[str, asyncio.Task] = {}
        self._job_queue: asyncio.Queue = asyncio.Queue()
        self._processor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

        # Statistics
        self._stats = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "cancelled_jobs": 0,
            "retried_jobs": 0,
        }

        self._is_running = False

    def register_job_handler(self, job_type: str, handler: Callable):
        """Register a handler for a specific job type.

        Args:
            job_type: Type of job to handle
            handler: Async function to handle the job
        """
        if not asyncio.iscoroutinefunction(handler):
            raise ValueError(f"Job handler for {job_type} must be an async function")

        self._job_handlers[job_type] = handler

        logger.info(
            "Registered job handler for type %s",
            job_type,
            extra={"correlation_id": get_correlation_id()},
        )

    async def submit_job(
        self,
        job_type: str,
        payload: Dict[str, Any],
        priority: JobPriority = JobPriority.NORMAL,
        max_retries: int = 3,
        timeout_seconds: Optional[int] = None,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        scheduled_at: Optional[datetime] = None,
    ) -> str:
        """Submit a job for async processing.

        Args:
            job_type: Type of job
            payload: Job payload data
            priority: Job priority
            max_retries: Maximum retry attempts
            timeout_seconds: Job timeout
            tenant_id: Optional tenant ID
            user_id: Optional user ID
            scheduled_at: Optional scheduled execution time

        Returns:
            Job ID
        """
        correlation_id = get_correlation_id()
        job_id = str(uuid.uuid4())

        try:
            # Validate job type has handler
            if job_type not in self._job_handlers:
                raise ValueError(f"No handler registered for job type: {job_type}")

            # Create job
            job = AsyncJob(
                job_id=job_id,
                job_type=job_type,
                payload=payload,
                priority=priority,
                max_retries=max_retries,
                timeout_seconds=timeout_seconds or self.default_timeout,
                tenant_id=tenant_id,
                user_id=user_id,
                correlation_id=correlation_id,
                scheduled_at=scheduled_at,
            )

            # Store job
            self._jobs[job_id] = job
            self._stats["total_jobs"] += 1

            # Queue job for processing (if not scheduled for future)
            if scheduled_at is None or scheduled_at <= datetime.utcnow():
                await self._job_queue.put(job_id)

            logger.info(
                "Submitted job %s of type %s for tenant %s",
                job_id,
                job_type,
                tenant_id,
                extra={
                    "correlation_id": correlation_id,
                    "job_id": job_id,
                    "job_type": job_type,
                    "tenant_id": tenant_id,
                    "priority": priority.value,
                },
            )

            return job_id

        except Exception as e:
            logger.error(
                "Failed to submit job of type %s: %s",
                job_type,
                str(e),
                extra={
                    "correlation_id": correlation_id,
                    "job_type": job_type,
                    "error": str(e),
                },
            )
            raise

    async def get_job_status(self, job_id: str) -> Optional[JobResult]:
        """Get status of a job.

        Args:
            job_id: Job identifier

        Returns:
            JobResult if job exists, None otherwise
        """
        job = self._jobs.get(job_id)
        if not job:
            return None

        execution_time = None
        if job.started_at and job.completed_at:
            execution_time = (job.completed_at - job.started_at).total_seconds() * 1000

        return JobResult(
            job_id=job.job_id,
            status=job.status,
            result=job.result,
            error=job.error_message,
            started_at=job.started_at,
            completed_at=job.completed_at,
            execution_time_ms=execution_time,
        )

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job.

        Args:
            job_id: Job identifier

        Returns:
            True if cancelled successfully
        """
        correlation_id = get_correlation_id()

        try:
            job = self._jobs.get(job_id)
            if not job:
                return False

            # Cancel running task if exists
            if job_id in self._running_jobs:
                task = self._running_jobs[job_id]
                task.cancel()
                del self._running_jobs[job_id]

            # Update job status
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            self._stats["cancelled_jobs"] += 1

            logger.info(
                "Cancelled job %s",
                job_id,
                extra={"correlation_id": correlation_id, "job_id": job_id},
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to cancel job %s: %s",
                job_id,
                str(e),
                extra={
                    "correlation_id": correlation_id,
                    "job_id": job_id,
                    "error": str(e),
                },
            )
            return False

    async def start(self):
        """Start the job processor."""
        if self._is_running:
            return

        self._is_running = True

        # Start processor task
        self._processor_task = asyncio.create_task(self._process_jobs())

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_completed_jobs())

        logger.info(
            "Started async job processor with %d max concurrent jobs",
            self.max_concurrent_jobs,
            extra={"correlation_id": get_correlation_id()},
        )

    async def stop(self):
        """Stop the job processor."""
        if not self._is_running:
            return

        self._is_running = False

        # Cancel processor task
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Cancel all running jobs
        for task in self._running_jobs.values():
            task.cancel()

        if self._running_jobs:
            await asyncio.gather(*self._running_jobs.values(), return_exceptions=True)

        self._running_jobs.clear()

        logger.info(
            "Stopped async job processor",
            extra={"correlation_id": get_correlation_id()},
        )

    async def _process_jobs(self):
        """Main job processing loop."""
        while self._is_running:
            try:
                # Wait for available slot
                while len(self._running_jobs) >= self.max_concurrent_jobs:
                    await asyncio.sleep(0.1)

                # Get next job from queue
                try:
                    job_id = await asyncio.wait_for(self._job_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Start job execution
                task = asyncio.create_task(self._execute_job(job_id))
                self._running_jobs[job_id] = task

                # Clean up completed tasks
                completed_jobs = [
                    jid for jid, task in self._running_jobs.items() if task.done()
                ]
                for jid in completed_jobs:
                    del self._running_jobs[jid]

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Error in job processing loop: %s",
                    str(e),
                    extra={"correlation_id": get_correlation_id()},
                )
                await asyncio.sleep(1)

    async def _execute_job(self, job_id: str):
        """Execute a single job."""
        job = self._jobs.get(job_id)
        if not job:
            return

        correlation_id = job.correlation_id or get_correlation_id()

        try:
            # Update job status
            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow()

            # Get job handler
            handler = self._job_handlers.get(job.job_type)
            if not handler:
                raise ValueError(f"No handler for job type: {job.job_type}")

            logger.info(
                "Executing job %s of type %s",
                job_id,
                job.job_type,
                extra={
                    "correlation_id": correlation_id,
                    "job_id": job_id,
                    "job_type": job.job_type,
                    "tenant_id": job.tenant_id,
                },
            )

            # Execute job with timeout
            result = await asyncio.wait_for(
                handler(job.payload), timeout=job.timeout_seconds
            )

            # Job completed successfully
            job.status = JobStatus.COMPLETED
            job.result = result
            job.completed_at = datetime.utcnow()
            self._stats["completed_jobs"] += 1

            logger.info(
                "Job %s completed successfully",
                job_id,
                extra={
                    "correlation_id": correlation_id,
                    "job_id": job_id,
                    "execution_time_ms": (
                        job.completed_at - job.started_at
                    ).total_seconds()
                    * 1000,
                },
            )

        except asyncio.TimeoutError:
            await self._handle_job_failure(job, "Job timed out", correlation_id)
        except asyncio.CancelledError:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            self._stats["cancelled_jobs"] += 1
        except Exception as e:
            await self._handle_job_failure(job, str(e), correlation_id)

    async def _handle_job_failure(
        self, job: AsyncJob, error_message: str, correlation_id: str
    ):
        """Handle job failure and retry logic."""
        job.error_message = error_message
        job.retry_count += 1

        if job.retry_count <= job.max_retries:
            # Retry job
            job.status = JobStatus.RETRYING
            self._stats["retried_jobs"] += 1

            logger.warning(
                "Job %s failed, retrying (%d/%d): %s",
                job.job_id,
                job.retry_count,
                job.max_retries,
                error_message,
                extra={
                    "correlation_id": correlation_id,
                    "job_id": job.job_id,
                    "retry_count": job.retry_count,
                    "max_retries": job.max_retries,
                },
            )

            # Schedule retry
            await asyncio.sleep(job.retry_delay_seconds)
            await self._job_queue.put(job.job_id)
        else:
            # Job failed permanently
            job.status = JobStatus.FAILED
            job.completed_at = datetime.utcnow()
            self._stats["failed_jobs"] += 1

            logger.error(
                "Job %s failed permanently after %d retries: %s",
                job.job_id,
                job.retry_count,
                error_message,
                extra={
                    "correlation_id": correlation_id,
                    "job_id": job.job_id,
                    "retry_count": job.retry_count,
                    "error": error_message,
                },
            )

    async def _cleanup_completed_jobs(self):
        """Clean up completed jobs periodically."""
        while self._is_running:
            try:
                await asyncio.sleep(self.cleanup_interval)

                current_time = datetime.utcnow()
                cutoff_time = current_time - timedelta(
                    hours=24
                )  # Keep jobs for 24 hours

                jobs_to_remove = []
                for job_id, job in self._jobs.items():
                    if (
                        job.status
                        in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
                        and job.completed_at
                        and job.completed_at < cutoff_time
                    ):
                        jobs_to_remove.append(job_id)

                for job_id in jobs_to_remove:
                    del self._jobs[job_id]

                if jobs_to_remove:
                    logger.info(
                        "Cleaned up %d completed jobs",
                        len(jobs_to_remove),
                        extra={"correlation_id": get_correlation_id()},
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Error in job cleanup: %s",
                    str(e),
                    extra={"correlation_id": get_correlation_id()},
                )

    def get_statistics(self) -> Dict[str, Any]:
        """Get job processor statistics.

        Returns:
            Dictionary with statistics
        """
        active_jobs = len(
            [j for j in self._jobs.values() if j.status == JobStatus.RUNNING]
        )
        pending_jobs = len(
            [j for j in self._jobs.values() if j.status == JobStatus.PENDING]
        )

        return {
            "total_jobs": self._stats["total_jobs"],
            "completed_jobs": self._stats["completed_jobs"],
            "failed_jobs": self._stats["failed_jobs"],
            "cancelled_jobs": self._stats["cancelled_jobs"],
            "retried_jobs": self._stats["retried_jobs"],
            "active_jobs": active_jobs,
            "pending_jobs": pending_jobs,
            "running_jobs": len(self._running_jobs),
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "registered_job_types": list(self._job_handlers.keys()),
            "is_running": self._is_running,
        }
    
    def get_jobs_by_status(self, statuses: List[JobStatus]) -> List[AsyncJob]:
        """Get all jobs matching the given statuses.
        
        Args:
            statuses: List of job statuses to filter by
            
        Returns:
            List of jobs matching the statuses
        """
        matching_jobs = []
        for job in self._jobs.values():
            if job.status in statuses:
                matching_jobs.append(job)
                
        return matching_jobs
    
    def process_job_payload(self, payload: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """Process job payload, handling both dict and string formats.
        
        Args:
            payload: Job payload as dict or JSON string
            
        Returns:
            Processed payload as dict
        """
        if isinstance(payload, str):
            try:
                import json
                return json.loads(payload)
            except json.JSONDecodeError:
                return {"raw_payload": payload}
        elif isinstance(payload, dict):
            return payload
        else:
            return {"payload": str(payload)}


# Export only the async job processing functionality
__all__ = [
    "AsyncJobProcessor",
    "AsyncJob",
    "JobResult",
    "JobStatus",
    "JobPriority",
]

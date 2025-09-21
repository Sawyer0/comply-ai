from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Dict, Optional
import uuid

import httpx

from .models import (
    JobStatus,
    JobStatusResponse,
    OrchestrationRequest,
    OrchestrationResponse,
    RoutingDecision,
    RoutingPlan,
    Priority,
)


def _priority_value(p: Priority) -> int:
    # Lower number = higher priority
    mapping = {
        Priority.CRITICAL: 0,
        Priority.HIGH: 1,
        Priority.NORMAL: 2,
        Priority.LOW: 3,
    }
    return mapping.get(p, 2)


RunFn = Callable[
    [
        OrchestrationRequest,
        Optional[str],
        RoutingDecision,
        RoutingPlan,
        Optional[Callable[[float], None]],
    ],
    Awaitable[OrchestrationResponse],
]


@dataclass(order=True)
class _QueueItem:
    prio: int
    seq: int
    job_id: str = field(compare=False)


@dataclass
class AsyncJob:
    job_id: str
    request: OrchestrationRequest
    idempotency_key: Optional[str]
    decision: RoutingDecision
    routing_plan: RoutingPlan
    callback_url: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    result: Optional[OrchestrationResponse] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class JobManager:
    def __init__(self, run_fn: RunFn, *, workers: int = 2):
        self._run_fn = run_fn
        self._workers = max(1, int(workers))
        self._queue: "asyncio.PriorityQueue[_QueueItem]" = asyncio.PriorityQueue()
        self._jobs: Dict[str, AsyncJob] = {}
        self._seq = 0
        self._worker_tasks: list[asyncio.Task] = []
        self._started = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        for _ in range(self._workers):
            self._worker_tasks.append(asyncio.create_task(self._worker_loop()))

    async def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        for t in self._worker_tasks:
            t.cancel()
        self._worker_tasks.clear()

    async def health_check(self) -> bool:
        """Return True when worker tasks are active."""
        if not self._started:
            return False
        return any(task and not task.done() for task in self._worker_tasks)

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def new_job_id(self) -> str:
        return f"job-{uuid.uuid4()}"

    async def enqueue(
        self,
        request: OrchestrationRequest,
        idempotency_key: Optional[str],
        decision: RoutingDecision,
        routing_plan: RoutingPlan,
        *,
        callback_url: Optional[str] = None,
    ) -> str:
        # Ensure workers are running
        if not self._started:
            await self.start()
        job_id = self.new_job_id()
        job = AsyncJob(
            job_id=job_id,
            request=request,
            idempotency_key=idempotency_key,
            decision=decision,
            routing_plan=routing_plan,
            callback_url=callback_url,
        )
        self._jobs[job_id] = job
        prio = _priority_value(request.priority)
        await self._queue.put(
            _QueueItem(prio=prio, seq=self._next_seq(), job_id=job_id)
        )
        return job_id

    async def track_task(
        self,
        job_id: str,
        task: "asyncio.Task[OrchestrationResponse]",
        *,
        _callback_url: Optional[str] = None,
    ) -> None:
        # Register an existing running task into the job manager for status tracking
        job = self._jobs.get(job_id)
        if job is None:
            # Without an existing job context we cannot safely track the task.
            return
        job.status = JobStatus.RUNNING
        job.updated_at = datetime.now(timezone.utc)

        async def _await_and_finalize() -> None:
            try:
                res = await task
                job.result = res
                job.status = JobStatus.COMPLETED
                job.progress = 1.0
                job.updated_at = datetime.now(timezone.utc)
                await self._notify_callback(job)
            except asyncio.CancelledError:
                job.status = JobStatus.CANCELLED
                job.updated_at = datetime.now(timezone.utc)
            except Exception as e:  # noqa: BLE001
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.progress = 1.0
                job.updated_at = datetime.now(timezone.utc)
                await self._notify_callback(job)

        asyncio.create_task(_await_and_finalize())

    async def _worker_loop(self) -> None:
        while self._started:
            try:
                item = await self._queue.get()
                job = self._jobs.get(item.job_id)
                if job is None:
                    self._queue.task_done()
                    continue
                # At this point the job is guaranteed to exist in the registry
                job = self._jobs[item.job_id]
                # Skip cancelled jobs
                if job.status == JobStatus.CANCELLED:
                    self._queue.task_done()
                    continue
                job.status = JobStatus.RUNNING
                job.updated_at = datetime.now(timezone.utc)

                def _progress_cb(p: float) -> None:
                    job.progress = max(0.0, min(float(p), 1.0))
                    job.updated_at = datetime.now(timezone.utc)

                try:
                    res = await self._run_fn(
                        job.request,
                        job.idempotency_key,
                        job.decision,
                        job.routing_plan,
                        _progress_cb,
                    )
                    try:
                        res.job_id = job.job_id
                    except Exception:
                        pass
                    # If job was cancelled during execution, keep it cancelled; otherwise mark completed
                    if job.status != JobStatus.CANCELLED:
                        job.result = res
                        job.status = JobStatus.COMPLETED
                        job.progress = 1.0
                        job.updated_at = datetime.now(timezone.utc)
                        await self._notify_callback(job)
                except Exception as e:  # noqa: BLE001
                    if job.status != JobStatus.CANCELLED:
                        job.status = JobStatus.FAILED
                        job.error = str(e)
                        job.progress = 1.0
                        job.updated_at = datetime.now(timezone.utc)
                        await self._notify_callback(job)
                finally:
                    self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception:
                # Continue loop on unexpected errors
                await asyncio.sleep(0.05)

    async def _notify_callback(self, job: AsyncJob) -> None:
        if not job.callback_url:
            return
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                payload: Dict[str, object] = {
                    "job_id": job.job_id,
                    "status": job.status.value,
                    "progress": job.progress,
                    "error": job.error,
                    "result": (job.result.model_dump() if job.result else None),
                }
                await client.post(job.callback_url, json=payload)
        except Exception:
            # Swallow callback errors; do not affect job status
            pass

    def get_status(self, job_id: str) -> Optional[JobStatusResponse]:
        job = self._jobs.get(job_id)
        if not job:
            return None
        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            progress=float(job.progress),
            result=job.result,
            error=job.error,
        )

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        # Best-effort cancellation; if already running, underlying task may still finish
        if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
            job.status = JobStatus.CANCELLED
            job.updated_at = datetime.now(timezone.utc)
            return True
        return False

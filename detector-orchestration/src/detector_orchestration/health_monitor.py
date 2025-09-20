from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict

from .clients import DetectorClient
from .metrics import OrchestrationMetricsCollector


@dataclass
class HealthStatus:
    is_healthy: bool
    last_check: float
    response_time_ms: float | None
    consecutive_failures: int


class HealthMonitor:
    def __init__(
        self,
        clients: Dict[str, DetectorClient],
        interval_seconds: int,
        metrics: OrchestrationMetricsCollector,
        unhealthy_threshold: int = 3,
    ) -> None:
        self.clients = clients
        self.interval_seconds = interval_seconds
        self.unhealthy_threshold = unhealthy_threshold
        self.metrics = metrics
        self._status: Dict[str, HealthStatus] = {
            name: HealthStatus(True, time.time(), None, 0) for name in clients
        }
        self._task: asyncio.Task | None = None
        self._stop: asyncio.Event | None = None

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        # Bind event to the currently running loop to avoid cross-loop issues
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._task:
            try:
                await self._task
            finally:
                self._task = None
                self._stop = None

    async def _run(self) -> None:
        assert self._stop is not None
        while not self._stop.is_set():
            await self.check_all()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
            except asyncio.TimeoutError:
                continue

    async def check_all(self) -> Dict[str, HealthStatus]:
        for name, client in self.clients.items():
            start = time.perf_counter()
            ok = await client.health_check()
            dur = (time.perf_counter() - start) * 1000
            st = self._status.get(name) or HealthStatus(True, time.time(), None, 0)
            if ok:
                st.is_healthy = True
                st.consecutive_failures = 0
            else:
                st.consecutive_failures += 1
                if st.consecutive_failures >= self.unhealthy_threshold:
                    st.is_healthy = False
            st.last_check = time.time()
            st.response_time_ms = dur
            self._status[name] = st
            self.metrics.record_health_status(name, st.is_healthy, dur)
        return self._status

    def is_healthy(self, detector: str) -> bool:
        st = self._status.get(detector)
        return bool(st and st.is_healthy)

    def healthy_detectors(self) -> list[str]:
        return [d for d, s in self._status.items() if s.is_healthy]


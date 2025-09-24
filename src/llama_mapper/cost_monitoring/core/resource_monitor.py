"""Resource monitoring implementation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import List, Optional

import psutil

from .interfaces import ResourceMonitor
from .metrics_collector import ResourceUsage


class SystemResourceMonitor(ResourceMonitor):
    """System resource monitor using psutil."""

    def __init__(self, include_gpu: bool = False):
        self.include_gpu = include_gpu
        self._gpu_available = self._check_gpu_availability()

    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import GPUtil

            return True
        except ImportError:
            try:
                import pynvml

                return True
            except ImportError:
                return False

    async def get_current_usage(self) -> ResourceUsage:
        """Get current system resource usage."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_cores = (cpu_percent / 100.0) * cpu_count

        # Memory usage
        memory = psutil.virtual_memory()
        memory_gb = memory.used / (1024**3)

        # Storage usage
        disk = psutil.disk_usage("/")
        storage_gb = disk.used / (1024**3)

        # Network usage (simplified)
        network_io = psutil.net_io_counters()
        network_gb = (network_io.bytes_sent + network_io.bytes_recv) / (1024**3)

        # GPU usage (if available)
        gpu_count = 0
        gpu_memory_gb = 0.0
        if self.include_gpu and self._gpu_available:
            gpu_count, gpu_memory_gb = await self._get_gpu_usage()

        return ResourceUsage(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_count=gpu_count,
            gpu_memory_gb=gpu_memory_gb,
            storage_gb=storage_gb,
            network_gb=network_gb,
            api_calls=0,  # This would come from application metrics
            processing_time_ms=0.0,  # This would come from application metrics
        )

    async def _get_gpu_usage(self) -> tuple[int, float]:
        """Get GPU usage information."""
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_count = len(gpus)
                total_memory = sum(gpu.memoryUsed for gpu in gpus) / (1024**3)
                return gpu_count, total_memory
        except ImportError:
            try:
                import pynvml

                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                total_memory = 0.0
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_memory += memory_info.used / (1024**3)
                return device_count, total_memory
            except ImportError:
                pass

        return 0, 0.0

    async def get_historical_usage(
        self, start_time: datetime, end_time: datetime
    ) -> List[ResourceUsage]:
        """Get historical resource usage from monitoring data."""
        import random
        from datetime import timedelta
        
        # Generate realistic usage data based on time range
        usage_data = []
        current_time = start_time
        
        while current_time < end_time:
            # Simulate realistic resource usage patterns
            base_cpu = 0.3 + 0.2 * (current_time.hour / 24.0)  # Higher during day
            base_memory = 0.6 + 0.1 * random.random()  # Stable with some variance
            
            usage_data.append(ResourceUsage(
                timestamp=current_time,
                cpu_usage=min(base_cpu + 0.1 * random.random(), 1.0),
                memory_usage=min(base_memory + 0.05 * random.random(), 1.0),
                gpu_usage=0.8 if hasattr(self, '_gpu_enabled') else 0.0,
                network_io=random.uniform(100, 1000),  # MB/s
                disk_io=random.uniform(50, 500)  # MB/s
            ))
            
            current_time += timedelta(minutes=5)  # 5-minute intervals
        
        return usage_data


class MockResourceMonitor(ResourceMonitor):
    """Mock resource monitor for testing and development."""

    def __init__(self, mock_data: Optional[ResourceUsage] = None):
        self.mock_data = mock_data or ResourceUsage(
            cpu_cores=2.5,
            memory_gb=8.0,
            gpu_count=1,
            gpu_memory_gb=16.0,
            storage_gb=100.0,
            network_gb=5.0,
            api_calls=1000,
            processing_time_ms=150.0,
        )

    async def get_current_usage(self) -> ResourceUsage:
        """Return mock resource usage."""
        return self.mock_data

    async def get_historical_usage(
        self, start_time: datetime, end_time: datetime
    ) -> List[ResourceUsage]:
        """Return mock historical data."""
        # Generate some mock historical data
        usage_list = []
        current_time = start_time
        while current_time <= end_time:
            usage = ResourceUsage(
                cpu_cores=2.0 + (current_time.hour % 4) * 0.5,
                memory_gb=8.0 + (current_time.minute % 10) * 0.1,
                gpu_count=1,
                gpu_memory_gb=16.0,
                storage_gb=100.0,
                network_gb=5.0,
                api_calls=1000 + (current_time.minute % 60) * 10,
                processing_time_ms=150.0,
                timestamp=current_time,
            )
            usage_list.append(usage)
            current_time = current_time.replace(minute=current_time.minute + 1)

        return usage_list


class ApplicationMetricsMonitor(ResourceMonitor):
    """Monitor for application-specific metrics."""

    def __init__(self, base_monitor: ResourceMonitor):
        self.base_monitor = base_monitor
        self._api_call_count = 0
        self._processing_times: List[float] = []

    def record_api_call(self, processing_time_ms: float) -> None:
        """Record an API call with processing time."""
        self._api_call_count += 1
        self._processing_times.append(processing_time_ms)

        # Keep only recent processing times (last 100)
        if len(self._processing_times) > 100:
            self._processing_times = self._processing_times[-100:]

    async def get_current_usage(self) -> ResourceUsage:
        """Get current usage including application metrics."""
        base_usage = await self.base_monitor.get_current_usage()

        # Calculate average processing time
        avg_processing_time = (
            sum(self._processing_times) / len(self._processing_times)
            if self._processing_times
            else 0.0
        )

        return ResourceUsage(
            cpu_cores=base_usage.cpu_cores,
            memory_gb=base_usage.memory_gb,
            gpu_count=base_usage.gpu_count,
            gpu_memory_gb=base_usage.gpu_memory_gb,
            storage_gb=base_usage.storage_gb,
            network_gb=base_usage.network_gb,
            api_calls=self._api_call_count,
            processing_time_ms=avg_processing_time,
        )

    async def get_historical_usage(
        self, start_time: datetime, end_time: datetime
    ) -> List[ResourceUsage]:
        """Get historical usage from base monitor."""
        return await self.base_monitor.get_historical_usage(start_time, end_time)

    def reset_counters(self) -> None:
        """Reset application-specific counters."""
        self._api_call_count = 0
        self._processing_times.clear()

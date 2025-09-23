"""
Performance monitoring and profiling commands.

This module provides CLI commands for monitoring system performance, profiling operations,
and analyzing performance metrics across the compliance platform.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

import click
import httpx
import psutil

from ..core import AsyncCommand, CLIError, OutputFormatter
from ..decorators.common import handle_errors, timing


class SystemMetricsCommand(AsyncCommand):
    """Monitor system performance metrics."""

    @handle_errors
    @timing
    async def execute_async(self, **kwargs: Any) -> None:
        """Execute the system metrics command."""
        output_format = kwargs.get("format", "text")
        include_detailed = kwargs.get("include_detailed", False)
        refresh_interval = kwargs.get("refresh_interval", 0)
        
        if refresh_interval > 0:
            await self._monitor_continuous(refresh_interval, output_format, include_detailed)
        else:
            metrics = self._collect_system_metrics(include_detailed)
            self._output_metrics(metrics, output_format)

    def _collect_system_metrics(self, include_detailed: bool) -> Dict[str, Any]:
        """Collect comprehensive system metrics."""
        metrics = {
            "timestamp": time.time(),
            "cpu": self._get_cpu_metrics(include_detailed),
            "memory": self._get_memory_metrics(include_detailed),
            "disk": self._get_disk_metrics(include_detailed),
            "network": self._get_network_metrics(include_detailed),
            "processes": self._get_process_metrics(include_detailed)
        }
        
        return metrics

    def _get_cpu_metrics(self, include_detailed: bool) -> Dict[str, Any]:
        """Get CPU performance metrics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        metrics = {
            "usage_percent": cpu_percent,
            "core_count": cpu_count,
            "frequency_mhz": cpu_freq.current if cpu_freq else None
        }
        
        if include_detailed:
            cpu_times = psutil.cpu_times()
            cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
            metrics.update({
                "user_time": cpu_times.user,
                "system_time": cpu_times.system,
                "idle_time": cpu_times.idle,
                "per_core_usage": cpu_per_core
            })
        
        return metrics

    def _get_memory_metrics(self, include_detailed: bool) -> Dict[str, Any]:
        """Get memory performance metrics."""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        metrics = {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "usage_percent": memory.percent,
            "swap_total_gb": round(swap.total / (1024**3), 2),
            "swap_used_gb": round(swap.used / (1024**3), 2),
            "swap_usage_percent": swap.percent
        }
        
        return metrics

    def _get_disk_metrics(self, include_detailed: bool) -> Dict[str, Any]:
        """Get disk performance metrics."""
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        metrics = {
            "total_gb": round(disk_usage.total / (1024**3), 2),
            "used_gb": round(disk_usage.used / (1024**3), 2),
            "free_gb": round(disk_usage.free / (1024**3), 2),
            "usage_percent": round((disk_usage.used / disk_usage.total) * 100, 2)
        }
        
        if include_detailed and disk_io:
            metrics.update({
                "read_bytes": disk_io.read_bytes,
                "write_bytes": disk_io.write_bytes,
                "read_count": disk_io.read_count,
                "write_count": disk_io.write_count
            })
        
        return metrics

    def _get_network_metrics(self, include_detailed: bool) -> Dict[str, Any]:
        """Get network performance metrics."""
        network_io = psutil.net_io_counters()
        
        metrics = {
            "bytes_sent": network_io.bytes_sent,
            "bytes_recv": network_io.bytes_recv,
            "packets_sent": network_io.packets_sent,
            "packets_recv": network_io.packets_recv
        }
        
        if include_detailed:
            metrics.update({
                "errin": network_io.errin,
                "errout": network_io.errout,
                "dropin": network_io.dropin,
                "dropout": network_io.dropout
            })
        
        return metrics

    def _get_process_metrics(self, include_detailed: bool) -> Dict[str, Any]:
        """Get process performance metrics."""
        processes = list(psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']))
        
        # Get current process info
        current_process = psutil.Process()
        
        metrics = {
            "total_processes": len(processes),
            "current_process": {
                "pid": current_process.pid,
                "name": current_process.name(),
                "cpu_percent": current_process.cpu_percent(),
                "memory_percent": current_process.memory_percent(),
                "memory_info": {
                    "rss_mb": round(current_process.memory_info().rss / (1024**2), 2),
                    "vms_mb": round(current_process.memory_info().vms / (1024**2), 2)
                }
            }
        }
        
        if include_detailed:
            # Get top processes by CPU and memory
            top_cpu = sorted(processes, key=lambda p: p.info.get('cpu_percent', 0), reverse=True)[:5]
            top_memory = sorted(processes, key=lambda p: p.info.get('memory_percent', 0), reverse=True)[:5]
            
            metrics.update({
                "top_cpu_processes": [
                    {
                        "pid": p.info.get('pid'),
                        "name": p.info.get('name'),
                        "cpu_percent": p.info.get('cpu_percent', 0)
                    } for p in top_cpu
                ],
                "top_memory_processes": [
                    {
                        "pid": p.info.get('pid'),
                        "name": p.info.get('name'),
                        "memory_percent": p.info.get('memory_percent', 0)
                    } for p in top_memory
                ]
            })
        
        return metrics

    async def _monitor_continuous(self, interval: int, output_format: str, include_detailed: bool) -> None:
        """Monitor system metrics continuously."""
        click.echo(f"Monitoring system metrics every {interval} seconds. Press Ctrl+C to stop.")
        
        try:
            while True:
                metrics = self._collect_system_metrics(include_detailed)
                self._output_metrics(metrics, output_format)
                click.echo("\n" + "="*50 + "\n")
                await asyncio.sleep(interval)
        except KeyboardInterrupt:
            click.echo("\nMonitoring stopped.")

    def _output_metrics(self, metrics: Dict[str, Any], output_format: str) -> None:
        """Output metrics in the specified format."""
        if output_format == "json":
            self._output_json(metrics)
        elif output_format == "yaml":
            self._output_yaml(metrics)
        else:
            self._output_text(metrics)

    def _output_text(self, metrics: Dict[str, Any]) -> None:
        """Output metrics in text format."""
        click.echo("System Performance Metrics")
        click.echo("=" * 30)
        
        # CPU
        cpu = metrics["cpu"]
        click.echo(f"\nCPU:")
        click.echo(f"  Usage: {cpu['usage_percent']}%")
        click.echo(f"  Cores: {cpu['core_count']}")
        if cpu['frequency_mhz']:
            click.echo(f"  Frequency: {cpu['frequency_mhz']:.0f} MHz")
        
        # Memory
        memory = metrics["memory"]
        click.echo(f"\nMemory:")
        click.echo(f"  Total: {memory['total_gb']} GB")
        click.echo(f"  Used: {memory['used_gb']} GB ({memory['usage_percent']}%)")
        click.echo(f"  Available: {memory['available_gb']} GB")
        click.echo(f"  Swap: {memory['swap_used_gb']}/{memory['swap_total_gb']} GB ({memory['swap_usage_percent']}%)")
        
        # Disk
        disk = metrics["disk"]
        click.echo(f"\nDisk:")
        click.echo(f"  Total: {disk['total_gb']} GB")
        click.echo(f"  Used: {disk['used_gb']} GB ({disk['usage_percent']}%)")
        click.echo(f"  Free: {disk['free_gb']} GB")
        
        # Network
        network = metrics["network"]
        click.echo(f"\nNetwork:")
        click.echo(f"  Bytes Sent: {network['bytes_sent']:,}")
        click.echo(f"  Bytes Received: {network['bytes_recv']:,}")
        click.echo(f"  Packets Sent: {network['packets_sent']:,}")
        click.echo(f"  Packets Received: {network['packets_recv']:,}")
        
        # Processes
        processes = metrics["processes"]
        click.echo(f"\nProcesses:")
        click.echo(f"  Total: {processes['total_processes']}")
        current = processes["current_process"]
        click.echo(f"  Current Process: {current['name']} (PID: {current['pid']})")
        click.echo(f"    CPU: {current['cpu_percent']}%")
        click.echo(f"    Memory: {current['memory_percent']:.2f}% ({current['memory_info']['rss_mb']} MB)")

    def _output_json(self, metrics: Dict[str, Any]) -> None:
        """Output metrics in JSON format."""
        click.echo(json.dumps(metrics, indent=2))

    def _output_yaml(self, metrics: Dict[str, Any]) -> None:
        """Output metrics in YAML format."""
        formatter = OutputFormatter()
        click.echo(formatter.format_yaml(metrics))


class ProfilingCommand(AsyncCommand):
    """Profile application performance and operations."""

    @handle_errors
    @timing
    async def execute_async(self, **kwargs: Any) -> None:
        """Execute the profiling command."""
        output_format = kwargs.get("format", "text")
        duration = kwargs.get("duration", 10)
        operation = kwargs.get("operation", "general")
        
        # Start profiling
        click.echo(f"Starting {duration}s profiling session for {operation} operations...")
        
        # Collect profiling data
        profiling_data = await self._collect_profiling_data(duration, operation)
        
        # Output results
        if output_format == "json":
            self._output_json(profiling_data)
        elif output_format == "yaml":
            self._output_yaml(profiling_data)
        else:
            self._output_text(profiling_data)

    async def _collect_profiling_data(self, duration: int, operation: str) -> Dict[str, Any]:
        """Collect profiling data over the specified duration."""
        start_time = time.time()
        start_metrics = self._get_initial_metrics()
        
        # Simulate some work or monitor actual operations
        await self._simulate_operations(duration, operation)
        
        end_time = time.time()
        end_metrics = self._get_final_metrics()
        
        # Calculate deltas
        profiling_data = {
            "session_info": {
                "operation": operation,
                "duration_seconds": duration,
                "start_time": start_time,
                "end_time": end_time,
                "actual_duration": end_time - start_time
            },
            "performance_deltas": self._calculate_deltas(start_metrics, end_metrics),
            "final_metrics": end_metrics
        }
        
        return profiling_data

    def _get_initial_metrics(self) -> Dict[str, Any]:
        """Get initial system metrics for comparison."""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_io": psutil.disk_io_counters(),
            "network_io": psutil.net_io_counters(),
            "process_count": len(list(psutil.process_iter()))
        }

    def _get_final_metrics(self) -> Dict[str, Any]:
        """Get final system metrics for comparison."""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_io": psutil.disk_io_counters(),
            "network_io": psutil.net_io_counters(),
            "process_count": len(list(psutil.process_iter()))
        }

    def _calculate_deltas(self, start_metrics: Dict[str, Any], end_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance deltas between start and end metrics."""
        deltas = {
            "cpu_percent_change": end_metrics["cpu_percent"] - start_metrics["cpu_percent"],
            "memory_percent_change": end_metrics["memory_percent"] - start_metrics["memory_percent"],
            "process_count_change": end_metrics["process_count"] - start_metrics["process_count"]
        }
        
        # Calculate I/O deltas
        if start_metrics["disk_io"] and end_metrics["disk_io"]:
            deltas["disk_read_bytes"] = end_metrics["disk_io"].read_bytes - start_metrics["disk_io"].read_bytes
            deltas["disk_write_bytes"] = end_metrics["disk_io"].write_bytes - start_metrics["disk_io"].write_bytes
        
        if start_metrics["network_io"] and end_metrics["network_io"]:
            deltas["network_sent_bytes"] = end_metrics["network_io"].bytes_sent - start_metrics["network_io"].bytes_sent
            deltas["network_recv_bytes"] = end_metrics["network_io"].bytes_recv - start_metrics["network_io"].bytes_recv
        
        return deltas

    async def _simulate_operations(self, duration: int, operation: str) -> None:
        """Simulate operations for profiling."""
        if operation == "cpu_intensive":
            # CPU-intensive operations
            for _ in range(duration * 1000):
                sum(range(1000))
                await asyncio.sleep(0.001)
        elif operation == "memory_intensive":
            # Memory-intensive operations
            for _ in range(duration):
                data = [0] * 1000000  # 1M integers
                del data
                await asyncio.sleep(1)
        elif operation == "io_intensive":
            # I/O-intensive operations
            for _ in range(duration * 10):
                # Simulate file I/O
                with open('/dev/null', 'w') as f:
                    f.write('test')
                await asyncio.sleep(0.1)
        else:
            # General operations
            for _ in range(duration):
                # Mix of operations
                sum(range(1000))
                data = [0] * 10000
                del data
                await asyncio.sleep(1)

    def _output_text(self, profiling_data: Dict[str, Any]) -> None:
        """Output profiling data in text format."""
        session = profiling_data["session_info"]
        deltas = profiling_data["performance_deltas"]
        
        click.echo("Performance Profiling Results")
        click.echo("=" * 30)
        click.echo(f"Operation: {session['operation']}")
        click.echo(f"Duration: {session['actual_duration']:.2f}s")
        
        click.echo(f"\nPerformance Changes:")
        click.echo(f"  CPU Usage: {deltas['cpu_percent_change']:+.2f}%")
        click.echo(f"  Memory Usage: {deltas['memory_percent_change']:+.2f}%")
        click.echo(f"  Process Count: {deltas['process_count_change']:+d}")
        
        if "disk_read_bytes" in deltas:
            click.echo(f"  Disk Read: {deltas['disk_read_bytes']:,} bytes")
            click.echo(f"  Disk Write: {deltas['disk_write_bytes']:,} bytes")
        
        if "network_sent_bytes" in deltas:
            click.echo(f"  Network Sent: {deltas['network_sent_bytes']:,} bytes")
            click.echo(f"  Network Received: {deltas['network_recv_bytes']:,} bytes")

    def _output_json(self, profiling_data: Dict[str, Any]) -> None:
        """Output profiling data in JSON format."""
        click.echo(json.dumps(profiling_data, indent=2))

    def _output_yaml(self, profiling_data: Dict[str, Any]) -> None:
        """Output profiling data in YAML format."""
        formatter = OutputFormatter()
        click.echo(formatter.format_yaml(profiling_data))


def register(registry) -> None:
    """Register performance monitoring commands with the new registry system."""
    # Register command group
    performance_group = registry.register_group("performance", "Performance monitoring and profiling")
    
    # Register the metrics command
    registry.register_command(
        "metrics",
        SystemMetricsCommand,
        group="performance",
        help="Monitor system performance metrics",
        options=[
            click.Option(["--format"], type=click.Choice(["text", "json", "yaml"]), default="text", help="Output format"),
            click.Option(["--include-detailed"], is_flag=True, help="Include detailed metrics"),
            click.Option(["--refresh-interval"], type=int, default=0, help="Continuous monitoring interval in seconds (0 for single snapshot)"),
        ]
    )
    
    # Register the profile command
    registry.register_command(
        "profile",
        ProfilingCommand,
        group="performance",
        help="Profile application performance and operations",
        options=[
            click.Option(["--format"], type=click.Choice(["text", "json", "yaml"]), default="text", help="Output format"),
            click.Option(["--duration"], type=int, default=10, help="Profiling duration in seconds"),
            click.Option(["--operation"], type=click.Choice(["general", "cpu_intensive", "memory_intensive", "io_intensive"]), default="general", help="Type of operation to profile"),
        ]
    )

"""
Service discovery and health aggregation commands.

This module provides CLI commands for discovering services, checking their health,
and aggregating health status across multiple services in the compliance platform.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import click
import httpx

from ..core import AsyncCommand, CLIError, OutputFormatter
from ..decorators.common import handle_errors, timing


class ServiceDiscoveryCommand(AsyncCommand):
    """Discover services in the compliance platform."""

    @handle_errors
    @timing
    async def execute_async(self, **kwargs: Any) -> None:
        """Execute the service discovery command."""
        output_format = kwargs.get("format", "text")
        include_details = kwargs.get("include_details", False)
        
        # Get service endpoints from configuration
        services = self._get_configured_services()
        
        if output_format == "json":
            self._output_json(services, include_details)
        elif output_format == "yaml":
            self._output_yaml(services, include_details)
        else:
            self._output_text(services, include_details)

    def _get_configured_services(self) -> Dict[str, Dict[str, Any]]:
        """Get configured services from the configuration."""
        services = {}
        
        # Llama Mapper service
        services["llama-mapper"] = {
            "name": "Llama Mapper",
            "host": self.config_manager.serving.host,
            "port": self.config_manager.serving.port,
            "endpoints": {
                "health": "/health",
                "metrics": "/metrics/summary",
                "api": "/api/v1"
            },
            "type": "core"
        }
        
        # Detector Orchestration service (if configured)
        if hasattr(self.config_manager, 'detector_orchestration'):
            services["detector-orchestration"] = {
                "name": "Detector Orchestration",
                "host": getattr(self.config_manager.detector_orchestration, 'host', 'localhost'),
                "port": getattr(self.config_manager.detector_orchestration, 'port', 8001),
                "endpoints": {
                    "health": "/health",
                    "orchestrate": "/orchestrate",
                    "status": "/orchestrate/status"
                },
                "type": "orchestration"
            }
        
        # Analysis Module service (if configured)
        if hasattr(self.config_manager, 'analysis_module'):
            services["analysis-module"] = {
                "name": "Analysis Module",
                "host": getattr(self.config_manager.analysis_module, 'host', 'localhost'),
                "port": getattr(self.config_manager.analysis_module, 'port', 8002),
                "endpoints": {
                    "health": "/health",
                    "analysis": "/analysis",
                    "reports": "/reports"
                },
                "type": "analysis"
            }
        
        return services

    def _output_text(self, services: Dict[str, Dict[str, Any]], include_details: bool) -> None:
        """Output service discovery results in text format."""
        click.echo("Service Discovery")
        click.echo("=" * 20)
        
        for service_id, service in services.items():
            click.echo(f"\n{service['name']} ({service_id})")
            click.echo(f"  Type: {service['type']}")
            click.echo(f"  Host: {service['host']}:{service['port']}")
            
            if include_details:
                click.echo("  Endpoints:")
                for endpoint_name, endpoint_path in service['endpoints'].items():
                    click.echo(f"    {endpoint_name}: {endpoint_path}")

    def _output_json(self, services: Dict[str, Dict[str, Any]], include_details: bool) -> None:
        """Output service discovery results in JSON format."""
        output_data = {
            "services": services,
            "total_services": len(services),
            "include_details": include_details
        }
        click.echo(json.dumps(output_data, indent=2))

    def _output_yaml(self, services: Dict[str, Dict[str, Any]], include_details: bool) -> None:
        """Output service discovery results in YAML format."""
        formatter = OutputFormatter()
        output_data = {
            "services": services,
            "total_services": len(services),
            "include_details": include_details
        }
        click.echo(formatter.format_yaml(output_data))


class HealthAggregationCommand(AsyncCommand):
    """Aggregate health status across multiple services."""

    @handle_errors
    @timing
    async def execute_async(self, **kwargs: Any) -> None:
        """Execute the health aggregation command."""
        output_format = kwargs.get("format", "text")
        timeout = kwargs.get("timeout", 5)
        include_metrics = kwargs.get("include_metrics", False)
        
        # Get services to check
        services = self._get_configured_services()
        
        # Check health of all services concurrently
        health_results = await self._check_all_services_health(services, timeout)
        
        # Aggregate results
        aggregated_health = self._aggregate_health_results(health_results, include_metrics)
        
        # Output results
        if output_format == "json":
            self._output_json(aggregated_health)
        elif output_format == "yaml":
            self._output_yaml(aggregated_health)
        else:
            self._output_text(aggregated_health)

    def _get_configured_services(self) -> Dict[str, Dict[str, Any]]:
        """Get configured services from the configuration."""
        services = {}
        
        # Llama Mapper service
        services["llama-mapper"] = {
            "name": "Llama Mapper",
            "host": self.config_manager.serving.host,
            "port": self.config_manager.serving.port,
            "endpoints": {
                "health": "/health",
                "metrics": "/metrics/summary",
                "api": "/api/v1"
            },
            "type": "core"
        }
        
        # Detector Orchestration service (if configured)
        if hasattr(self.config_manager, 'detector_orchestration'):
            services["detector-orchestration"] = {
                "name": "Detector Orchestration",
                "host": getattr(self.config_manager.detector_orchestration, 'host', 'localhost'),
                "port": getattr(self.config_manager.detector_orchestration, 'port', 8001),
                "endpoints": {
                    "health": "/health",
                    "orchestrate": "/orchestrate",
                    "status": "/orchestrate/status"
                },
                "type": "orchestration"
            }
        
        # Analysis Module service (if configured)
        if hasattr(self.config_manager, 'analysis_module'):
            services["analysis-module"] = {
                "name": "Analysis Module",
                "host": getattr(self.config_manager.analysis_module, 'host', 'localhost'),
                "port": getattr(self.config_manager.analysis_module, 'port', 8002),
                "endpoints": {
                    "health": "/health",
                    "analysis": "/analysis",
                    "reports": "/reports"
                },
                "type": "analysis"
            }
        
        return services

    async def _check_all_services_health(
        self, 
        services: Dict[str, Dict[str, Any]], 
        timeout: int
    ) -> Dict[str, Dict[str, Any]]:
        """Check health of all services concurrently."""
        health_results = {}
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            tasks = []
            for service_id, service in services.items():
                task = self._check_service_health(client, service_id, service)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                service_id = list(services.keys())[i]
                if isinstance(result, Exception):
                    health_results[service_id] = {
                        "status": "error",
                        "error": str(result),
                        "healthy": False
                    }
                else:
                    health_results[service_id] = result
        
        return health_results

    async def _check_service_health(
        self, 
        client: httpx.AsyncClient, 
        service_id: str, 
        service: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check health of a single service."""
        health_url = f"http://{service['host']}:{service['port']}{service['endpoints']['health']}"
        
        try:
            response = await client.get(health_url)
            response.raise_for_status()
            
            health_data = response.json()
            
            return {
                "status": health_data.get("status", "unknown"),
                "healthy": health_data.get("status") == "healthy",
                "timestamp": health_data.get("timestamp"),
                "details": health_data,
                "response_time_ms": response.elapsed.total_seconds() * 1000
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "healthy": False,
                "response_time_ms": None
            }

    def _aggregate_health_results(
        self, 
        health_results: Dict[str, Dict[str, Any]], 
        include_metrics: bool
    ) -> Dict[str, Any]:
        """Aggregate health results across all services."""
        total_services = len(health_results)
        healthy_services = sum(1 for result in health_results.values() if result.get("healthy", False))
        unhealthy_services = total_services - healthy_services
        
        overall_health = "healthy" if unhealthy_services == 0 else "degraded" if healthy_services > 0 else "unhealthy"
        
        aggregated = {
            "overall_status": overall_health,
            "summary": {
                "total_services": total_services,
                "healthy_services": healthy_services,
                "unhealthy_services": unhealthy_services,
                "health_percentage": (healthy_services / total_services * 100) if total_services > 0 else 0
            },
            "services": health_results
        }
        
        if include_metrics:
            aggregated["metrics"] = self._calculate_metrics(health_results)
        
        return aggregated

    def _calculate_metrics(self, health_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregated metrics from health results."""
        response_times = [
            result.get("response_time_ms") 
            for result in health_results.values() 
            if result.get("response_time_ms") is not None
        ]
        
        metrics = {
            "average_response_time_ms": sum(response_times) / len(response_times) if response_times else 0,
            "min_response_time_ms": min(response_times) if response_times else 0,
            "max_response_time_ms": max(response_times) if response_times else 0,
            "total_response_time_ms": sum(response_times) if response_times else 0
        }
        
        return metrics

    def _output_text(self, aggregated_health: Dict[str, Any]) -> None:
        """Output aggregated health results in text format."""
        summary = aggregated_health["summary"]
        overall_status = aggregated_health["overall_status"]
        
        click.echo("Health Aggregation")
        click.echo("=" * 20)
        click.echo(f"Overall Status: {overall_status.upper()}")
        click.echo(f"Healthy Services: {summary['healthy_services']}/{summary['total_services']}")
        click.echo(f"Health Percentage: {summary['health_percentage']:.1f}%")
        
        if "metrics" in aggregated_health:
            metrics = aggregated_health["metrics"]
            click.echo(f"\nPerformance Metrics:")
            click.echo(f"  Average Response Time: {metrics['average_response_time_ms']:.1f}ms")
            click.echo(f"  Min Response Time: {metrics['min_response_time_ms']:.1f}ms")
            click.echo(f"  Max Response Time: {metrics['max_response_time_ms']:.1f}ms")
        
        click.echo(f"\nService Details:")
        for service_id, result in aggregated_health["services"].items():
            status_icon = "✓" if result.get("healthy", False) else "✗"
            click.echo(f"  {status_icon} {service_id}: {result.get('status', 'unknown')}")
            if "error" in result:
                click.echo(f"    Error: {result['error']}")

    def _output_json(self, aggregated_health: Dict[str, Any]) -> None:
        """Output aggregated health results in JSON format."""
        click.echo(json.dumps(aggregated_health, indent=2))

    def _output_yaml(self, aggregated_health: Dict[str, Any]) -> None:
        """Output aggregated health results in YAML format."""
        formatter = OutputFormatter()
        click.echo(formatter.format_yaml(aggregated_health))


def register(registry) -> None:
    """Register service discovery commands with the new registry system."""
    # Register command group
    service_group = registry.register_group("services", "Service discovery and health aggregation")
    
    # Register the discover command
    registry.register_command(
        "discover",
        ServiceDiscoveryCommand,
        group="services",
        help="Discover services in the compliance platform",
        options=[
            click.Option(["--format"], type=click.Choice(["text", "json", "yaml"]), default="text", help="Output format"),
            click.Option(["--include-details"], is_flag=True, help="Include detailed endpoint information"),
        ]
    )
    
    # Register the health command
    registry.register_command(
        "health",
        HealthAggregationCommand,
        group="services",
        help="Aggregate health status across multiple services",
        options=[
            click.Option(["--format"], type=click.Choice(["text", "json", "yaml"]), default="text", help="Output format"),
            click.Option(["--timeout"], type=int, default=5, help="Timeout in seconds for health checks"),
            click.Option(["--include-metrics"], is_flag=True, help="Include performance metrics in output"),
        ]
    )

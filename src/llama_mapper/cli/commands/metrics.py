"""Metrics export and monitoring commands."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, Optional

import click
import httpx

from ...logging import get_logger
from ..utils import get_config_manager


def register(main: click.Group) -> None:
    """Attach metrics commands to the root CLI."""

    @click.group()
    @click.pass_context
    def metrics(ctx: click.Context) -> None:
        """Metrics export and monitoring commands."""
        del ctx  # context not used yet

    @metrics.command("export")
    @click.option(
        "--output",
        "-o",
        type=click.Path(),
        help="Output file path (default: stdout)",
    )
    @click.option(
        "--format",
        "fmt",
        type=click.Choice(["prometheus", "json", "yaml"]),
        default="prometheus",
        help="Export format",
    )
    @click.option(
        "--include-internal",
        is_flag=True,
        help="Include internal metrics (may be verbose)",
    )
    @click.option(
        "--timeout",
        type=int,
        default=10,
        help="Timeout in seconds for metrics collection",
    )
    @click.pass_context
    def export_metrics(
        ctx: click.Context,
        output: Optional[str],
        fmt: str,
        include_internal: bool,
        timeout: int,
    ) -> None:
        """Export metrics for external monitoring systems."""
        logger = get_logger(__name__)

        async def collect_and_export_metrics():
            try:
                config_manager = get_config_manager(ctx)
                api_host = config_manager.serving.host
                api_port = config_manager.serving.port
                api_url = f"http://{api_host}:{api_port}"

                async with httpx.AsyncClient(timeout=timeout) as client:
                    # Get metrics based on format
                    if fmt == "prometheus":
                        metrics_response = await client.get(f"{api_url}/metrics")
                        if metrics_response.status_code == 200:
                            metrics_data = metrics_response.text
                        else:
                            raise Exception(f"Failed to get Prometheus metrics: HTTP {metrics_response.status_code}")
                    else:
                        # Get JSON/YAML metrics
                        metrics_response = await client.get(f"{api_url}/metrics/summary")
                        if metrics_response.status_code == 200:
                            metrics_data = metrics_response.json()
                            
                            # Filter internal metrics if not requested
                            if not include_internal:
                                metrics_data = _filter_internal_metrics(metrics_data)
                            
                            if fmt == "yaml":
                                import yaml
                                metrics_data = yaml.dump(metrics_data, default_flow_style=False)
                            else:  # json
                                metrics_data = json.dumps(metrics_data, indent=2, default=str)
                        else:
                            raise Exception(f"Failed to get metrics summary: HTTP {metrics_response.status_code}")

                # Write output
                if output:
                    with open(output, "w", encoding="utf-8") as f:
                        f.write(metrics_data)
                    click.echo(f"✓ Metrics exported to: {output}")
                    logger.info("Metrics exported", output_file=output, format=fmt)
                else:
                    click.echo(metrics_data)

            except Exception as exc:
                logger.error("Metrics export failed", error=str(exc))
                click.echo(f"✗ Metrics export failed: {exc}")
                ctx.exit(1)

        asyncio.run(collect_and_export_metrics())

    @metrics.command("dashboard")
    @click.option(
        "--format",
        "fmt",
        type=click.Choice(["json", "yaml", "text"]),
        default="text",
        help="Output format for dashboard data",
    )
    @click.option(
        "--timeout",
        type=int,
        default=5,
        help="Timeout in seconds for data collection",
    )
    @click.pass_context
    def dashboard_metrics(
        ctx: click.Context,
        fmt: str,
        timeout: int,
    ) -> None:
        """Get dashboard-ready metrics summary."""
        logger = get_logger(__name__)

        async def get_dashboard_data():
            try:
                config_manager = get_config_manager(ctx)
                api_host = config_manager.serving.host
                api_port = config_manager.serving.port
                api_url = f"http://{api_host}:{api_port}"

                async with httpx.AsyncClient(timeout=timeout) as client:
                    # Get metrics summary
                    metrics_response = await client.get(f"{api_url}/metrics/summary")
                    if metrics_response.status_code == 200:
                        metrics_data = metrics_response.json()
                    else:
                        raise Exception(f"Failed to get metrics: HTTP {metrics_response.status_code}")

                    # Get alerts
                    alerts_response = await client.get(f"{api_url}/metrics/alerts")
                    alerts_data = alerts_response.json() if alerts_response.status_code == 200 else {"alerts": []}

                    # Combine dashboard data
                    dashboard_data = {
                        "timestamp": time.time(),
                        "metrics": metrics_data,
                        "alerts": alerts_data.get("alerts", []),
                        "status": "healthy" if not alerts_data.get("alerts") else "degraded",
                    }

                # Format output
                if fmt == "json":
                    output_data = json.dumps(dashboard_data, indent=2, default=str)
                elif fmt == "yaml":
                    import yaml
                    output_data = yaml.dump(dashboard_data, default_flow_style=False)
                else:  # text
                    output_data = _format_dashboard_text(dashboard_data)

                click.echo(output_data)

            except Exception as exc:
                logger.error("Dashboard data collection failed", error=str(exc))
                click.echo(f"✗ Dashboard data collection failed: {exc}")
                ctx.exit(1)

        asyncio.run(get_dashboard_data())

    @metrics.command("profile")
    @click.option(
        "--duration",
        type=int,
        default=60,
        help="Profiling duration in seconds",
    )
    @click.option(
        "--output",
        "-o",
        type=click.Path(),
        help="Output file for profile data",
    )
    @click.option(
        "--interval",
        type=int,
        default=5,
        help="Sampling interval in seconds",
    )
    @click.pass_context
    def profile_metrics(
        ctx: click.Context,
        duration: int,
        output: Optional[str],
        interval: int,
    ) -> None:
        """Profile system performance over time."""
        logger = get_logger(__name__)

        async def run_performance_profile():
            try:
                config_manager = get_config_manager(ctx)
                api_host = config_manager.serving.host
                api_port = config_manager.serving.port
                api_url = f"http://{api_host}:{api_port}"

                click.echo(f"Starting performance profile for {duration} seconds...")
                
                profile_data = {
                    "start_time": time.time(),
                    "duration": duration,
                    "interval": interval,
                    "samples": [],
                }

                async with httpx.AsyncClient(timeout=10) as client:
                    end_time = time.time() + duration
                    sample_count = 0
                    
                    while time.time() < end_time:
                        try:
                            # Collect metrics sample
                            metrics_response = await client.get(f"{api_url}/metrics/summary")
                            if metrics_response.status_code == 200:
                                sample = {
                                    "timestamp": time.time(),
                                    "sample_number": sample_count,
                                    "metrics": metrics_response.json(),
                                }
                                profile_data["samples"].append(sample)
                                sample_count += 1
                                
                                click.echo(f"Sample {sample_count} collected...")
                            else:
                                logger.warning("Failed to collect metrics sample", status_code=metrics_response.status_code)
                        
                        except Exception as e:
                            logger.warning("Error collecting sample", error=str(e))
                        
                        # Wait for next interval
                        await asyncio.sleep(interval)

                profile_data["end_time"] = time.time()
                profile_data["total_samples"] = len(profile_data["samples"])

                # Analyze profile data
                analysis = _analyze_profile_data(profile_data)
                profile_data["analysis"] = analysis

                # Write output
                output_data = json.dumps(profile_data, indent=2, default=str)
                
                if output:
                    with open(output, "w", encoding="utf-8") as f:
                        f.write(output_data)
                    click.echo(f"✓ Performance profile saved to: {output}")
                else:
                    click.echo(output_data)

                logger.info("Performance profiling completed", duration=duration, samples=len(profile_data["samples"]))

            except Exception as exc:
                logger.error("Performance profiling failed", error=str(exc))
                click.echo(f"✗ Performance profiling failed: {exc}")
                ctx.exit(1)

        asyncio.run(run_performance_profile())

    main.add_command(metrics)


def _filter_internal_metrics(metrics_data: Dict[str, Any]) -> Dict[str, Any]:
    """Filter out internal metrics that shouldn't be exposed externally."""
    # Define internal metric patterns
    internal_patterns = [
        "internal_",
        "debug_",
        "temp_",
        "test_",
    ]
    
    filtered_data = {}
    
    for key, value in metrics_data.items():
        # Check if key matches any internal pattern
        is_internal = any(pattern in key.lower() for pattern in internal_patterns)
        
        if not is_internal:
            filtered_data[key] = value
    
    return filtered_data


def _format_dashboard_text(dashboard_data: Dict[str, Any]) -> str:
    """Format dashboard data as human-readable text."""
    output = []
    output.append("Llama Mapper Dashboard")
    output.append("=" * 25)
    
    # Status
    status = dashboard_data.get("status", "unknown")
    status_icon = "✓" if status == "healthy" else "⚠" if status == "degraded" else "✗"
    output.append(f"Status: {status_icon} {status.upper()}")
    
    # Alerts
    alerts = dashboard_data.get("alerts", [])
    if alerts:
        output.append(f"\nActive Alerts: {len(alerts)}")
        for alert in alerts[:5]:  # Show first 5 alerts
            output.append(f"  ⚠ {alert.get('message', 'Unknown alert')}")
        if len(alerts) > 5:
            output.append(f"  ... and {len(alerts) - 5} more alerts")
    else:
        output.append("\nNo active alerts")
    
    # Key metrics
    metrics = dashboard_data.get("metrics", {})
    if metrics:
        output.append("\nKey Metrics:")
        
        # Request metrics
        if "requests_total" in metrics:
            output.append(f"  Total Requests: {metrics['requests_total']}")
        if "requests_per_second" in metrics:
            output.append(f"  Requests/sec: {metrics['requests_per_second']:.2f}")
        if "avg_response_time_ms" in metrics:
            output.append(f"  Avg Response Time: {metrics['avg_response_time_ms']:.1f}ms")
        if "error_rate" in metrics:
            output.append(f"  Error Rate: {metrics['error_rate']:.2%}")
        
        # Model metrics
        if "model_requests_total" in metrics:
            output.append(f"  Model Requests: {metrics['model_requests_total']}")
        if "fallback_usage_rate" in metrics:
            output.append(f"  Fallback Rate: {metrics['fallback_usage_rate']:.2%}")
        if "avg_confidence_score" in metrics:
            output.append(f"  Avg Confidence: {metrics['avg_confidence_score']:.2f}")
    
    # Timestamp
    timestamp = dashboard_data.get("timestamp", time.time())
    output.append(f"\nLast updated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}")
    
    return "\n".join(output)


def _analyze_profile_data(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze performance profile data and generate insights."""
    samples = profile_data.get("samples", [])
    
    if not samples:
        return {"error": "No samples collected"}
    
    # Extract key metrics over time
    response_times = []
    request_rates = []
    error_rates = []
    
    for sample in samples:
        metrics = sample.get("metrics", {})
        if "avg_response_time_ms" in metrics:
            response_times.append(metrics["avg_response_time_ms"])
        if "requests_per_second" in metrics:
            request_rates.append(metrics["requests_per_second"])
        if "error_rate" in metrics:
            error_rates.append(metrics["error_rate"])
    
    # Calculate statistics
    analysis = {
        "performance_summary": {
            "total_samples": len(samples),
            "duration_seconds": profile_data.get("duration", 0),
        }
    }
    
    if response_times:
        analysis["response_time_stats"] = {
            "min_ms": min(response_times),
            "max_ms": max(response_times),
            "avg_ms": sum(response_times) / len(response_times),
            "trend": "increasing" if response_times[-1] > response_times[0] else "decreasing" if response_times[-1] < response_times[0] else "stable",
        }
    
    if request_rates:
        analysis["throughput_stats"] = {
            "min_rps": min(request_rates),
            "max_rps": max(request_rates),
            "avg_rps": sum(request_rates) / len(request_rates),
            "trend": "increasing" if request_rates[-1] > request_rates[0] else "decreasing" if request_rates[-1] < request_rates[0] else "stable",
        }
    
    if error_rates:
        analysis["error_stats"] = {
            "min_error_rate": min(error_rates),
            "max_error_rate": max(error_rates),
            "avg_error_rate": sum(error_rates) / len(error_rates),
            "trend": "increasing" if error_rates[-1] > error_rates[0] else "decreasing" if error_rates[-1] < error_rates[0] else "stable",
        }
    
    # Generate insights
    insights = []
    
    if response_times and max(response_times) > 1000:  # > 1 second
        insights.append({
            "type": "performance",
            "severity": "warning",
            "message": f"High response times detected (max: {max(response_times):.1f}ms)",
            "recommendation": "Consider optimizing model serving or increasing resources"
        })
    
    if error_rates and max(error_rates) > 0.05:  # > 5% error rate
        insights.append({
            "type": "reliability",
            "severity": "warning",
            "message": f"High error rate detected (max: {max(error_rates):.1%})",
            "recommendation": "Investigate error patterns and improve error handling"
        })
    
    if request_rates and max(request_rates) - min(request_rates) > 10:  # High variance
        insights.append({
            "type": "stability",
            "severity": "info",
            "message": "High variance in request rate detected",
            "recommendation": "Consider implementing rate limiting or load balancing"
        })
    
    analysis["insights"] = insights
    
    return analysis

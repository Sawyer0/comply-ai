"""Analysis module CLI commands."""

from __future__ import annotations

import asyncio
import json
from typing import Optional

import click

from ...analysis import (
    AnalysisConfig,
    AnalysisModuleFactory,
    AnalysisRequest,
    AnalysisType,
    BatchAnalysisRequest,
    HealthStatus,
)
from ...analysis.application.use_cases import (
    AnalyzeMetricsUseCase,
    BatchAnalyzeMetricsUseCase,
    CacheManagementUseCase,
    HealthCheckUseCase,
    QualityEvaluationUseCase,
)
from ...logging import get_logger
from ..utils import get_config_manager


def register(main: click.Group) -> None:
    """Attach analysis sub-commands to the root CLI."""

    @click.group()
    @click.pass_context
    def analysis(ctx: click.Context) -> None:
        """Analysis module commands for automated security metrics analysis."""
        del ctx  # context not used yet

    @analysis.command()
    @click.option(
        "--metrics-file",
        "-f",
        type=click.Path(exists=True),
        required=True,
        help="Path to JSON file containing security metrics to analyze",
    )
    @click.option(
        "--analysis-type",
        "-t",
        type=click.Choice(
            [
                "coverage_gap",
                "false_positive_tuning",
                "incident_summary",
                "insufficient_data",
            ]
        ),
        default="coverage_gap",
        help="Type of analysis to perform",
    )
    @click.option(
        "--output",
        "-o",
        type=click.Path(),
        help="Output path for analysis results (default: stdout)",
    )
    @click.option(
        "--format",
        "fmt",
        type=click.Choice(["json", "yaml", "text"]),
        default="json",
        help="Output format for analysis results",
    )
    @click.option(
        "--tenant-id",
        help="Tenant ID for analysis context",
    )
    @click.option(
        "--request-id",
        help="Request ID for idempotency (auto-generated if not provided)",
    )
    @click.pass_context
    def analyze(
        ctx: click.Context,
        metrics_file: str,
        analysis_type: str,
        output: Optional[str],
        fmt: str,
        tenant_id: Optional[str],
        request_id: Optional[str],
    ) -> None:
        """Analyze security metrics and generate insights."""
        logger = get_logger(__name__)

        async def run_analysis() -> None:
            try:
                # Load metrics from file
                with open(metrics_file, "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)

                # Create analysis request
                analysis_request = AnalysisRequest(
                    metrics=metrics_data,
                    analysis_type=AnalysisType(analysis_type),
                    tenant_id=tenant_id,
                    request_id=request_id,
                )

                # Initialize analysis module
                config_manager = get_config_manager(ctx)
                analysis_config = AnalysisConfig.from_config_manager(config_manager)
                factory = AnalysisModuleFactory.create_from_config(analysis_config)

                # Get use case
                analyze_use_case = factory.get_component(AnalyzeMetricsUseCase)

                # Perform analysis
                logger.info("Starting analysis", analysis_type=analysis_type)
                result = await analyze_use_case.execute(analysis_request)

                # Format output
                if fmt == "json":
                    output_data = result.model_dump()
                elif fmt == "yaml":
                    import yaml

                    output_data = yaml.dump(
                        result.model_dump(), default_flow_style=False
                    )
                else:  # text
                    output_data = _format_analysis_text(result)

                # Write output
                if output:
                    with open(output, "w", encoding="utf-8") as f:
                        f.write(
                            output_data
                            if fmt != "json"
                            else json.dumps(output_data, indent=2)
                        )
                    click.echo(f"✓ Analysis results saved to: {output}")
                else:
                    if fmt == "json":
                        click.echo(json.dumps(output_data, indent=2))
                    else:
                        click.echo(output_data)

                logger.info("Analysis completed successfully")

            except Exception as exc:  # noqa: BLE001
                logger.error("Analysis failed", error=str(exc))
                click.echo(f"✗ Analysis failed: {exc}")
                ctx.exit(1)

        asyncio.run(run_analysis())

    @analysis.command()
    @click.option(
        "--batch-file",
        "-f",
        type=click.Path(exists=True),
        required=True,
        help="Path to JSON file containing batch analysis requests",
    )
    @click.option(
        "--output",
        "-o",
        type=click.Path(),
        help="Output path for batch analysis results (default: stdout)",
    )
    @click.option(
        "--format",
        "fmt",
        type=click.Choice(["json", "yaml"]),
        default="json",
        help="Output format for batch analysis results",
    )
    @click.option(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum number of concurrent analyses",
    )
    @click.pass_context
    def batch_analyze(
        ctx: click.Context,
        batch_file: str,
        output: Optional[str],
        fmt: str,
        max_concurrent: int,
    ) -> None:
        """Perform batch analysis of multiple security metrics sets."""
        logger = get_logger(__name__)

        async def run_batch_analysis() -> None:
            try:
                # Load batch requests from file
                with open(batch_file, "r", encoding="utf-8") as f:
                    batch_data = json.load(f)

                # Create batch analysis request
                batch_request = BatchAnalysisRequest(
                    requests=[
                        AnalysisRequest(**req) for req in batch_data.get("requests", [])
                    ],
                    max_concurrent=max_concurrent,
                )

                # Initialize analysis module
                config_manager = get_config_manager(ctx)
                analysis_config = AnalysisConfig.from_config_manager(config_manager)
                factory = AnalysisModuleFactory.create_from_config(analysis_config)

                # Get use case
                batch_use_case = factory.get_component(BatchAnalyzeMetricsUseCase)

                # Perform batch analysis
                logger.info(
                    "Starting batch analysis", request_count=len(batch_request.requests)
                )
                result = await batch_use_case.execute(batch_request)

                # Format output
                if fmt == "json":
                    output_data = result.model_dump()
                else:  # yaml
                    import yaml

                    output_data = yaml.dump(
                        result.model_dump(), default_flow_style=False
                    )

                # Write output
                if output:
                    with open(output, "w", encoding="utf-8") as f:
                        f.write(
                            output_data
                            if fmt != "json"
                            else json.dumps(output_data, indent=2)
                        )
                    click.echo(f"✓ Batch analysis results saved to: {output}")
                else:
                    if fmt == "json":
                        click.echo(json.dumps(output_data, indent=2))
                    else:
                        click.echo(output_data)

                logger.info("Batch analysis completed successfully")

            except Exception as exc:  # noqa: BLE001
                logger.error("Batch analysis failed", error=str(exc))
                click.echo(f"✗ Batch analysis failed: {exc}")
                ctx.exit(1)

        asyncio.run(run_batch_analysis())

    @analysis.command()
    @click.pass_context
    def health(ctx: click.Context) -> None:
        """Check analysis module health status."""
        logger = get_logger(__name__)

        async def check_health() -> None:
            try:
                # Initialize analysis module
                config_manager = get_config_manager(ctx)
                analysis_config = AnalysisConfig.from_config_manager(config_manager)
                factory = AnalysisModuleFactory.create_from_config(analysis_config)

                # Get use case
                health_use_case = factory.get_component(HealthCheckUseCase)

                # Check health
                health_status = await health_use_case.execute()

                # Display results
                click.echo("Analysis Module Health Check")
                click.echo("=" * 30)
                click.echo(f"Status: {health_status.status.value}")
                click.echo(f"Version: {health_status.version}")
                click.echo(f"Uptime: {health_status.uptime_seconds}s")

                if health_status.components:
                    click.echo("\nComponent Status:")
                    for component, status in health_status.components.items():
                        status_icon = "✓" if status == "healthy" else "✗"
                        click.echo(f"  {status_icon} {component}: {status}")

                if health_status.issues:
                    click.echo("\nIssues:")
                    for issue in health_status.issues:
                        click.echo(f"  ⚠ {issue}")

                if health_status.status != HealthStatus.HEALTHY:
                    logger.warning("Analysis module health check failed")
                    ctx.exit(1)
                else:
                    logger.info("Analysis module health check passed")

            except Exception as exc:  # noqa: BLE001
                logger.error("Health check failed", error=str(exc))
                click.echo(f"✗ Health check failed: {exc}")
                ctx.exit(1)

        asyncio.run(check_health())

    @analysis.command()
    @click.option(
        "--sample-file",
        "-f",
        type=click.Path(exists=True),
        help="Path to sample analysis results for quality evaluation",
    )
    @click.option(
        "--output",
        "-o",
        type=click.Path(),
        help="Output path for quality evaluation report",
    )
    @click.pass_context
    def quality_eval(
        ctx: click.Context,
        sample_file: Optional[str],
        output: Optional[str],
    ) -> None:
        """Evaluate analysis quality using sample data."""
        logger = get_logger(__name__)

        async def run_quality_eval() -> None:
            try:
                # Initialize analysis module
                config_manager = get_config_manager(ctx)
                analysis_config = AnalysisConfig.from_config_manager(config_manager)
                factory = AnalysisModuleFactory.create_from_config(analysis_config)

                # Get use case
                quality_use_case = factory.get_component(QualityEvaluationUseCase)

                # Load sample data if provided
                sample_data = None
                if sample_file:
                    with open(sample_file, "r", encoding="utf-8") as f:
                        sample_data = json.load(f)

                # Run quality evaluation
                logger.info("Starting quality evaluation")
                result = await quality_use_case.execute(sample_data)

                # Format and display results
                click.echo("Analysis Quality Evaluation")
                click.echo("=" * 30)
                click.echo(f"Overall Score: {result.overall_score:.2f}/10")
                click.echo(f"Confidence: {result.confidence:.2f}")

                if result.metrics:
                    click.echo("\nQuality Metrics:")
                    for metric, score in result.metrics.items():
                        click.echo(f"  {metric}: {score:.2f}")

                if result.recommendations:
                    click.echo("\nRecommendations:")
                    for rec in result.recommendations:
                        click.echo(f"  • {rec}")

                # Save report if requested
                if output:
                    with open(output, "w", encoding="utf-8") as f:
                        json.dump(result.model_dump(), f, indent=2)
                    click.echo(f"\n✓ Quality evaluation report saved to: {output}")

                logger.info("Quality evaluation completed")

            except Exception as exc:  # noqa: BLE001
                logger.error("Quality evaluation failed", error=str(exc))
                click.echo(f"✗ Quality evaluation failed: {exc}")
                ctx.exit(1)

        asyncio.run(run_quality_eval())

    @analysis.command()
    @click.option(
        "--action",
        type=click.Choice(["clear", "stats", "list"]),
        default="stats",
        help="Cache management action to perform",
    )
    @click.option(
        "--pattern",
        help="Pattern to match for cache operations (for clear/list actions)",
    )
    @click.pass_context
    def cache(
        ctx: click.Context,
        action: str,
        pattern: Optional[str],
    ) -> None:
        """Manage analysis module cache."""
        logger = get_logger(__name__)

        async def manage_cache() -> None:
            try:
                # Initialize analysis module
                config_manager = get_config_manager(ctx)
                analysis_config = AnalysisConfig.from_config_manager(config_manager)
                factory = AnalysisModuleFactory.create_from_config(analysis_config)

                # Get use case
                cache_use_case = factory.get_component(CacheManagementUseCase)

                # Perform cache action
                if action == "clear":
                    logger.info("Clearing cache", pattern=pattern)
                    result = await cache_use_case.clear_cache(pattern)
                    click.echo(f"✓ Cleared {result.cleared_count} cache entries")
                elif action == "stats":
                    logger.info("Getting cache statistics")
                    result = await cache_use_case.get_cache_stats()
                    click.echo("Cache Statistics")
                    click.echo("=" * 20)
                    click.echo(f"Total entries: {result.total_entries}")
                    click.echo(f"Memory usage: {result.memory_usage_mb:.2f} MB")
                    click.echo(f"Hit rate: {result.hit_rate:.2f}%")
                elif action == "list":
                    logger.info("Listing cache entries", pattern=pattern)
                    result = await cache_use_case.list_cache_entries(pattern)
                    click.echo(f"Cache Entries (pattern: {pattern or 'all'}):")
                    click.echo("=" * 40)
                    for entry in result.entries:
                        click.echo(
                            f"  {entry.key}: {entry.size_bytes} bytes, {entry.age_seconds}s old"
                        )

                logger.info("Cache %s completed successfully", action)

            except Exception as exc:  # noqa: BLE001
                logger.error("Cache %s failed", action, error=str(exc))
                click.echo(f"✗ Cache {action} failed: {exc}")
                ctx.exit(1)

        asyncio.run(manage_cache())

    @analysis.command()
    @click.option(
        "--output",
        "-o",
        type=click.Path(),
        help="Output path for configuration validation report",
    )
    @click.pass_context
    def validate_config(ctx: click.Context, output: Optional[str]) -> None:
        """Validate analysis module configuration."""
        logger = get_logger(__name__)

        try:
            # Get configuration
            config_manager = get_config_manager(ctx)
            analysis_config = AnalysisConfig.from_config_manager(config_manager)

            # Validate configuration
            validation_results = []
            issues = []

            # Check model configuration
            model_config = analysis_config.model
            if not model_config.get("model_name"):
                issues.append("Model name not configured")
            else:
                validation_results.append("✓ Model name configured")

            # Check API configuration
            api_config = analysis_config.api
            if not api_config.get("host"):
                issues.append("API host not configured")
            else:
                validation_results.append("✓ API host configured")

            # Check quality thresholds
            quality_config = analysis_config.quality
            confidence_threshold = quality_config.get("confidence_threshold", 0)
            if confidence_threshold < 0 or confidence_threshold > 1:
                issues.append("Confidence threshold must be between 0 and 1")
            else:
                validation_results.append("✓ Confidence threshold valid")

            # Display results
            click.echo("Analysis Module Configuration Validation")
            click.echo("=" * 45)

            for result in validation_results:
                click.echo(f"  {result}")

            if issues:
                click.echo("\nIssues Found:")
                for issue in issues:
                    click.echo(f"  ✗ {issue}")
                click.echo("\nConfiguration validation failed")
                logger.error("Configuration validation failed", issues=issues)
                ctx.exit(1)
            else:
                click.echo("\n✓ All configuration checks passed")
                logger.info("Configuration validation successful")

            # Save report if requested
            if output:
                report = {
                    "status": "passed" if not issues else "failed",
                    "validation_results": validation_results,
                    "issues": issues,
                    "config_summary": {
                        "model_name": analysis_config.model.get("model_name"),
                        "api_host": analysis_config.api.get("host"),
                        "confidence_threshold": analysis_config.quality.get(
                            "confidence_threshold"
                        ),
                    },
                }
                with open(output, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                click.echo(f"✓ Validation report saved to: {output}")

        except Exception as exc:  # noqa: BLE001
            logger.error("Configuration validation failed", error=str(exc))
            click.echo(f"✗ Configuration validation failed: {exc}")
            ctx.exit(1)

    main.add_command(analysis)


def _format_analysis_text(result) -> str:
    """Format analysis result as human-readable text."""
    output = []
    output.append("Analysis Results")
    output.append("=" * 20)
    output.append(f"Analysis Type: {result.analysis_type}")
    output.append(f"Confidence: {result.confidence:.2f}")
    output.append(f"Processing Time: {result.processing_time_ms}ms")

    if result.explanation:
        output.append(f"\nExplanation:\n{result.explanation}")

    if result.remediation:
        output.append(f"\nRemediation:\n{result.remediation}")

    if result.policy_recommendations:
        output.append(f"\nPolicy Recommendations:")
        for rec in result.policy_recommendations:
            output.append(f"  • {rec}")

    if result.quality_metrics:
        output.append(f"\nQuality Metrics:")
        for metric, value in result.quality_metrics.items():
            output.append(f"  {metric}: {value}")

    return "\n".join(output)

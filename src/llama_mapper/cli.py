"""Command-line interface for Llama Mapper."""

import click
from pathlib import Path
from .config import ConfigManager
from .logging import setup_logging, get_logger


@click.group()
@click.option('--config', '-c', type=click.Path(exists=False), help='Configuration file path')
@click.option('--log-level', default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']))
@click.pass_context
def main(ctx, config, log_level):
    """Llama Mapper CLI - Fine-tuned model for detector output mapping."""
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Initialize configuration
    config_manager = ConfigManager(config_path=config)
    ctx.obj['config'] = config_manager
    
    # Setup logging with privacy-first approach
    setup_logging(
        log_level=log_level or config_manager.monitoring.log_level,
        log_format="console",
        enable_privacy_filter=True
    )
    
    logger = get_logger(__name__)
    logger.info("Llama Mapper CLI initialized", 
                config_path=str(config_manager.config_path),
                log_level=log_level)


@main.command()
@click.pass_context
def validate_config(ctx):
    """Validate the current configuration."""
    config_manager = ctx.obj['config']
    logger = get_logger(__name__)
    
    if config_manager.validate_config():
        logger.info("Configuration validation successful")
        click.echo("✓ Configuration is valid")
    else:
        logger.error("Configuration validation failed")
        click.echo("✗ Configuration validation failed")
        ctx.exit(1)


@main.command()
@click.pass_context
def show_config(ctx):
    """Display current configuration."""
    config_manager = ctx.obj['config']
    config_dict = config_manager.get_config_dict()
    
    click.echo("Current Configuration:")
    click.echo("=" * 50)
    
    for section, values in config_dict.items():
        click.echo(f"\n[{section.upper()}]")
        for key, value in values.items():
            # Mask sensitive values
            if key.lower() in ['token', 'password', 'key', 'vault_token', 'api_key', 'secret_key']:
                value = "***MASKED***" if value else None
            click.echo(f"  {key}: {value}")


@main.command()
@click.option('--output', '-o', type=click.Path(), help='Output path for configuration file')
@click.pass_context
def init_config(ctx, output):
    """Initialize a new configuration file with defaults."""
    config_manager = ctx.obj['config']
    logger = get_logger(__name__)
    
    output_path = Path(output) if output else config_manager.config_path
    
    try:
        config_manager.save_config(output_path)
        logger.info("Configuration file created", path=str(output_path))
        click.echo(f"✓ Configuration file created at: {output_path}")
    except Exception as e:
        logger.error("Failed to create configuration file", error=str(e))
        click.echo(f"✗ Failed to create configuration file: {e}")
        ctx.exit(1)


@main.command()
@click.pass_context
def train(ctx):
    """Train the LoRA fine-tuned model."""
    config_manager = ctx.obj['config']
    logger = get_logger(__name__)
    
    logger.info("Training command invoked", 
                model_name=config_manager.model.name,
                lora_r=config_manager.model.lora_r,
                lora_alpha=config_manager.model.lora_alpha)
    
    click.echo("Training not implemented yet")
    click.echo(f"Model: {config_manager.model.name}")
    click.echo(f"LoRA config: r={config_manager.model.lora_r}, α={config_manager.model.lora_alpha}")


@main.command()
@click.option('--host', help='Host to bind to')
@click.option('--port', type=int, help='Port to bind to')
@click.pass_context
def serve(ctx, host, port):
    """Start the FastAPI server."""
    config_manager = ctx.obj['config']
    logger = get_logger(__name__)
    
    # Override config with CLI options
    serve_host = host or config_manager.serving.host
    serve_port = port or config_manager.serving.port
    
    logger.info("Serve command invoked",
                host=serve_host,
                port=serve_port,
                backend=config_manager.serving.backend)
    
    click.echo("Serving not implemented yet")
    click.echo(f"Would serve on {serve_host}:{serve_port}")
    click.echo(f"Backend: {config_manager.serving.backend}")


@main.group()
def quality():
    """Quality gate commands for CI/CD pipeline."""
    pass


@quality.command()
@click.option('--golden-cases', type=click.Path(exists=True), help='Path to golden test cases file')
@click.option('--output', '-o', type=click.Path(), help='Output path for quality report')
@click.option('--fail-on-error', is_flag=True, default=True, help='Exit with error code if quality gates fail')
@click.option('--environment', '-e', default='production', help='Environment (development, staging, production)')
@click.option('--config', type=click.Path(exists=True), help='Path to quality gates configuration file')
@click.pass_context
def validate(ctx, golden_cases, output, fail_on_error, environment, config):
    """Run quality gate validation."""
    import asyncio
    import json
    from pathlib import Path
    from .monitoring.quality_gates import QualityGateValidator
    
    config_manager = ctx.obj['config']
    logger = get_logger(__name__)
    
    logger.info("Quality gate validation started")
    click.echo("Running quality gate validation...")
    
    # Initialize validator
    validator = QualityGateValidator(
        golden_test_cases_path=Path(golden_cases) if golden_cases else None,
        config_path=Path(config) if config else None,
        environment=environment
    )
    
    async def run_validation():
        try:
            passed, results, metrics = await validator.validate_all_quality_gates()
            
            # Generate report
            report = validator.generate_quality_report(results, metrics)
            
            # Output report
            if output:
                with open(output, 'w') as f:
                    json.dump(report, f, indent=2)
                click.echo(f"Quality report saved to: {output}")
            
            # Display summary
            click.echo(f"\nQuality Gate Results: {'PASSED' if passed else 'FAILED'}")
            click.echo(f"Total checks: {report['summary']['total_checks']}")
            click.echo(f"Passed: {report['summary']['passed_checks']}")
            click.echo(f"Failed: {report['summary']['failed_checks']}")
            
            # Display failed checks
            if report['summary']['failed_checks'] > 0:
                click.echo("\nFailed Checks:")
                for result in report['results']:
                    if result['status'] == 'FAILED':
                        click.echo(f"  ✗ {result['metric']}: {result['message']}")
            
            # Display recommendations
            if report['recommendations']:
                click.echo("\nRecommendations:")
                for rec in report['recommendations']:
                    click.echo(f"  • {rec}")
            
            return passed
            
        except Exception as e:
            logger.error("Quality gate validation failed", error=str(e))
            click.echo(f"✗ Quality gate validation failed: {e}")
            return False
    
    # Run validation
    passed = asyncio.run(run_validation())
    
    if not passed and fail_on_error:
        ctx.exit(1)


@quality.command()
@click.option('--output', '-o', type=click.Path(), default='tests/golden_test_cases.json', help='Output path for sample file')
@click.pass_context
def init_golden_cases(ctx, output):
    """Initialize sample golden test cases file."""
    from pathlib import Path
    from .monitoring.quality_gates import create_sample_golden_test_cases
    
    logger = get_logger(__name__)
    output_path = Path(output)
    
    try:
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        create_sample_golden_test_cases(output_path)
        click.echo(f"✓ Sample golden test cases created at: {output_path}")
        click.echo("Edit this file to add your specific test cases.")
        
    except Exception as e:
        logger.error("Failed to create golden test cases", error=str(e))
        click.echo(f"✗ Failed to create golden test cases: {e}")
        ctx.exit(1)


@quality.command()
@click.option('--golden-cases', type=click.Path(exists=True), help='Path to golden test cases file')
@click.option('--environment', '-e', default='development', help='Environment (development, staging, production)')
@click.option('--config', type=click.Path(exists=True), help='Path to quality gates configuration file')
@click.pass_context
def check_coverage(ctx, golden_cases, environment, config):
    """Check golden test case coverage."""
    from pathlib import Path
    from .monitoring.quality_gates import QualityGateValidator
    
    logger = get_logger(__name__)
    
    validator = QualityGateValidator(
        golden_test_cases_path=Path(golden_cases) if golden_cases else None,
        config_path=Path(config) if config else None,
        environment=environment
    )
    
    try:
        test_cases = validator.load_golden_test_cases()
        coverage_results = validator.validate_golden_test_coverage(test_cases)
        
        click.echo("Golden Test Case Coverage:")
        click.echo("=" * 40)
        
        for result in coverage_results:
            status = "✓" if result.passed else "✗"
            click.echo(f"{status} {result.message}")
        
        failed_count = len([r for r in coverage_results if not r.passed])
        if failed_count > 0:
            click.echo(f"\n{failed_count} coverage issues found")
            ctx.exit(1)
        else:
            click.echo("\n✓ All coverage requirements met")
            
    except Exception as e:
        logger.error("Coverage check failed", error=str(e))
        click.echo(f"✗ Coverage check failed: {e}")
        ctx.exit(1)


if __name__ == '__main__':
    main()
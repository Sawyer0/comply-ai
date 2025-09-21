"""Quality gate related CLI commands."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import click

from ...logging import get_logger
from ...monitoring.quality_gates import (
    QualityGateValidator,
    create_sample_golden_test_cases,
)


def register(main: click.Group) -> None:
    """Attach quality sub-commands to the root CLI."""

    @click.group()
    @click.pass_context
    def quality(ctx: click.Context) -> None:
        """Quality gate commands for CI/CD pipeline."""
        del ctx  # context not used yet

    @quality.command()
    @click.option(
        "--golden-cases",
        type=click.Path(exists=True),
        help="Path to golden test cases file",
    )
    @click.option(
        "--output", "-o", type=click.Path(), help="Output path for quality report"
    )
    @click.option(
        "--fail-on-error",
        is_flag=True,
        default=True,
        help="Exit with error code if quality gates fail",
    )
    @click.option(
        "--environment",
        "-e",
        default="production",
        help="Environment (development, staging, production)",
    )
    @click.option(
        "--config",
        type=click.Path(exists=True),
        help="Path to quality gates configuration file",
    )
    @click.pass_context
    def validate(
        ctx: click.Context,
        golden_cases: Optional[str],
        output: Optional[str],
        fail_on_error: bool,
        environment: str,
        config: Optional[str],
    ) -> None:
        """Run quality gate validation."""
        logger = get_logger(__name__)

        logger.info("Quality gate validation started")
        click.echo("Running quality gate validation...")

        validator = QualityGateValidator(
            golden_test_cases_path=Path(golden_cases) if golden_cases else None,
            config_path=Path(config) if config else None,
            environment=environment,
        )

        async def run_validation() -> bool:
            try:
                passed, results, metrics = await validator.validate_all_quality_gates()
                report = validator.generate_quality_report(results, metrics)

                if output:
                    with open(output, "w", encoding="utf-8") as file:
                        json.dump(report, file, indent=2)
                    click.echo(f"Quality report saved to: {output}")

                click.echo(
                    f"\nQuality Gate Results: {'PASSED' if passed else 'FAILED'}"
                )
                click.echo(f"Total checks: {report['summary']['total_checks']}")
                click.echo(f"Passed: {report['summary']['passed_checks']}")
                click.echo(f"Failed: {report['summary']['failed_checks']}")

                if report["summary"]["failed_checks"] > 0:
                    click.echo("\nFailed Checks:")
                    for result in report["results"]:
                        if result["status"] == "FAILED":
                            click.echo(f"  ✗ {result['metric']}: {result['message']}")

                if report["recommendations"]:
                    click.echo("\nRecommendations:")
                    for rec in report["recommendations"]:
                        click.echo(f"  • {rec}")

                return passed

            except Exception as exc:  # noqa: BLE001
                logger.error("Quality gate validation failed", error=str(exc))
                click.echo(f"✗ Quality gate validation failed: {exc}")
                return False

        passed = asyncio.run(run_validation())

        if not passed and fail_on_error:
            ctx.exit(1)

    @quality.command()
    @click.option(
        "--output",
        "-o",
        type=click.Path(),
        default="tests/golden_test_cases.json",
        help="Output path for sample file",
    )
    @click.pass_context
    def init_golden_cases(ctx: click.Context, output: str) -> None:
        """Initialize sample golden test cases file."""
        logger = get_logger(__name__)
        output_path = Path(output)

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            create_sample_golden_test_cases(output_path)
            click.echo(f"✓ Sample golden test cases created at: {output_path}")
            click.echo("Edit this file to add your specific test cases.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to create golden test cases", error=str(exc))
            click.echo(f"✗ Failed to create golden test cases: {exc}")
            ctx.exit(1)

    @quality.command()
    @click.option(
        "--golden-cases",
        type=click.Path(exists=True),
        help="Path to golden test cases file",
    )
    @click.option(
        "--environment",
        "-e",
        default="development",
        help="Environment (development, staging, production)",
    )
    @click.option(
        "--config",
        type=click.Path(exists=True),
        help="Path to quality gates configuration file",
    )
    @click.pass_context
    def check_coverage(
        ctx: click.Context,
        golden_cases: Optional[str],
        environment: str,
        config: Optional[str],
    ) -> None:
        """Check golden test case coverage."""
        logger = get_logger(__name__)

        validator = QualityGateValidator(
            golden_test_cases_path=Path(golden_cases) if golden_cases else None,
            config_path=Path(config) if config else None,
            environment=environment,
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

        except Exception as exc:  # noqa: BLE001
            logger.error("Coverage check failed", error=str(exc))
            click.echo(f"✗ Coverage check failed: {exc}")
            ctx.exit(1)

    main.add_command(quality)

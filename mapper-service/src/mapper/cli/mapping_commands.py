"""
Mapping CLI commands for Mapper Service.

Single Responsibility: Handle mapping operations via CLI.
Provides user-facing commands for detector output mapping and validation.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import click

from ..core.mapper import CoreMapper
from ..config.settings import MapperSettings
from ..schemas.models import MappingRequest


@click.group()
def mapping():
    """Mapping operations - map detector outputs to canonical taxonomy."""


@mapping.command("map")
@click.option(
    "--detector",
    "-d",
    required=True,
    help="Detector type (e.g., presidio, deberta, custom)",
)
@click.option(
    "--input",
    "-i",
    "input_data",
    required=True,
    help="Input data (JSON string or file path)",
)
@click.option(
    "--framework",
    "-f",
    help="Target compliance framework (e.g., soc2, iso27001, hipaa)",
)
@click.option("--output", "-o", help="Output file path (default: stdout)")
@click.option(
    "--confidence-threshold",
    "-c",
    type=float,
    default=0.8,
    help="Minimum confidence threshold (0.0-1.0)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "yaml", "table"]),
    default="json",
    help="Output format",
)
def map_detector_output(
    detector: str,
    input_data: str,
    framework: Optional[str],
    output: Optional[str],
    confidence_threshold: float,
    output_format: str,
):
    """Map detector output to canonical taxonomy.

    Examples:
        mapper mapping map -d presidio -i '{"entities": [{"type": "PERSON", "text": "John"}]}'
        mapper mapping map -d deberta -i /path/to/output.json -f soc2 -o result.json
    """

    async def _map():
        try:
            # Load settings
            settings = MapperSettings()
            settings.confidence_threshold = confidence_threshold

            # Initialize mapper
            mapper = CoreMapper(settings)
            await mapper.initialize()

            # Parse input data
            if Path(input_data).exists():
                with open(input_data, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = json.loads(input_data)

            # Create mapping request
            request = MappingRequest(
                detector=detector,
                output=data,
                framework=framework,
                metadata={"cli_request": True},
            )

            # Perform mapping
            click.echo(f"Mapping {detector} output...", err=True)
            response = await mapper.map_detector_output(request)

            # Format output
            result = response.to_dict()
            if output_format == "json":
                output_text = json.dumps(result, indent=2)
            elif output_format == "yaml":
                try:
                    import yaml

                    output_text = yaml.dump(result, default_flow_style=False)
                except ImportError:
                    click.echo(
                        "Error: PyYAML not installed. Install with: pip install pyyaml",
                        err=True,
                    )
                    output_text = json.dumps(result, indent=2)  # Fallback to JSON
            elif output_format == "table":
                output_text = _format_table_output(response)
            else:
                output_text = json.dumps(result, indent=2)

            # Write output
            if output:
                with open(output, "w", encoding="utf-8") as f:
                    f.write(output_text)
                click.echo(f"Results written to {output}", err=True)
            else:
                click.echo(output_text)

            # Show summary
            click.echo(
                f"✓ Mapping completed with confidence: {response.confidence:.3f}",
                err=True,
            )

        except (json.JSONDecodeError, FileNotFoundError, ValueError, RuntimeError) as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        finally:
            if "mapper" in locals():
                await mapper.shutdown()

    asyncio.run(_map())


@mapping.command()
@click.option(
    "--input-file",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="JSON file with batch mapping requests",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    help="Output directory for results (default: current directory)",
)
@click.option(
    "--parallel",
    "-p",
    type=int,
    default=5,
    help="Number of parallel mappings (currently unused)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "jsonl"]),
    default="json",
    help="Output format",
)
def batch(
    input_file: str, output_dir: Optional[str], parallel: int, output_format: str
):
    """Process multiple mapping requests in batch.

    Input file should contain a JSON array of mapping requests:
    [
        {"detector": "presidio", "output": {...}, "framework": "soc2"},
        {"detector": "deberta", "output": {...}, "framework": "hipaa"}
    ]
    """

    async def _batch():
        try:
            # Note: parallel parameter is currently unused - future enhancement
            _ = parallel  # Acknowledge unused parameter

            # Load settings
            settings = MapperSettings()

            # Initialize mapper
            mapper = CoreMapper(settings)
            await mapper.initialize()

            # Load batch requests
            with open(input_file, "r", encoding="utf-8") as f:
                requests_data = json.load(f)

            requests = [
                MappingRequest(
                    detector=req["detector"],
                    output=req["output"],
                    framework=req.get("framework"),
                    metadata={"cli_batch": True, "batch_index": i},
                )
                for i, req in enumerate(requests_data)
            ]

            click.echo(f"Processing {len(requests)} mapping requests...", err=True)

            # Process batch
            responses = await mapper.batch_map(requests)

            # Prepare output
            output_path = Path(output_dir) if output_dir else Path.cwd()
            output_path.mkdir(parents=True, exist_ok=True)

            if output_format == "json":
                results = [response.to_dict() for response in responses]
                output_file = output_path / "batch_results.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
            elif output_format == "jsonl":
                output_file = output_path / "batch_results.jsonl"
                with open(output_file, "w", encoding="utf-8") as f:
                    for response in responses:
                        f.write(json.dumps(response.to_dict()) + "\n")

            click.echo(
                f"✓ Batch processing completed. Results saved to {output_file}",
                err=True,
            )

            # Show summary
            successful = sum(1 for r in responses if r.confidence > 0.5)
            click.echo(
                f"Summary: {successful}/{len(responses)} successful mappings", err=True
            )

        except (json.JSONDecodeError, FileNotFoundError, ValueError, RuntimeError) as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        finally:
            if "mapper" in locals():
                await mapper.shutdown()

    asyncio.run(_batch())


@mapping.command()
@click.option("--detector", "-d", help="Filter by detector type")
@click.option("--framework", "-f", help="Filter by framework")
def validate(detector: Optional[str], framework: Optional[str]):
    """Validate mapping configurations and taxonomy files.

    Checks detector configurations, taxonomy mappings, and framework definitions.
    """

    async def _validate():
        try:
            settings = MapperSettings()
            mapper = CoreMapper(settings)

            click.echo("Validating mapper configuration...", err=True)

            # Validate supported detectors
            detectors = await mapper.get_supported_detectors()
            if detector and detector not in detectors:
                click.echo(f"❌ Detector '{detector}' not supported", err=True)
                click.echo(f"Available detectors: {', '.join(detectors)}", err=True)
                sys.exit(1)

            # Validate supported frameworks
            frameworks = await mapper.get_supported_frameworks()
            if framework and framework not in frameworks:
                click.echo(f"❌ Framework '{framework}' not supported", err=True)
                click.echo(f"Available frameworks: {', '.join(frameworks)}", err=True)
                sys.exit(1)

            # Health check
            health = await mapper.health_check()
            if health["status"] != "healthy":
                click.echo("❌ Mapper health check failed", err=True)
                click.echo(json.dumps(health, indent=2), err=True)
                sys.exit(1)

            click.echo("✓ All validations passed", err=True)
            click.echo(f"Supported detectors: {', '.join(detectors)}")
            click.echo(f"Supported frameworks: {', '.join(frameworks)}")

        except (ValueError, RuntimeError) as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        finally:
            if "mapper" in locals():
                await mapper.shutdown()

    asyncio.run(_validate())


def _format_table_output(response) -> str:
    """Format mapping response as a table."""

    lines = []
    lines.append("Mapping Results")
    lines.append("=" * 50)
    lines.append(f"Confidence: {response.confidence:.3f}")
    lines.append(f"Detector: {response.provenance.detector}")
    lines.append("")

    lines.append("Taxonomy Mappings:")
    lines.append("-" * 20)
    for item in response.taxonomy:
        lines.append(f"  • {item}")

    if response.scores:
        lines.append("")
        lines.append("Confidence Scores:")
        lines.append("-" * 20)
        for key, score in response.scores.items():
            lines.append(f"  {key}: {score:.3f}")

    if response.notes:
        lines.append("")
        lines.append("Notes:")
        lines.append("-" * 10)
        lines.append(f"  {response.notes}")

    return "\n".join(lines)

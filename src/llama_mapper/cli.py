"""Command-line interface for Llama Mapper."""

import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, cast

import click

from .config import ConfigManager
from .config.manager import APIKeyInfo, ServingConfig
from .logging import get_logger, setup_logging
from .versioning import TaxonomyMigrator, VersionManager
from .serving.model_server import create_model_server, GenerationConfig
from .serving.json_validator import JSONValidator
from .serving.fallback_mapper import FallbackMapper
from .api.mapper import create_app
from .monitoring.metrics_collector import MetricsCollector


@click.group()
@click.option(
    "--config", "-c", type=click.Path(exists=False), help="Configuration file path"
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
)
@click.pass_context
def main(ctx: click.Context, config: Optional[str], log_level: str) -> None:
    """Llama Mapper CLI - Fine-tuned model for detector output mapping."""
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Initialize configuration
    config_manager = ConfigManager(config_path=config)
    ctx.obj["config"] = config_manager

    # Setup logging with privacy-first approach
    setup_logging(
        log_level=log_level or config_manager.monitoring.log_level,
        log_format="console",
        enable_privacy_filter=True,
    )

    logger = get_logger(__name__)
    logger.info(
        "Llama Mapper CLI initialized",
        config_path=str(config_manager.config_path),
        log_level=log_level,
    )


@main.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=False, file_okay=False),
    help="Directory containing taxonomy.yaml, frameworks.yaml, and detector YAMLs",
)
@click.pass_context
def validate_config(ctx: click.Context, data_dir: Optional[str]) -> None:
    """Validate taxonomy, frameworks, and detector mappings."""
    from pathlib import Path as _Path

    from .config.validator import validate_configuration

    logger = get_logger(__name__)
    base = _Path(data_dir) if data_dir else None

    click.echo("Validating configuration (taxonomy/frameworks/detectors)...\n")
    result = validate_configuration(base)

    click.echo(f"Data directory: {result.data_dir}")

    # Taxonomy
    click.echo("\n[Taxonomy]")
    if result.taxonomy.ok:
        click.echo("  ✓ taxonomy.yaml loaded successfully")
    else:
        click.echo("  ✗ taxonomy.yaml validation failed")
        for err in result.taxonomy.errors:
            click.echo(f"    - {err}")

    # Frameworks
    click.echo("\n[Frameworks]")
    if result.frameworks.ok:
        frameworks = ", ".join(cast(list[str], result.frameworks.details.get("frameworks", [])))
        click.echo(f"  ✓ frameworks.yaml valid (frameworks: {frameworks})")
    else:
        click.echo("  ✗ frameworks.yaml validation failed")
        for err in result.frameworks.errors:
            click.echo(f"    - {err}")

    # Detectors
    click.echo("\n[Detectors]")
    if result.detectors.ok:
        detectors_found = cast(list[str], result.detectors.details.get("detectors_found", []))
        dets = ", ".join(detectors_found)
        click.echo(
            f"  ✓ detector YAMLs valid ({len(detectors_found)} found)"
        )
        if dets:
            click.echo(f"    - {dets}")
    else:
        click.echo("  ✗ detector YAML validation failed")
        for err in result.detectors.errors:
            click.echo(f"    - {err}")

    if not result.ok:
        logger.error("Configuration validation failed")
        ctx.exit(1)
    else:
        logger.info("Configuration validation successful")
        click.echo("\nAll configuration checks passed ✓")


@main.command()
@click.option("--tenant", type=str, help="Tenant ID to apply tenant overrides")
@click.option(
    "--environment",
    type=str,
    help="Environment name to apply environment overrides (development|staging|production)",
)
@click.option("--format", "fmt", type=click.Choice(["json"]), default=None)
@click.pass_context
def show_config(ctx: click.Context, tenant: Optional[str], environment: Optional[str], fmt: Optional[str]) -> None:
    """Display current configuration with optional tenant/environment overlays."""
    # Recreate config manager with overlays if flags are provided
    base_cm: ConfigManager = ctx.obj["config"]
    if tenant or environment:
        config_manager = ConfigManager(
            config_path=base_cm.config_path, tenant_id=tenant, environment=environment
        )
    else:
        config_manager = base_cm

    config_dict = config_manager.get_config_dict()

    # Overlay header
    active_env = config_dict.get("environment") or getattr(
        getattr(config_manager, "_config_data", {}), "get", lambda *_: None
    )("environment")

    if fmt == "json":
        # Create a masked copy for JSON output
        masked = {}
        for section, values in config_dict.items():
            if section == "environment":
                masked[section] = values
                continue
            sec_out = {}
            for key, value in values.items():
                if key.lower() in [
                    "token",
                    "password",
                    "key",
                    "vault_token",
                    "api_key",
                    "secret_key",
                ]:
                    value = "***MASKED***" if value else None
                sec_out[key] = value
            masked[section] = sec_out
        import json as _json

        payload = {
            "tenant": tenant,
            "environment": active_env,
            "config": masked,
        }
        click.echo(_json.dumps(payload, indent=2))
        return

    click.echo("Current Configuration:")
    click.echo("=" * 50)
    click.echo(
        f"Active overlays: tenant={tenant or '-'} | environment={active_env or '-'}"
    )

    for section, values in config_dict.items():
        if section == "environment":
            # already printed in header; skip duplicate top-level line
            continue
        click.echo(f"\n[{section.upper()}]")
        for key, value in values.items():
            # Mask sensitive values
            if key.lower() in [
                "token",
                "password",
                "key",
                "vault_token",
                "api_key",
                "secret_key",
            ]:
                value = "***MASKED***" if value else None
            click.echo(f"  {key}: {value}")


@main.command()
@click.option(
    "--output", "-o", type=click.Path(), help="Output path for configuration file"
)
@click.pass_context
def init_config(ctx: click.Context, output: Optional[str]) -> None:
    """Initialize a new configuration file with defaults."""
    config_manager = ctx.obj["config"]
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
def train(ctx: click.Context) -> None:
    """Train the LoRA fine-tuned model."""
    config_manager = ctx.obj["config"]
    logger = get_logger(__name__)

    logger.info(
        "Training command invoked",
        model_name=config_manager.model.name,
        lora_r=config_manager.model.lora_r,
        lora_alpha=config_manager.model.lora_alpha,
    )

    click.echo("Training not implemented yet")
    click.echo(f"Model: {config_manager.model.name}")
    click.echo(
        f"LoRA config: r={config_manager.model.lora_r}, α={config_manager.model.lora_alpha}"
    )


@main.command()
@click.option("--host", help="Host to bind to")
@click.option("--port", type=int, help="Port to bind to")
@click.pass_context
def serve(ctx: click.Context, host: Optional[str], port: Optional[int]) -> None:
    """Start the FastAPI server."""
    import uvicorn

    config_manager = ctx.obj["config"]
    logger = get_logger(__name__)

    # Override config with CLI options
    serve_host = host or config_manager.serving.host
    serve_port = port or config_manager.serving.port

    logger.info(
        "Serve command invoked",
        host=serve_host,
        port=serve_port,
        backend=config_manager.serving.backend,
    )

    # Build dependencies
    try:
        # Prefer local .kiro artifacts if present
        from pathlib import Path as _Path

        schema_path = str(_Path('.kiro/pillars-detectors/schema.json'))
        detectors_dir = str(_Path('.kiro/pillars-detectors'))

        gen_cfg = GenerationConfig(
            temperature=config_manager.model.temperature,
            top_p=config_manager.model.top_p,
            max_new_tokens=config_manager.model.max_new_tokens,
        )
        model_server = create_model_server(
            backend=config_manager.serving.backend,
            model_path=config_manager.model.name,
            generation_config=gen_cfg,
            gpu_memory_utilization=config_manager.serving.gpu_memory_utilization,
        )
        json_validator = JSONValidator(schema_path=schema_path)
        fallback_mapper = FallbackMapper(detector_configs_path=detectors_dir)

        metrics = MetricsCollector()

        app = create_app(
            model_server=model_server,
            json_validator=json_validator,
            fallback_mapper=fallback_mapper,
            config_manager=config_manager,
            metrics_collector=metrics,
        )
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to initialize app", error=str(e))
        raise SystemExit(1)

    uvicorn.run(
        app,
        host=serve_host,
        port=serve_port,
        loop="uvloop",
        http="httptools",
        access_log=False,
        workers=config_manager.serving.workers,
        timeout_keep_alive=5,
    )


@main.group()
def quality() -> None:
    """Quality gate commands for CI/CD pipeline."""
    pass


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
    import asyncio
    import json
    from pathlib import Path

    from .monitoring.quality_gates import QualityGateValidator

    config_manager = ctx.obj["config"]
    logger = get_logger(__name__)

    logger.info("Quality gate validation started")
    click.echo("Running quality gate validation...")

    # Initialize validator
    validator = QualityGateValidator(
        golden_test_cases_path=Path(golden_cases) if golden_cases else None,
        config_path=Path(config) if config else None,
        environment=environment,
    )

    async def run_validation() -> bool:
        try:
            passed, results, metrics = await validator.validate_all_quality_gates()

            # Generate report
            report = validator.generate_quality_report(results, metrics)

            # Output report
            if output:
                with open(output, "w") as f:
                    json.dump(report, f, indent=2)
                click.echo(f"Quality report saved to: {output}")

            # Display summary
            click.echo(f"\nQuality Gate Results: {'PASSED' if passed else 'FAILED'}")
            click.echo(f"Total checks: {report['summary']['total_checks']}")
            click.echo(f"Passed: {report['summary']['passed_checks']}")
            click.echo(f"Failed: {report['summary']['failed_checks']}")

            # Display failed checks
            if report["summary"]["failed_checks"] > 0:
                click.echo("\nFailed Checks:")
                for result in report["results"]:
                    if result["status"] == "FAILED":
                        click.echo(f"  ✗ {result['metric']}: {result['message']}")

            # Display recommendations
            if report["recommendations"]:
                click.echo("\nRecommendations:")
                for rec in report["recommendations"]:
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
    from pathlib import Path

    from .monitoring.quality_gates import QualityGateValidator

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

    except Exception as e:
        logger.error("Coverage check failed", error=str(e))
        click.echo(f"✗ Coverage check failed: {e}")
        ctx.exit(1)


@main.group()
@click.pass_context
def auth(ctx: click.Context) -> None:
    """Authentication and API key management commands."""
    pass


@main.group()
def detectors() -> None:
    """Detector configuration commands."""
    pass


@detectors.command("add")
@click.option("--name", required=True, help="Detector name (e.g., openai-moderation)")
@click.option(
    "--version", default="v1", show_default=True, help="Detector config version"
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False),
    help="Directory to place YAML (defaults to pillars-detectors or .kiro/pillars-detectors)",
)
@click.pass_context
def detectors_add(
    ctx: click.Context, name: str, version: str, output_dir: Optional[str]
) -> None:
    """Scaffold a new detector YAML with required fields and guidance."""
    from pathlib import Path as _Path

    from .config.validator import scaffold_detector_yaml

    try:
        path, guidance = scaffold_detector_yaml(
            name=name,
            version=version,
            output_dir=_Path(output_dir) if output_dir else None,
        )
        click.echo(f"✓ Created: {path}")
        click.echo(guidance)
    except FileExistsError as e:
        click.echo(f"✗ {e}")
        ctx.exit(1)
    except Exception as e:
        click.echo(f"✗ Failed to scaffold detector: {e}")
        ctx.exit(1)


@detectors.command("lint")
@click.option(
    "--data-dir",
    type=click.Path(exists=False, file_okay=False),
    help="Directory with taxonomy.yaml and detector YAMLs",
)
@click.option("--format", "fmt", type=click.Choice(["json"]), default=None)
@click.option("--strict", is_flag=True, default=False, help="Treat warnings as errors")
def detectors_lint(data_dir: Optional[str], fmt: Optional[str], strict: bool) -> None:
    """Lint detector YAMLs against the taxonomy and report issues."""
    import json as _json
    from pathlib import Path as _Path

    from .config.validator import validate_configuration

    result = validate_configuration(_Path(data_dir) if data_dir else None)

    # Summarize
    issues = []
    if not result.taxonomy.ok:
        issues.append("Taxonomy invalid; fix taxonomy.yaml before linting detectors.")
    if not result.detectors.ok:
        issues.extend(result.detectors.errors)

    # JSON output if requested
    if fmt == "json":
        payload = {
            "data_dir": str(result.data_dir),
            "taxonomy_ok": result.taxonomy.ok,
            "detectors_ok": result.detectors.ok,
            "detectors_found": result.detectors.details.get("detectors_found", []),
            "errors": issues,
            "warnings": result.detectors.warnings,
        }
        click.echo(_json.dumps(payload, indent=2))
    else:
        click.echo(f"Data directory: {result.data_dir}")
        click.echo("\n[Detectors Lint]")
        if result.taxonomy.ok:
            click.echo("  ✓ taxonomy.yaml valid")
        else:
            click.echo("  ✗ taxonomy.yaml invalid")
            for err in result.taxonomy.errors:
                click.echo(f"    - {err}")
        if result.detectors.ok:
            detectors_found = cast(list[str], result.detectors.details.get("detectors_found", []))
            names = ", ".join(detectors_found)
            click.echo(
                f"  ✓ detector YAMLs valid ({len(detectors_found)} found)"
            )
            if names:
                click.echo(f"    - {names}")
        else:
            click.echo("  ✗ detector YAML validation failed")
            for err in result.detectors.errors:
                click.echo(f"    - {err}")

    # Exit code logic
    if (not result.taxonomy.ok) or (not result.detectors.ok) or (strict and result.detectors.warnings):
        raise SystemExit(1)


@detectors.command("fix")
@click.option(
    "--data-dir",
    type=click.Path(exists=False, file_okay=False),
    help="Directory with taxonomy.yaml and detector YAMLs",
)
@click.option("--apply/--dry-run", default=False, help="Apply suggested fixes where confident enough")
@click.option(
    "--threshold",
    type=float,
    default=0.86,
    show_default=True,
    help="Confidence threshold (0-1) to auto-apply a suggestion",
)
@click.option("--format", "fmt", type=click.Choice(["json"]), default=None)
def detectors_fix(data_dir: Optional[str], apply: bool, threshold: float, fmt: Optional[str]) -> None:
    """Suggest (and optionally apply) fixes for invalid detector canonical labels."""
    import json as _json
    from pathlib import Path as _Path

    from .config.validator import build_detector_fix_plan, apply_detector_fix_plan

    plan = build_detector_fix_plan(_Path(data_dir) if data_dir else None)

    if fmt == "json":
        click.echo(_json.dumps(plan, indent=2))
    else:
        click.echo(f"Data directory: {plan['data_dir']}")
        click.echo(f"Total invalid labels: {plan['total_invalid']}")
        items = cast(list[dict], plan.get("items", []))
        for item in items:
            det = item.get("detector")
            file = item.get("file")
            click.echo(f"\nDetector: {det}\nFile: {file}")
            for bad in item.get("invalid", []):
                click.echo(f"  - {bad}")
                for s in item.get("suggestions", {}).get(bad, []):
                    click.echo(f"      suggestion: {s['label']} (score={s['score']})")

    if apply:
        summary = apply_detector_fix_plan(plan, apply_threshold=threshold)
        if fmt == "json":
            click.echo(_json.dumps({"apply_summary": summary}, indent=2))
        else:
            click.echo("\nApply summary:")
            click.echo(f"  applied: {summary['applied']}")
            click.echo(f"  skipped: {summary['skipped']}")
            if summary.get("updated_files"):
                click.echo("  updated_files:")
                updated_files = cast(dict[str, int], summary.get("updated_files", {}))
                for fp, count in updated_files.items():
                    click.echo(f"    - {fp}: {count} change(s)")
        # Non-zero exit if there are still invalids after apply
        # Rebuild plan to see remaining invalid
        new_plan = build_detector_fix_plan(_Path(data_dir) if data_dir else None)
        total_invalid = cast(int, (new_plan.get("total_invalid", 0) or 0))
        if total_invalid > 0:
            raise SystemExit(1)


@auth.command("rotate-key")
@click.option("--tenant", required=True, help="Tenant ID to associate with the new key")
@click.option(
    "--scope",
    "scopes",
    multiple=True,
    help="Scope to grant (repeat for multiple), e.g., map:write",
)
@click.option(
    "--revoke-old/--keep-old", default=True, help="Revoke existing keys for this tenant"
)
@click.option(
    "--print-key/--no-print-key",
    default=False,
    help="Print the new API key to stdout (be careful)",
)
@click.pass_context
def rotate_key(
    ctx: click.Context,
    tenant: str,
    scopes: Tuple[str, ...],
    revoke_old: bool,
    print_key: bool,
) -> None:
    """Generate a new API key for a tenant and optionally revoke old keys."""
    config_manager: ConfigManager = ctx.obj["config"]
    logger = get_logger(__name__)

    # Ensure auth section exists
    auth_cfg = getattr(config_manager, "auth", None)
    if auth_cfg is None:
        click.echo("✗ Auth configuration not available in ConfigManager")
        ctx.exit(1)

    if revoke_old:
        for key, info in list(auth_cfg.api_keys.items()):
            try:
                if info.tenant_id == tenant and info.active:
                    auth_cfg.api_keys[key] = APIKeyInfo(
                        tenant_id=info.tenant_id, scopes=info.scopes, active=False
                    )
            except Exception:
                # Backward compatibility if info is a dict
                if (
                    isinstance(info, dict)
                    and info.get("tenant_id") == tenant
                    and info.get("active", True)
                ):
                    info["active"] = False
                    auth_cfg.api_keys[key] = APIKeyInfo(**info)

    new_key = secrets.token_urlsafe(32)
    granted_scopes = list(scopes) if scopes else ["map:write"]
    auth_cfg.api_keys[new_key] = APIKeyInfo(
        tenant_id=tenant, scopes=granted_scopes, active=True
    )

    # Persist
    try:
        config_manager.save_config()
    except Exception as e:
        logger.error("Failed to save config after key rotation", error=str(e))
        click.echo(f"✗ Failed to save configuration: {e}")
        ctx.exit(1)

    logger.info(
        "Rotated API key",
        tenant=tenant,
        scopes=granted_scopes,
        revoked_old=revoke_old,
        at=datetime.now(timezone.utc).isoformat(),
    )
    click.echo("✓ API key rotated successfully")
    if print_key:
        click.echo(f"New API Key: {new_key}")


@main.group()
def versions() -> None:
    """Version management commands."""
    pass


@versions.command("show")
@click.option(
    "--data-dir",
    type=click.Path(exists=False),
    help="Directory with taxonomy/frameworks/detectors",
)
@click.option(
    "--registry",
    type=click.Path(exists=False),
    help="Path to model versions registry (versions.json)",
)
def versions_show(data_dir: Optional[str], registry: Optional[str]) -> None:
    """Show current version snapshot as JSON."""
    import json as _json
    from pathlib import Path as _Path

    vm = VersionManager(
        _Path(data_dir) if data_dir else None,
        _Path(registry) if registry else None,
    )
    snap = vm.snapshot().to_dict()
    click.echo(_json.dumps(snap, indent=2))


@main.group()
def taxonomy() -> None:
    """Taxonomy migration commands."""
    pass


@taxonomy.command("migrate-plan")
@click.option(
    "--from-taxonomy",
    "from_taxonomy",
    required=True,
    type=click.Path(exists=True),
    help="Path to old taxonomy.yaml",
)
@click.option(
    "--to-taxonomy",
    "to_taxonomy",
    required=True,
    type=click.Path(exists=True),
    help="Path to new taxonomy.yaml",
)
@click.option(
    "--output", "-o", type=click.Path(), help="Path to write migration plan JSON"
)
def taxonomy_migrate_plan(
    from_taxonomy: str, to_taxonomy: str, output: Optional[str]
) -> None:
    """Compute a migration plan between taxonomy versions and optionally write it to a file."""
    import json as _json
    from pathlib import Path as _Path

    migrator = TaxonomyMigrator(_Path(from_taxonomy), _Path(to_taxonomy))
    plan = migrator.compute_plan()
    summary = migrator.validate_plan_completeness(plan)
    result = {
        "plan": {
            "from_version": plan.from_version,
            "to_version": plan.to_version,
            "created_at": plan.created_at,
            "label_map": plan.label_map,
            "label_map_count": len(plan.label_map),
            "unmapped_old_labels": plan.unmapped_old_labels,
            "new_labels_without_source": plan.new_labels_without_source,
        },
        "summary": summary,
    }
    text = _json.dumps(result, indent=2)
    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(text)
        click.echo(f"✓ Migration plan written to {output}")
    else:
        click.echo(text)


@taxonomy.command("migrate-apply")
@click.option(
    "--plan",
    "plan_path",
    required=True,
    type=click.Path(exists=True),
    help="Migration plan JSON produced by migrate-plan",
)
@click.option(
    "--detectors-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing detector YAMLs",
)
@click.option(
    "--write-dir",
    type=click.Path(),
    help="Optional directory to write migrated detector YAMLs (dry-run if omitted)",
)
def taxonomy_migrate_apply(
    plan_path: str, detectors_dir: str, write_dir: Optional[str]
) -> None:
    """Apply a migration plan to detector YAMLs. Writes outputs to a separate directory if provided; otherwise prints a report (dry-run)."""
    import json as _json
    from pathlib import Path as _Path

    import yaml as _yaml

    # Load plan JSON
    with open(plan_path, "r", encoding="utf-8") as f:
        data = _json.load(f)
    plan_obj = data.get("plan") or {}
    from .versioning import MigrationPlan as _MigrationPlan

    plan = _MigrationPlan(
        from_version=plan_obj.get("from_version", ""),
        to_version=plan_obj.get("to_version", ""),
        created_at=plan_obj.get("created_at", ""),
        label_map=plan_obj.get("label_map", {}) if "label_map" in plan_obj else {},
        unmapped_old_labels=plan_obj.get("unmapped_old_labels", []),
        new_labels_without_source=plan_obj.get("new_labels_without_source", []),
    )

    # Build in-memory detector mappings from files
    det_dir = _Path(detectors_dir)
    yaml_files = [
        p
        for p in det_dir.glob("*.yaml")
        if p.name not in {"taxonomy.yaml", "frameworks.yaml"}
    ]
    detector_maps = {}
    for yf in yaml_files:
        with open(yf, "r", encoding="utf-8") as f:
            y = _yaml.safe_load(f)
        if not isinstance(y, dict) or "detector" not in y or "maps" not in y:
            continue
        detector_maps[y["detector"]] = y["maps"]

    # Apply plan
    from .versioning import TaxonomyMigrator as _TaxonomyMigrator

    # We do not need full migrator instance to apply; fake minimal structure
    migrator = _TaxonomyMigrator(_Path("noop_old.yaml"), _Path("noop_new.yaml"))
    # Bypass loader usage when applying; only plan is used
    report = migrator.apply_to_detector_mappings(detector_maps, plan)

    click.echo(
        _json.dumps(
            {
                "summary": {
                    "total_mappings": report.total_mappings,
                    "remapped": report.remapped,
                    "unchanged": report.unchanged,
                    "unknown_after_migration": report.unknown_after_migration,
                },
                "details": report.details,
            },
            indent=2,
        )
    )

    # Optionally write new YAMLs
    if write_dir:
        out = _Path(write_dir)
        out.mkdir(parents=True, exist_ok=True)
        for yf in yaml_files:
            with open(yf, "r", encoding="utf-8") as f:
                y = _yaml.safe_load(f)
            det = y.get("detector")
            if det and det in detector_maps:
                # Build new mapping by applying plan label_map
                new_maps = {
                    k: plan.label_map.get(v, v) for k, v in detector_maps[det].items()
                }
                y["maps"] = new_maps
                # Bump version minor (e.g., v1 -> v1.1) if present
                ver = str(y.get("version", "v1"))
                if ver.startswith("v"):
                    try:
                        parts = ver[1:].split(".")
                        if len(parts) == 1:
                            y["version"] = f"v{int(parts[0]) + 1}"
                        else:
                            major = int(parts[0])
                            minor = int(parts[1]) if len(parts) > 1 else 0
                            y["version"] = f"v{major}.{minor + 1}"
                    except Exception:
                        pass
                out_file = out / yf.name
                with open(out_file, "w", encoding="utf-8") as f:
                    _yaml.safe_dump(y, f, sort_keys=False)
        click.echo(f"✓ Migrated detector YAMLs written to {out}")


@main.group()
@click.pass_context
def runtime(ctx: click.Context) -> None:
    """Runtime controls (kill-switch, modes)."""
    pass


@runtime.command("show-mode")
@click.pass_context
def runtime_show_mode(ctx: click.Context) -> None:
    """Show current runtime mode (hybrid or rules_only)."""
    config_manager: ConfigManager = ctx.obj["config"]
    mode = getattr(config_manager.serving, "mode", "hybrid")
    click.echo(f"Runtime mode: {mode}")


@runtime.command("set-mode")
@click.argument("mode", type=click.Choice(["hybrid", "rules_only"]))
@click.pass_context
def runtime_set_mode(ctx: click.Context, mode: str) -> None:
    """Set runtime mode. 'rules_only' enables kill-switch; 'hybrid' re-enables model."""
    config_manager: ConfigManager = ctx.obj["config"]
    # mutate in-memory and persist
    try:
        current = config_manager.serving
        # Pydantic models are immutable by default; reconstruct
        new_serving = ServingConfig(
            backend=current.backend,
            host=current.host,
            port=current.port,
            workers=current.workers,
            batch_size=current.batch_size,
            device=current.device,
            gpu_memory_utilization=current.gpu_memory_utilization,
            mode=mode,
        )
        config_manager.serving = new_serving
        config_manager.save_config()
        click.echo(f"✓ Runtime mode set to {mode}")
    except Exception as e:
        click.echo(f"✗ Failed to set runtime mode: {e}")
        raise


@runtime.command("kill-switch")
@click.argument("state", type=click.Choice(["on", "off"]))
@click.pass_context
def runtime_kill_switch(ctx: click.Context, state: str) -> None:
    """Alias for set-mode: 'on' => rules_only, 'off' => hybrid."""
    mode = "rules_only" if state == "on" else "hybrid"
    ctx.invoke(runtime_set_mode, mode=mode)


@main.group()
def tenant() -> None:
    """Tenant configuration tools (migration and validation)."""
    pass


@tenant.command("migrate-config")
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing tenant config YAMLs",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory to write migrated configs",
)
@click.pass_context
def tenant_migrate_config(ctx: click.Context, input_dir: str, output_dir: str) -> None:
    """Migrate tenant config files to the latest schema and validate."""
    from pathlib import Path as _Path

    import yaml as _yaml

    input_path = _Path(input_dir)
    output_path = _Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    migrated = 0
    for yf in input_path.glob("*.yaml"):
        try:
            with open(yf, "r", encoding="utf-8") as f:
                y = _yaml.safe_load(f) or {}
            # Minimal schema normalization: ensure required keys exist
            y.setdefault("tenant_id", yf.stem)
            y.setdefault("overrides", {})
            # Validate precedence fields
            overrides = y.get("overrides") or {}
            for level in ["global", "tenant", "environment"]:
                overrides.setdefault(level, {})
            # Write migrated file
            out_file = output_path / yf.name
            with open(out_file, "w", encoding="utf-8") as f:
                _yaml.safe_dump(y, f, sort_keys=False)
            migrated += 1
        except Exception as e:
            click.echo(f"✗ Failed to migrate {yf.name}: {e}")
    click.echo(f"✓ Migrated {migrated} tenant config(s) to {output_path}")


@tenant.command("validate-config")
@click.option(
    "--dir",
    "dir_path",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing tenant config YAMLs",
)
@click.pass_context
def tenant_validate_config(ctx: click.Context, dir_path: str) -> None:
    """Validate tenant config files for required structure and precedence rules."""
    from pathlib import Path as _Path

    import yaml as _yaml

    dirp = _Path(dir_path)
    errors = 0
    for yf in dirp.glob("*.yaml"):
        try:
            with open(yf, "r", encoding="utf-8") as f:
                y = _yaml.safe_load(f) or {}
            if "tenant_id" not in y:
                raise ValueError("missing tenant_id")
            overrides = y.get("overrides") or {}
            for level in ["global", "tenant", "environment"]:
                if level not in overrides or not isinstance(overrides[level], dict):
                    raise ValueError(f"missing overrides.{level}")
        except Exception as e:
            errors += 1
            click.echo(f"✗ {yf.name}: {e}")
    if errors == 0:
        click.echo("✓ All tenant configs valid")
    else:
        click.echo(f"✗ {errors} invalid tenant config(s)")
        ctx.exit(1)


if __name__ == "__main__":
    main()

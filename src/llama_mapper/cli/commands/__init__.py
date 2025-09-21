"""Command registration helpers for the CLI."""

from __future__ import annotations

import click

from . import auth, config, detectors, quality, runtime, serve, taxonomy, tenant, versions


def register_all(main: click.Group) -> None:
    """Attach all command groups to the root CLI."""
    config.register(main)
    serve.register(main)
    quality.register(main)
    detectors.register(main)
    auth.register(main)
    versions.register(main)
    taxonomy.register(main)
    runtime.register(main)
    tenant.register(main)

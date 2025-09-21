"""Shared helpers for the CLI commands."""

from __future__ import annotations

from typing import cast

import click

from ..config import ConfigManager


def get_config_manager(ctx: click.Context) -> ConfigManager:
    """Return the config manager stored on the click context."""
    return cast(ConfigManager, ctx.obj["config"])

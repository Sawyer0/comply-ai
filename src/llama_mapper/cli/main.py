"""Root Click group for the Llama Mapper CLI."""

from __future__ import annotations

from typing import Optional

import click

from ..config import ConfigManager
from ..logging import get_logger, setup_logging
from .commands import register_all


@click.group()
@click.option("--config", "-c", type=click.Path(exists=False), help="Configuration file path")
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
)
@click.pass_context
def main(ctx: click.Context, config: Optional[str], log_level: str) -> None:
    """Llama Mapper CLI - Fine-tuned model for detector output mapping."""
    ctx.ensure_object(dict)

    config_manager = ConfigManager(config_path=config)
    ctx.obj["config"] = config_manager

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


register_all(main)


if __name__ == "__main__":  # pragma: no cover
    main()

"""Runtime control commands."""

from __future__ import annotations

import click

from ...config.manager import ServingConfig
from ..utils import get_config_manager


def register(main: click.Group) -> None:
    """Attach runtime subcommands."""

    @click.group()
    @click.pass_context
    def runtime(ctx: click.Context) -> None:
        """Runtime controls (kill-switch, modes)."""
        del ctx

    @runtime.command("show-mode")
    @click.pass_context
    def runtime_show_mode(ctx: click.Context) -> None:
        """Show current runtime mode (hybrid or rules_only)."""
        config_manager = get_config_manager(ctx)
        mode = getattr(config_manager.serving, "mode", "hybrid")
        click.echo(f"Runtime mode: {mode}")

    @runtime.command("set-mode")
    @click.argument("mode", type=click.Choice(["hybrid", "rules_only"]))
    @click.pass_context
    def runtime_set_mode(ctx: click.Context, mode: str) -> None:
        """Set runtime mode. 'rules_only' enables kill-switch; 'hybrid' re-enables model."""
        config_manager = get_config_manager(ctx)
        try:
            current = config_manager.serving
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
        except Exception as exc:  # noqa: BLE001
            click.echo(f"✗ Failed to set runtime mode: {exc}")
            raise

    @runtime.command("kill-switch")
    @click.argument("state", type=click.Choice(["on", "off"]))
    @click.pass_context
    def runtime_kill_switch(ctx: click.Context, state: str) -> None:
        """Alias for set-mode: 'on' => rules_only, 'off' => hybrid."""
        mode = "rules_only" if state == "on" else "hybrid"
        ctx.invoke(runtime_set_mode, mode=mode)

    main.add_command(runtime)

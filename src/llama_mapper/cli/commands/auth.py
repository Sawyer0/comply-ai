"""Authentication-focused CLI commands."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import MutableMapping, Tuple, cast

import click

from ...config.manager import APIKeyInfo
from ...logging import get_logger
from ..utils import get_config_manager


def register(main: click.Group) -> None:
    """Attach authentication subcommands."""

    @click.group()
    @click.pass_context
    def auth(ctx: click.Context) -> None:
        """Authentication and API key management commands."""
        del ctx

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
        config_manager = get_config_manager(ctx)
        logger = get_logger(__name__)

        auth_cfg = getattr(config_manager, "auth", None)
        if auth_cfg is None:
            click.echo("✗ Auth configuration not available in ConfigManager")
            ctx.exit(1)

        api_keys_attr = getattr(auth_cfg, "api_keys", None)
        if not isinstance(api_keys_attr, MutableMapping):
            click.echo("✗ Auth configuration missing API key registry")
            ctx.exit(1)
        api_keys = cast(MutableMapping[str, APIKeyInfo], api_keys_attr)

        if revoke_old:
            for key, info in list(api_keys.items()):
                try:
                    if info.tenant_id == tenant and info.active:
                        api_keys[key] = APIKeyInfo(
                            tenant_id=info.tenant_id, scopes=info.scopes, active=False
                        )
                except Exception:
                    if (
                        isinstance(info, dict)
                        and info.get("tenant_id") == tenant
                        and info.get("active", True)
                    ):
                        info["active"] = False
                        api_keys[key] = APIKeyInfo(**info)

        new_key = secrets.token_urlsafe(32)
        granted_scopes = list(scopes) if scopes else ["map:write"]
        api_keys[new_key] = APIKeyInfo(tenant_id=tenant, scopes=granted_scopes, active=True)

        try:
            config_manager.save_config()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save config after key rotation", error=str(exc))
            click.echo(f"✗ Failed to save configuration: {exc}")
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

    main.add_command(auth)

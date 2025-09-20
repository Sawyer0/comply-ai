"""Tests for tenant and environment override precedence in ConfigManager."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import yaml

from src.llama_mapper.config import ConfigManager


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def test_tenant_and_environment_overrides_precedence():
    with tempfile.TemporaryDirectory() as tmp:
        base_dir = Path(tmp)
        base_cfg = base_dir / "config.yaml"

        # Base config
        write_yaml(
            base_cfg,
            {
                "environment": "development",
                "model": {"name": "base-model", "temperature": 0.1},
                "serving": {"port": 8000},
            },
        )

        # Tenant override (should override model.name)
        write_yaml(
            base_dir / "tenants" / "tenantA.yaml",
            {
                "model": {"name": "tenant-model"},
            },
        )

        # Environment override (should override serving.port and not reset tenant override)
        write_yaml(
            base_dir / "environments" / "staging.yaml",
            {
                "serving": {"port": 9001},
            },
        )

        # Instantiate with tenant and environment
        cm = ConfigManager(
            config_path=base_cfg, tenant_id="tenantA", environment="staging"
        )

        assert cm.model.name == "tenant-model"  # from tenant overlay
        assert cm.serving.port == 9001  # from environment overlay


def test_env_var_override_wins_over_files():
    with tempfile.TemporaryDirectory() as tmp:
        base_dir = Path(tmp)
        base_cfg = base_dir / "config.yaml"

        write_yaml(base_cfg, {"serving": {"port": 8000}})
        write_yaml(
            base_dir / "environments" / "production.yaml", {"serving": {"port": 9001}}
        )

        os.environ["MAPPER_SERVING_PORT"] = "7777"
        try:
            cm = ConfigManager(config_path=base_cfg, environment="production")
            assert cm.serving.port == 7777
        finally:
            os.environ.pop("MAPPER_SERVING_PORT", None)

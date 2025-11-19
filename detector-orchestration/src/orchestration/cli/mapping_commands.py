from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from shared.exceptions.base import BaseServiceException

from ..config.mapping_schemas import get_mapping_schema, validate_mapping_rules

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

logger = logging.getLogger(__name__)


class MappingCLI:
    def __init__(self, orchestration_service) -> None:
        self.service = orchestration_service

    def _get_repository(self):
        components = getattr(self.service, "components", None)
        if not components:
            return "Service components are not initialized"
        repo = getattr(components, "detector_mapping_repository", None)
        if not repo:
            return "Detector mapping repository is not initialized"
        return repo

    async def list_configs(
        self,
        tenant_id: str,
        detector_type: str,
        output_format: str = "table",
    ) -> str:
        repo = self._get_repository()
        if isinstance(repo, str):
            return repo

        try:
            configs = await repo.list_configs(
                tenant_id=tenant_id,
                detector_type=detector_type,
            )
        except (BaseServiceException, RuntimeError) as exc:
            return self._format_error("list mapping configs", exc)

        if not configs:
            return "No mapping configs found."

        payload = [
            {
                "id": c.id,
                "tenant_id": c.tenant_id,
                "detector_type": c.detector_type,
                "detector_version": c.detector_version,
                "version": c.version,
                "schema_version": c.schema_version,
                "status": c.status,
                "is_active": c.is_active,
                "backward_compatible": c.backward_compatible,
                "rollback_of_version": c.rollback_of_version,
                "created_at": c.created_at,
                "activated_at": c.activated_at,
                "deactivated_at": c.deactivated_at,
                "created_by": c.created_by,
            }
            for c in configs
        ]

        if output_format == "json":
            return json.dumps(payload, indent=2)

        if output_format == "yaml":
            if yaml is None:
                return "PyYAML not installed; use --format json instead"
            return yaml.dump(payload, default_flow_style=False)

        lines = [
            f"Mapping configs for tenant='{tenant_id}', detector_type='{detector_type}':",
            "=" * 50,
        ]
        for index, config in enumerate(configs, start=1):
            lines.append(
                f"{index:2d}. version={config.version} status={config.status} active={config.is_active}"
            )
        return "\n".join(lines)

    async def show_active_config(
        self,
        tenant_id: str,
        detector_type: str,
        output_format: str = "json",
    ) -> str:
        repo = self._get_repository()
        if isinstance(repo, str):
            return repo

        try:
            config = await repo.get_active_config(
                tenant_id=tenant_id,
                detector_type=detector_type,
            )
        except (BaseServiceException, RuntimeError) as exc:
            return self._format_error("get active mapping config", exc)

        if not config:
            return "No active mapping config found."

        payload = {
            "id": config.id,
            "tenant_id": config.tenant_id,
            "detector_type": config.detector_type,
            "detector_version": config.detector_version,
            "version": config.version,
            "schema_version": config.schema_version,
            "status": config.status,
            "is_active": config.is_active,
            "backward_compatible": config.backward_compatible,
            "rollback_of_version": config.rollback_of_version,
            "created_at": config.created_at,
            "activated_at": config.activated_at,
            "deactivated_at": config.deactivated_at,
            "created_by": config.created_by,
            "mapping_rules": config.mapping_rules,
            "validation_schema": config.validation_schema,
        }

        if output_format == "yaml":
            if yaml is None:
                return "PyYAML not installed; use --format json instead"
            return yaml.dump(payload, default_flow_style=False)

        return json.dumps(payload, indent=2)

    async def create_config(
        self,
        *,
        tenant_id: str,
        detector_type: str,
        version: str,
        schema_version: str,
        mapping_rules_path: str,
        detector_version: Optional[str] = None,
        validation_schema_path: Optional[str] = None,
        activate: bool = False,
        created_by: Optional[str] = None,
    ) -> str:
        repo = self._get_repository()
        if isinstance(repo, str):
            return repo

        try:
            mapping_rules = self._load_struct(mapping_rules_path)
            validation_schema = (
                self._load_struct(validation_schema_path)
                if validation_schema_path
                else None
            )
        except (OSError, ValueError) as exc:
            return f"Error loading configuration files: {exc}"

        if validation_schema is None:
            try:
                validate_mapping_rules(mapping_rules, schema_version)
                validation_schema = get_mapping_schema(schema_version)
            except Exception as exc:  # pragma: no cover - defensive validation
                return (
                    "Error validating mapping rules for schema_version '"
                    + schema_version
                    + f"': {exc}"
                )

        fields: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "detector_type": detector_type,
            "version": version,
            "schema_version": schema_version,
            "mapping_rules": mapping_rules,
            "validation_schema": validation_schema,
            "status": "inactive",
            "is_active": False,
            "backward_compatible": True,
            "detector_version": detector_version,
            "created_by": created_by,
        }

        try:
            new_id = await repo.create_config(fields=fields)
        except (BaseServiceException, RuntimeError) as exc:
            return self._format_error("create mapping config", exc)

        if activate:
            activated = await self.activate_version(
                tenant_id=tenant_id,
                detector_type=detector_type,
                version=version,
            )
            if activated.startswith("Error"):
                return (
                    f"Created mapping config '{new_id}' but failed to activate version "
                    f"'{version}': {activated}"
                )
            return (
                f"Created and activated mapping config '{new_id}' for detector_type="
                f"'{detector_type}' version='{version}'"
            )

        return (
            f"Created mapping config '{new_id}' for detector_type="
            f"'{detector_type}' version='{version}'"
        )

    async def activate_version(
        self,
        *,
        tenant_id: str,
        detector_type: str,
        version: str,
    ) -> str:
        repo = self._get_repository()
        if isinstance(repo, str):
            return repo

        try:
            result = await repo.activate_version(
                tenant_id=tenant_id,
                detector_type=detector_type,
                version=version,
            )
        except (BaseServiceException, RuntimeError) as exc:
            return self._format_error("activate mapping version", exc)

        if not result:
            return (
                f"No mapping config found for tenant='{tenant_id}', detector_type="
                f"'{detector_type}', version='{version}'"
            )

        return (
            f"Activated mapping config for tenant='{tenant_id}', detector_type="
            f"'{detector_type}', version='{version}'"
        )

    async def rollback_to_version(
        self,
        *,
        tenant_id: str,
        detector_type: str,
        version: str,
    ) -> str:
        repo = self._get_repository()
        if isinstance(repo, str):
            return repo

        try:
            result = await repo.rollback_to_version(
                tenant_id=tenant_id,
                detector_type=detector_type,
                version=version,
            )
        except (BaseServiceException, RuntimeError) as exc:
            return self._format_error("rollback mapping version", exc)

        if not result:
            return (
                f"No mapping config found to rollback to for tenant='{tenant_id}', "
                f"detector_type='{detector_type}', version='{version}'"
            )

        return (
            f"Rolled back to mapping version '{version}' for tenant='{tenant_id}', "
            f"detector_type='{detector_type}'"
        )

    def _load_struct(self, path: str) -> Dict[str, Any]:
        content = Path(path).read_text(encoding="utf-8")
        suffix = Path(path).suffix.lower()
        if suffix in {".yml", ".yaml"} and yaml is not None:
            data = yaml.safe_load(content)
        else:
            data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("Configuration file must contain a JSON or YAML object")
        return data

    def _format_error(self, action: str, exc: Exception) -> str:
        logger.error("Failed to %s: %s", action, exc)
        return f"Error attempting to {action}: {exc}"


__all__ = ["MappingCLI"]

"""Policy management CLI commands following SRP.

This module provides ONLY policy management CLI commands.
Single Responsibility: Handle CLI commands for policy operations.
"""

from __future__ import annotations

from pathlib import Path
import json
import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from shared.exceptions.base import BaseServiceException

try:
    import yaml
except ImportError:  # pragma: no cover - yaml optional for CLI formatting
    yaml = None

if TYPE_CHECKING:
    from ..policy.policy_manager import PolicyManager

logger = logging.getLogger(__name__)


class PolicyCLI:
    """CLI commands for policy management."""

    def __init__(self, orchestration_service) -> None:
        self.service = orchestration_service

    async def list_policies(self, output_format: str = "table") -> str:
        """List all policies."""

        manager = self._policy_manager_available()
        if isinstance(manager, str):
            return manager

        try:
            policies = await manager.list_policies()
        except BaseServiceException as exc:
            return self._format_error("list policies", exc)

        result: str
        if output_format == "json":
            result = json.dumps(policies, indent=2)
        elif output_format == "yaml":
            if yaml is None:
                result = "PyYAML not installed; use --format json instead"
            else:
                result = yaml.dump(policies, default_flow_style=False)  # type: ignore[arg-type]
        elif not policies:
            result = "No policies configured."
        else:
            lines = ["Active Policies:", "=" * 50]
            for index, policy in enumerate(policies, start=1):
                lines.append(f"{index:2d}. {policy.get('name', 'Unnamed')}")
                lines.append(f"    Type: {policy.get('type', 'Unknown')}")
                lines.append(f"    Enabled: {policy.get('enabled', False)}")
                lines.append(f"    Priority: {policy.get('priority', 0)}")
            result = "\n".join(lines)

        return result

    async def validate_policy(self, policy_file: str) -> str:
        """Validate a policy file."""

        manager = self._policy_manager_available()
        if isinstance(manager, str):
            return manager

        try:
            policy_content = Path(policy_file).read_text(encoding="utf-8")
        except FileNotFoundError:
            return f"Policy file '{policy_file}' not found"

        try:
            validation_result = await manager.validate_policy(policy_content)
        except BaseServiceException as exc:
            return self._format_error("validate policy", exc)

        if validation_result.is_valid:
            return f"Policy '{policy_file}' is valid"

        lines = [f"Policy '{policy_file}' validation failed:"]
        lines.extend(f"  - {error}" for error in validation_result.errors)
        return "\n".join(lines)

    async def load_policy(
        self, policy_file: str, policy_name: Optional[str] = None
    ) -> str:
        """Load a policy from file."""

        manager = self._policy_manager_available()
        if isinstance(manager, str):
            return manager

        try:
            policy_content = Path(policy_file).read_text(encoding="utf-8")
        except FileNotFoundError:
            return f"Policy file '{policy_file}' not found"

        try:
            was_loaded = await manager.load_policy(policy_content, policy_name)
        except BaseServiceException as exc:
            return self._format_error("load policy", exc)

        if not was_loaded:
            return f"Failed to load policy from '{policy_file}'"

        name = policy_name or policy_file
        return f"Successfully loaded policy '{name}'"

    async def unload_policy(self, policy_name: str) -> str:
        """Unload a policy."""

        manager = self._policy_manager_available()
        if isinstance(manager, str):
            return manager

        try:
            was_unloaded = await manager.unload_policy(policy_name)
        except BaseServiceException as exc:
            return self._format_error("unload policy", exc)

        if not was_unloaded:
            return f"Policy '{policy_name}' not found"

        return f"Successfully unloaded policy '{policy_name}'"

    async def policy_status(self, policy_name: Optional[str] = None) -> str:
        """Show policy status."""

        manager = self._policy_manager_available()
        if isinstance(manager, str):
            return manager

        try:
            if policy_name:
                policy = await manager.get_policy(policy_name)
                if not policy:
                    return f"Policy '{policy_name}' not found"

                return json.dumps(
                    {
                        "name": policy.get("name"),
                        "type": policy.get("type"),
                        "enabled": policy.get("enabled"),
                        "priority": policy.get("priority"),
                        "last_updated": policy.get("last_updated"),
                    },
                    indent=2,
                )

            policies = await manager.list_policies()
            stats = await manager.get_policy_statistics()
        except BaseServiceException as exc:
            return self._format_error("retrieve policy status", exc)

        lines = ["Policy System Status:", "=" * 50]
        lines.append(f"Total Policies: {len(policies)}")
        lines.append(f"Active Policies: {stats.get('active_policies', 0)}")
        lines.append(f"Policy Evaluations: {stats.get('total_evaluations', 0)}")
        lines.append(f"Policy Violations: {stats.get('total_violations', 0)}")
        return "\n".join(lines)

    async def test_policy(self, policy_name: str, test_data: Dict[str, Any]) -> str:
        """Test a policy with sample data."""

        manager = self._policy_manager_available()
        if isinstance(manager, str):
            return manager

        try:
            result = await manager.evaluate_policy(policy_name, test_data)
        except BaseServiceException as exc:
            return self._format_error("test policy", exc)

        lines = [f"Policy '{policy_name}' Test Result:"]
        lines.append(f"Decision: {result.decision}")
        lines.append(f"Confidence: {result.confidence}")
        if result.violations:
            lines.append("Violations:")
            lines.extend(f"  - {violation}" for violation in result.violations)
        return "\n".join(lines)

    def _policy_manager_available(self) -> "PolicyManager | str":
        manager = getattr(self.service, "policy_manager", None)
        if not manager:
            return "Policy management is disabled"
        return manager

    @staticmethod
    def _format_error(action: str, exc: BaseServiceException) -> str:
        logger.error("Failed to %s: %s", action, exc)
        return f"Error attempting to {action}: {exc}"


__all__ = ["PolicyCLI"]

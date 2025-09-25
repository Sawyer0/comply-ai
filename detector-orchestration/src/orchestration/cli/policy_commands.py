"""Policy management CLI commands following SRP.

This module provides ONLY policy management CLI commands.
Single Responsibility: Handle CLI commands for policy operations.
"""

import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class PolicyCLI:
    """CLI commands for policy management.

    Single Responsibility: Provide CLI interface for policy operations.
    Does NOT handle: business logic, validation, orchestration.
    """

    def __init__(self, orchestration_service):
        """Initialize policy CLI.

        Args:
            orchestration_service: OrchestrationService instance
        """
        self.service = orchestration_service

    async def list_policies(self, output_format: str = "table") -> str:
        """List all policies.

        Args:
            output_format: Output format (table, json, yaml)

        Returns:
            Formatted policy list
        """
        try:
            if not self.service.policy_manager:
                return "Policy management is disabled"

            policies = await self.service.policy_manager.list_policies()

            if output_format == "json":
                return json.dumps(policies, indent=2)

            if output_format == "yaml":
                import yaml

                return yaml.dump(policies, default_flow_style=False)

            # Table format
            if not policies:
                return "No policies configured."

            output = "Active Policies:\\n"
            output += "=" * 50 + "\\n"
            for i, policy in enumerate(policies, 1):
                output += f"{i:2d}. {policy.get('name', 'Unnamed')}\\n"
                output += f"    Type: {policy.get('type', 'Unknown')}\\n"
                output += f"    Enabled: {policy.get('enabled', False)}\\n"
                output += f"    Priority: {policy.get('priority', 0)}\\n\\n"

            return output

        except Exception as e:
            logger.error("Failed to list policies: %s", str(e))
            return f"Error listing policies: {str(e)}"

    async def validate_policy(self, policy_file: str) -> str:
        """Validate a policy file.

        Args:
            policy_file: Path to policy file

        Returns:
            Validation result
        """
        try:
            if not self.service.policy_manager:
                return "Policy management is disabled"

            with open(policy_file, "r", encoding="utf-8") as f:
                policy_content = f.read()

            validation_result = await self.service.policy_manager.validate_policy(
                policy_content
            )

            if validation_result.is_valid:
                return f"Policy '{policy_file}' is valid"

            output = f"Policy '{policy_file}' validation failed:\\n"
            for error in validation_result.errors:
                output += f"  - {error}\\n"
            return output

        except FileNotFoundError:
            return f"Policy file '{policy_file}' not found"
        except Exception as e:
            logger.error("Failed to validate policy: %s", str(e))
            return f"Error validating policy: {str(e)}"

    async def load_policy(
        self, policy_file: str, policy_name: Optional[str] = None
    ) -> str:
        """Load a policy from file.

        Args:
            policy_file: Path to policy file
            policy_name: Optional policy name override

        Returns:
            Load result
        """
        try:
            if not self.service.policy_manager:
                return "Policy management is disabled"

            with open(policy_file, "r", encoding="utf-8") as f:
                policy_content = f.read()

            success = await self.service.policy_manager.load_policy(
                policy_content, policy_name
            )

            if success:
                name = policy_name or policy_file
                return f"Successfully loaded policy '{name}'"

            return f"Failed to load policy from '{policy_file}'"

        except FileNotFoundError:
            return f"Policy file '{policy_file}' not found"
        except Exception as e:
            logger.error("Failed to load policy: %s", str(e))
            return f"Error loading policy: {str(e)}"

    async def unload_policy(self, policy_name: str) -> str:
        """Unload a policy.

        Args:
            policy_name: Name of policy to unload

        Returns:
            Unload result
        """
        try:
            if not self.service.policy_manager:
                return "Policy management is disabled"

            success = await self.service.policy_manager.unload_policy(policy_name)

            if success:
                return f"Successfully unloaded policy '{policy_name}'"

            return f"Failed to unload policy '{policy_name}' (not found)"

        except Exception as e:
            logger.error("Failed to unload policy: %s", str(e))
            return f"Error unloading policy: {str(e)}"

    async def policy_status(self, policy_name: Optional[str] = None) -> str:
        """Show policy status.

        Args:
            policy_name: Optional specific policy to check

        Returns:
            Policy status report
        """
        try:
            if not self.service.policy_manager:
                return "Policy management is disabled"

            if policy_name:
                # Check specific policy
                policy = await self.service.policy_manager.get_policy(policy_name)
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

            # Show all policy status
            policies = await self.service.policy_manager.list_policies()
            stats = await self.service.policy_manager.get_policy_statistics()

            output = "Policy System Status:\\n"
            output += "=" * 50 + "\\n"
            output += f"Total Policies: {len(policies)}\\n"
            output += f"Active Policies: {stats.get('active_policies', 0)}\\n"
            output += f"Policy Evaluations: {stats.get('total_evaluations', 0)}\\n"
            output += f"Policy Violations: {stats.get('total_violations', 0)}\\n"

            return output

        except Exception as e:
            logger.error("Failed to get policy status: %s", str(e))
            return f"Error getting policy status: {str(e)}"

    async def test_policy(self, policy_name: str, test_data: Dict[str, Any]) -> str:
        """Test a policy with sample data.

        Args:
            policy_name: Name of policy to test
            test_data: Test data to evaluate

        Returns:
            Test result
        """
        try:
            if not self.service.policy_manager:
                return "Policy management is disabled"

            result = await self.service.policy_manager.evaluate_policy(
                policy_name, test_data
            )

            output = f"Policy '{policy_name}' Test Result:\\n"
            output += f"Decision: {result.decision}\\n"
            output += f"Confidence: {result.confidence}\\n"

            if result.violations:
                output += "Violations:\\n"
                for violation in result.violations:
                    output += f"  - {violation}\\n"

            return output

        except Exception as e:
            logger.error("Failed to test policy: %s", str(e))
            return f"Error testing policy: {str(e)}"


# Export only the policy CLI functionality
__all__ = [
    "PolicyCLI",
]

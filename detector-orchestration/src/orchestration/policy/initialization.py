"""Policy initialization and template loading.

Extracted from PolicyManager to handle only initialization concerns:
template loading, tenant data loading, and OPA bootstrapping.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict

from shared.utils.correlation import get_correlation_id
from shared.clients.opa_client import OPAError
from shared.exceptions.base import ValidationError, ServiceUnavailableError

from .opa_adapter import load_policy as _opa_load_policy, load_data as _opa_load_data
from .policy_loader import PolicyLoader, PolicyLoadError

logger = logging.getLogger(__name__)

__all__ = [
    "PolicyInitializer",
]


class PolicyInitializer:
    """Handles policy initialization and template loading.
    
    Single Responsibility: Bootstrap policy system with templates and tenant data.
    Does NOT handle: policy CRUD, evaluation, or runtime operations.
    """

    def __init__(self, policy_loader: PolicyLoader, opa_client):
        """Initialize policy initializer.
        
        Args:
            policy_loader: PolicyLoader instance for loading templates
            opa_client: OPA client for loading policies and data
        """
        self.policy_loader = policy_loader
        self.opa_client = opa_client
        self._template_status: Dict[str, Dict[str, Any]] = {}

    def load_policy_templates(self) -> None:
        """Load policy templates from directory and queue them for OPA loading."""
        try:
            loaded_policies = self.policy_loader.load_all_policies()

            if loaded_policies:
                # Queue policy templates for loading to OPA
                for policy_name, policy_content in loaded_policies.items():
                    self._template_status[policy_name] = {
                        "content": policy_content,
                        "status": "template_loaded",
                        "loaded_at": time.time(),
                    }

                logger.info(
                    "Policy templates loaded successfully",
                    extra={
                        "count": len(loaded_policies),
                        "templates": list(loaded_policies.keys()),
                    },
                )

                # Schedule loading templates to OPA in background
                asyncio.create_task(self._load_templates_to_opa(loaded_policies))
            else:
                logger.warning("No policy templates found")

        except PolicyLoadError as e:
            logger.error("Failed to load policy templates", extra={"error": str(e)})

    async def _load_templates_to_opa(self, templates: Dict[str, str]):
        """Load policy templates to OPA in background.

        Single Responsibility: Load templates to OPA without blocking initialization.
        """
        correlation_id = get_correlation_id()

        for policy_name, policy_content in templates.items():
            try:
                success = await _opa_load_policy(
                    client=self.opa_client,
                    policy_content=policy_content,
                    policy_name=policy_name,
                    correlation_id=correlation_id,
                )

                if success and policy_name in self._template_status:
                    self._template_status[policy_name]["status"] = "loaded_to_opa"

                logger.debug(
                    "Template loaded to OPA",
                    extra={
                        "policy_name": policy_name,
                        "success": success,
                        "correlation_id": correlation_id,
                    },
                )

            except (OPAError, ValidationError, ServiceUnavailableError) as e:
                logger.error(
                    "Failed to load template to OPA",
                    extra={
                        "policy_name": policy_name,
                        "correlation_id": correlation_id,
                        "error": str(e),
                    },
                )

                if policy_name in self._template_status:
                    self._template_status[policy_name]["status"] = "load_failed"

    async def load_tenant_policies_to_opa(self):
        """Load tenant policy data into OPA for policy evaluation.

        Single Responsibility: Initialize OPA with tenant configuration data.
        """
        correlation_id = get_correlation_id()

        try:
            # Load tenant policy data from policy loader
            tenant_data = self.policy_loader.load_tenant_policy_data()

            if tenant_data:
                # Load tenant policies into OPA data store
                success = await _opa_load_data(
                    client=self.opa_client,
                    data_path="tenant_policies",
                    data=tenant_data,
                    correlation_id=correlation_id,
                )

                if success:
                    logger.info(
                        "Tenant policy data loaded to OPA successfully",
                        extra={
                            "correlation_id": correlation_id,
                            "tenant_count": len(tenant_data),
                            "tenants": list(tenant_data.keys()),
                        },
                    )
                else:
                    logger.error(
                        "Failed to load tenant policy data to OPA",
                        extra={"correlation_id": correlation_id},
                    )
            else:
                logger.warning(
                    "No tenant policy data found to load",
                    extra={"correlation_id": correlation_id},
                )

        except (PolicyLoadError, OPAError, ValidationError, ServiceUnavailableError) as e:
            logger.error(
                "Failed to load tenant policies to OPA",
                extra={
                    "correlation_id": correlation_id,
                    "error": str(e),
                },
            )

    async def reload_all_policies(self) -> bool:
        """Reload all policies from templates and tenant data.

        Single Responsibility: Refresh all policy data in OPA.
        """
        correlation_id = get_correlation_id()

        try:
            # Reload policy templates
            self.load_policy_templates()

            # Reload tenant data to OPA
            await self.load_tenant_policies_to_opa()

            logger.info(
                "Policies reloaded successfully",
                extra={
                    "correlation_id": correlation_id,
                    "template_count": len(self._template_status),
                },
            )

            return True

        except (PolicyLoadError, OPAError, ValidationError, ServiceUnavailableError) as e:
            logger.error(
                "Failed to reload policies",
                extra={
                    "correlation_id": correlation_id,
                    "error": str(e),
                },
            )
            return False

    def get_template_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current template loading status.
        
        Returns:
            Dictionary mapping template names to their status
        """
        return self._template_status.copy()

    def get_template_count(self) -> int:
        """Get number of loaded templates.
        
        Returns:
            Number of templates currently tracked
        """
        return len(self._template_status)

    def clear_template_status(self) -> None:
        """Clear all template status tracking.
        
        Warning: This removes all template status. Use with caution.
        """
        self._template_status.clear()
        logger.warning("All template status cleared")

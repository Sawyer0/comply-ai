"""OPA interaction helpers for policy management.

This adapter wraps the shared `opa_client` with thin convenience
functions so that the bulky transport logic can be moved out of
`policy_manager.py`.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from shared.clients.opa_client import OPAError, OPAPolicyError
from shared.exceptions.base import ServiceUnavailableError, ValidationError

logger = logging.getLogger(__name__)

# Public API re-export
__all__ = [
    "query_policy",
    "load_policy",
    "unload_policy",
    "load_data",
]


async def query_policy(
    *,
    client,
    tenant_id: str,
    bundle: str,
    correlation_id: str,
) -> Dict[str, Any]:
    """Retrieve a tenant/bundle policy from OPA, returning an empty mapping if not found."""

    policy_path = f"tenant_policies/{tenant_id}/{bundle}"
    input_data = {
        "tenant_id": tenant_id,
        "bundle": bundle,
        "query_type": "policy_retrieval",
        "timestamp": correlation_id,
    }

    try:
        result = await client.evaluate_policy(
            policy_path=policy_path,
            input_data=input_data,
            tenant_id=tenant_id,
            correlation_id=correlation_id,
        )
        return result.get("result", {}) if isinstance(result, dict) else {}

    except (OPAError, ValidationError, ServiceUnavailableError) as err:
        logger.warning(
            "OPA query failed",
            extra={
                "tenant_id": tenant_id,
                "bundle": bundle,
                "correlation_id": correlation_id,
                "error": str(err),
            },
        )
        raise


async def load_policy(
    *,
    client,
    policy_content: str,
    policy_name: str,
    correlation_id: str,
) -> bool:
    """Load a Rego policy into OPA."""
    try:
        return await client.load_policy(
            policy_content=policy_content,
            policy_name=policy_name,
            correlation_id=correlation_id,
        )
    except (
        OPAError,
        OPAPolicyError,
        ValidationError,
        ServiceUnavailableError,
    ) as err:
        logger.error(
            "OPA policy load failed",
            extra={"policy_name": policy_name, "correlation_id": correlation_id, "error": str(err)},
        )
        return False


async def unload_policy(
    *,
    client,
    policy_name: str,
    correlation_id: str,
) -> bool:
    """Remove a policy from OPA."""
    try:
        return await client.unload_policy(
            policy_name=policy_name, correlation_id=correlation_id
        )
    except (
        OPAError,
        OPAPolicyError,
        ValidationError,
        ServiceUnavailableError,
    ) as err:
        logger.error(
            "OPA policy unload failed",
            extra={"policy_name": policy_name, "correlation_id": correlation_id, "error": str(err)},
        )
        return False


async def load_data(
    *,
    client,
    data_path: str,
    data: Any,
    correlation_id: str,
) -> bool:
    """Load JSON data into OPA data store."""
    try:
        return await client.load_data(
            data_path=data_path, data=data, correlation_id=correlation_id
        )
    except (OPAError, ValidationError, ServiceUnavailableError) as err:
        logger.error(
            "OPA data load failed",
            extra={"data_path": data_path, "correlation_id": correlation_id, "error": str(err)},
        )
        return False

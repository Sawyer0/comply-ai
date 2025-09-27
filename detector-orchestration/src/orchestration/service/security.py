"""Security helpers for the orchestration service."""

from __future__ import annotations

import logging
from typing import Optional

from shared.exceptions.base import BaseServiceException
from shared.interfaces.orchestration import OrchestrationRequest
from shared.utils.correlation import get_correlation_id

from ..security import Permission

logger = logging.getLogger(__name__)


def deny_security_request(
    service,
    *,
    message: str,
    log_args,
    correlation_id: str,
    tenant_id: str,
    violation_type: str = "security_violation",
    severity: str = "medium",
    **details,
) -> bool:
    """Log and record a standardized security denial."""
    logger.warning(
        message,
        *log_args,
        extra=service._log_extra(  # noqa: SLF001
            correlation_id,
            tenant_id=tenant_id,
            violation_type=violation_type,
            severity=severity,
            **details,
        ),
    )

    metrics = service.components.metrics_collector
    if metrics:
        metrics.record_security_violation(
            tenant_id=tenant_id, violation_type=violation_type, severity=severity
        )

    return False


async def validate_request_security(
    service,
    request: OrchestrationRequest,
    tenant_id: str,
    api_key: Optional[str] = None,
    user_id: Optional[str] = None,
) -> bool:
    """Validate request security including authorization and sanitization."""
    correlation_id = get_correlation_id()

    try:
        if (
            service.components.tenant_manager
            and not service.components.tenant_manager.is_tenant_active(tenant_id)
        ):
            return deny_security_request(
                service,
                message="Request denied: tenant %s is not active",
                log_args=(tenant_id,),
                correlation_id=correlation_id,
                tenant_id=tenant_id,
                violation_type="inactive_tenant",
                severity="high",
            )

        if api_key:
            api_key_manager = service.components.api_key_manager
            if api_key_manager:
                api_key_obj = api_key_manager.validate_api_key(api_key)
                key_valid = api_key_obj and api_key_obj.tenant_id == tenant_id

                if service.metrics_collector:
                    service.metrics_collector.record_api_key_validation(
                        tenant_id=tenant_id,
                        status="success" if key_valid else "failure",
                    )

                if not key_valid:
                    return deny_security_request(
                        service,
                        message="Request denied: invalid API key for tenant %s",
                        log_args=(tenant_id,),
                        correlation_id=correlation_id,
                        tenant_id=tenant_id,
                        violation_type="invalid_api_key",
                        severity="medium",
                    )

        attack_detector = service.components.attack_detector
        if attack_detector:
            attack_detections = attack_detector.detect_attacks(request.content)
            if attack_detector.has_high_severity_attacks(request.content):
                attack_types = [d.attack_type.value for d in attack_detections]
                return deny_security_request(
                    service,
                    message="Request denied: high-severity attack patterns for tenant %s",
                    log_args=(tenant_id,),
                    correlation_id=correlation_id,
                    tenant_id=tenant_id,
                    violation_type="attack_detected",
                    severity="critical",
                    attack_types=attack_types,
                )

        if service.components.input_sanitizer:
            request.content = service.components.input_sanitizer.sanitize(
                request.content
            )

        if (
            user_id
            and service.components.rbac_manager
            and not service.components.rbac_manager.check_permission(
                user_id, tenant_id, Permission.ORCHESTRATE_DETECTORS
            )
        ):
            return deny_security_request(
                service,
                message="Request denied: user %s lacks orchestration access for tenant %s",
                log_args=(user_id, tenant_id),
                correlation_id=correlation_id,
                tenant_id=tenant_id,
                violation_type="permission_denied",
                severity="medium",
                user_id=user_id,
            )

        return True

    except (
        BaseServiceException,
        ValueError,
    ) as exc:  # pragma: no cover - logged side effect
        logger.error(
            "Security validation failed: %s",
            str(exc),
            extra=service._log_extra(  # noqa: SLF001
                correlation_id,
                tenant_id=tenant_id,
                error=str(exc),
            ),
        )
        return False

"""Detector registration helpers."""

# pylint: disable=protected-access
from __future__ import annotations

import logging

from shared.exceptions.base import BaseServiceException
from shared.utils.correlation import get_correlation_id

from ..core import CustomerDetectorClient, DetectorClientConfig, DetectorConfig
from ..discovery.service_discovery import ServiceMetadata, ServiceRegistration
from ..utils.detector_response import get_response_parser
from .models import DetectorRegistrationConfig

logger = logging.getLogger(__name__)


async def register_detector(service, registration: DetectorRegistrationConfig) -> bool:
    """Register a new detector with all relevant components."""
    correlation_id = get_correlation_id()

    logger.info(
        "Registering detector %s at %s",
        registration.detector_id,
        registration.endpoint,
        extra=service._log_extra(  # noqa: SLF001
            correlation_id,
            detector_id=registration.detector_id,
            endpoint=registration.endpoint,
            detector_type=registration.detector_type,
        ),
    )

    analyze_endpoint = (
        registration.endpoint.rstrip("/")
        if registration.analyze_path
        else registration.endpoint
    )
    if registration.analyze_path:
        analyze_endpoint = f"{analyze_endpoint}/{registration.analyze_path.lstrip('/')}"

    try:
        if service.components.content_router:
            router_success = service.components.content_router.register_detector(
                DetectorConfig(
                    name=registration.detector_id,
                    endpoint=registration.endpoint,
                    timeout_ms=registration.timeout_ms,
                    max_retries=registration.max_retries,
                    supported_content_types=registration.supported_content_types or ["text"],
                )
            )

            if not router_success:
                logger.error(
                    "Failed to register detector with router: %s",
                    registration.detector_id,
                )
                return False

        if service.components.service_discovery:
            metadata = ServiceMetadata(
                timeout_ms=registration.timeout_ms,
                max_retries=registration.max_retries,
                supported_content_types=registration.supported_content_types or [],
                analyze_path=registration.analyze_path,
                response_parser=registration.response_parser,
                auth_headers=registration.auth_headers,
            )
            discovery_success = service.components.service_discovery.register_service(
                ServiceRegistration(
                    service_id=registration.detector_id,
                    endpoint_url=registration.endpoint,
                    service_type=registration.detector_type,
                    metadata=metadata.to_dict(),
                )
            )

            if not discovery_success:
                logger.error(
                    "Failed to register detector with service discovery: %s",
                    registration.detector_id,
                )
                return False

        timeout_seconds = max(registration.timeout_ms / 1000.0, 0.1)
        existing_client = service.components.detector_clients.get(
            registration.detector_id
        )
        if existing_client:
            await existing_client.close()

        parser = get_response_parser(registration.response_parser)
        client_config = DetectorClientConfig(
            name=registration.detector_id,
            endpoint=analyze_endpoint,
            timeout=timeout_seconds,
            max_retries=registration.max_retries,
            default_headers=registration.auth_headers or {},
            response_parser=parser,
        )

        service.components.detector_clients[registration.detector_id] = (
            CustomerDetectorClient(client_config)
        )

        logger.info(
            "Successfully registered detector %s",
            registration.detector_id,
            extra=service._log_extra(  # noqa: SLF001
                correlation_id,
                detector_id=registration.detector_id,
                analyze_endpoint=analyze_endpoint,
                response_parser=registration.response_parser or "default",
            ),
        )

        return True

    except (BaseServiceException, ValueError) as exc:
        logger.error(
            "Failed to register detector %s: %s",
            registration.detector_id,
            str(exc),
            extra=service._log_extra(  # noqa: SLF001
                correlation_id, detector_id=registration.detector_id, error=str(exc)
            ),
        )
        return False


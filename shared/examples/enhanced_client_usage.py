"""Example usage of enhanced request/response models with proper validation."""

import asyncio
import logging
from datetime import datetime
from typing import List

from ..clients.client_factory import ClientFactory, ClientConfig
from ..interfaces.orchestration import (
    OrchestrationRequest,
    ProcessingMode,
    OrchestrationResponse,
)
from ..interfaces.analysis import AnalysisRequest, AnalysisType, AnalysisResponse
from ..interfaces.mapper import MappingRequest, MappingMode, MappingResponse
from ..validation.decorators import validate_request_response, validate_tenant_access
from ..validation.schemas import ValidationContext, default_validator
from ..utils.correlation import set_correlation_id
from ..exceptions.base import ValidationError, ServiceUnavailableError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedMicroserviceClient:
    """Example client demonstrating enhanced validation and error handling."""

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id

        # Create client factory with enhanced configuration
        config = ClientConfig(
            timeout=60.0,
            max_retries=3,
            enable_circuit_breaker=True,
            circuit_breaker_threshold=5,
            circuit_breaker_timeout=60.0,
        )

        self.factory = ClientFactory(default_config=config)

        # Register Pydantic models with validator
        default_validator.register_pydantic_model(
            "OrchestrationRequest", OrchestrationRequest
        )
        default_validator.register_pydantic_model("AnalysisRequest", AnalysisRequest)
        default_validator.register_pydantic_model("MappingRequest", MappingRequest)

    @validate_request_response(
        request_model=OrchestrationRequest,
        response_model=OrchestrationResponse,
        validate_tenant=True,
    )
    async def orchestrate_detectors(
        self,
        content: str,
        detector_types: List[str],
        tenant_id: str,
        request: OrchestrationRequest = None,
    ) -> OrchestrationResponse:
        """Orchestrate detector execution with enhanced validation."""

        # Create request if not provided
        if request is None:
            request = OrchestrationRequest(
                content=content,
                detector_types=detector_types,
                processing_mode=ProcessingMode.STANDARD,
            )

        # Create orchestration client
        orchestration_client = self.factory.create_orchestration_client()

        try:
            # Call orchestration service
            response = await orchestration_client.orchestrate_detectors(
                request=request, tenant_id=tenant_id
            )

            logger.info(
                "Orchestration completed successfully",
                extra={
                    "tenant_id": tenant_id,
                    "detector_count": len(response.detector_results),
                    "success": response.success,
                },
            )

            return response

        except ValidationError as e:
            logger.error("Validation error in orchestration: %s", e.message)
            raise
        except ServiceUnavailableError as e:
            logger.error("Orchestration service unavailable: %s", e.message)
            raise
        finally:
            await orchestration_client.close()

    @validate_request_response(
        request_model=AnalysisRequest,
        response_model=AnalysisResponse,
        validate_tenant=True,
    )
    async def analyze_content(
        self,
        orchestration_response: OrchestrationResponse,
        analysis_types: List[AnalysisType],
        tenant_id: str,
        request: AnalysisRequest = None,
    ) -> AnalysisResponse:
        """Perform content analysis with enhanced validation."""

        # Create request if not provided
        if request is None:
            request = AnalysisRequest(
                orchestration_response=orchestration_response,
                analysis_types=analysis_types,
                include_recommendations=True,
            )

        # Create analysis client
        analysis_client = self.factory.create_analysis_client()

        try:
            # Call analysis service
            response = await analysis_client.analyze_content(
                request=request, tenant_id=tenant_id
            )

            logger.info(
                "Analysis completed successfully",
                extra={
                    "tenant_id": tenant_id,
                    "canonical_results_count": len(response.canonical_results),
                    "success": response.success,
                },
            )

            return response

        except ValidationError as e:
            logger.error("Validation error in analysis: %s", e.message)
            raise
        except ServiceUnavailableError as e:
            logger.error("Analysis service unavailable: %s", e.message)
            raise
        finally:
            await analysis_client.close()

    @validate_request_response(
        request_model=MappingRequest,
        response_model=MappingResponse,
        validate_tenant=True,
    )
    async def map_content(
        self,
        analysis_response: AnalysisResponse,
        target_frameworks: List[str],
        tenant_id: str,
        request: MappingRequest = None,
    ) -> MappingResponse:
        """Perform content mapping with enhanced validation."""

        # Create request if not provided
        if request is None:
            request = MappingRequest(
                analysis_response=analysis_response,
                target_frameworks=target_frameworks,
                mapping_mode=MappingMode.STANDARD,
                include_validation=True,
            )

        # Create mapper client
        mapper_client = self.factory.create_mapper_client()

        try:
            # Call mapper service
            response = await mapper_client.map_content(
                request=request, tenant_id=tenant_id
            )

            logger.info(
                "Mapping completed successfully",
                extra={
                    "tenant_id": tenant_id,
                    "mapping_results_count": len(response.mapping_results),
                    "overall_confidence": response.overall_confidence,
                    "success": response.success,
                },
            )

            return response

        except ValidationError as e:
            logger.error("Validation error in mapping: %s", e.message)
            raise
        except ServiceUnavailableError as e:
            logger.error("Mapper service unavailable: %s", e.message)
            raise
        finally:
            await mapper_client.close()

    @validate_tenant_access(require_admin=False)
    async def process_complete_workflow(
        self,
        content: str,
        detector_types: List[str],
        analysis_types: List[AnalysisType],
        target_frameworks: List[str],
        tenant_id: str,
    ) -> MappingResponse:
        """Process complete workflow from orchestration to mapping."""

        # Set correlation ID for the entire workflow
        correlation_id = set_correlation_id(f"workflow_{datetime.utcnow().isoformat()}")

        logger.info(
            "Starting complete workflow",
            extra={
                "tenant_id": tenant_id,
                "correlation_id": correlation_id,
                "detector_types": detector_types,
                "analysis_types": analysis_types,
                "target_frameworks": target_frameworks,
            },
        )

        try:
            # Step 1: Orchestrate detectors
            orchestration_response = await self.orchestrate_detectors(
                content=content, detector_types=detector_types, tenant_id=tenant_id
            )

            # Step 2: Analyze results
            analysis_response = await self.analyze_content(
                orchestration_response=orchestration_response,
                analysis_types=analysis_types,
                tenant_id=tenant_id,
            )

            # Step 3: Map to compliance frameworks
            mapping_response = await self.map_content(
                analysis_response=analysis_response,
                target_frameworks=target_frameworks,
                tenant_id=tenant_id,
            )

            logger.info(
                "Complete workflow finished successfully",
                extra={
                    "tenant_id": tenant_id,
                    "correlation_id": correlation_id,
                    "final_confidence": mapping_response.overall_confidence,
                },
            )

            return mapping_response

        except Exception as e:
            logger.error(
                "Workflow failed: %s",
                str(e),
                extra={"tenant_id": tenant_id, "correlation_id": correlation_id},
            )
            raise


async def example_usage():
    """Example usage of the enhanced client."""

    # Create client
    client = EnhancedMicroserviceClient(tenant_id="example-tenant")

    try:
        # Process complete workflow
        result = await client.process_complete_workflow(
            content="This is sensitive data that needs to be analyzed",
            detector_types=["presidio", "deberta", "custom"],
            analysis_types=[
                AnalysisType.PATTERN,
                AnalysisType.RISK,
                AnalysisType.COMPLIANCE,
            ],
            target_frameworks=["SOC2", "ISO27001", "HIPAA"],
            tenant_id="example-tenant",
        )

        print(f"Workflow completed with confidence: {result.overall_confidence}")
        print(f"Number of mapping results: {len(result.mapping_results)}")

        # Display results
        for i, mapping_result in enumerate(result.mapping_results):
            print(f"\nMapping Result {i + 1}:")
            print(f"  Category: {mapping_result.canonical_result.category}")
            print(f"  Confidence: {mapping_result.confidence}")
            print(f"  Framework mappings: {len(mapping_result.framework_mappings)}")
            print(f"  Validation passed: {mapping_result.validation_result.is_valid}")

    except ValidationError as e:
        print(f"Validation error: {e.message}")
        print(f"Details: {e.details}")
    except ServiceUnavailableError as e:
        print(f"Service unavailable: {e.message}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def example_validation_context():
    """Example of using validation context."""

    # Create validation context
    context = ValidationContext(
        tenant_id="example-tenant", strict_mode=True, allow_extra_fields=False
    )

    # Example request data
    request_data = {
        "content": "Test content",
        "detector_types": ["presidio", "deberta"],
        "processing_mode": "standard",
    }

    try:
        # Validate request
        validated_request = default_validator.validate_request(
            data=request_data, model_name="OrchestrationRequest", context=context
        )

        print(f"Validation successful: {validated_request}")

    except ValidationError as e:
        print(f"Validation failed: {e.message}")
        print(f"Details: {e.details}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())

    # Run validation example
    example_validation_context()

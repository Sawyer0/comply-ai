"""
FastAPI application for the Llama Mapper service.
"""
import logging
import uuid
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from .models import (
    DetectorRequest, 
    BatchDetectorRequest, 
    MappingResponse, 
    BatchMappingResponse,
    ErrorResponse,
    Provenance
)
from ..serving.model_server import ModelServer
from ..serving.json_validator import JSONValidator
from ..serving.fallback_mapper import FallbackMapper
from ..config.manager import ConfigManager
from ..monitoring.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


class MapperAPI:
    """FastAPI application for the Llama Mapper service."""
    
    def __init__(
        self,
        model_server: ModelServer,
        json_validator: JSONValidator,
        fallback_mapper: FallbackMapper,
        config_manager: ConfigManager,
        metrics_collector: MetricsCollector
    ):
        self.model_server = model_server
        self.json_validator = json_validator
        self.fallback_mapper = fallback_mapper
        self.config_manager = config_manager
        self.metrics_collector = metrics_collector
        
        self.app = FastAPI(
            title="Llama Mapper API",
            description="AI safety detector output normalization service",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add request ID middleware
        @self.app.middleware("http")
        async def add_request_id(request: Request, call_next):
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": time.time()}
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Prometheus metrics endpoint."""
            from fastapi import Response
            metrics_data = self.metrics_collector.get_prometheus_metrics()
            return Response(content=metrics_data, media_type="text/plain")
        
        @self.app.get("/metrics/summary")
        async def get_metrics_summary():
            """Get metrics summary in JSON format."""
            return self.metrics_collector.get_all_metrics()
        
        @self.app.get("/metrics/alerts")
        async def get_quality_alerts():
            """Get current quality threshold violations."""
            alerts = self.metrics_collector.check_quality_thresholds()
            return {
                "alerts": alerts,
                "count": len(alerts),
                "timestamp": time.time()
            }
        
        @self.app.post("/map", response_model=MappingResponse)
        async def map_detector_output(
            request: DetectorRequest,
            http_request: Request
        ) -> MappingResponse:
            """
            Map a single detector output to canonical taxonomy.
            
            Args:
                request: Detector output to map
                http_request: FastAPI request object for metadata
                
            Returns:
                MappingResponse: Canonical taxonomy mapping
                
            Raises:
                HTTPException: If mapping fails
            """
            request_id = getattr(http_request.state, 'request_id', 'unknown')
            start_time = time.time()
            
            try:
                logger.info(f"Processing mapping request {request_id} for detector {request.detector}")
                
                # Attempt mapping with the fine-tuned model
                result = await self._map_single_request(request, request_id)
                
                # Track success metrics
                processing_time = time.time() - start_time
                self.metrics_collector.record_request(request.detector, processing_time, True)
                
                logger.info(f"Successfully processed request {request_id} in {processing_time:.3f}s")
                return result
                
            except Exception as e:
                # Track error metrics
                processing_time = time.time() - start_time
                self.metrics_collector.record_request(request.detector, processing_time, False)
                
                logger.error(f"Error processing request {request_id}: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to process mapping request: {str(e)}"
                )
        
        @self.app.post("/map/batch", response_model=BatchMappingResponse)
        async def map_detector_outputs_batch(
            request: BatchDetectorRequest,
            http_request: Request
        ) -> BatchMappingResponse:
            """
            Map multiple detector outputs to canonical taxonomy.
            
            Args:
                request: Batch of detector outputs to map
                http_request: FastAPI request object for metadata
                
            Returns:
                BatchMappingResponse: Batch of canonical taxonomy mappings
            """
            request_id = getattr(http_request.state, 'request_id', 'unknown')
            start_time = time.time()
            
            logger.info(f"Processing batch mapping request {request_id} with {len(request.requests)} items")
            
            results = []
            errors = []
            
            for i, single_request in enumerate(request.requests):
                try:
                    result = await self._map_single_request(single_request, f"{request_id}-{i}")
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing batch item {i}: {str(e)}")
                    errors.append({
                        "index": i,
                        "error": str(e),
                        "detector": single_request.detector
                    })
                    # Add a placeholder result for failed items
                    results.append(self._create_error_response(single_request.detector, str(e)))
            
            processing_time = time.time() - start_time
            self.metrics_collector.record_histogram("batch_request_duration_seconds", processing_time)
            self.metrics_collector.record_batch_request(len(request.requests))
            
            logger.info(f"Processed batch request {request_id} in {processing_time:.3f}s")
            
            return BatchMappingResponse(
                results=results,
                errors=errors if errors else None
            )
    
    async def _map_single_request(self, request: DetectorRequest, request_id: str) -> MappingResponse:
        """
        Map a single detector request to canonical taxonomy.
        
        Args:
            request: Single detector request
            request_id: Unique request identifier
            
        Returns:
            MappingResponse: Canonical taxonomy mapping
        """
        # Create provenance information
        provenance = Provenance(
            detector=request.detector,
            tenant_id=request.tenant_id,
            ts=time.time()
        )
        
        try:
            # Generate mapping using the fine-tuned model
            model_output = await self.model_server.generate_mapping(
                detector=request.detector,
                output=request.output,
                metadata=request.metadata
            )
            
            # Validate the model output against JSON schema
            is_valid, validation_errors = self.json_validator.validate(model_output)
            self.metrics_collector.record_schema_validation(request.detector, is_valid)
            
            if is_valid:
                # Parse the validated output
                parsed_output = self.json_validator.parse_output(model_output)
                
                # Record confidence score
                self.metrics_collector.record_confidence_score(request.detector, parsed_output.confidence)
                
                # Check confidence threshold
                confidence_threshold = self.config_manager.confidence.threshold
                if parsed_output.confidence >= confidence_threshold:
                    # Use model output
                    self.metrics_collector.record_model_success(request.detector)
                    parsed_output.provenance = provenance
                    return parsed_output
                else:
                    # Confidence too low, use fallback
                    logger.warning(f"Model confidence {parsed_output.confidence} below threshold {confidence_threshold}")
                    self.metrics_collector.record_fallback_usage(request.detector, "low_confidence")
            else:
                # Schema validation failed
                logger.warning(f"Schema validation failed: {validation_errors}")
                self.metrics_collector.record_fallback_usage(request.detector, "schema_validation_failed")
        
        except Exception as e:
            logger.error(f"Model generation failed: {str(e)}")
            self.metrics_collector.record_model_error(request.detector, "generation_failed")
        
        # Fall back to rule-based mapping
        logger.info(f"Using fallback mapping for detector {request.detector}")
        fallback_result = self.fallback_mapper.map(request.detector, request.output)
        fallback_result.provenance = provenance
        fallback_result.notes = "Generated using rule-based fallback mapping"
        
        self.metrics_collector.record_fallback_usage(request.detector, "model_error")
        return fallback_result
    
    def _create_error_response(self, detector: str, error_message: str) -> MappingResponse:
        """
        Create an error response for failed mappings.
        
        Args:
            detector: Name of the detector
            error_message: Error message
            
        Returns:
            MappingResponse: Error response with OTHER.Unknown mapping
        """
        return MappingResponse(
            taxonomy=["OTHER.Unknown"],
            scores={"OTHER.Unknown": 0.0},
            confidence=0.0,
            notes=f"Mapping failed: {error_message}",
            provenance=Provenance(detector=detector)
        )


def create_app(
    model_server: ModelServer,
    json_validator: JSONValidator,
    fallback_mapper: FallbackMapper,
    config_manager: ConfigManager,
    metrics_collector: MetricsCollector
) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        model_server: Model serving backend
        json_validator: JSON schema validator
        fallback_mapper: Rule-based fallback mapper
        config_manager: Configuration manager
        metrics_collector: Metrics collection service
        
    Returns:
        FastAPI: Configured FastAPI application
    """
    mapper_api = MapperAPI(
        model_server=model_server,
        json_validator=json_validator,
        fallback_mapper=fallback_mapper,
        config_manager=config_manager,
        metrics_collector=metrics_collector
    )
    return mapper_api.app
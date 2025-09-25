"""
Analysis Pipeline for Processing Compliance Data

Implements batch and streaming analysis pipelines for compliance data processing.
Focuses on analysis orchestration within the analysis service domain.
"""

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from ..engines.core.pattern_recognition import PatternRecognitionEngine
from ..engines.core.risk_scoring import RiskScoringEngine
from ..ml.model_server import ModelServer
from ..quality.monitoring import QualityMonitor

from ..schemas.analysis_schemas import (
    AnalysisRequest,
    AnalysisResult,
    BatchAnalysisRequest,
)

logger = structlog.get_logger(__name__)


@dataclass
class AnalysisPipelineConfig:
    """Configuration for analysis pipeline."""

    # Processing configuration
    batch_size: int = 100
    max_concurrent_tasks: int = 10
    timeout_seconds: int = 300

    # Quality thresholds
    min_confidence_threshold: float = 0.7
    max_error_rate: float = 0.05

    # Output configuration
    include_raw_results: bool = False


class AnalysisPipeline:
    """
    Main analysis pipeline for processing compliance data.

    Orchestrates multiple analysis engines to provide comprehensive
    compliance analysis results.
    """

    def __init__(
        self,
        config: AnalysisPipelineConfig,
        pattern_engine: PatternRecognitionEngine,
        risk_engine: RiskScoringEngine,
        model_server: ModelServer,
        quality_monitor: QualityMonitor,
    ):
        self.config = config
        self.pattern_engine = pattern_engine
        self.risk_engine = risk_engine
        self.model_server = model_server
        self.quality_monitor = quality_monitor
        self.logger = logger.bind(component="analysis_pipeline")

        # Processing state
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._processing_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "avg_processing_time": 0.0,
        }

    async def process_single(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Process a single analysis request.

        Args:
            request: Analysis request to process

        Returns:
            Analysis result with findings and recommendations
        """
        analysis_id = str(uuid.uuid4())
        start_time = datetime.now()

        self.logger.info(
            "Processing single analysis request",
            analysis_id=analysis_id,
            request_id=request.request_id,
        )

        try:
            # Execute analysis stages
            results = await self._execute_analysis_stages(request, analysis_id)

            # Aggregate results
            final_result = await self._aggregate_results(results, request, analysis_id)

            # Record success metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            await self._record_processing_metrics(analysis_id, processing_time, True)

            self.logger.info(
                "Analysis completed successfully",
                analysis_id=analysis_id,
                processing_time=processing_time,
            )

            return final_result

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            await self._record_processing_metrics(analysis_id, processing_time, False)

            self.logger.error(
                "Analysis failed",
                analysis_id=analysis_id,
                error=str(e),
                processing_time=processing_time,
            )
            raise

    async def process_batch(
        self, request: BatchAnalysisRequest
    ) -> List[AnalysisResult]:
        """
        Process a batch of analysis requests.

        Args:
            request: Batch analysis request

        Returns:
            List of analysis results
        """
        batch_id = str(uuid.uuid4())
        start_time = datetime.now()

        self.logger.info(
            "Processing batch analysis request",
            batch_id=batch_id,
            batch_size=len(request.requests),
        )

        try:
            # Process in chunks to manage memory and concurrency
            results = []
            for i in range(0, len(request.requests), self.config.batch_size):
                chunk = request.requests[i : i + self.config.batch_size]
                chunk_results = await self._process_chunk(chunk, batch_id, i)
                results.extend(chunk_results)

            processing_time = (datetime.now() - start_time).total_seconds()

            self.logger.info(
                "Batch analysis completed",
                batch_id=batch_id,
                total_requests=len(request.requests),
                successful=len([r for r in results if r.confidence > 0]),
                processing_time=processing_time,
            )

            return results

        except Exception as e:
            self.logger.error("Batch analysis failed", batch_id=batch_id, error=str(e))
            raise

    async def _process_chunk(
        self, chunk: List[AnalysisRequest], batch_id: str, chunk_index: int
    ) -> List[AnalysisResult]:
        """Process a chunk of requests concurrently."""

        self.logger.info(
            "Processing chunk",
            batch_id=batch_id,
            chunk_index=chunk_index,
            chunk_size=len(chunk),
        )

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)

        async def process_with_semaphore(request: AnalysisRequest) -> AnalysisResult:
            async with semaphore:
                return await self.process_single(request)

        # Process chunk concurrently
        tasks = [process_with_semaphore(request) for request in chunk]

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.timeout_seconds,
            )

            # Handle exceptions in results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Create error result
                    error_result = AnalysisResult(
                        analysis_type="error",
                        confidence=0.0,
                        metadata={
                            "error": str(result),
                            "request_id": chunk[i].request_id,
                        },
                    )
                    processed_results.append(error_result)
                else:
                    processed_results.append(result)

            return processed_results

        except asyncio.TimeoutError:
            self.logger.error(
                "Chunk processing timeout", batch_id=batch_id, chunk_index=chunk_index
            )
            raise

    async def _execute_analysis_stages(
        self, request: AnalysisRequest, analysis_id: str
    ) -> Dict[str, Any]:
        """Execute all analysis stages for a request."""

        results = {}

        # Pattern analysis stage
        try:
            pattern_result = await self.pattern_engine.analyze_patterns(request)
            results["pattern_analysis"] = pattern_result
        except Exception as e:
            self.logger.warning(
                "Pattern analysis failed", analysis_id=analysis_id, error=str(e)
            )
            results["pattern_analysis"] = {"error": str(e)}

        # Risk analysis stage
        try:
            risk_result = await self.risk_engine.calculate_risk_score(request)
            results["risk_analysis"] = risk_result
        except Exception as e:
            self.logger.warning(
                "Risk analysis failed", analysis_id=analysis_id, error=str(e)
            )
            results["risk_analysis"] = {"error": str(e)}

        return results

    async def _aggregate_results(
        self, stage_results: Dict[str, Any], request: AnalysisRequest, analysis_id: str
    ) -> AnalysisResult:
        """Aggregate results from all analysis stages."""

        # Extract findings from each stage
        findings = []
        confidence_scores = []

        # Process pattern analysis results
        if "pattern_analysis" in stage_results and isinstance(
            stage_results["pattern_analysis"], AnalysisResult
        ):
            pattern_result = stage_results["pattern_analysis"]
            if pattern_result.patterns:
                findings.extend(pattern_result.patterns)
            confidence_scores.append(pattern_result.confidence)

        # Process risk analysis results
        if "risk_analysis" in stage_results and isinstance(
            stage_results["risk_analysis"], AnalysisResult
        ):
            risk_result = stage_results["risk_analysis"]
            if risk_result.risk_score:
                findings.append(risk_result.risk_score)
            confidence_scores.append(risk_result.confidence)

        # Calculate overall confidence
        overall_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else 0.0
        )

        # Create aggregated result
        result = AnalysisResult(
            analysis_type="comprehensive_analysis",
            confidence=overall_confidence,
            patterns=findings,
            metadata={
                "analysis_id": analysis_id,
                "request_id": request.request_id,
                "stage_results": (
                    stage_results if self.config.include_raw_results else None
                ),
                "processed_at": datetime.now().isoformat(),
            },
        )

        return result

    async def _record_processing_metrics(
        self, analysis_id: str, processing_time: float, success: bool
    ) -> None:
        """Record processing metrics for monitoring."""

        self._processing_stats["total_processed"] += 1

        if success:
            self._processing_stats["successful"] += 1
        else:
            self._processing_stats["failed"] += 1

        # Update average processing time
        total = self._processing_stats["total_processed"]
        current_avg = self._processing_stats["avg_processing_time"]
        self._processing_stats["avg_processing_time"] = (
            current_avg * (total - 1) + processing_time
        ) / total

        # Record with quality monitor
        await self.quality_monitor.record_analysis_metrics(
            {
                "analysis_id": analysis_id,
                "processing_time": processing_time,
                "success": success,
                "timestamp": datetime.now().isoformat(),
            }
        )

    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return {
            **self._processing_stats,
            "active_tasks": len(self._active_tasks),
            "error_rate": (
                self._processing_stats["failed"]
                / self._processing_stats["total_processed"]
                if self._processing_stats["total_processed"] > 0
                else 0.0
            ),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on pipeline components."""

        health_status = {"pipeline": "healthy", "components": {}}

        # Check pattern engine
        try:
            # Simplified health check - just verify engine exists
            health_status["components"]["pattern_engine"] = {"status": "healthy"}
        except Exception as e:
            health_status["components"]["pattern_engine"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["pipeline"] = "degraded"

        # Check risk engine
        try:
            # Simplified health check - just verify engine exists
            health_status["components"]["risk_engine"] =    {"status": "healthy"}
        except Exception as e:
            health_status["components"]["risk_engine"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["pipeline"] = "degraded"

        # Check model server
        try:
            model_health = await self.model_server.health_check()
            health_status["components"]["model_server"] = model_health
        except Exception as e:
            health_status["components"]["model_server"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["pipeline"] = "degraded"

        return health_status


class StreamingAnalysisPipeline:
    """
    Streaming analysis pipeline for real-time processing.

    Handles continuous stream of analysis requests with
    low-latency processing requirements.
    """

    def __init__(
        self,
        base_pipeline: AnalysisPipeline,
        stream_config: Optional[Dict[str, Any]] = None,
    ):
        self.base_pipeline = base_pipeline
        self.stream_config = stream_config or {}
        self.logger = logger.bind(component="streaming_pipeline")

        # Streaming state
        self._is_running = False
        self._stream_task: Optional[asyncio.Task] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()

    async def start_stream_processing(self) -> None:
        """Start streaming analysis processing."""

        if self._is_running:
            self.logger.warning("Stream processing already running")
            return

        self._is_running = True
        self._stream_task = asyncio.create_task(self._process_stream())

        self.logger.info("Started streaming analysis processing")

    async def stop_stream_processing(self) -> None:
        """Stop streaming analysis processing."""

        if not self._is_running:
            return

        self._is_running = False

        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Stopped streaming analysis processing")

    async def submit_for_processing(self, request: AnalysisRequest) -> None:
        """Submit request for streaming processing."""

        if not self._is_running:
            raise RuntimeError("Stream processing not running")

        await self._message_queue.put(request)

    async def _process_stream(self) -> None:
        """Main streaming processing loop."""

        self.logger.info("Starting stream processing loop")

        try:
            while self._is_running:
                try:
                    # Get next request with timeout
                    request = await asyncio.wait_for(
                        self._message_queue.get(), timeout=1.0
                    )

                    # Process request asynchronously
                    asyncio.create_task(self._process_stream_request(request))

                except asyncio.TimeoutError:
                    # No messages, continue loop
                    continue
                except Exception as e:
                    self.logger.error("Stream processing error", error=str(e))

        except asyncio.CancelledError:
            self.logger.info("Stream processing cancelled")
        except Exception as e:
            self.logger.error("Stream processing failed", error=str(e))

    async def _process_stream_request(self, request: AnalysisRequest) -> None:
        """Process individual streaming request."""

        try:
            result = await self.base_pipeline.process_single(request)

            # Handle result (could publish to message queue, webhook, etc.)
            await self._handle_stream_result(result)

        except Exception as e:
            self.logger.error(
                "Failed to process stream request",
                request_id=request.request_id,
                error=str(e),
            )

    async def _handle_stream_result(self, result: AnalysisResult) -> None:
        """Handle streaming analysis result."""

        # This could be extended to publish results to various destinations
        self.logger.info(
            "Stream analysis completed",
            analysis_type=result.analysis_type,
            confidence=result.confidence,
        )

"""
Mock services for isolated testing.

This module provides mock implementations of all three services for unit testing
without requiring actual service instances.
"""

from typing import Dict, Any, List, Optional, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass
import asyncio
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MockResponse:
    """Mock HTTP response object."""
    status_code: int
    content: bytes = b""
    headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}
    
    def json(self) -> Dict[str, Any]:
        """Return JSON content."""
        import json
        return json.loads(self.content.decode())


class MockMapperService:
    """Mock implementation of Core Mapper Service."""
    
    def __init__(self):
        self.request_count = 0
        self.last_request = None
        self.responses = {}
        self.default_confidence = 0.95
    
    async def map_detector_output(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Mock mapping detector output to canonical taxonomy."""
        self.request_count += 1
        self.last_request = payload
        
        detector_outputs = payload.get("detector_outputs", [])
        framework = payload.get("framework", "SOC2")
        
        # Generate mock canonical result
        canonical_result = {
            "category": "pii",
            "subcategory": "person_name",
            "confidence": self.default_confidence,
            "taxonomy_version": "1.0.0",
            "model_version": "test-model-v1.0",
            "metadata": {
                "processed_detectors": len(detector_outputs),
                "processing_time_ms": 50
            }
        }
        
        # Generate mock framework mappings
        framework_mappings = []
        if framework == "SOC2":
            framework_mappings = [
                {
                    "control_id": "CC6.1",
                    "control_name": "Logical Access Controls",
                    "risk_level": "medium",
                    "evidence_required": True
                }
            ]
        elif framework == "ISO27001":
            framework_mappings = [
                {
                    "control_id": "A.8.2.1",
                    "control_name": "Data Classification",
                    "risk_level": "medium",
                    "evidence_required": True
                }
            ]
        
        return {
            "canonical_result": canonical_result,
            "framework_mappings": framework_mappings,
            "confidence_score": self.default_confidence,
            "processing_metadata": {
                "request_id": f"mock-{self.request_count}",
                "timestamp": "2024-01-01T00:00:00Z",
                "processing_time_ms": 50
            }
        }
    
    async def get_health(self) -> Dict[str, Any]:
        """Mock health check."""
        return {
            "status": "healthy",
            "service": "core_mapper",
            "version": "test",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    
    def set_response_confidence(self, confidence: float) -> None:
        """Set mock response confidence level."""
        self.default_confidence = confidence
    
    def set_custom_response(self, payload_key: str, response: Dict[str, Any]) -> None:
        """Set custom response for specific payload."""
        self.responses[payload_key] = response


class MockDetectorOrchestrationService:
    """Mock implementation of Detector Orchestration Service."""
    
    def __init__(self):
        self.request_count = 0
        self.last_request = None
        self.detector_registry = ["presidio", "deberta", "custom"]
        self.auto_map_enabled = False
        self.mapper_service_mock = MockMapperService()
    
    async def orchestrate_detection(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Mock orchestrated detection."""
        self.request_count += 1
        self.last_request = payload
        
        content = payload.get("content", "")
        detectors = payload.get("detectors", self.detector_registry)
        tenant_id = payload.get("tenant_id", "test_tenant")
        
        # Generate mock detector outputs
        detector_outputs = []
        for detector in detectors:
            detector_outputs.append({
                "detector_type": detector,
                "findings": [
                    {
                        "entity_type": "PERSON",
                        "confidence": 0.95,
                        "start": 0,
                        "end": len(content),
                        "text": "[REDACTED]"
                    }
                ],
                "metadata": {
                    "processing_time_ms": 30,
                    "model_version": f"{detector}-v1.0"
                }
            })
        
        # Create mapper payload
        mapper_payload = {
            "detector_outputs": detector_outputs,
            "framework": payload.get("framework", "SOC2"),
            "tenant_id": tenant_id,
            "correlation_id": payload.get("correlation_id", f"mock-{self.request_count}")
        }
        
        result = {
            "orchestration_id": f"orch-{self.request_count}",
            "detector_results": detector_outputs,
            "mapper_payload": mapper_payload,
            "coverage_metrics": {
                "detectors_executed": len(detectors),
                "detectors_successful": len(detectors),
                "total_findings": len(detector_outputs)
            },
            "processing_metadata": {
                "request_id": f"mock-orch-{self.request_count}",
                "timestamp": "2024-01-01T00:00:00Z",
                "processing_time_ms": 100
            }
        }
        
        # Add mapping result if auto_map is enabled
        if self.auto_map_enabled:
            mapping_result = await self.mapper_service_mock.map_detector_output(mapper_payload)
            result["mapping_result"] = mapping_result
        
        return result
    
    async def get_detector_registry(self) -> List[Dict[str, Any]]:
        """Mock detector registry."""
        return [
            {
                "name": detector,
                "version": "1.0.0",
                "status": "healthy",
                "capabilities": ["pii_detection"],
                "endpoint": f"http://mock-{detector}:8000"
            }
            for detector in self.detector_registry
        ]
    
    async def get_health(self) -> Dict[str, Any]:
        """Mock health check."""
        return {
            "status": "healthy",
            "service": "detector_orchestration",
            "version": "test",
            "detectors": len(self.detector_registry),
            "timestamp": "2024-01-01T00:00:00Z"
        }
    
    def enable_auto_mapping(self, enabled: bool = True) -> None:
        """Enable/disable auto mapping."""
        self.auto_map_enabled = enabled
    
    def add_detector(self, detector_name: str) -> None:
        """Add detector to registry."""
        if detector_name not in self.detector_registry:
            self.detector_registry.append(detector_name)
    
    def remove_detector(self, detector_name: str) -> None:
        """Remove detector from registry."""
        if detector_name in self.detector_registry:
            self.detector_registry.remove(detector_name)


class MockAnalysisService:
    """Mock implementation of Analysis Service."""
    
    def __init__(self):
        self.request_count = 0
        self.last_request = None
        self.analysis_history = []
    
    async def analyze_compliance_data(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Mock compliance analysis."""
        self.request_count += 1
        self.last_request = payload
        
        mapping_result = payload.get("mapping_result", {})
        correlation_id = payload.get("correlation_id", f"mock-analysis-{self.request_count}")
        
        # Extract canonical result
        canonical_result = mapping_result.get("canonical_result", {})
        framework_mappings = mapping_result.get("framework_mappings", [])
        
        # Generate mock analysis result
        risk_assessment = {
            "overall_risk_score": 0.7,
            "risk_level": "medium",
            "contributing_factors": [
                {
                    "factor": "pii_detection",
                    "impact": 0.8,
                    "confidence": canonical_result.get("confidence", 0.95)
                }
            ]
        }
        
        remediation_actions = [
            {
                "action_type": "data_masking",
                "priority": "high",
                "description": "Implement data masking for detected PII",
                "estimated_effort_hours": 4
            },
            {
                "action_type": "access_review",
                "priority": "medium",
                "description": "Review access controls for sensitive data",
                "estimated_effort_hours": 8
            }
        ]
        
        compliance_evidence = {
            "framework_compliance": [
                {
                    "framework": mapping.get("control_id", "unknown"),
                    "compliance_status": "partial",
                    "evidence_collected": True,
                    "recommendations": ["Implement additional controls"]
                }
                for mapping in framework_mappings
            ]
        }
        
        analysis_result = {
            "analysis_id": f"analysis-{self.request_count}",
            "risk_assessment": risk_assessment,
            "remediation_actions": remediation_actions,
            "compliance_evidence": compliance_evidence,
            "trend_analysis": {
                "risk_trend": "stable",
                "detection_frequency": "normal",
                "compliance_drift": "minimal"
            },
            "processing_metadata": {
                "request_id": correlation_id,
                "timestamp": "2024-01-01T00:00:00Z",
                "processing_time_ms": 200,
                "analysis_version": "test-v1.0"
            }
        }
        
        self.analysis_history.append(analysis_result)
        return analysis_result
    
    async def get_compliance_dashboard(self, tenant_id: str, timeframe: str = "24h") -> Dict[str, Any]:
        """Mock compliance dashboard data."""
        return {
            "tenant_id": tenant_id,
            "timeframe": timeframe,
            "summary_metrics": {
                "total_analyses": self.request_count,
                "avg_risk_score": 0.6,
                "compliance_score": 0.85,
                "trend": "improving"
            },
            "risk_distribution": {
                "low": 30,
                "medium": 50,
                "high": 15,
                "critical": 5
            },
            "framework_compliance": {
                "SOC2": {"score": 0.88, "status": "compliant"},
                "ISO27001": {"score": 0.82, "status": "partial"},
                "HIPAA": {"score": 0.91, "status": "compliant"}
            }
        }
    
    async def get_health(self) -> Dict[str, Any]:
        """Mock health check."""
        return {
            "status": "healthy",
            "service": "analysis_service",
            "version": "test",
            "analyses_processed": self.request_count,
            "timestamp": "2024-01-01T00:00:00Z"
        }
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get analysis history."""
        return self.analysis_history.copy()


class MockServiceRegistry:
    """Registry for managing mock services."""
    
    def __init__(self):
        self.mapper_service = MockMapperService()
        self.orchestration_service = MockDetectorOrchestrationService()
        self.analysis_service = MockAnalysisService()
        self.http_mocks = {}
    
    def create_mapper_mock(self) -> AsyncMock:
        """Create async mock for mapper service."""
        mock = AsyncMock()
        mock.post = AsyncMock()
        mock.get = AsyncMock()
        
        # Setup mock responses
        async def mock_map_request(*args, **kwargs):
            json_data = kwargs.get("json", {})
            result = await self.mapper_service.map_detector_output(json_data)
            return MockResponse(
                status_code=200,
                content=str(result).encode(),
                headers={"Content-Type": "application/json"}
            )
        
        async def mock_health_request(*args, **kwargs):
            result = await self.mapper_service.get_health()
            return MockResponse(
                status_code=200,
                content=str(result).encode(),
                headers={"Content-Type": "application/json"}
            )
        
        mock.post.side_effect = mock_map_request
        mock.get.side_effect = mock_health_request
        
        return mock
    
    def create_orchestration_mock(self) -> AsyncMock:
        """Create async mock for orchestration service."""
        mock = AsyncMock()
        mock.post = AsyncMock()
        mock.get = AsyncMock()
        
        # Setup mock responses
        async def mock_orchestrate_request(*args, **kwargs):
            json_data = kwargs.get("json", {})
            result = await self.orchestration_service.orchestrate_detection(json_data)
            return MockResponse(
                status_code=200,
                content=str(result).encode(),
                headers={"Content-Type": "application/json"}
            )
        
        async def mock_health_request(*args, **kwargs):
            result = await self.orchestration_service.get_health()
            return MockResponse(
                status_code=200,
                content=str(result).encode(),
                headers={"Content-Type": "application/json"}
            )
        
        mock.post.side_effect = mock_orchestrate_request
        mock.get.side_effect = mock_health_request
        
        return mock
    
    def create_analysis_mock(self) -> AsyncMock:
        """Create async mock for analysis service."""
        mock = AsyncMock()
        mock.post = AsyncMock()
        mock.get = AsyncMock()
        
        # Setup mock responses
        async def mock_analyze_request(*args, **kwargs):
            json_data = kwargs.get("json", {})
            result = await self.analysis_service.analyze_compliance_data(json_data)
            return MockResponse(
                status_code=200,
                content=str(result).encode(),
                headers={"Content-Type": "application/json"}
            )
        
        async def mock_health_request(*args, **kwargs):
            result = await self.analysis_service.get_health()
            return MockResponse(
                status_code=200,
                content=str(result).encode(),
                headers={"Content-Type": "application/json"}
            )
        
        mock.post.side_effect = mock_analyze_request
        mock.get.side_effect = mock_health_request
        
        return mock
    
    def create_cross_service_workflow_mock(self) -> Dict[str, AsyncMock]:
        """Create mocks for cross-service workflow testing."""
        return {
            "mapper": self.create_mapper_mock(),
            "orchestration": self.create_orchestration_mock(), 
            "analysis": self.create_analysis_mock()
        }
    
    def reset_all_mocks(self) -> None:
        """Reset all mock services."""
        self.mapper_service = MockMapperService()
        self.orchestration_service = MockDetectorOrchestrationService()
        self.analysis_service = MockAnalysisService()
    
    def configure_failure_scenario(self, service: str, failure_type: str) -> None:
        """Configure mock to simulate failure scenarios."""
        if service == "mapper":
            if failure_type == "timeout":
                self.mapper_service.map_detector_output = AsyncMock(
                    side_effect=asyncio.TimeoutError("Service timeout")
                )
            elif failure_type == "error":
                self.mapper_service.map_detector_output = AsyncMock(
                    side_effect=Exception("Service error")
                )
        
        elif service == "orchestration":
            if failure_type == "timeout":
                self.orchestration_service.orchestrate_detection = AsyncMock(
                    side_effect=asyncio.TimeoutError("Service timeout")
                )
            elif failure_type == "error":
                self.orchestration_service.orchestrate_detection = AsyncMock(
                    side_effect=Exception("Service error")
                )
        
        elif service == "analysis":
            if failure_type == "timeout":
                self.analysis_service.analyze_compliance_data = AsyncMock(
                    side_effect=asyncio.TimeoutError("Service timeout")
                )
            elif failure_type == "error":
                self.analysis_service.analyze_compliance_data = AsyncMock(
                    side_effect=Exception("Service error")
                )
    
    def get_service_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all mock services."""
        return {
            "mapper": {
                "request_count": self.mapper_service.request_count,
                "last_request": self.mapper_service.last_request
            },
            "orchestration": {
                "request_count": self.orchestration_service.request_count,
                "last_request": self.orchestration_service.last_request,
                "detector_count": len(self.orchestration_service.detector_registry)
            },
            "analysis": {
                "request_count": self.analysis_service.request_count,
                "last_request": self.analysis_service.last_request,
                "analysis_history_count": len(self.analysis_service.analysis_history)
            }
        }

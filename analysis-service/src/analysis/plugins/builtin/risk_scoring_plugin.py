"""
Built-in risk scoring plugin for Analysis Service.

This plugin provides comprehensive risk scoring capabilities using multiple
risk factors and statistical analysis.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from ..interfaces import (
    IRiskScorerPlugin,
    PluginMetadata,
    PluginType,
    PluginCapability,
    AnalysisRequest,
    AnalysisResult,
)

logger = logging.getLogger(__name__)


class RiskScoringPlugin(IRiskScorerPlugin):
    """Built-in risk scoring plugin."""

    def __init__(self):
        self.initialized = False
        self.config = {}
        self.risk_factors = {}

    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="builtin_risk_scoring",
            version="1.0.0",
            description="Built-in risk scoring using multi-factor analysis",
            author="Analysis Service Team",
            plugin_type=PluginType.ANALYSIS_ENGINE,
            capabilities=[
                PluginCapability.RISK_SCORING,
                PluginCapability.STATISTICAL_ANALYSIS,
                PluginCapability.BATCH_PROCESSING,
                PluginCapability.REAL_TIME_ANALYSIS,
            ],
            dependencies=[],
            supported_frameworks=["SOC2", "ISO27001", "HIPAA", "GDPR"],
            min_confidence_threshold=0.7,
            max_batch_size=50,
        )

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin."""
        try:
            self.config = config

            # Initialize risk scoring parameters
            self.base_risk_weight = config.get("base_risk_weight", 0.3)
            self.temporal_risk_weight = config.get("temporal_risk_weight", 0.2)
            self.frequency_risk_weight = config.get("frequency_risk_weight", 0.2)
            self.severity_risk_weight = config.get("severity_risk_weight", 0.3)

            # Risk thresholds
            self.low_risk_threshold = config.get("low_risk_threshold", 0.3)
            self.medium_risk_threshold = config.get("medium_risk_threshold", 0.6)
            self.high_risk_threshold = config.get("high_risk_threshold", 0.8)

            # Risk categories
            self.risk_categories = [
                "technical_risk",
                "business_risk",
                "regulatory_risk",
                "operational_risk",
                "reputational_risk",
            ]

            self.initialized = True
            logger.info("Risk scoring plugin initialized")
            return True

        except Exception as e:
            logger.error("Failed to initialize risk scoring plugin", error=str(e))
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "status": "healthy" if self.initialized else "not_initialized",
            "initialized": self.initialized,
            "risk_categories": len(self.risk_categories),
            "last_check": datetime.now(timezone.utc).isoformat(),
        }

    async def cleanup(self) -> bool:
        """Cleanup plugin resources."""
        try:
            self.risk_factors.clear()
            self.initialized = False
            logger.info("Risk scoring plugin cleaned up")
            return True
        except Exception as e:
            logger.error("Failed to cleanup risk scoring plugin", error=str(e))
            return False

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Perform risk scoring analysis."""
        start_time = time.time()

        try:
            if not self.initialized:
                raise RuntimeError("Plugin not initialized")

            # Calculate risk score
            risk_score_data = await self.calculate_risk_score(request)

            # Get risk factors
            risk_factors = await self.get_risk_factors(request)

            # Calculate overall confidence
            confidence = self._calculate_confidence(risk_score_data, request.metadata)

            # Prepare result
            result_data = {
                "risk_score": risk_score_data,
                "risk_factors": risk_factors,
                "risk_categories": self.get_risk_categories(),
                "analysis_type": "risk_scoring",
                "framework_mappings": self._map_to_frameworks(risk_score_data),
            }

            processing_time = (time.time() - start_time) * 1000

            return AnalysisResult(
                request_id=request.request_id,
                plugin_name="builtin_risk_scoring",
                plugin_version="1.0.0",
                confidence=confidence,
                result_data=result_data,
                processing_time_ms=processing_time,
                metadata={
                    "tenant_id": request.tenant_id,
                    "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(
                "Risk scoring analysis failed",
                request_id=request.request_id,
                error=str(e),
            )

            return AnalysisResult(
                request_id=request.request_id,
                plugin_name="builtin_risk_scoring",
                plugin_version="1.0.0",
                confidence=0.0,
                result_data={},
                processing_time_ms=processing_time,
                errors=[str(e)],
            )

    async def batch_analyze(
        self, requests: List[AnalysisRequest]
    ) -> List[AnalysisResult]:
        """Perform batch risk scoring analysis."""
        results = []

        # Process requests in parallel batches
        batch_size = min(self.get_metadata().max_batch_size or 10, len(requests))

        for i in range(0, len(requests), batch_size):
            batch = requests[i : i + batch_size]
            batch_results = await asyncio.gather(
                *[self.analyze(request) for request in batch], return_exceptions=True
            )

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error("Batch risk scoring item failed", error=str(result))
                    error_result = AnalysisResult(
                        request_id="unknown",
                        plugin_name="builtin_risk_scoring",
                        plugin_version="1.0.0",
                        confidence=0.0,
                        result_data={},
                        processing_time_ms=0.0,
                        errors=[str(result)],
                    )
                    results.append(error_result)
                else:
                    results.append(result)

        return results

    def get_supported_analysis_types(self) -> List[str]:
        """Get supported analysis types."""
        return ["risk_scoring", "risk_assessment", "threat_analysis"]

    async def validate_request(self, request: AnalysisRequest) -> bool:
        """Validate if the plugin can handle the request."""
        try:
            # Check if analysis type is supported
            if request.analysis_type not in self.get_supported_analysis_types():
                return False

            # Check if metadata contains required fields
            required_fields = ["severity", "category"]
            for field in required_fields:
                if field not in request.metadata:
                    logger.warning(
                        "Missing required field for risk scoring", field=field
                    )
                    # Don't fail validation, use defaults

            return True

        except Exception as e:
            logger.error("Request validation failed", error=str(e))
            return False

    async def calculate_risk_score(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Calculate comprehensive risk score."""
        try:
            metadata = request.metadata

            # Base risk factors
            severity = metadata.get("severity", "medium")
            category = metadata.get("category", "unknown")

            # Calculate component risk scores
            technical_risk = self._calculate_technical_risk(metadata)
            business_risk = self._calculate_business_risk(metadata)
            regulatory_risk = self._calculate_regulatory_risk(metadata)
            operational_risk = self._calculate_operational_risk(metadata)

            # Calculate weighted overall risk score
            overall_risk = (
                technical_risk * 0.3
                + business_risk * 0.25
                + regulatory_risk * 0.25
                + operational_risk * 0.2
            )

            # Determine risk level
            if overall_risk >= self.high_risk_threshold:
                risk_level = "high"
            elif overall_risk >= self.medium_risk_threshold:
                risk_level = "medium"
            elif overall_risk >= self.low_risk_threshold:
                risk_level = "low"
            else:
                risk_level = "minimal"

            return {
                "overall_risk_score": min(1.0, max(0.0, overall_risk)),
                "risk_level": risk_level,
                "component_scores": {
                    "technical_risk": technical_risk,
                    "business_risk": business_risk,
                    "regulatory_risk": regulatory_risk,
                    "operational_risk": operational_risk,
                },
                "risk_factors": {
                    "severity": severity,
                    "category": category,
                    "temporal_factor": self._calculate_temporal_factor(metadata),
                    "frequency_factor": self._calculate_frequency_factor(metadata),
                },
                "mitigation_priority": self._calculate_mitigation_priority(
                    overall_risk, severity
                ),
            }

        except Exception as e:
            logger.error("Risk score calculation failed", error=str(e))
            return {"overall_risk_score": 0.5, "risk_level": "medium", "error": str(e)}

    async def get_risk_factors(self, request: AnalysisRequest) -> List[Dict[str, Any]]:
        """Get risk factors contributing to the score."""
        try:
            metadata = request.metadata
            factors = []

            # Severity factor
            severity = metadata.get("severity", "medium")
            severity_impact = {
                "low": 0.2,
                "medium": 0.5,
                "high": 0.8,
                "critical": 1.0,
            }.get(severity, 0.5)
            factors.append(
                {
                    "factor": "severity",
                    "value": severity,
                    "impact": severity_impact,
                    "weight": 0.3,
                    "description": f"Incident severity level: {severity}",
                }
            )

            # Category factor
            category = metadata.get("category", "unknown")
            category_impact = self._get_category_risk_impact(category)
            factors.append(
                {
                    "factor": "category",
                    "value": category,
                    "impact": category_impact,
                    "weight": 0.25,
                    "description": f"Risk category: {category}",
                }
            )

            # Temporal factor
            temporal_factor = self._calculate_temporal_factor(metadata)
            factors.append(
                {
                    "factor": "temporal",
                    "value": temporal_factor,
                    "impact": temporal_factor,
                    "weight": 0.2,
                    "description": "Time-based risk assessment",
                }
            )

            # Frequency factor
            frequency_factor = self._calculate_frequency_factor(metadata)
            factors.append(
                {
                    "factor": "frequency",
                    "value": frequency_factor,
                    "impact": frequency_factor,
                    "weight": 0.25,
                    "description": "Frequency-based risk assessment",
                }
            )

            return factors

        except Exception as e:
            logger.error("Risk factors calculation failed", error=str(e))
            return []

    def get_risk_categories(self) -> List[str]:
        """Get supported risk categories."""
        return self.risk_categories

    # Private helper methods

    def _calculate_technical_risk(self, metadata: Dict[str, Any]) -> float:
        """Calculate technical risk component."""
        base_risk = 0.3

        # Adjust based on technical factors
        if metadata.get("system_critical", False):
            base_risk += 0.3

        if metadata.get("data_sensitive", False):
            base_risk += 0.2

        if metadata.get("external_facing", False):
            base_risk += 0.2

        return min(1.0, base_risk)

    def _calculate_business_risk(self, metadata: Dict[str, Any]) -> float:
        """Calculate business risk component."""
        base_risk = 0.2

        # Adjust based on business factors
        business_impact = metadata.get("business_impact", "medium")
        impact_multiplier = {
            "low": 0.1,
            "medium": 0.3,
            "high": 0.5,
            "critical": 0.8,
        }.get(business_impact, 0.3)
        base_risk += impact_multiplier

        if metadata.get("revenue_affecting", False):
            base_risk += 0.3

        return min(1.0, base_risk)

    def _calculate_regulatory_risk(self, metadata: Dict[str, Any]) -> float:
        """Calculate regulatory risk component."""
        base_risk = 0.1

        # Adjust based on regulatory factors
        frameworks = metadata.get("applicable_frameworks", [])
        if "HIPAA" in frameworks:
            base_risk += 0.4
        if "SOX" in frameworks:
            base_risk += 0.3
        if "GDPR" in frameworks:
            base_risk += 0.3

        return min(1.0, base_risk)

    def _calculate_operational_risk(self, metadata: Dict[str, Any]) -> float:
        """Calculate operational risk component."""
        base_risk = 0.2

        # Adjust based on operational factors
        if metadata.get("requires_manual_intervention", False):
            base_risk += 0.3

        if metadata.get("affects_multiple_systems", False):
            base_risk += 0.2

        return min(1.0, base_risk)

    def _calculate_temporal_factor(self, metadata: Dict[str, Any]) -> float:
        """Calculate temporal risk factor."""
        # Simple temporal analysis - in production would be more sophisticated
        time_since_detection = metadata.get("time_since_detection_hours", 1)

        # Risk increases with time
        if time_since_detection > 24:
            return 0.8
        elif time_since_detection > 8:
            return 0.6
        elif time_since_detection > 2:
            return 0.4
        else:
            return 0.2

    def _calculate_frequency_factor(self, metadata: Dict[str, Any]) -> float:
        """Calculate frequency risk factor."""
        # Simple frequency analysis
        occurrence_count = metadata.get("occurrence_count_24h", 1)

        if occurrence_count > 10:
            return 0.9
        elif occurrence_count > 5:
            return 0.7
        elif occurrence_count > 2:
            return 0.5
        else:
            return 0.3

    def _get_category_risk_impact(self, category: str) -> float:
        """Get risk impact for category."""
        category_impacts = {
            "security": 0.9,
            "privacy": 0.8,
            "compliance": 0.7,
            "operational": 0.6,
            "performance": 0.4,
            "unknown": 0.5,
        }
        return category_impacts.get(category.lower(), 0.5)

    def _calculate_mitigation_priority(self, risk_score: float, severity: str) -> str:
        """Calculate mitigation priority."""
        if risk_score >= 0.8 or severity == "critical":
            return "immediate"
        elif risk_score >= 0.6 or severity == "high":
            return "urgent"
        elif risk_score >= 0.4 or severity == "medium":
            return "normal"
        else:
            return "low"

    def _calculate_confidence(
        self, risk_data: Dict[str, Any], metadata: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence score."""
        try:
            base_confidence = 0.7

            # Adjust based on data quality
            if "severity" in metadata:
                base_confidence += 0.1
            if "category" in metadata:
                base_confidence += 0.1
            if "business_impact" in metadata:
                base_confidence += 0.1

            # Adjust based on risk score consistency
            overall_risk = risk_data.get("overall_risk_score", 0.5)
            if 0.3 <= overall_risk <= 0.8:  # Reasonable range
                base_confidence += 0.1

            return min(0.95, max(0.0, base_confidence))

        except Exception as e:
            logger.error("Confidence calculation failed", error=str(e))
            return 0.5

    def _map_to_frameworks(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map risk assessment to compliance frameworks."""
        framework_mappings = {}

        try:
            overall_risk = risk_data.get("overall_risk_score", 0.5)
            risk_level = risk_data.get("risk_level", "medium")

            for framework in ["SOC2", "ISO27001", "HIPAA", "GDPR"]:
                framework_mappings[framework] = {
                    "applicable_controls": [],
                    "risk_level": risk_level,
                    "recommendations": [],
                }

                # Map to framework-specific controls
                if framework == "SOC2":
                    if overall_risk >= 0.6:
                        framework_mappings[framework]["applicable_controls"].extend(
                            ["CC6.1", "CC7.1"]
                        )
                    if overall_risk >= 0.8:
                        framework_mappings[framework]["applicable_controls"].append(
                            "CC8.1"
                        )

                elif framework == "ISO27001":
                    if overall_risk >= 0.6:
                        framework_mappings[framework]["applicable_controls"].extend(
                            ["A.16.1.2", "A.16.1.4"]
                        )
                    if overall_risk >= 0.8:
                        framework_mappings[framework]["applicable_controls"].append(
                            "A.16.1.1"
                        )

                elif framework == "HIPAA":
                    if overall_risk >= 0.6:
                        framework_mappings[framework]["applicable_controls"].extend(
                            ["164.308(a)(1)", "164.312(b)"]
                        )

                elif framework == "GDPR":
                    if overall_risk >= 0.6:
                        framework_mappings[framework]["applicable_controls"].extend(
                            ["Art. 32", "Art. 33"]
                        )

                # Add recommendations based on risk level
                if risk_level in ["high", "critical"]:
                    framework_mappings[framework]["recommendations"] = [
                        "Immediate incident response required",
                        "Escalate to senior management",
                        "Consider regulatory notification",
                    ]
                elif risk_level == "medium":
                    framework_mappings[framework]["recommendations"] = [
                        "Implement additional monitoring",
                        "Review and update controls",
                        "Schedule risk assessment",
                    ]

        except Exception as e:
            logger.error("Framework mapping failed", error=str(e))

        return framework_mappings

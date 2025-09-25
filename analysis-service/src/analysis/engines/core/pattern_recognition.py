"""
Consolidated Pattern Recognition Engine

This engine consolidates all pattern recognition capabilities from the original analysis module,
including temporal, frequency, correlation, and anomaly detection patterns.
Follows SRP by focusing solely on pattern detection and classification.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from ...schemas.analysis_schemas import AnalysisRequest, AnalysisResult
from ...taxonomy.pattern_taxonomy import PatternTaxonomy
from ..statistical import (
    TemporalAnalyzer,
    FrequencyAnalyzer,
    CorrelationAnalyzer,
    AnomalyDetector,
    PatternClassifier,
    PatternStrengthCalculator,
    BusinessRelevanceAssessor,
    PatternConfidenceCalculator,
    MultiPatternAnalyzer,
    PatternEvolutionTracker,
    PatternInteractionMatrix,
)

logger = logging.getLogger(__name__)


class PatternRecognitionEngine:
    """
    Consolidated pattern recognition engine using advanced statistical methods.

    Consolidates all pattern detection capabilities:
    - Temporal pattern analysis
    - Frequency pattern detection
    - Correlation pattern identification
    - Anomaly pattern detection
    - Multi-pattern relationship analysis
    - Pattern evolution tracking
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pattern_config = config.get("pattern_recognition", {})
        self.taxonomy = PatternTaxonomy()

        # Initialize core analyzers (reduced to stay under attribute limit)
        self._analyzers = {
            "temporal": TemporalAnalyzer(self.pattern_config),
            "frequency": FrequencyAnalyzer(self.pattern_config),
            "correlation": CorrelationAnalyzer(self.pattern_config),
            "anomaly": AnomalyDetector(self.pattern_config),
        }

        # Initialize enhancement components
        self._enhancers = {
            "classifier": PatternClassifier(self.pattern_config),
            "strength": PatternStrengthCalculator(self.pattern_config),
            "relevance": BusinessRelevanceAssessor(self.pattern_config),
            "confidence": PatternConfidenceCalculator(self.pattern_config),
        }

        # Initialize advanced components
        self._advanced = {
            "multi_pattern": MultiPatternAnalyzer(self.pattern_config),
            "evolution": PatternEvolutionTracker(self.pattern_config),
            "interaction": PatternInteractionMatrix(self.pattern_config),
        }

        logger.info(
            "Pattern Recognition Engine initialized with consolidated analyzers"
        )

    def get_engine_capabilities(self) -> Dict[str, Any]:
        """Get information about the pattern recognition engine capabilities."""
        return {
            "engine_type": "consolidated_pattern_recognition",
            "version": "1.0.0",
            "analyzers": list(self._analyzers.keys()),
            "enhancers": list(self._enhancers.keys()),
            "advanced_features": list(self._advanced.keys()),
            "supported_patterns": [
                "temporal_clustering",
                "frequency_spike",
                "data_correlation",
                "behavioral_anomaly",
                "multi_pattern_interactions",
            ],
        }

    async def analyze_patterns(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Perform comprehensive pattern analysis on the request data.

        Args:
            request: Analysis request containing security data

        Returns:
            AnalysisResult with detected patterns and correlations
        """
        try:
            # Extract security data from request
            security_data = self._extract_security_data(request)

            # Detect patterns using all analyzers
            patterns = await self._detect_all_patterns(security_data)

            # Enhance patterns with classification and assessment
            enhanced_patterns = await self._enhance_patterns(patterns, security_data)

            # Perform cross-pattern analysis
            cross_pattern_analysis = await self._analyze_pattern_relationships(
                enhanced_patterns
            )

            # Calculate overall confidence
            confidence = self._calculate_overall_confidence(enhanced_patterns)

            # Create comprehensive result
            result = AnalysisResult(
                analysis_type="pattern_recognition",
                confidence=confidence,
                patterns=enhanced_patterns,
                evidence=self._create_evidence(
                    enhanced_patterns, cross_pattern_analysis
                ),
                metadata=self._create_metadata(
                    enhanced_patterns, cross_pattern_analysis
                ),
            )

            logger.info(
                "Pattern analysis completed",
                patterns_detected=len(enhanced_patterns),
                confidence=confidence,
            )

            return result

        except (ValueError, KeyError, AttributeError) as e:
            logger.error("Pattern analysis failed", error=str(e))
            return self._create_error_result(str(e))

    async def _detect_all_patterns(
        self, security_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect patterns using all available analyzers."""
        all_patterns = []

        # Temporal patterns
        if security_data.get("time_series"):
            temporal_patterns = await self._analyzers[
                "temporal"
            ].analyze_temporal_patterns(security_data["time_series"])
            all_patterns.extend(temporal_patterns)

        # Frequency patterns
        if security_data.get("events"):
            frequency_patterns = await self._analyzers[
                "frequency"
            ].analyze_frequency_patterns(security_data["events"])
            all_patterns.extend(frequency_patterns)

        # Correlation patterns
        if security_data.get("multi_dimensional"):
            correlation_patterns = await self._analyzers[
                "correlation"
            ].analyze_correlation_patterns(security_data["multi_dimensional"])
            all_patterns.extend(correlation_patterns)

        # Anomaly patterns
        if security_data.get("metrics"):
            anomaly_patterns = await self._analyzers["anomaly"].detect_anomalies(
                security_data["metrics"]
            )
            all_patterns.extend(anomaly_patterns)

        return all_patterns

    async def _enhance_patterns(
        self, patterns: List[Dict[str, Any]], security_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Enhance patterns with classification and assessment."""
        enhanced_patterns = []

        for pattern in patterns:
            # Classify pattern characteristics
            classified_patterns = await self._enhancers["classifier"].classify_patterns(
                [pattern]
            )
            classified_pattern = (
                classified_patterns.get("security", [pattern])[0]
                if classified_patterns
                else pattern
            )

            # Calculate pattern strength
            pattern_strength = await self._enhancers["strength"].calculate_strength(
                pattern, security_data
            )
            classified_pattern.strength = pattern_strength

            # Assess business relevance
            business_relevance = await self._enhancers["relevance"].assess_relevance(
                pattern, security_data
            )
            classified_pattern.metadata = classified_pattern.metadata or {}
            classified_pattern.metadata["business_relevance"] = business_relevance

            # Calculate enhanced confidence
            enhanced_confidence = await self._enhancers[
                "confidence"
            ].calculate_confidence(pattern, [])
            classified_pattern.confidence = enhanced_confidence

            enhanced_patterns.append(classified_pattern)

        # Filter by confidence threshold
        confidence_threshold = self.pattern_config.get("confidence_threshold", 0.6)
        filtered_patterns = [
            pattern
            for pattern in enhanced_patterns
            if pattern.get("confidence", 0) >= confidence_threshold
        ]

        return filtered_patterns

    async def _analyze_pattern_relationships(
        self, patterns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze relationships between patterns."""
        if len(patterns) < 2:
            return {"message": "Insufficient patterns for relationship analysis"}

        try:
            # Multi-pattern relationship analysis
            relationship_analysis = await self._advanced[
                "multi_pattern"
            ].analyze_pattern_interactions(patterns)

            # Pattern evolution tracking
            evolution_analysis = await self._advanced["evolution"].track_evolution(
                patterns, []  # Empty historical patterns for now
            )

            # Interaction matrix creation
            interaction_matrices = await self._advanced[
                "interaction"
            ].build_interaction_matrix(patterns)

            return {
                "relationships": relationship_analysis,
                "evolution": evolution_analysis,
                "interactions": interaction_matrices,
                "summary": {
                    "total_patterns": len(patterns),
                    "relationships_found": len(
                        relationship_analysis.get("relationships", [])
                    ),
                    "clusters_identified": len(
                        relationship_analysis.get("clusters", [])
                    ),
                    "emerging_patterns": len(
                        evolution_analysis.get("emerging_patterns", [])
                    ),
                    "declining_patterns": len(
                        evolution_analysis.get("declining_patterns", [])
                    ),
                },
            }

        except (ValueError, KeyError, AttributeError) as e:
            logger.error("Cross-pattern analysis failed", error=str(e))
            return {"error": str(e)}

    def _extract_security_data(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Extract security data from analysis request."""
        time_series = []
        events = []
        multi_dimensional = []
        metrics = []

        # Extract from high severity hits
        for hit in request.high_sev_hits:
            events.append(
                {
                    "detector": hit.get("detector", "unknown"),
                    "type": "high_severity",
                    "timestamp": datetime.now(timezone.utc),
                    "severity": hit.get("severity", "unknown"),
                    "metadata": hit,
                }
            )

        # Extract from false positive bands
        for band in request.false_positive_bands:
            metrics.append(
                {
                    "type": "false_positive_rate",
                    "value": band.get("false_positive_rate", 0.0),
                    "detector": band.get("detector", "unknown"),
                    "metadata": band,
                }
            )

        # Extract from coverage data
        for detector, coverage in request.observed_coverage.items():
            time_series.append(
                {
                    "detector": detector,
                    "timestamp": datetime.now(timezone.utc),
                    "value": coverage,
                    "type": "coverage",
                }
            )

            multi_dimensional.append(
                {
                    "detector": detector,
                    "observed_coverage": coverage,
                    "required_coverage": request.required_coverage.get(detector, 0.0),
                    "coverage_gap": request.required_coverage.get(detector, 0.0)
                    - coverage,
                }
            )

        return {
            "time_series": time_series,
            "events": events,
            "multi_dimensional": multi_dimensional,
            "metrics": metrics,
            "metadata": {"request_id": request.request_id},
        }

    def _calculate_overall_confidence(self, patterns: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence for pattern analysis."""
        if not patterns:
            return 0.0

        confidences = [pattern.get("confidence", 0.5) for pattern in patterns]
        return sum(confidences) / len(confidences)

    def _create_evidence(
        self, patterns: List[Dict[str, Any]], cross_pattern_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create evidence for the analysis result."""
        return [
            {
                "type": "pattern_detection",
                "patterns_found": len(patterns),
                "relationships_found": len(
                    cross_pattern_analysis.get("relationships", {}).get(
                        "relationships", []
                    )
                ),
                "analysis_methods": [
                    "temporal",
                    "frequency",
                    "correlation",
                    "anomaly",
                    "multi_pattern",
                    "evolution_tracking",
                    "interaction_matrix",
                ],
            }
        ]

    def _create_metadata(
        self, patterns: List[Dict[str, Any]], cross_pattern_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create metadata for the analysis result."""
        return {
            "engine": "consolidated_pattern_recognition",
            "version": "1.0.0",
            "patterns_by_type": self._group_patterns_by_type(patterns),
            "cross_pattern_analysis": cross_pattern_analysis,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _group_patterns_by_type(self, patterns: List[Dict[str, Any]]) -> Dict[str, int]:
        """Group patterns by type for metadata."""
        type_counts = {}
        for pattern in patterns:
            pattern_type = pattern.get("type", "unknown")
            type_counts[pattern_type] = type_counts.get(pattern_type, 0) + 1
        return type_counts

    def _create_error_result(self, error_message: str) -> AnalysisResult:
        """Create error result when analysis fails."""
        return AnalysisResult(
            analysis_type="pattern_recognition",
            confidence=0.0,
            patterns=[],
            evidence=[{"type": "error", "message": error_message}],
            metadata={
                "error": error_message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

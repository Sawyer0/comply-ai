"""
Pattern Recognition Engine for advanced pattern detection in security data.

This engine uses sophisticated statistical methods to detect, classify,
and correlate patterns in security data without requiring AI/ML models.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..domain import (
    AnalysisConfiguration,
    AnalysisResult,
    IPatternRecognitionEngine,
    Pattern,
    PatternCorrelation,
    PatternStrength,
    PatternType,
    SecurityData,
    StatisticalAnalysisEngine,
    TimeRange,
)
from ..domain.entities import AnalysisRequest
from .analyzers import (
    TemporalAnalyzer,
    FrequencyAnalyzer,
    CorrelationAnalyzer,
    AnomalyDetector,
    PatternClassifier,
    PatternStrengthCalculator,
    BusinessRelevanceAssessor,
    PatternConfidenceCalculator,
    MultiPatternAnalyzer,
    CompoundRiskCalculator,
    PatternInteractionMatrix,
    PatternEvolutionTracker,
)

logger = logging.getLogger(__name__)


class PatternRecognitionEngine(StatisticalAnalysisEngine, IPatternRecognitionEngine):
    """
    Advanced pattern recognition engine using statistical methods.

    Detects temporal, frequency, correlation, and anomaly patterns
    in security data using sophisticated statistical analysis.
    """

    def __init__(self, config: AnalysisConfiguration):
        super().__init__(config)
        self.pattern_config = config.parameters.get("pattern_recognition", {})
        self.temporal_window = self.pattern_config.get("temporal_window_hours", 24)
        self.correlation_threshold = self.pattern_config.get(
            "correlation_threshold", 0.7
        )
        self.anomaly_sensitivity = self.pattern_config.get("anomaly_sensitivity", 2.0)

        # Initialize specialized analyzers
        self.temporal_analyzer = TemporalAnalyzer(self.pattern_config)
        self.frequency_analyzer = FrequencyAnalyzer(self.pattern_config)
        self.correlation_analyzer = CorrelationAnalyzer(self.pattern_config)
        self.anomaly_detector = AnomalyDetector(self.pattern_config)

        # Initialize pattern classification and assessment components
        self.pattern_classifier = PatternClassifier(self.pattern_config)
        self.strength_calculator = PatternStrengthCalculator(self.pattern_config)
        self.relevance_assessor = BusinessRelevanceAssessor(self.pattern_config)
        self.confidence_calculator = PatternConfidenceCalculator(self.pattern_config)

        # Initialize cross-pattern correlation detection components
        self.multi_pattern_analyzer = MultiPatternAnalyzer(self.pattern_config)
        self.compound_risk_calculator = CompoundRiskCalculator(self.pattern_config)
        self.interaction_matrix = PatternInteractionMatrix(self.pattern_config)
        self.evolution_tracker = PatternEvolutionTracker(self.pattern_config)

    def get_engine_name(self) -> str:
        """Get the name of this analysis engine."""
        return "pattern_recognition"

    async def _perform_analysis(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Perform pattern recognition analysis on the request data.

        Args:
            request: Analysis request containing security data

        Returns:
            AnalysisResult with detected patterns and correlations
        """
        # Extract security data from request
        security_data = self._extract_security_data(request)

        # Detect various types of patterns
        patterns = await self.detect_patterns(security_data)

        # Correlate patterns to find relationships
        correlations = await self.correlate_patterns(patterns)

        # Perform advanced cross-pattern correlation detection
        cross_pattern_analysis = await self._perform_cross_pattern_analysis(
            patterns, security_data
        )

        # Calculate overall confidence
        confidence = self._calculate_pattern_confidence(patterns, correlations)

        # Create analysis result
        result = AnalysisResult(
            analysis_type="pattern_recognition",
            confidence=confidence,
            patterns=patterns,
            evidence=[
                {
                    "type": "pattern_detection",
                    "patterns_found": len(patterns),
                    "correlations_found": len(correlations),
                    "analysis_window": f"{self.temporal_window} hours",
                }
            ],
            metadata={
                "correlations": [corr.dict() for corr in correlations],
                "cross_pattern_analysis": cross_pattern_analysis,
                "detection_methods": [
                    "temporal",
                    "frequency",
                    "correlation",
                    "anomaly",
                    "multi_pattern",
                    "compound_risk",
                    "interaction_matrix",
                    "evolution_tracking",
                ],
                "statistical_significance": await self._calculate_overall_significance(
                    patterns
                ),
            },
        )

        return result

    async def detect_patterns(self, data: SecurityData) -> List[Pattern]:
        """
        Detect patterns in security data using statistical analysis.

        Args:
            data: Security data to analyze for patterns

        Returns:
            List of detected patterns with confidence scores
        """
        patterns = []

        # Use specialized analyzers for pattern detection
        temporal_patterns = await self.temporal_analyzer.analyze(data.time_series)
        patterns.extend(temporal_patterns)

        frequency_patterns = await self.frequency_analyzer.analyze(data.events)
        patterns.extend(frequency_patterns)

        correlation_patterns = await self.correlation_analyzer.analyze(
            data.multi_dimensional
        )
        patterns.extend(correlation_patterns)

        anomaly_patterns = await self.anomaly_detector.analyze(data.metrics)
        patterns.extend(anomaly_patterns)

        # Enhance patterns with classification and assessment
        enhanced_patterns = []
        for pattern in patterns:
            # Classify pattern characteristics
            classified_pattern = self.pattern_classifier.classify_pattern(pattern, data)

            # Calculate pattern strength using statistical methods
            pattern_strength = self.strength_calculator.calculate_pattern_strength(
                classified_pattern, data
            )
            classified_pattern.strength = pattern_strength

            # Assess business relevance
            business_relevance = self.relevance_assessor.assess_business_relevance(
                classified_pattern, data
            )
            classified_pattern.business_relevance = business_relevance

            # Calculate enhanced confidence score
            enhanced_confidence = (
                self.confidence_calculator.calculate_pattern_confidence(
                    classified_pattern,
                    data,
                    {"original_confidence": pattern.confidence},
                )
            )
            classified_pattern.confidence = enhanced_confidence

            enhanced_patterns.append(classified_pattern)

        # Filter patterns by confidence threshold
        filtered_patterns = [
            pattern
            for pattern in enhanced_patterns
            if pattern.confidence >= self.config.confidence_threshold
        ]

        logger.info(
            "Pattern detection and classification completed",
            total_patterns=len(patterns),
            enhanced_patterns=len(enhanced_patterns),
            filtered_patterns=len(filtered_patterns),
            confidence_threshold=self.config.confidence_threshold,
        )

        return filtered_patterns

    async def classify_pattern_strength(self, pattern: Pattern) -> PatternStrength:
        """
        Classify the strength and significance of a detected pattern.

        Args:
            pattern: The pattern to classify

        Returns:
            PatternStrength classification with statistical significance
        """
        # Combine confidence and statistical significance
        combined_score = (pattern.confidence + pattern.statistical_significance) / 2

        if combined_score >= 0.8:
            return PatternStrength.STRONG
        elif combined_score >= 0.6:
            return PatternStrength.MODERATE
        else:
            return PatternStrength.WEAK

    async def correlate_patterns(
        self, patterns: List[Pattern]
    ) -> List[PatternCorrelation]:
        """
        Identify correlations between multiple patterns.

        Args:
            patterns: List of patterns to correlate

        Returns:
            List of pattern correlations with strength indicators
        """
        correlations = []

        # Compare each pattern with every other pattern
        for i, pattern1 in enumerate(patterns):
            for j, pattern2 in enumerate(patterns[i + 1 :], i + 1):
                correlation = await self._calculate_pattern_correlation(
                    pattern1, pattern2
                )
                if (
                    correlation
                    and correlation.correlation_strength >= self.correlation_threshold
                ):
                    correlations.append(correlation)

        return correlations

    def _extract_security_data(self, request: AnalysisRequest) -> SecurityData:
        """Extract security data from analysis request."""
        # Convert request data to SecurityData format
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

        return SecurityData(
            time_series=time_series,
            events=events,
            multi_dimensional=multi_dimensional,
            metrics=metrics,
            metadata={"request_id": request.request_id},
        )

    async def _calculate_pattern_correlation(
        self, pattern1: Pattern, pattern2: Pattern
    ) -> Optional[PatternCorrelation]:
        """Calculate correlation between two patterns."""
        try:
            # Check temporal overlap
            temporal_overlap = self._calculate_temporal_overlap(
                pattern1.time_range, pattern2.time_range
            )

            # Check detector overlap
            detector_overlap = len(
                set(pattern1.affected_detectors) & set(pattern2.affected_detectors)
            )
            detector_similarity = detector_overlap / max(
                len(pattern1.affected_detectors), len(pattern2.affected_detectors)
            )

            # Calculate overall correlation strength
            correlation_strength = (temporal_overlap + detector_similarity) / 2

            if correlation_strength >= self.correlation_threshold:
                return PatternCorrelation(
                    primary_pattern=pattern1.pattern_id,
                    secondary_pattern=pattern2.pattern_id,
                    correlation_strength=correlation_strength,
                    correlation_type="temporal_detector_overlap",
                    confidence=min(pattern1.confidence, pattern2.confidence),
                )

        except Exception as e:
            self.logger.error(f"Pattern correlation calculation failed: {e}")

        return None

    def _calculate_temporal_overlap(
        self, range1: TimeRange, range2: TimeRange
    ) -> float:
        """Calculate temporal overlap between two time ranges."""
        try:
            # Calculate overlap duration
            overlap_start = max(range1.start, range2.start)
            overlap_end = min(range1.end, range2.end)

            if overlap_start >= overlap_end:
                return 0.0

            overlap_duration = (overlap_end - overlap_start).total_seconds()

            # Calculate total duration
            total_start = min(range1.start, range2.start)
            total_end = max(range1.end, range2.end)
            total_duration = (total_end - total_start).total_seconds()

            if total_duration == 0:
                return 0.0

            return overlap_duration / total_duration

        except Exception:
            return 0.0

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        try:
            if len(x) != len(y) or len(x) < 2:
                return 0.0

            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(xi * yi for xi, yi in zip(x, y))
            sum_x2 = sum(xi * xi for xi in x)
            sum_y2 = sum(yi * yi for yi in y)

            numerator = n * sum_xy - sum_x * sum_y
            denominator = (
                (n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)
            ) ** 0.5

            if denominator == 0:
                return 0.0

            return numerator / denominator

        except Exception:
            return 0.0

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values."""
        try:
            if len(values) < 2:
                return 0.0

            mean = sum(values) / len(values)
            return sum((x - mean) ** 2 for x in values) / (len(values) - 1)

        except Exception:
            return 0.0

    def _calculate_pattern_confidence(
        self, patterns: List[Pattern], correlations: List[PatternCorrelation]
    ) -> float:
        """Calculate overall confidence for pattern analysis."""
        if not patterns:
            return 0.0

        # Base confidence from individual patterns
        pattern_confidences = [p.confidence for p in patterns]
        base_confidence = sum(pattern_confidences) / len(pattern_confidences)

        # Boost confidence if correlations are found
        correlation_boost = min(0.2, len(correlations) * 0.05)

        return min(1.0, base_confidence + correlation_boost)

    async def _calculate_overall_significance(self, patterns: List[Pattern]) -> float:
        """Calculate overall statistical significance."""
        if not patterns:
            return 0.0

        significances = [p.statistical_significance for p in patterns]
        return sum(significances) / len(significances)

    async def _perform_cross_pattern_analysis(
        self, patterns: List[Pattern], security_data: SecurityData
    ) -> Dict[str, any]:
        """
        Perform comprehensive cross-pattern correlation detection.

        Args:
            patterns: List of detected patterns
            security_data: Security data for context

        Returns:
            Dictionary containing cross-pattern analysis results
        """
        try:
            if len(patterns) < 2:
                return {"message": "Insufficient patterns for cross-pattern analysis"}

            # Analyze pattern relationships
            relationship_analysis = (
                await self.multi_pattern_analyzer.analyze_pattern_relationships(
                    patterns, security_data
                )
            )

            # Calculate compound risk
            compound_risk = await self.compound_risk_calculator.calculate_compound_risk(
                patterns, relationship_analysis.get("relationships", []), security_data
            )

            # Create interaction matrices
            interaction_matrices = (
                await self.interaction_matrix.create_interaction_matrices(
                    patterns,
                    relationship_analysis.get("relationships", []),
                    security_data,
                )
            )

            # Track pattern evolution (using empty historical data for now)
            evolution_analysis = await self.evolution_tracker.track_pattern_evolution(
                patterns,
                [],
                security_data,  # In production, would pass historical patterns
            )

            cross_pattern_result = {
                "multi_pattern_relationships": relationship_analysis,
                "compound_risk_assessment": compound_risk,
                "interaction_matrices": interaction_matrices,
                "pattern_evolution": evolution_analysis,
                "analysis_summary": {
                    "total_patterns_analyzed": len(patterns),
                    "relationships_found": len(
                        relationship_analysis.get("relationships", [])
                    ),
                    "clusters_identified": len(
                        relationship_analysis.get("clusters", [])
                    ),
                    "compound_risk_level": compound_risk.get("risk_level", "unknown"),
                    "matrices_created": len(
                        [
                            k
                            for k in interaction_matrices.keys()
                            if k not in ["statistics", "visualization_metadata"]
                        ]
                    ),
                    "emerging_patterns": len(
                        evolution_analysis.get("emerging_patterns", [])
                    ),
                    "declining_patterns": len(
                        evolution_analysis.get("declining_patterns", [])
                    ),
                },
            }

            logger.info(
                "Cross-pattern analysis completed",
                total_patterns=len(patterns),
                relationships_found=len(relationship_analysis.get("relationships", [])),
                compound_risk_level=compound_risk.get("risk_level", "unknown"),
            )

            return cross_pattern_result

        except Exception as e:
            logger.error("Cross-pattern analysis failed", error=str(e))
            return {"error": str(e), "message": "Cross-pattern analysis failed"}

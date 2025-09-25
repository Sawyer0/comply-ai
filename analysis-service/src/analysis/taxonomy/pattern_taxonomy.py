"""
Pattern taxonomy for the analysis service.

This module defines the pattern classification system and taxonomy
used for pattern recognition and analysis operations.
"""

from enum import Enum
from typing import Dict, List, Any
from dataclasses import dataclass


class PatternType(Enum):
    """Pattern types for classification."""

    TEMPORAL = "temporal"
    FREQUENCY = "frequency"
    CORRELATION = "correlation"
    ANOMALY = "anomaly"
    BEHAVIORAL = "behavioral"
    STRUCTURAL = "structural"


class PatternSeverity(Enum):
    """Pattern severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


@dataclass
class PatternDefinition:
    """Individual pattern definition."""

    name: str
    pattern_type: PatternType
    severity: PatternSeverity
    description: str
    detection_method: str
    confidence_threshold: float


class PatternTaxonomy:
    """
    Pattern taxonomy management system.

    Provides classification and categorization of patterns
    for consistent pattern recognition across the analysis service.
    """

    def __init__(self):
        self.pattern_definitions = self._initialize_pattern_definitions()
        self.type_weights = self._initialize_type_weights()
        self.severity_scores = self._initialize_severity_scores()

    def _initialize_pattern_definitions(self) -> Dict[str, PatternDefinition]:
        """Initialize standard pattern definitions."""
        return {
            "data_exfiltration": PatternDefinition(
                name="data_exfiltration",
                pattern_type=PatternType.BEHAVIORAL,
                severity=PatternSeverity.CRITICAL,
                description="Pattern indicating potential data exfiltration",
                detection_method="behavioral_analysis",
                confidence_threshold=0.8,
            ),
            "access_anomaly": PatternDefinition(
                name="access_anomaly",
                pattern_type=PatternType.ANOMALY,
                severity=PatternSeverity.HIGH,
                description="Unusual access patterns",
                detection_method="statistical_analysis",
                confidence_threshold=0.7,
            ),
            "frequency_spike": PatternDefinition(
                name="frequency_spike",
                pattern_type=PatternType.FREQUENCY,
                severity=PatternSeverity.MEDIUM,
                description="Unusual frequency of events",
                detection_method="frequency_analysis",
                confidence_threshold=0.6,
            ),
            "temporal_clustering": PatternDefinition(
                name="temporal_clustering",
                pattern_type=PatternType.TEMPORAL,
                severity=PatternSeverity.MEDIUM,
                description="Events clustered in time",
                detection_method="temporal_analysis",
                confidence_threshold=0.6,
            ),
        }

    def _initialize_type_weights(self) -> Dict[PatternType, float]:
        """Initialize pattern type weights for scoring."""
        return {
            PatternType.BEHAVIORAL: 0.9,
            PatternType.ANOMALY: 0.8,
            PatternType.CORRELATION: 0.7,
            PatternType.FREQUENCY: 0.6,
            PatternType.TEMPORAL: 0.6,
            PatternType.STRUCTURAL: 0.5,
        }

    def _initialize_severity_scores(self) -> Dict[PatternSeverity, float]:
        """Initialize severity score mappings."""
        return {
            PatternSeverity.CRITICAL: 1.0,
            PatternSeverity.HIGH: 0.8,
            PatternSeverity.MEDIUM: 0.5,
            PatternSeverity.LOW: 0.3,
            PatternSeverity.INFORMATIONAL: 0.1,
        }

    def get_pattern_definition(self, name: str) -> PatternDefinition:
        """Get pattern definition by name."""
        return self.pattern_definitions.get(name)

    def get_type_weight(self, pattern_type: PatternType) -> float:
        """Get weight for pattern type."""
        return self.type_weights.get(pattern_type, 0.5)

    def get_severity_score(self, severity: PatternSeverity) -> float:
        """Get numeric score for severity level."""
        return self.severity_scores.get(severity, 0.5)

    def classify_pattern_severity(
        self, confidence: float, impact: float
    ) -> PatternSeverity:
        """Classify pattern severity based on confidence and impact."""
        combined_score = (confidence + impact) / 2

        if combined_score >= 0.9:
            return PatternSeverity.CRITICAL
        if combined_score >= 0.7:
            return PatternSeverity.HIGH
        if combined_score >= 0.4:
            return PatternSeverity.MEDIUM
        if combined_score >= 0.2:
            return PatternSeverity.LOW
        return PatternSeverity.INFORMATIONAL

    def get_patterns_by_type(
        self, pattern_type: PatternType
    ) -> List[PatternDefinition]:
        """Get all pattern definitions for a specific type."""
        return [
            pattern
            for pattern in self.pattern_definitions.values()
            if pattern.pattern_type == pattern_type
        ]

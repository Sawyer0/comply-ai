"""Content analysis ML component following SRP.

This module provides ONLY content analysis capabilities:
- Content type classification
- Content complexity scoring
- Feature extraction for routing
- Lightweight content characterization
"""

import hashlib
import json
import logging
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ContentFeatures:  # pylint: disable=too-many-instance-attributes
    """Content features for routing decisions."""

    content_type: str
    content_length: int
    complexity_score: float
    language: str
    encoding: str
    structure_type: str
    sensitive_indicators: List[str]
    processing_priority: int = 5  # Default priority


@dataclass
class ContentProfile:
    """Content profile for detector matching."""

    profile_id: str
    content_characteristics: Dict[str, float]
    recommended_detectors: List[str]
    processing_hints: Dict[str, Any]
    confidence: float


class ContentAnalyzer:
    """Lightweight content analyzer for routing optimization following SRP."""

    def __init__(self, max_content_length: int = 10000):
        """Initialize content analyzer.

        Args:
            max_content_length: Maximum content length to analyze
        """
        self.max_content_length = max_content_length
        self.is_trained = False

        # Content type patterns
        self.content_patterns = {
            "json": [r"^\s*[\{\[]", r'["\']:\s*["\']', r'["\'],\s*["\']'],
            "xml": [r"<\?xml", r"<[^>]+>", r"</[^>]+>"],
            "html": [r"<!DOCTYPE", r"<html", r"<body", r"<div"],
            "email": [r"@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", r"Subject:", r"From:"],
            "code": [r"function\s+\w+", r"class\s+\w+", r"import\s+\w+", r"def\s+\w+"],
            "sql": [r"SELECT\s+", r"INSERT\s+INTO", r"UPDATE\s+", r"DELETE\s+FROM"],
            "log": [r"\d{4}-\d{2}-\d{2}", r"\[INFO\]", r"\[ERROR\]", r"\[DEBUG\]"],
        }

        # Sensitive content indicators
        self.sensitive_patterns = {
            "pii": [
                r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
                r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Credit card
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            ],
            "financial": [
                r"\$\d+(?:,\d{3})*(?:\.\d{2})?",  # Currency
                r"\b(?:account|routing)\s+number\b",
                r"\biban\b",
                r"\bswift\b",
            ],
            "medical": [
                r"\b(?:patient|diagnosis|prescription|medication)\b",
                r"\b(?:hipaa|phi|medical record)\b",
            ],
            "security": [
                r"\b(?:password|token|key|secret)\b",
                r"\b(?:api[_-]?key|access[_-]?token)\b",
            ],
        }

    def analyze_content(
        self, content: str, content_id: Optional[str] = None
    ) -> ContentFeatures:
        """Analyze content and extract features for routing.

        Args:
            content: Content to analyze
            content_id: Optional content identifier

        Returns:
            Content features
        """
        try:
            # Truncate content if too long
            analyzed_content = content[: self.max_content_length]

            # Extract basic features
            content_type = self._detect_content_type(analyzed_content)
            complexity_score = self._calculate_complexity_score(analyzed_content)
            language = self._detect_language(analyzed_content)
            encoding = self._detect_encoding(analyzed_content)
            structure_type = self._detect_structure_type(analyzed_content)
            sensitive_indicators = self._detect_sensitive_content(analyzed_content)
            processing_priority = self._calculate_processing_priority(
                content_type, complexity_score, sensitive_indicators
            )

            features = ContentFeatures(
                content_type=content_type,
                content_length=len(content),
                complexity_score=complexity_score,
                language=language,
                encoding=encoding,
                structure_type=structure_type,
                sensitive_indicators=sensitive_indicators,
                processing_priority=processing_priority,
            )

            logger.debug(
                "Analyzed content features",
                extra={
                    "content_id": content_id,
                    "content_type": content_type,
                    "complexity_score": complexity_score,
                    "sensitive_indicators": len(sensitive_indicators),
                },
            )

            return features

        except (ValueError, TypeError) as e:
            logger.error("Content analysis failed: %s", str(e))
            return self._fallback_features(content)

    def _detect_content_type(self, content: str) -> str:
        """Detect content type based on patterns.

        Args:
            content: Content to analyze

        Returns:
            Detected content type
        """
        content_lower = content.lower().strip()

        # Check each content type pattern
        for content_type, patterns in self.content_patterns.items():
            matches = sum(
                1 for pattern in patterns if re.search(pattern, content_lower)
            )
            if matches >= len(patterns) // 2:  # Majority of patterns match
                return content_type

        # Default to text if no specific type detected
        return "text"

    def _calculate_complexity_score(self, content: str) -> float:
        """Calculate content complexity score.

        Args:
            content: Content to analyze

        Returns:
            Complexity score between 0 and 1
        """
        if not content:
            return 0.0

        # Factors contributing to complexity
        factors = []

        # Length factor
        length_factor = min(1.0, len(content) / 5000.0)
        factors.append(length_factor)

        # Unique character ratio
        unique_chars = len(set(content))
        char_diversity = min(1.0, unique_chars / 100.0)
        factors.append(char_diversity)

        # Nested structure depth (for structured content)
        nesting_depth = self._calculate_nesting_depth(content)
        nesting_factor = min(1.0, nesting_depth / 10.0)
        factors.append(nesting_factor)

        # Special character density
        special_chars = len(re.findall(r"[^\w\s]", content))
        special_density = min(1.0, special_chars / len(content))
        factors.append(special_density)

        # Average complexity
        return float(np.mean(factors)) if factors else 0.0

    def _calculate_nesting_depth(self, content: str) -> int:
        """Calculate nesting depth for structured content.

        Args:
            content: Content to analyze

        Returns:
            Maximum nesting depth
        """
        max_depth = 0
        current_depth = 0

        for char in content:
            if char in "({[<":
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char in ")}]>":
                current_depth = max(0, current_depth - 1)

        return max_depth

    def _detect_language(self, content: str) -> str:
        """Detect content language (simplified).

        Args:
            content: Content to analyze

        Returns:
            Detected language code
        """
        # Simple heuristic-based language detection
        # In production, could use more sophisticated language detection

        # Check for common English words
        english_words = ["the", "and", "or", "but", "in", "on", "at", "to", "for"]
        english_count = sum(1 for word in english_words if word in content.lower())

        if english_count >= 3:
            return "en"

        # Check for other language indicators
        if re.search(r"[àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]", content.lower()):
            return "es"  # Spanish/Romance languages

        if re.search(r"[äöüß]", content.lower()):
            return "de"  # German

        return "unknown"

    def _detect_encoding(self, content: str) -> str:
        """Detect content encoding.

        Args:
            content: Content to analyze

        Returns:
            Detected encoding
        """
        try:
            # Check if content is valid UTF-8
            content.encode("utf-8")
            return "utf-8"
        except UnicodeEncodeError:
            return "unknown"

    def _detect_structure_type(self, content: str) -> str:
        """Detect content structure type.

        Args:
            content: Content to analyze

        Returns:
            Structure type
        """
        # Check for structured formats
        if self._is_json_like(content):
            return "structured"
        if self._is_tabular(content):
            return "tabular"
        if self._is_hierarchical(content):
            return "hierarchical"
        return "unstructured"

    def _is_json_like(self, content: str) -> bool:
        """Check if content is JSON-like."""
        try:
            json.loads(content)
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    def _is_tabular(self, content: str) -> bool:
        """Check if content is tabular."""
        lines = content.split("\n")
        if len(lines) < 2:
            return False

        # Check for consistent delimiters
        delimiters = [",", "\t", "|", ";"]
        for delimiter in delimiters:
            if all(delimiter in line for line in lines[:5]):
                return True

        return False

    def _is_hierarchical(self, content: str) -> bool:
        """Check if content is hierarchical."""
        # Look for XML/HTML tags or indentation patterns
        if re.search(r"<[^>]+>.*</[^>]+>", content):
            return True

        # Check for consistent indentation
        lines = content.split("\n")
        indented_lines = [line for line in lines if line.startswith((" ", "\t"))]
        return len(indented_lines) > len(lines) * 0.3

    def _detect_sensitive_content(self, content: str) -> List[str]:
        """Detect sensitive content indicators.

        Args:
            content: Content to analyze

        Returns:
            List of sensitive content types detected
        """
        detected_types = []

        for sensitive_type, patterns in self.sensitive_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    detected_types.append(sensitive_type)
                    break  # One match per type is enough

        return detected_types

    def _calculate_processing_priority(
        self,
        content_type: str,
        complexity_score: float,
        sensitive_indicators: List[str],
    ) -> int:
        """Calculate processing priority based on content characteristics following SRP.

        This method has a single responsibility: determine processing priority
        based on content analysis results.

        Args:
            content_type: Detected content type
            complexity_score: Content complexity score (0.0-1.0)
            sensitive_indicators: List of sensitive content types detected

        Returns:
            Processing priority (1-10, higher = more urgent processing needed)
        """
        # Base priority mapping for different content types
        base_priority = self._get_content_type_priority(content_type)

        # Complexity adjustment
        complexity_adjustment = self._get_complexity_priority_adjustment(
            complexity_score
        )

        # Sensitivity adjustment
        sensitivity_adjustment = self._get_sensitivity_priority_adjustment(
            sensitive_indicators
        )

        # Calculate final priority
        final_priority = base_priority + complexity_adjustment + sensitivity_adjustment

        # Clamp to valid range (1-10)
        return max(1, min(10, final_priority))

    def _get_content_type_priority(self, content_type: str) -> int:
        """Get base priority for content type following DRY principle.

        Args:
            content_type: Content type string

        Returns:
            Base priority for the content type
        """
        # Centralized content type priority mapping
        type_priorities = {
            "sql": 9,  # High priority - potential security risk
            "email": 8,  # High priority - often contains PII
            "json": 7,  # Medium-high priority - structured data
            "xml": 6,  # Medium-high priority - structured data
            "html": 6,  # Medium-high priority - web content
            "text": 5,  # Medium priority - general text
            "code": 4,  # Medium-low priority - source code
            "log": 3,  # Low priority - log files
        }
        return type_priorities.get(content_type, 5)  # Default medium priority

    def _get_complexity_priority_adjustment(self, complexity_score: float) -> int:
        """Get priority adjustment based on complexity following SRP.

        Args:
            complexity_score: Complexity score (0.0-1.0)

        Returns:
            Priority adjustment (-2 to +2)
        """
        if complexity_score > 0.8:
            return 2  # Very complex content needs more processing
        if complexity_score > 0.6:
            return 1  # Moderately complex content
        if complexity_score < 0.2:
            return -1  # Simple content can be deprioritized
        return 0  # Normal complexity

    def _get_sensitivity_priority_adjustment(
        self, sensitive_indicators: List[str]
    ) -> int:
        """Get priority adjustment based on sensitive content following SRP.

        Args:
            sensitive_indicators: List of detected sensitive content types

        Returns:
            Priority adjustment (0 to +3)
        """
        if not sensitive_indicators:
            return 0

        # Different sensitive content types have different urgency levels
        high_priority_types = {"financial", "medical", "security"}
        medium_priority_types = {"pii"}

        high_priority_count = sum(
            1 for indicator in sensitive_indicators if indicator in high_priority_types
        )
        medium_priority_count = sum(
            1
            for indicator in sensitive_indicators
            if indicator in medium_priority_types
        )

        # Calculate adjustment based on sensitivity
        adjustment = high_priority_count * 2 + medium_priority_count * 1

        # Cap the adjustment to prevent extreme priorities
        return min(3, adjustment)

    def _fallback_features(self, content: str) -> ContentFeatures:
        """Generate fallback features when analysis fails.

        Args:
            content: Original content

        Returns:
            Fallback content features
        """
        return ContentFeatures(
            content_type="text",
            content_length=len(content),
            complexity_score=0.5,
            language="unknown",
            encoding="utf-8",
            structure_type="unstructured",
            sensitive_indicators=[],
            processing_priority=5,  # Default medium priority
        )

    def create_content_profile(
        self, features: ContentFeatures, detector_performance: Dict[str, float]
    ) -> ContentProfile:
        """Create content profile for detector matching.

        Args:
            features: Content features
            detector_performance: Historical detector performance data

        Returns:
            Content profile with detector recommendations
        """
        # Create content characteristics vector
        characteristics = {
            "complexity": features.complexity_score,
            "length_factor": min(1.0, features.content_length / 10000.0),
            "structure_score": self._get_structure_score(features.structure_type),
            "sensitivity_score": len(features.sensitive_indicators) / 4.0,
            "priority_score": features.processing_priority / 10.0,
        }

        # Recommend detectors based on content characteristics
        recommended_detectors = self._recommend_detectors(
            features, detector_performance
        )

        # Generate processing hints
        processing_hints = {
            "timeout_multiplier": 1.0 + features.complexity_score,
            "parallel_processing": features.complexity_score < 0.3,
            "cache_results": features.content_length < 1000,
            "priority_boost": len(features.sensitive_indicators) > 0,
        }

        # Calculate confidence based on feature clarity
        confidence = self._calculate_profile_confidence(features)

        profile_data = "_".join(
            [
                features.content_type,
                f"{features.complexity_score}",
                str(features.content_length),
            ]
        )
        profile_id = hashlib.md5(profile_data.encode()).hexdigest()[:8]

        return ContentProfile(
            profile_id=profile_id,
            content_characteristics=characteristics,
            recommended_detectors=recommended_detectors,
            processing_hints=processing_hints,
            confidence=confidence,
        )

    def _get_structure_score(self, structure_type: str) -> float:
        """Get numeric score for structure type.

        Args:
            structure_type: Structure type string

        Returns:
            Numeric structure score
        """
        structure_scores = {
            "structured": 1.0,
            "hierarchical": 0.8,
            "tabular": 0.6,
            "unstructured": 0.2,
        }
        return structure_scores.get(structure_type, 0.5)

    def _recommend_detectors(
        self, features: ContentFeatures, detector_performance: Dict[str, float]
    ) -> List[str]:
        """Recommend detectors based on content features.

        Args:
            features: Content features
            detector_performance: Detector performance data

        Returns:
            List of recommended detector IDs
        """
        recommendations = []

        # Content type specific recommendations
        type_detectors = {
            "json": ["json-validator", "structure-analyzer"],
            "xml": ["xml-parser", "schema-validator"],
            "html": ["html-parser", "xss-detector"],
            "email": ["email-validator", "spam-detector"],
            "code": ["code-analyzer", "security-scanner"],
            "sql": ["sql-parser", "injection-detector"],
            "text": ["text-analyzer", "sentiment-analyzer"],
        }

        content_specific = type_detectors.get(features.content_type, ["text-analyzer"])
        recommendations.extend(content_specific)

        # Sensitive content specific detectors
        if "pii" in features.sensitive_indicators:
            recommendations.append("pii-detector")
        if "financial" in features.sensitive_indicators:
            recommendations.append("financial-detector")
        if "medical" in features.sensitive_indicators:
            recommendations.append("medical-detector")
        if "security" in features.sensitive_indicators:
            recommendations.append("security-detector")

        # Filter by performance and remove duplicates
        available_detectors = list(detector_performance.keys())
        recommendations = [d for d in recommendations if d in available_detectors]

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for detector in recommendations:
            if detector not in seen:
                seen.add(detector)
                unique_recommendations.append(detector)

        return unique_recommendations[:5]  # Limit to top 5

    def _calculate_profile_confidence(self, features: ContentFeatures) -> float:
        """Calculate confidence in content profile.

        Args:
            features: Content features

        Returns:
            Confidence score between 0 and 1
        """
        confidence_factors = []

        # Content type detection confidence
        if features.content_type != "text":
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)

        # Language detection confidence
        if features.language != "unknown":
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.4)

        # Structure detection confidence
        if features.structure_type != "unstructured":
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)

        # Sensitive content detection confidence
        if features.sensitive_indicators:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)

        if not confidence_factors:
            return 0.0
        return float(np.mean(confidence_factors))

    def get_analyzer_stats(self) -> Dict[str, Any]:
        """Get content analyzer statistics.

        Returns:
            Analyzer statistics
        """
        return {
            "max_content_length": self.max_content_length,
            "supported_content_types": list(self.content_patterns.keys()),
            "sensitive_pattern_types": list(self.sensitive_patterns.keys()),
            "is_trained": self.is_trained,
            "analyzer_version": "1.0.0",
        }

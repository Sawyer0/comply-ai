"""
Built-in pattern recognition plugin for Analysis Service.

This plugin provides basic pattern recognition capabilities using statistical analysis
and rule-based pattern detection.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from ..interfaces import (
    IPatternDetectorPlugin,
    PluginMetadata,
    PluginType,
    PluginCapability,
    AnalysisRequest,
    AnalysisResult,
)

logger = logging.getLogger(__name__)


class PatternRecognitionPlugin(IPatternDetectorPlugin):
    """Built-in pattern recognition plugin."""

    def __init__(self):
        self.initialized = False
        self.config = {}
        self.pattern_cache = {}

    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="builtin_pattern_recognition",
            version="1.0.0",
            description="Built-in pattern recognition using statistical analysis",
            author="Analysis Service Team",
            plugin_type=PluginType.ANALYSIS_ENGINE,
            capabilities=[
                PluginCapability.PATTERN_RECOGNITION,
                PluginCapability.STATISTICAL_ANALYSIS,
                PluginCapability.BATCH_PROCESSING,
                PluginCapability.REAL_TIME_ANALYSIS,
            ],
            dependencies=[],
            supported_frameworks=["SOC2", "ISO27001", "HIPAA"],
            min_confidence_threshold=0.6,
            max_batch_size=100,
        )

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin."""
        try:
            self.config = config

            # Initialize pattern detection parameters
            self.confidence_threshold = config.get("confidence_threshold", 0.7)
            self.pattern_window_size = config.get("pattern_window_size", 10)
            self.statistical_threshold = config.get("statistical_threshold", 2.0)

            # Initialize pattern types
            self.pattern_types = [
                "temporal_anomaly",
                "frequency_spike",
                "correlation_break",
                "threshold_violation",
                "sequence_anomaly",
            ]

            self.initialized = True
            logger.info("Pattern recognition plugin initialized")
            return True

        except Exception as e:
            logger.error(
                "Failed to initialize pattern recognition plugin", error=str(e)
            )
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "status": "healthy" if self.initialized else "not_initialized",
            "initialized": self.initialized,
            "pattern_cache_size": len(self.pattern_cache),
            "supported_patterns": len(self.pattern_types),
            "last_check": datetime.now(timezone.utc).isoformat(),
        }

    async def cleanup(self) -> bool:
        """Cleanup plugin resources."""
        try:
            self.pattern_cache.clear()
            self.initialized = False
            logger.info("Pattern recognition plugin cleaned up")
            return True
        except Exception as e:
            logger.error("Failed to cleanup pattern recognition plugin", error=str(e))
            return False

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Perform pattern recognition analysis."""
        start_time = time.time()

        try:
            if not self.initialized:
                raise RuntimeError("Plugin not initialized")

            # Detect patterns
            patterns = await self.detect_patterns(request)

            # Classify pattern strength
            pattern_strength = await self.classify_pattern_strength(patterns)

            # Calculate confidence based on pattern strength and metadata
            confidence = self._calculate_confidence(patterns, request.metadata)

            # Prepare result
            result_data = {
                "patterns": patterns,
                "pattern_strength": pattern_strength,
                "pattern_count": len(patterns.get("detected_patterns", [])),
                "analysis_type": "pattern_recognition",
                "framework_mappings": self._map_to_frameworks(patterns),
            }

            processing_time = (time.time() - start_time) * 1000

            return AnalysisResult(
                request_id=request.request_id,
                plugin_name="builtin_pattern_recognition",
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
                "Pattern recognition analysis failed",
                request_id=request.request_id,
                error=str(e),
            )

            return AnalysisResult(
                request_id=request.request_id,
                plugin_name="builtin_pattern_recognition",
                plugin_version="1.0.0",
                confidence=0.0,
                result_data={},
                processing_time_ms=processing_time,
                errors=[str(e)],
            )

    async def batch_analyze(
        self, requests: List[AnalysisRequest]
    ) -> List[AnalysisResult]:
        """Perform batch pattern recognition analysis."""
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
                    logger.error("Batch analysis item failed", error=str(result))
                    # Create error result
                    error_result = AnalysisResult(
                        request_id="unknown",
                        plugin_name="builtin_pattern_recognition",
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
        return ["pattern_recognition", "temporal_analysis", "statistical_analysis"]

    async def validate_request(self, request: AnalysisRequest) -> bool:
        """Validate if the plugin can handle the request."""
        try:
            # Check if analysis type is supported
            if request.analysis_type not in self.get_supported_analysis_types():
                return False

            # Check if metadata contains required fields
            required_fields = ["timestamp", "data_points"]
            for field in required_fields:
                if field not in request.metadata:
                    return False

            return True

        except Exception as e:
            logger.error("Request validation failed", error=str(e))
            return False

    async def detect_patterns(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Detect patterns in the analysis request."""
        try:
            metadata = request.metadata
            data_points = metadata.get("data_points", [])

            detected_patterns = []

            # Temporal anomaly detection
            temporal_patterns = self._detect_temporal_anomalies(data_points)
            detected_patterns.extend(temporal_patterns)

            # Frequency spike detection
            frequency_patterns = self._detect_frequency_spikes(data_points)
            detected_patterns.extend(frequency_patterns)

            # Correlation break detection
            correlation_patterns = self._detect_correlation_breaks(data_points)
            detected_patterns.extend(correlation_patterns)

            # Threshold violation detection
            threshold_patterns = self._detect_threshold_violations(
                data_points, metadata
            )
            detected_patterns.extend(threshold_patterns)

            return {
                "detected_patterns": detected_patterns,
                "total_patterns": len(detected_patterns),
                "pattern_types": list(set(p["type"] for p in detected_patterns)),
                "analysis_metadata": {
                    "data_point_count": len(data_points),
                    "analysis_window": self.pattern_window_size,
                    "threshold": self.statistical_threshold,
                },
            }

        except Exception as e:
            logger.error("Pattern detection failed", error=str(e))
            return {"detected_patterns": [], "error": str(e)}

    async def classify_pattern_strength(self, patterns: Dict[str, Any]) -> str:
        """Classify the strength of detected patterns."""
        try:
            detected_patterns = patterns.get("detected_patterns", [])

            if not detected_patterns:
                return "none"

            # Calculate average confidence
            confidences = [p.get("confidence", 0.0) for p in detected_patterns]
            avg_confidence = sum(confidences) / len(confidences)

            # Classify based on confidence and pattern count
            pattern_count = len(detected_patterns)

            if avg_confidence >= 0.8 and pattern_count >= 3:
                return "strong"
            elif avg_confidence >= 0.6 and pattern_count >= 2:
                return "moderate"
            elif avg_confidence >= 0.4 or pattern_count >= 1:
                return "weak"
            else:
                return "none"

        except Exception as e:
            logger.error("Pattern strength classification failed", error=str(e))
            return "unknown"

    def get_pattern_types(self) -> List[str]:
        """Get supported pattern types."""
        return self.pattern_types

    # Private helper methods

    def _detect_temporal_anomalies(
        self, data_points: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect temporal anomalies in data points."""
        patterns = []

        try:
            if len(data_points) < self.pattern_window_size:
                return patterns

            # Simple temporal anomaly detection based on time gaps
            timestamps = [
                dp.get("timestamp") for dp in data_points if dp.get("timestamp")
            ]

            if len(timestamps) < 2:
                return patterns

            # Calculate time differences
            time_diffs = []
            for i in range(1, len(timestamps)):
                try:
                    t1 = datetime.fromisoformat(
                        timestamps[i - 1].replace("Z", "+00:00")
                    )
                    t2 = datetime.fromisoformat(timestamps[i].replace("Z", "+00:00"))
                    diff = (t2 - t1).total_seconds()
                    time_diffs.append(diff)
                except:
                    continue

            if not time_diffs:
                return patterns

            # Detect anomalous gaps
            avg_diff = sum(time_diffs) / len(time_diffs)
            std_diff = (
                sum((x - avg_diff) ** 2 for x in time_diffs) / len(time_diffs)
            ) ** 0.5

            for i, diff in enumerate(time_diffs):
                if abs(diff - avg_diff) > self.statistical_threshold * std_diff:
                    patterns.append(
                        {
                            "type": "temporal_anomaly",
                            "confidence": min(
                                0.9, abs(diff - avg_diff) / (std_diff + 1e-6)
                            ),
                            "description": f"Unusual time gap detected: {diff:.2f}s vs avg {avg_diff:.2f}s",
                            "position": i,
                            "severity": (
                                "high" if diff > avg_diff + 2 * std_diff else "medium"
                            ),
                        }
                    )

        except Exception as e:
            logger.error("Temporal anomaly detection failed", error=str(e))

        return patterns

    def _detect_frequency_spikes(
        self, data_points: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect frequency spikes in data points."""
        patterns = []

        try:
            # Count events per time window
            time_windows = {}

            for dp in data_points:
                timestamp = dp.get("timestamp")
                if not timestamp:
                    continue

                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    # Group by hour
                    window = dt.replace(minute=0, second=0, microsecond=0)
                    time_windows[window] = time_windows.get(window, 0) + 1
                except:
                    continue

            if len(time_windows) < 2:
                return patterns

            # Detect spikes
            counts = list(time_windows.values())
            avg_count = sum(counts) / len(counts)
            std_count = (sum((x - avg_count) ** 2 for x in counts) / len(counts)) ** 0.5

            for window, count in time_windows.items():
                if count > avg_count + self.statistical_threshold * std_count:
                    patterns.append(
                        {
                            "type": "frequency_spike",
                            "confidence": min(
                                0.9, (count - avg_count) / (std_count + 1e-6) / 3
                            ),
                            "description": f"Frequency spike detected: {count} events vs avg {avg_count:.1f}",
                            "window": window.isoformat(),
                            "severity": (
                                "high"
                                if count > avg_count + 3 * std_count
                                else "medium"
                            ),
                        }
                    )

        except Exception as e:
            logger.error("Frequency spike detection failed", error=str(e))

        return patterns

    def _detect_correlation_breaks(
        self, data_points: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect correlation breaks in data points."""
        patterns = []

        try:
            # Simple correlation analysis between numeric fields
            numeric_fields = []
            for dp in data_points:
                for key, value in dp.items():
                    if isinstance(value, (int, float)) and key not in ["timestamp"]:
                        if key not in numeric_fields:
                            numeric_fields.append(key)

            if len(numeric_fields) < 2:
                return patterns

            # Calculate correlations between field pairs
            for i in range(len(numeric_fields)):
                for j in range(i + 1, len(numeric_fields)):
                    field1, field2 = numeric_fields[i], numeric_fields[j]

                    values1 = [dp.get(field1, 0) for dp in data_points if field1 in dp]
                    values2 = [dp.get(field2, 0) for dp in data_points if field2 in dp]

                    if len(values1) != len(values2) or len(values1) < 3:
                        continue

                    # Simple correlation coefficient
                    mean1 = sum(values1) / len(values1)
                    mean2 = sum(values2) / len(values2)

                    numerator = sum(
                        (v1 - mean1) * (v2 - mean2) for v1, v2 in zip(values1, values2)
                    )
                    denom1 = sum((v1 - mean1) ** 2 for v1 in values1) ** 0.5
                    denom2 = sum((v2 - mean2) ** 2 for v2 in values2) ** 0.5

                    if denom1 > 0 and denom2 > 0:
                        correlation = numerator / (denom1 * denom2)

                        # Detect significant correlation changes (simplified)
                        if (
                            abs(correlation) < 0.3
                        ):  # Weak correlation might indicate a break
                            patterns.append(
                                {
                                    "type": "correlation_break",
                                    "confidence": 0.6,
                                    "description": f"Weak correlation between {field1} and {field2}: {correlation:.3f}",
                                    "fields": [field1, field2],
                                    "correlation": correlation,
                                    "severity": "medium",
                                }
                            )

        except Exception as e:
            logger.error("Correlation break detection failed", error=str(e))

        return patterns

    def _detect_threshold_violations(
        self, data_points: List[Dict[str, Any]], metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect threshold violations in data points."""
        patterns = []

        try:
            thresholds = metadata.get("thresholds", {})

            for dp in data_points:
                for field, value in dp.items():
                    if field in thresholds and isinstance(value, (int, float)):
                        threshold_config = thresholds[field]

                        if isinstance(threshold_config, dict):
                            min_val = threshold_config.get("min")
                            max_val = threshold_config.get("max")

                            if min_val is not None and value < min_val:
                                patterns.append(
                                    {
                                        "type": "threshold_violation",
                                        "confidence": 0.8,
                                        "description": f"{field} below minimum: {value} < {min_val}",
                                        "field": field,
                                        "value": value,
                                        "threshold": min_val,
                                        "violation_type": "minimum",
                                        "severity": "high",
                                    }
                                )

                            if max_val is not None and value > max_val:
                                patterns.append(
                                    {
                                        "type": "threshold_violation",
                                        "confidence": 0.8,
                                        "description": f"{field} above maximum: {value} > {max_val}",
                                        "field": field,
                                        "value": value,
                                        "threshold": max_val,
                                        "violation_type": "maximum",
                                        "severity": "high",
                                    }
                                )

        except Exception as e:
            logger.error("Threshold violation detection failed", error=str(e))

        return patterns

    def _calculate_confidence(
        self, patterns: Dict[str, Any], metadata: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence score."""
        try:
            detected_patterns = patterns.get("detected_patterns", [])

            if not detected_patterns:
                return 0.0

            # Base confidence on pattern confidences and metadata quality
            pattern_confidences = [p.get("confidence", 0.0) for p in detected_patterns]
            avg_confidence = sum(pattern_confidences) / len(pattern_confidences)

            # Adjust based on data quality
            data_points = metadata.get("data_points", [])
            data_quality_factor = min(1.0, len(data_points) / self.pattern_window_size)

            # Adjust based on pattern diversity
            pattern_types = set(p.get("type") for p in detected_patterns)
            diversity_factor = min(1.0, len(pattern_types) / len(self.pattern_types))

            final_confidence = (
                avg_confidence * data_quality_factor * (0.8 + 0.2 * diversity_factor)
            )

            return min(0.95, max(0.0, final_confidence))

        except Exception as e:
            logger.error("Confidence calculation failed", error=str(e))
            return 0.0

    def _map_to_frameworks(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Map detected patterns to compliance frameworks."""
        framework_mappings = {}

        try:
            detected_patterns = patterns.get("detected_patterns", [])

            for framework in ["SOC2", "ISO27001", "HIPAA"]:
                framework_mappings[framework] = {
                    "applicable_controls": [],
                    "risk_level": "low",
                    "recommendations": [],
                }

                # Map patterns to framework controls
                for pattern in detected_patterns:
                    pattern_type = pattern.get("type")
                    severity = pattern.get("severity", "medium")

                    if framework == "SOC2":
                        if pattern_type in ["temporal_anomaly", "frequency_spike"]:
                            framework_mappings[framework]["applicable_controls"].append(
                                "CC6.1"
                            )
                        if pattern_type == "threshold_violation":
                            framework_mappings[framework]["applicable_controls"].append(
                                "CC7.1"
                            )

                    elif framework == "ISO27001":
                        if pattern_type in ["correlation_break", "threshold_violation"]:
                            framework_mappings[framework]["applicable_controls"].append(
                                "A.12.6.1"
                            )
                        if pattern_type == "frequency_spike":
                            framework_mappings[framework]["applicable_controls"].append(
                                "A.16.1.2"
                            )

                    elif framework == "HIPAA":
                        if pattern_type in ["temporal_anomaly", "frequency_spike"]:
                            framework_mappings[framework]["applicable_controls"].append(
                                "164.308(a)(1)"
                            )
                        if pattern_type == "threshold_violation":
                            framework_mappings[framework]["applicable_controls"].append(
                                "164.312(b)"
                            )

                # Determine overall risk level
                high_severity_count = sum(
                    1 for p in detected_patterns if p.get("severity") == "high"
                )
                if high_severity_count > 0:
                    framework_mappings[framework]["risk_level"] = "high"
                elif len(detected_patterns) > 2:
                    framework_mappings[framework]["risk_level"] = "medium"

                # Add recommendations
                if framework_mappings[framework]["risk_level"] != "low":
                    framework_mappings[framework]["recommendations"] = [
                        "Review and adjust monitoring thresholds",
                        "Implement additional alerting for detected patterns",
                        "Conduct root cause analysis for anomalies",
                    ]

        except Exception as e:
            logger.error("Framework mapping failed", error=str(e))

        return framework_mappings

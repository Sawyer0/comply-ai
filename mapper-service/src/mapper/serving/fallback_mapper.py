"""
Rule-based fallback mapper for when model confidence is low or validation fails.

This consolidates the fallback mapping functionality from the original llama-mapper
implementation with enhanced metrics tracking and mapping statistics.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class FallbackMapper:
    """Rule-based mapper using detector YAML configurations."""

    def __init__(self, detector_configs_path: str = "pillars-detectors"):
        """
        Initialize the fallback mapper.

        Args:
            detector_configs_path: Path to detector configuration directory
        """
        self.detector_configs_path = Path(detector_configs_path)
        self.detector_mappings: Dict[str, Dict[str, str]] = {}
        self.usage_stats: Dict[str, int] = {}
        self.load_detector_mappings()

    def load_detector_mappings(self) -> None:
        """Load all detector mapping configurations."""
        try:
            if not self.detector_configs_path.exists():
                logger.warning(
                    f"Detector configs path does not exist: {self.detector_configs_path}"
                )
                return

            config_files = list(self.detector_configs_path.glob("*.yaml"))
            config_files.extend(list(self.detector_configs_path.glob("*.yml")))

            for config_file in config_files:
                if config_file.name in [
                    "taxonomy.yaml",
                    "frameworks.yaml",
                    "schema.json",
                ]:
                    continue  # Skip non-detector files

                try:
                    with open(config_file, "r") as f:
                        config = yaml.safe_load(f)

                    detector_name = config.get("detector")
                    if detector_name and "maps" in config:
                        self.detector_mappings[detector_name] = config["maps"]
                        logger.info("Loaded mappings for detector: %s", detector_name)

                except Exception as e:
                    logger.warning(
                        f"Failed to load detector config {config_file}: {str(e)}"
                    )

            logger.info("Loaded %s detector mappings", len(self.detector_mappings))

        except Exception as e:
            logger.error("Failed to load detector mappings: %s", str(e))
            self.detector_mappings = {}

    def map(
        self, detector: str, output: str, reason: str = "low_confidence"
    ) -> "MappingResponse":
        """
        Map detector output using rule-based mappings.

        Args:
            detector: Name of the detector
            output: Raw detector output
            reason: Reason for using fallback (for tracking)

        Returns:
            MappingResponse: Mapped response
        """
        # Import here to avoid circular dependency
        from ..schemas.models import MappingResponse, Provenance

        # Track fallback usage
        self._track_usage(detector, reason)

        logger.info(
            "Using fallback mapper for detector %s, reason: %s", detector, reason
        )

        # Get detector mappings
        detector_maps = self.detector_mappings.get(detector, {})

        if not detector_maps:
            logger.warning("No mappings found for detector: %s", detector)
            return self._create_unknown_response(detector, output, reason)

        # Try exact match first
        canonical_label = detector_maps.get(output)
        if canonical_label:
            return MappingResponse(
                taxonomy=[canonical_label],
                scores={canonical_label: 1.0},
                confidence=0.8,  # High confidence for exact rule match
                notes=f"Rule-based mapping (exact): {output} -> {canonical_label}",
                provenance=Provenance(detector=detector, raw_ref=output),
                version_info=None,
            )

        # Try case-insensitive match
        output_lower = output.lower()
        for rule_output, canonical_label in detector_maps.items():
            if rule_output.lower() == output_lower:
                return MappingResponse(
                    taxonomy=[canonical_label],
                    scores={canonical_label: 1.0},
                    confidence=0.7,  # Slightly lower confidence for case-insensitive match
                    notes=f"Rule-based mapping (case-insensitive): {output} -> {canonical_label}",
                    provenance=Provenance(detector=detector, raw_ref=output),
                    version_info=None,
                )

        # Try partial match
        for rule_output, canonical_label in detector_maps.items():
            if (
                rule_output.lower() in output_lower
                or output_lower in rule_output.lower()
            ):
                return MappingResponse(
                    taxonomy=[canonical_label],
                    scores={canonical_label: 0.8},
                    confidence=0.5,  # Lower confidence for partial match
                    notes=f"Rule-based mapping (partial): {output} -> {canonical_label}",
                    provenance=Provenance(detector=detector, raw_ref=output),
                    version_info=None,
                )

        # No match found
        logger.warning(
            f"No rule mapping found for detector {detector} output: {output}"
        )
        return self._create_unknown_response(detector, output, reason)

    def _create_unknown_response(
        self, detector: str, output: str, reason: str = "unknown"
    ) -> "MappingResponse":
        """
        Create response for unknown/unmapped outputs.

        Args:
            detector: Detector name
            output: Original output
            reason: Reason for fallback

        Returns:
            MappingResponse: Unknown response
        """
        # Import here to avoid circular dependency
        from ..schemas.models import MappingResponse, Provenance

        return MappingResponse(
            taxonomy=["OTHER.Unknown"],
            scores={"OTHER.Unknown": 0.0},
            confidence=0.0,
            notes=f"No mapping found for detector {detector} output: {output} (reason: {reason})",
            provenance=Provenance(detector=detector, raw_ref=output),
            version_info=None,
        )

    def _track_usage(self, detector: str, reason: str) -> None:
        """Track fallback usage for analytics."""
        key = f"{detector}:{reason}"
        self.usage_stats[key] = self.usage_stats.get(key, 0) + 1

    def get_supported_detectors(self) -> List[str]:
        """
        Get list of supported detector names.

        Returns:
            List[str]: List of detector names
        """
        return list(self.detector_mappings.keys())

    def get_detector_mappings(self, detector: str) -> Optional[Dict[str, str]]:
        """
        Get mappings for a specific detector.

        Args:
            detector: Detector name

        Returns:
            Optional[Dict[str, str]]: Detector mappings or None
        """
        return self.detector_mappings.get(detector)

    def reload_mappings(self) -> None:
        """Reload detector mappings from disk."""
        logger.info("Reloading detector mappings")
        self.detector_mappings.clear()
        self.load_detector_mappings()

    def add_detector_mapping(self, detector: str, mappings: Dict[str, str]) -> None:
        """
        Add or update detector mappings programmatically.

        Args:
            detector: Detector name
            mappings: Dictionary of output -> canonical_label mappings
        """
        self.detector_mappings[detector] = mappings
        logger.info("Added/updated mappings for detector: %s", detector)

    def get_mapping_stats(self) -> Dict[str, Any]:
        """
        Get statistics about loaded mappings.

        Returns:
            Dict[str, Any]: Mapping statistics
        """
        total_mappings = sum(len(maps) for maps in self.detector_mappings.values())
        return {
            "total_detectors": len(self.detector_mappings),
            "total_mappings": total_mappings,
            "detectors": {
                detector: len(maps) for detector, maps in self.detector_mappings.items()
            },
        }

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about fallback usage.

        Returns:
            Dict[str, Any]: Usage statistics
        """
        total_usage = sum(self.usage_stats.values())

        by_detector = {}
        by_reason = {}

        for key, count in self.usage_stats.items():
            if ":" in key:
                detector, reason = key.split(":", 1)
                by_detector[detector] = by_detector.get(detector, 0) + count
                by_reason[reason] = by_reason.get(reason, 0) + count

        return {
            "total_fallback_usage": total_usage,
            "by_detector": by_detector,
            "by_reason": by_reason,
            "detailed_stats": self.usage_stats,
        }

    def log_improvement_suggestions(self) -> None:
        """
        Log suggestions for model improvement based on fallback usage patterns.
        """
        stats = self.get_usage_stats()

        if stats["total_fallback_usage"] == 0:
            logger.info("No fallback usage recorded - model performing well!")
            return

        logger.info("=== Fallback Usage Analysis for Model Improvement ===")
        logger.info("Total fallback usage: %s", stats["total_fallback_usage"])

        # Analyze by detector
        if stats["by_detector"]:
            logger.info("Top detectors requiring fallback:")
            sorted_detectors = sorted(
                stats["by_detector"].items(), key=lambda x: x[1], reverse=True
            )
            for detector, count in sorted_detectors[:5]:
                logger.info("  %s: %s fallbacks", detector, count)

        # Analyze by reason
        if stats["by_reason"]:
            logger.info("Fallback reasons:")
            for reason, count in stats["by_reason"].items():
                logger.info("  %s: %s cases", reason, count)

        # Provide improvement suggestions
        total_usage = stats["total_fallback_usage"]
        low_confidence_usage = stats["by_reason"].get("low_confidence", 0)

        if low_confidence_usage > total_usage * 0.5:
            logger.warning(
                "High rate of low-confidence fallbacks - consider retraining "
                "model or adjusting confidence threshold"
            )

        # Check for detectors with high fallback rates
        for detector, count in stats["by_detector"].items():
            if count > total_usage * 0.2:  # More than 20% of total fallbacks
                logger.warning(
                    f"Detector {detector} has high fallback rate ({count} uses) - "
                    "consider adding more training data for this detector type"
                )

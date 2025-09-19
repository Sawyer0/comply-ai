"""
Rule-based fallback mapper for when model confidence is low or validation fails.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from ..api.models import MappingResponse, Provenance

# from ..api.models import MappingResponse, Provenance  # Moved to local imports
from ..monitoring.metrics_collector import get_metrics_collector

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
        self.metrics_collector = get_metrics_collector()
        self.load_detector_mappings()

    def load_detector_mappings(self) -> None:
        """Load all detector mapping configurations."""
        try:
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
                        logger.info(f"Loaded mappings for detector: {detector_name}")

                except Exception as e:
                    logger.warning(
                        f"Failed to load detector config {config_file}: {str(e)}"
                    )

            logger.info(f"Loaded {len(self.detector_mappings)} detector mappings")

        except Exception as e:
            logger.error(f"Failed to load detector mappings: {str(e)}")
            self.detector_mappings = {}

    def map(self, detector: str, output: str, reason: str = "low_confidence") -> "MappingResponse":
        """
        Map detector output using rule-based mappings.

        Args:
            detector: Name of the detector
            output: Raw detector output
            reason: Reason for using fallback (for tracking)

        Returns:
            MappingResponse: Mapped response
        """
        # Local import to avoid circular dependency
        from ..api.models import MappingResponse, Provenance

        # Track fallback usage for model improvement
        self.metrics_collector.increment_counter(
            "fallback_usage_total", {"detector": detector, "reason": reason}
        )

        logger.info(f"Using fallback mapper for detector {detector}, reason: {reason}")

        # Get detector mappings
        detector_maps = self.detector_mappings.get(detector, {})

        if not detector_maps:
            logger.warning(f"No mappings found for detector: {detector}")
            self.metrics_collector.increment_counter(
                "fallback_no_mapping_total", {"detector": detector}
            )
            return self._create_unknown_response(detector, output, reason)

        # Try exact match first
        canonical_label = detector_maps.get(output)
        if canonical_label:
            self.metrics_collector.increment_counter(
                "fallback_exact_match_total", {"detector": detector}
            )
            return MappingResponse(
                taxonomy=[canonical_label],
                scores={canonical_label: 1.0},
                confidence=0.8,  # High confidence for exact rule match
                notes=f"Rule-based mapping (exact): {output} -> {canonical_label}",
                provenance=Provenance(detector=detector, raw_ref=None),
            )

        # Try case-insensitive match
        output_lower = output.lower()
        for rule_output, canonical_label in detector_maps.items():
            if rule_output.lower() == output_lower:
                self.metrics_collector.increment_counter(
                    "fallback_case_insensitive_match_total", {"detector": detector}
                )
                return MappingResponse(
                    taxonomy=[canonical_label],
                    scores={canonical_label: 1.0},
                    confidence=0.7,  # Slightly lower confidence for case-insensitive match
                    notes=f"Rule-based mapping (case-insensitive): {output} -> {canonical_label}",
                    provenance=Provenance(detector=detector, raw_ref=None),
                )

        # Try partial match
        for rule_output, canonical_label in detector_maps.items():
            if (
                rule_output.lower() in output_lower
                or output_lower in rule_output.lower()
            ):
                self.metrics_collector.increment_counter(
                    "fallback_partial_match_total", {"detector": detector}
                )
                return MappingResponse(
                    taxonomy=[canonical_label],
                    scores={canonical_label: 0.8},
                    confidence=0.5,  # Lower confidence for partial match
                    notes=f"Rule-based mapping (partial): {output} -> {canonical_label}",
                    provenance=Provenance(detector=detector, raw_ref=None),
                )

        # No match found
        logger.warning(
            f"No rule mapping found for detector {detector} output: {output}"
        )
        self.metrics_collector.increment_counter(
            "fallback_no_match_total", {"detector": detector}
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
        # Local import to avoid circular dependency
        from ..api.models import MappingResponse, Provenance

        return MappingResponse(
            taxonomy=["OTHER.Unknown"],
            scores={"OTHER.Unknown": 0.0},
            confidence=0.0,
            notes=f"No mapping found for detector {detector} output: {output} (reason: {reason})",
            provenance=Provenance(detector=detector, raw_ref=None),
        )

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
        logger.info(f"Added/updated mappings for detector: {detector}")

    def get_mapping_stats(self) -> Dict[str, Any]:
        """
        Get statistics about loaded mappings.

        Returns:
            Dict[str, int]: Mapping statistics
        """
        total_mappings = sum(len(maps) for maps in self.detector_mappings.values())
        return {
            "total_detectors": len(self.detector_mappings),
            "total_mappings": total_mappings,
            "detectors": {
                detector: len(maps) for detector, maps in self.detector_mappings.items()
            },
        }

    def get_fallback_usage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about fallback usage for model improvement tracking.

        Returns:
            Dict[str, any]: Fallback usage statistics
        """
        metrics: Dict[str, Any] = self.metrics_collector.get_all_metrics()

        fallback_stats: Dict[str, Any] = {
            "total_fallback_usage": 0,
            "by_detector": {},
            "by_reason": {},
            "match_types": {
                "exact_matches": 0,
                "case_insensitive_matches": 0,
                "partial_matches": 0,
                "no_matches": 0,
                "no_mapping": 0,
            },
        }

        # Extract fallback-related counters
        for counter_name, label_dict in metrics.get("counters", {}).items():
            if counter_name.startswith("fallback_"):
                for label_key, count in label_dict.items():
                    if counter_name == "fallback_usage_total":
                        fallback_stats["total_fallback_usage"] += count

                        # Parse labels to extract detector and reason
                        if "detector=" in label_key:
                            parts = label_key.split(",")
                            detector = None
                            reason = None
                            for part in parts:
                                if part.startswith("detector="):
                                    detector = part.split("=", 1)[1]
                                elif part.startswith("reason="):
                                    reason = part.split("=", 1)[1]

                            if detector:
                                fallback_stats["by_detector"][detector] = (
                                    fallback_stats["by_detector"].get(detector, 0)
                                    + count
                                )
                            if reason:
                                fallback_stats["by_reason"][reason] = (
                                    fallback_stats["by_reason"].get(reason, 0) + count
                                )

                    elif counter_name == "fallback_exact_match_total":
                        fallback_stats["match_types"]["exact_matches"] += count
                    elif counter_name == "fallback_case_insensitive_match_total":
                        fallback_stats["match_types"][
                            "case_insensitive_matches"
                        ] += count
                    elif counter_name == "fallback_partial_match_total":
                        fallback_stats["match_types"]["partial_matches"] += count
                    elif counter_name == "fallback_no_match_total":
                        fallback_stats["match_types"]["no_matches"] += count
                    elif counter_name == "fallback_no_mapping_total":
                        fallback_stats["match_types"]["no_mapping"] += count

        return fallback_stats

    def log_fallback_improvement_suggestions(self) -> None:
        """
        Log suggestions for model improvement based on fallback usage patterns.
        """
        stats = self.get_fallback_usage_stats()

        if stats["total_fallback_usage"] == 0:
            logger.info("No fallback usage recorded - model performing well!")
            return

        logger.info("=== Fallback Usage Analysis for Model Improvement ===")
        logger.info(f"Total fallback usage: {stats['total_fallback_usage']}")

        # Analyze by detector
        if stats["by_detector"]:
            logger.info("Top detectors requiring fallback:")
            sorted_detectors = sorted(
                stats["by_detector"].items(), key=lambda x: x[1], reverse=True
            )
            for detector, count in sorted_detectors[:5]:
                logger.info(f"  {detector}: {count} fallbacks")

        # Analyze by reason
        if stats["by_reason"]:
            logger.info("Fallback reasons:")
            for reason, count in stats["by_reason"].items():
                logger.info(f"  {reason}: {count} cases")

        # Analyze match types
        match_types = stats["match_types"]
        total_matches = sum(match_types.values())
        if total_matches > 0:
            logger.info("Fallback match success rates:")
            logger.info(
                f"  Exact matches: {match_types['exact_matches']} ({match_types['exact_matches']/total_matches*100:.1f}%)"
            )
            logger.info(
                f"  Case-insensitive: {match_types['case_insensitive_matches']} ({match_types['case_insensitive_matches']/total_matches*100:.1f}%)"
            )
            logger.info(
                f"  Partial matches: {match_types['partial_matches']} ({match_types['partial_matches']/total_matches*100:.1f}%)"
            )
            logger.info(
                f"  No matches: {match_types['no_matches']} ({match_types['no_matches']/total_matches*100:.1f}%)"
            )
            logger.info(
                f"  No mapping available: {match_types['no_mapping']} ({match_types['no_mapping']/total_matches*100:.1f}%)"
            )

        # Provide improvement suggestions
        if match_types["no_matches"] > match_types["exact_matches"]:
            logger.warning(
                "High rate of unmatched outputs - consider expanding detector mappings"
            )

        if match_types["partial_matches"] > match_types["exact_matches"]:
            logger.warning(
                "High rate of partial matches - consider adding exact mappings for common outputs"
            )

        if (
            stats["by_reason"].get("low_confidence", 0)
            > stats["total_fallback_usage"] * 0.5
        ):
            logger.warning(
                "High rate of low-confidence fallbacks - consider retraining model or adjusting confidence threshold"
            )

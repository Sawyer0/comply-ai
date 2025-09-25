"""
Rule-based fallback mapping system.

Single responsibility: Provide deterministic rule-based mappings when model fails.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..schemas.models import MappingResponse, Provenance

logger = logging.getLogger(__name__)


class RuleBasedFallback:
    """
    Rule-based fallback mapper using detector configurations.

    Single responsibility: Execute rule-based mapping logic.
    """

    def __init__(self, detector_configs_path: str = "config/detectors"):
        """
        Initialize rule-based fallback.

        Args:
            detector_configs_path: Path to detector configuration directory
        """
        self.detector_configs_path = Path(detector_configs_path)
        self.detector_mappings: Dict[str, Dict[str, str]] = {}
        self.usage_stats: Dict[str, int] = {}
        self.load_detector_mappings()

    def load_detector_mappings(self) -> None:
        """Load detector mappings from configuration files."""
        try:
            if not self.detector_configs_path.exists():
                logger.warning(
                    f"Detector configs path not found: {self.detector_configs_path}"
                )
                self._load_default_mappings()
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

            logger.info("Loaded %d detector mappings", len(self.detector_mappings))

        except Exception as e:
            logger.error("Failed to load detector mappings: %s", str(e))
            self._load_default_mappings()

    def _load_default_mappings(self) -> None:
        """Load default detector mappings."""
        self.detector_mappings = {
            "presidio": {
                "EMAIL_ADDRESS": "PII.Contact.Email",
                "PHONE_NUMBER": "PII.Contact.Phone",
                "PERSON": "PII.Identity.Name",
                "US_SSN": "PII.Identity.SSN",
                "CREDIT_CARD": "PII.Financial.CreditCard",
                "LOCATION": "PII.Contact.Address",
            },
            "deberta": {
                "toxic": "CONTENT.Harmful.Toxic",
                "severe_toxic": "CONTENT.Harmful.Toxic",
                "obscene": "CONTENT.Harmful.Toxic",
                "threat": "CONTENT.Harmful.Violence",
                "insult": "CONTENT.Harmful.Toxic",
                "identity_hate": "CONTENT.Harmful.Hate",
            },
            "custom_detector": {
                "api_key": "SECURITY.Credentials.APIKey",
                "password": "SECURITY.Credentials.Password",
                "token": "SECURITY.Credentials.Token",
                "unauthorized_access": "SECURITY.Access.Unauthorized",
            },
        }
        logger.info("Loaded default detector mappings")

    def map(
        self, detector: str, output: str, reason: str = "model_failure"
    ) -> MappingResponse:
        """
        Map detector output using rule-based mappings.

        Args:
            detector: Name of the detector
            output: Raw detector output
            reason: Reason for using fallback

        Returns:
            MappingResponse: Mapped response
        """
        # Track usage
        self._track_usage(detector, reason)

        logger.info(
            "Using rule-based fallback for detector %s, reason: %s", detector, reason
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
                scores={canonical_label: 0.9},  # High confidence for exact rule match
                confidence=0.9,
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
                    scores={canonical_label: 0.8},
                    confidence=0.8,  # Slightly lower for case-insensitive
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
                    scores={canonical_label: 0.6},
                    confidence=0.6,  # Lower confidence for partial match
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
        self, detector: str, output: str, reason: str
    ) -> MappingResponse:
        """Create response for unknown/unmapped outputs."""
        return MappingResponse(
            taxonomy=["OTHER.Unknown"],
            scores={"OTHER.Unknown": 0.1},
            confidence=0.1,
            notes=f"No rule mapping found for detector {detector} output: {output} (reason: {reason})",
            provenance=Provenance(detector=detector, raw_ref=output),
            version_info=None,
        )

    def _track_usage(self, detector: str, reason: str) -> None:
        """Track fallback usage for analytics."""
        key = f"{detector}:{reason}"
        self.usage_stats[key] = self.usage_stats.get(key, 0) + 1

    def get_supported_detectors(self) -> List[str]:
        """Get list of supported detector names."""
        return list(self.detector_mappings.keys())

    def get_detector_mappings(self, detector: str) -> Optional[Dict[str, str]]:
        """Get mappings for a specific detector."""
        return self.detector_mappings.get(detector)

    def add_detector_mapping(self, detector: str, mappings: Dict[str, str]) -> None:
        """Add or update detector mappings programmatically."""
        self.detector_mappings[detector] = mappings
        logger.info("Added/updated mappings for detector: %s", detector)

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        total_usage = sum(self.usage_stats.values())

        by_detector = {}
        by_reason = {}

        for key, count in self.usage_stats.items():
            if ":" in key:
                detector, reason = key.split(":", 1)
                by_detector[detector] = by_detector.get(detector, 0) + count
                by_reason[reason] = by_reason.get(reason, 0) + count

        return {
            "total_usage": total_usage,
            "by_detector": by_detector,
            "by_reason": by_reason,
            "detailed_stats": self.usage_stats,
        }

    def reload_mappings(self) -> None:
        """Reload detector mappings from disk."""
        logger.info("Reloading detector mappings")
        self.detector_mappings.clear()
        self.load_detector_mappings()

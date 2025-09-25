"""
Framework adaptation for compliance mappings.

Single responsibility: Adapt canonical taxonomy to specific compliance frameworks.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ..schemas.models import MappingResponse

logger = logging.getLogger(__name__)


class FrameworkAdapter:
    """
    Adapts canonical taxonomy to compliance frameworks.

    Single responsibility: Framework adaptation only.
    """

    def __init__(self, frameworks_path: str = "config/frameworks"):
        """
        Initialize framework adapter.

        Args:
            frameworks_path: Path to framework configuration directory
        """
        self.frameworks_path = Path(frameworks_path)
        self.framework_mappings: Dict[str, Dict[str, str]] = {}
        self.load_frameworks()

    def load_frameworks(self) -> None:
        """Load framework mappings from configuration files."""
        try:
            if not self.frameworks_path.exists():
                logger.warning(f"Frameworks path not found: {self.frameworks_path}")
                self._load_default_frameworks()
                return

            framework_files = list(self.frameworks_path.glob("*.yaml"))
            framework_files.extend(list(self.frameworks_path.glob("*.yml")))

            for framework_file in framework_files:
                try:
                    with open(framework_file, "r") as f:
                        config = yaml.safe_load(f)

                    framework_name = config.get("framework", {}).get("name")
                    if framework_name and "mappings" in config:
                        self.framework_mappings[framework_name] = config["mappings"]
                        logger.info("Loaded framework: %s", framework_name)

                except Exception as e:
                    logger.warning(
                        "Failed to load framework file %s: %s", framework_file, str(e)
                    )

            logger.info("Loaded %d frameworks", len(self.framework_mappings))

        except Exception as e:
            logger.error("Failed to load frameworks: %s", str(e))
            self._load_default_frameworks()

    def _load_default_frameworks(self) -> None:
        """Load default framework mappings."""
        self.framework_mappings = {
            "GDPR": {
                "PII.Contact.Email": "GDPR.PersonalData.ContactDetails",
                "PII.Contact.Phone": "GDPR.PersonalData.ContactDetails",
                "PII.Contact.Address": "GDPR.PersonalData.ContactDetails",
                "PII.Identity.Name": "GDPR.PersonalData.IdentifyingInformation",
                "PII.Identity.SSN": "GDPR.PersonalData.IdentifyingInformation",
                "PII.Financial.CreditCard": "GDPR.SpecialCategory.FinancialData",
                "PII.Financial.BankAccount": "GDPR.SpecialCategory.FinancialData",
            },
            "HIPAA": {
                "PII.Identity.Name": "HIPAA.PHI.IndividuallyIdentifiableHealthInformation",
                "PII.Identity.SSN": "HIPAA.PHI.IndividuallyIdentifiableHealthInformation",
                "PII.Contact.Email": "HIPAA.PHI.ContactInformation",
                "PII.Contact.Phone": "HIPAA.PHI.ContactInformation",
                "PII.Contact.Address": "HIPAA.PHI.GeographicInformation",
            },
            "SOC2": {
                "SECURITY.Access.Unauthorized": "SOC2.CC6.LogicalAccess",
                "SECURITY.Access.Privilege": "SOC2.CC6.LogicalAccess",
                "SECURITY.Credentials.Password": "SOC2.CC6.LogicalAccess",
                "SECURITY.Credentials.APIKey": "SOC2.CC6.LogicalAccess",
                "PII.Contact.Email": "SOC2.CC6.DataProtection",
                "PII.Financial.CreditCard": "SOC2.CC6.DataProtection",
            },
            "ISO27001": {
                "PII.Contact.Email": "ISO27001.A.8.2.1.DataClassification",
                "PII.Identity.Name": "ISO27001.A.8.2.1.DataClassification",
                "PII.Financial.CreditCard": "ISO27001.A.8.2.1.DataClassification",
                "SECURITY.Access.Unauthorized": "ISO27001.A.9.1.1.AccessControlPolicy",
                "SECURITY.Credentials.Password": "ISO27001.A.9.2.1.UserRegistration",
            },
        }
        logger.info("Loaded default framework mappings")

    def adapt_to_framework(
        self, response: MappingResponse, framework: str
    ) -> MappingResponse:
        """
        Adapt canonical taxonomy response to specific framework.

        Args:
            response: Original mapping response
            framework: Target framework name

        Returns:
            MappingResponse: Framework-adapted response
        """
        framework_mappings = self.framework_mappings.get(framework)

        if not framework_mappings:
            logger.warning(f"No mappings found for framework: {framework}")
            return response

        # Adapt taxonomy labels
        adapted_taxonomy = []
        adapted_scores = {}

        for canonical_label in response.taxonomy:
            framework_label = framework_mappings.get(canonical_label)
            if framework_label:
                adapted_taxonomy.append(framework_label)
                adapted_scores[framework_label] = response.scores.get(
                    canonical_label, 0.0
                )
            else:
                # Keep original if no mapping exists
                adapted_taxonomy.append(canonical_label)
                adapted_scores[canonical_label] = response.scores.get(
                    canonical_label, 0.0
                )

        # Create adapted response
        adapted_response = MappingResponse(
            taxonomy=adapted_taxonomy,
            scores=adapted_scores,
            confidence=response.confidence
            * 0.95,  # Slight confidence reduction for adaptation
            notes=f"{response.notes or ''} [Adapted to {framework}]".strip(),
            provenance=response.provenance,
            version_info=response.version_info,
        )

        return adapted_response

    def get_supported_frameworks(self) -> List[str]:
        """
        Get list of supported frameworks.

        Returns:
            List[str]: List of framework names
        """
        return list(self.framework_mappings.keys())

    def get_framework_mappings(self, framework: str) -> Optional[Dict[str, str]]:
        """
        Get mappings for a specific framework.

        Args:
            framework: Framework name

        Returns:
            Optional[Dict[str, str]]: Framework mappings or None
        """
        return self.framework_mappings.get(framework)

    def add_framework_mapping(self, framework: str, mappings: Dict[str, str]) -> None:
        """
        Add or update framework mappings.

        Args:
            framework: Framework name
            mappings: Dictionary of canonical -> framework label mappings
        """
        self.framework_mappings[framework] = mappings
        logger.info("Added/updated framework mappings: %s", framework)

    def get_reverse_mapping(self, framework: str) -> Dict[str, str]:
        """
        Get reverse mapping (framework -> canonical).

        Args:
            framework: Framework name

        Returns:
            Dict[str, str]: Reverse mapping
        """
        forward_mapping = self.framework_mappings.get(framework, {})
        return {v: k for k, v in forward_mapping.items()}

    def reload_frameworks(self) -> None:
        """Reload framework mappings from files."""
        logger.info("Reloading framework mappings")
        self.framework_mappings.clear()
        self.load_frameworks()

"""
Framework adaptation for compliance mappings.

Single responsibility: Adapt canonical taxonomy to specific compliance frameworks.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from shared.taxonomy import framework_mapping_registry

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
        # Use centralized framework mapping system
        self.framework_registry = framework_mapping_registry

        # Legacy support - maintain interface compatibility
        self.frameworks_path = Path(frameworks_path)
        self.framework_mappings: Dict[str, Dict[str, str]] = {}
        self._sync_from_registry()

    def _sync_from_registry(self) -> None:
        """Sync from centralized framework mapping registry."""
        try:
            # Get all supported frameworks from registry
            supported_frameworks = self.framework_registry.get_supported_frameworks()

            self.framework_mappings = {}
            for framework_name in supported_frameworks:
                mappings = self.framework_registry.get_framework_mappings(
                    framework_name
                )
                if mappings:
                    self.framework_mappings[framework_name] = mappings

            logger.info(
                "Synced %d frameworks from registry", len(self.framework_mappings)
            )

        except (AttributeError, KeyError, TypeError) as e:
            logger.error("Failed to sync from framework registry: %s", str(e))
            self._load_default_frameworks()

    def load_frameworks(self) -> None:
        """Load frameworks - delegates to centralized registry."""
        self._sync_from_registry()

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
        return self.framework_registry.get_supported_frameworks()

    def get_framework_mappings(self, framework: str) -> Optional[Dict[str, str]]:
        """
        Get mappings for a specific framework.

        Args:
            framework: Framework name

        Returns:
            Optional[Dict[str, str]]: Framework mappings or None
        """
        return self.framework_registry.get_framework_mappings(framework)

    def add_framework_mapping(self, framework: str, mappings: Dict[str, str]) -> None:
        """
        Add or update framework mappings.

        Args:
            framework: Framework name
            mappings: Dictionary of canonical -> framework label mappings
        """
        # Delegate to centralized registry
        from shared.taxonomy import ChangeType

        self.framework_registry.create_new_framework_version(
            framework,
            mappings,
            ChangeType.MINOR,
            [f"Updated mappings via framework adapter"],
            "mapper-service",
        )
        self._sync_from_registry()
        logger.info("Added/updated framework mappings: %s", framework)

    def get_reverse_mapping(self, framework: str) -> Dict[str, str]:
        """
        Get reverse mapping (framework -> canonical).

        Args:
            framework: Framework name

        Returns:
            Dict[str, str]: Reverse mapping
        """
        return self.framework_registry.get_reverse_mapping(framework)

    def reload_frameworks(self) -> None:
        """Reload framework mappings from centralized registry."""
        logger.info("Reloading framework mappings from registry")
        self._sync_from_registry()

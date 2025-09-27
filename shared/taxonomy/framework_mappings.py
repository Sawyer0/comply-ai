"""
Framework mapping registry with versioning and migration tools.

Single responsibility: Manage compliance framework mappings and their evolution.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

from .base_models import ChangeType
from .version_manager import VersionManager

logger = logging.getLogger(__name__)


class FrameworkMappingRegistry:
    """
    Framework mapping registry with versioning.

    Single responsibility: Manage framework mappings and their versions.
    """

    def __init__(self, frameworks_path: Optional[Path] = None):
        """Initialize framework mapping registry."""
        self.frameworks_path = frameworks_path or Path("config/frameworks")
        self.framework_mappings: Dict[str, Dict[str, str]] = (
            {}
        )  # framework -> canonical_label -> framework_label
        self.version_managers: Dict[str, VersionManager] = {}

        self._load_frameworks()

    def _load_frameworks(self) -> None:
        """Load framework mappings from configuration."""
        if not self.frameworks_path.exists():
            self._create_default_frameworks()
            return

        for framework_file in self.frameworks_path.glob("*.json"):
            try:
                with open(framework_file, "r") as f:
                    framework_data = json.load(f)
                    framework_name = framework_data.get("framework_name")
                    if framework_name:
                        self._load_framework(framework_name, framework_data)
            except Exception as e:
                logger.error("Failed to load framework %s: %s", framework_file, str(e))

    def _load_framework(self, framework_name: str, data: Dict[str, any]) -> None:
        """Load individual framework data."""
        if framework_name not in self.framework_mappings:
            self.framework_mappings[framework_name] = {}
            self.version_managers[framework_name] = VersionManager(
                f"framework_{framework_name}", self.frameworks_path / framework_name
            )

        # Load current mappings
        current_mappings = data.get("current_mappings", {})
        self.framework_mappings[framework_name] = current_mappings

    def _create_default_frameworks(self) -> None:
        """Create default framework mappings."""
        # GDPR mappings
        gdpr_mappings = {
            "PII.Contact.Email": "GDPR.PersonalData.ContactDetails",
            "PII.Contact.Phone": "GDPR.PersonalData.ContactDetails",
            "PII.Contact.Address": "GDPR.PersonalData.ContactDetails",
            "PII.Identity.Name": "GDPR.PersonalData.IdentifyingInformation",
            "PII.Identity.SSN": "GDPR.PersonalData.IdentifyingInformation",
            "PII.Financial.CreditCard": "GDPR.SpecialCategory.FinancialData",
            "PII.Financial.BankAccount": "GDPR.SpecialCategory.FinancialData",
        }

        self.register_framework("GDPR", "1.0.0", gdpr_mappings, "system")

        # HIPAA mappings
        hipaa_mappings = {
            "PII.Identity.Name": "HIPAA.PHI.IndividuallyIdentifiableHealthInformation",
            "PII.Identity.SSN": "HIPAA.PHI.IndividuallyIdentifiableHealthInformation",
            "PII.Contact.Email": "HIPAA.PHI.ContactInformation",
            "PII.Contact.Phone": "HIPAA.PHI.ContactInformation",
            "PII.Contact.Address": "HIPAA.PHI.GeographicInformation",
        }

        self.register_framework("HIPAA", "1.0.0", hipaa_mappings, "system")

        # SOC2 mappings
        soc2_mappings = {
            "SECURITY.Access.Unauthorized": "SOC2.CC6.LogicalAccess",
            "SECURITY.Access.Privilege": "SOC2.CC6.LogicalAccess",
            "SECURITY.Credentials.Password": "SOC2.CC6.LogicalAccess",
            "SECURITY.Credentials.APIKey": "SOC2.CC6.LogicalAccess",
            "PII.Contact.Email": "SOC2.CC6.DataProtection",
            "PII.Financial.CreditCard": "SOC2.CC6.DataProtection",
        }

        self.register_framework("SOC2", "1.0.0", soc2_mappings, "system")

        # ISO27001 mappings
        iso27001_mappings = {
            "PII.Contact.Email": "ISO27001.A.8.2.1.DataClassification",
            "PII.Identity.Name": "ISO27001.A.8.2.1.DataClassification",
            "PII.Financial.CreditCard": "ISO27001.A.8.2.1.DataClassification",
            "SECURITY.Access.Unauthorized": "ISO27001.A.9.1.1.AccessControlPolicy",
            "SECURITY.Credentials.Password": "ISO27001.A.9.2.1.UserRegistration",
        }

        self.register_framework("ISO27001", "1.0.0", iso27001_mappings, "system")

        self._save_all_frameworks()
        logger.info("Created default framework mappings")

    def register_framework(
        self,
        framework_name: str,
        version: str,
        mappings: Dict[str, str],
        created_by: str,
    ) -> None:
        """Register a new framework version."""
        if framework_name not in self.framework_mappings:
            self.framework_mappings[framework_name] = {}
            self.version_managers[framework_name] = VersionManager(
                f"framework_{framework_name}", self.frameworks_path / framework_name
            )

        self.framework_mappings[framework_name] = mappings

        # Create version in version manager
        checksum = self.version_managers[framework_name].calculate_checksum(mappings)
        self.version_managers[framework_name].create_version(
            ChangeType.MINOR,
            [f"Registered framework {framework_name} v{version}"],
            created_by,
            checksum,
        )

    def create_new_framework_version(
        self,
        framework_name: str,
        mappings: Dict[str, str],
        change_type: ChangeType,
        changes: List[str],
        created_by: str,
    ) -> str:
        """Create a new version of framework mappings."""
        if framework_name not in self.version_managers:
            raise ValueError(f"Framework '{framework_name}' not found")

        # Validate mappings
        self._validate_mappings(mappings)

        # Create new version
        checksum = self.version_managers[framework_name].calculate_checksum(mappings)
        new_version = self.version_managers[framework_name].create_version(
            change_type, changes, created_by, checksum
        )

        # Update current mappings
        self.framework_mappings[framework_name] = mappings

        # Save to file
        self._save_framework(framework_name)

        return new_version

    def _validate_mappings(self, mappings: Dict[str, str]) -> None:
        """Validate framework mappings."""
        if not mappings:
            raise ValueError("Mappings cannot be empty")

        for canonical_label, framework_label in mappings.items():
            if not canonical_label or not framework_label:
                raise ValueError("Labels cannot be empty")

            # Basic format validation
            if "." not in canonical_label:
                raise ValueError(f"Invalid canonical label format: {canonical_label}")

    def map_to_framework(
        self, canonical_labels: List[str], framework_name: str
    ) -> Dict[str, str]:
        """Map canonical labels to framework labels."""
        if framework_name not in self.framework_mappings:
            raise ValueError(f"Framework '{framework_name}' not found")

        framework_mappings = self.framework_mappings[framework_name]
        result = {}

        for canonical_label in canonical_labels:
            framework_label = framework_mappings.get(canonical_label)
            if framework_label:
                result[canonical_label] = framework_label
            else:
                # Keep original if no mapping exists
                result[canonical_label] = canonical_label

        return result

    def get_reverse_mapping(self, framework_name: str) -> Dict[str, str]:
        """Get reverse mapping (framework -> canonical)."""
        if framework_name not in self.framework_mappings:
            return {}

        forward_mapping = self.framework_mappings[framework_name]
        return {v: k for k, v in forward_mapping.items()}

    def get_supported_frameworks(self) -> List[str]:
        """Get list of supported frameworks."""
        return list(self.framework_mappings.keys())

    def get_framework_mappings(self, framework_name: str) -> Optional[Dict[str, str]]:
        """Get mappings for a specific framework."""
        return self.framework_mappings.get(framework_name)

    def add_mapping(
        self, framework_name: str, canonical_label: str, framework_label: str
    ) -> None:
        """Add a single mapping to a framework."""
        if framework_name not in self.framework_mappings:
            raise ValueError(f"Framework '{framework_name}' not found")

        self.framework_mappings[framework_name][canonical_label] = framework_label
        logger.info(
            "Added mapping: %s -> %s for %s",
            canonical_label,
            framework_label,
            framework_name,
        )

    def remove_mapping(self, framework_name: str, canonical_label: str) -> None:
        """Remove a mapping from a framework."""
        if framework_name not in self.framework_mappings:
            raise ValueError(f"Framework '{framework_name}' not found")

        if canonical_label in self.framework_mappings[framework_name]:
            del self.framework_mappings[framework_name][canonical_label]
            logger.info("Removed mapping: %s from %s", canonical_label, framework_name)

    def get_coverage_stats(
        self, framework_name: str, canonical_labels: Set[str]
    ) -> Dict[str, any]:
        """Get coverage statistics for a framework."""
        if framework_name not in self.framework_mappings:
            return {"coverage": 0.0, "mapped": 0, "total": 0}

        framework_mappings = self.framework_mappings[framework_name]
        mapped_labels = set(framework_mappings.keys()) & canonical_labels

        total = len(canonical_labels)
        mapped = len(mapped_labels)
        coverage = (mapped / total) * 100 if total > 0 else 0.0

        return {
            "coverage": coverage,
            "mapped": mapped,
            "total": total,
            "unmapped_labels": list(canonical_labels - mapped_labels),
        }

    def _save_framework(self, framework_name: str) -> None:
        """Save individual framework to file."""
        try:
            self.frameworks_path.mkdir(parents=True, exist_ok=True)

            framework_data = {
                "framework_name": framework_name,
                "current_mappings": self.framework_mappings[framework_name],
            }

            framework_file = self.frameworks_path / f"{framework_name}.json"
            with open(framework_file, "w") as f:
                json.dump(framework_data, f, indent=2)

        except Exception as e:
            logger.error("Failed to save framework %s: %s", framework_name, str(e))
            raise

    def _save_all_frameworks(self) -> None:
        """Save all frameworks to files."""
        for framework_name in self.framework_mappings:
            self._save_framework(framework_name)


# Global framework mapping registry instance
framework_mapping_registry = FrameworkMappingRegistry()

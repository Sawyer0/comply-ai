"""
Canonical taxonomy management.

Single responsibility: Manage canonical taxonomy definitions and validation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .base_models import TaxonomyCategory, TaxonomySubcategory, ChangeType
from .version_manager import TaxonomyVersionManager

logger = logging.getLogger(__name__)


class CanonicalTaxonomy:
    """
    Canonical taxonomy manager.

    Single responsibility: Manage taxonomy categories and labels.
    """

    def __init__(self, taxonomy_path: Optional[Path] = None):
        """Initialize canonical taxonomy manager."""
        self.taxonomy_path = taxonomy_path or Path("config/taxonomy")
        self.version_manager = TaxonomyVersionManager()
        self.categories: Dict[str, TaxonomyCategory] = {}
        self.valid_labels: Set[str] = set()

        self._load_taxonomy()

    def _load_taxonomy(self) -> None:
        """Load taxonomy from configuration or create default."""
        taxonomy_file = self.taxonomy_path / "taxonomy.json"

        if taxonomy_file.exists():
            try:
                with open(taxonomy_file, "r") as f:
                    data = json.load(f)
                    self._parse_taxonomy_data(data)
            except Exception as e:
                logger.error("Failed to load taxonomy: %s", str(e))
                self._create_default_taxonomy()
        else:
            self._create_default_taxonomy()

        self._build_valid_labels()

    def _parse_taxonomy_data(self, data: Dict[str, any]) -> None:
        """Parse taxonomy data from JSON."""
        self.categories.clear()

        for category_name, category_data in data.get("categories", {}).items():
            subcategories = {}

            for subcat_name, subcat_data in category_data.get(
                "subcategories", {}
            ).items():
                subcategory = TaxonomySubcategory(
                    name=subcat_name,
                    description=subcat_data.get("description", ""),
                    types=subcat_data.get("types", []),
                    metadata=subcat_data.get("metadata", {}),
                    deprecated=subcat_data.get("deprecated", False),
                    deprecated_since=subcat_data.get("deprecated_since"),
                    replacement=subcat_data.get("replacement"),
                )
                subcategories[subcat_name] = subcategory

            category = TaxonomyCategory(
                name=category_name,
                description=category_data.get("description", ""),
                subcategories=subcategories,
                metadata=category_data.get("metadata", {}),
                deprecated=category_data.get("deprecated", False),
                deprecated_since=category_data.get("deprecated_since"),
                replacement=category_data.get("replacement"),
            )
            self.categories[category_name] = category

    def _create_default_taxonomy(self) -> None:
        """Create default canonical taxonomy."""
        # Core security categories
        self.categories = {
            "PII": TaxonomyCategory(
                name="PII",
                description="Personally Identifiable Information",
                subcategories={
                    "Contact": TaxonomySubcategory(
                        name="Contact",
                        description="Contact information",
                        types=["Email", "Phone", "Address"],
                    ),
                    "Identity": TaxonomySubcategory(
                        name="Identity",
                        description="Identity information",
                        types=["Name", "SSN", "ID"],
                    ),
                    "Financial": TaxonomySubcategory(
                        name="Financial",
                        description="Financial information",
                        types=["CreditCard", "BankAccount"],
                    ),
                },
            ),
            "SECURITY": TaxonomyCategory(
                name="SECURITY",
                description="Security-related content",
                subcategories={
                    "Credentials": TaxonomySubcategory(
                        name="Credentials",
                        description="Authentication credentials",
                        types=["Password", "APIKey", "Token"],
                    ),
                    "Access": TaxonomySubcategory(
                        name="Access",
                        description="Access control violations",
                        types=["Unauthorized", "Privilege"],
                    ),
                },
            ),
            "CONTENT": TaxonomyCategory(
                name="CONTENT",
                description="Content moderation",
                subcategories={
                    "Harmful": TaxonomySubcategory(
                        name="Harmful",
                        description="Harmful content",
                        types=["Toxic", "Hate", "Violence"],
                    )
                },
            ),
        }

        # Create initial version
        checksum = self.version_manager.calculate_checksum(self._serialize_taxonomy())
        self.version_manager.create_version(
            ChangeType.MAJOR,
            ["Initial canonical taxonomy creation"],
            "system",
            checksum,
            "Initial version - no migration needed",
        )

        self._save_taxonomy()
        logger.info("Created default canonical taxonomy")

    def _build_valid_labels(self) -> None:
        """Build set of valid taxonomy labels."""
        self.valid_labels.clear()

        for category_name, category in self.categories.items():
            if category.deprecated:
                continue

            for subcat_name, subcategory in category.subcategories.items():
                if subcategory.deprecated:
                    continue

                # Add subcategory-level labels
                label = f"{category_name}.{subcat_name}"
                self.valid_labels.add(label)

                # Add type-level labels
                for type_name in subcategory.types:
                    label = f"{category_name}.{subcat_name}.{type_name}"
                    self.valid_labels.add(label)

    def _serialize_taxonomy(self) -> Dict[str, any]:
        """Serialize taxonomy to dictionary."""
        return {
            "categories": {
                name: {
                    "description": cat.description,
                    "subcategories": {
                        sub_name: {
                            "description": sub.description,
                            "types": sub.types,
                            "metadata": sub.metadata,
                            "deprecated": sub.deprecated,
                            "deprecated_since": sub.deprecated_since,
                            "replacement": sub.replacement,
                        }
                        for sub_name, sub in cat.subcategories.items()
                    },
                    "metadata": cat.metadata,
                    "deprecated": cat.deprecated,
                    "deprecated_since": cat.deprecated_since,
                    "replacement": cat.replacement,
                }
                for name, cat in self.categories.items()
            }
        }

    def _save_taxonomy(self) -> None:
        """Save taxonomy to file."""
        try:
            self.taxonomy_path.mkdir(parents=True, exist_ok=True)

            taxonomy_file = self.taxonomy_path / "taxonomy.json"
            with open(taxonomy_file, "w") as f:
                json.dump(self._serialize_taxonomy(), f, indent=2)

        except Exception as e:
            logger.error("Failed to save taxonomy: %s", str(e))
            raise

    def is_valid_label(self, label: str) -> bool:
        """Check if a taxonomy label is valid."""
        return label in self.valid_labels

    def get_category(self, label: str) -> Optional[str]:
        """Get category from taxonomy label."""
        parts = label.split(".")
        return parts[0] if parts else None

    def get_subcategory(self, label: str) -> Optional[str]:
        """Get subcategory from taxonomy label."""
        parts = label.split(".")
        return parts[1] if len(parts) >= 2 else None

    def get_type(self, label: str) -> Optional[str]:
        """Get type from taxonomy label."""
        parts = label.split(".")
        return parts[2] if len(parts) >= 3 else None

    def get_labels_by_category(self, category: str) -> List[str]:
        """Get all labels for a specific category."""
        return [
            label for label in self.valid_labels if label.startswith(f"{category}.")
        ]

    def add_category(self, name: str, description: str) -> None:
        """Add a new category."""
        if name in self.categories:
            raise ValueError(f"Category '{name}' already exists")

        self.categories[name] = TaxonomyCategory(name=name, description=description)
        self._build_valid_labels()
        logger.info("Added category: %s", name)

    def add_subcategory(
        self,
        category_name: str,
        subcategory_name: str,
        description: str,
        types: Optional[List[str]] = None,
    ) -> None:
        """Add a new subcategory."""
        if category_name not in self.categories:
            raise ValueError(f"Category '{category_name}' does not exist")

        category = self.categories[category_name]
        if subcategory_name in category.subcategories:
            raise ValueError(f"Subcategory '{subcategory_name}' already exists")

        category.subcategories[subcategory_name] = TaxonomySubcategory(
            name=subcategory_name, description=description, types=types or []
        )
        self._build_valid_labels()
        logger.info("Added subcategory: %s.%s", category_name, subcategory_name)

    def create_new_version(
        self, change_type: ChangeType, changes: List[str], created_by: str
    ) -> str:
        """Create a new version of the taxonomy."""
        checksum = self.version_manager.calculate_checksum(self._serialize_taxonomy())
        version = self.version_manager.create_version(
            change_type, changes, created_by, checksum
        )
        self._save_taxonomy()
        return version

    def get_taxonomy_stats(self) -> Dict[str, any]:
        """Get taxonomy statistics."""
        active_categories = sum(
            1 for cat in self.categories.values() if not cat.deprecated
        )
        active_subcategories = 0
        total_types = 0

        for category in self.categories.values():
            if category.deprecated:
                continue
            for subcategory in category.subcategories.values():
                if not subcategory.deprecated:
                    active_subcategories += 1
                    total_types += len(subcategory.types)

        return {
            "version": self.version_manager.current_version,
            "total_labels": len(self.valid_labels),
            "active_categories": active_categories,
            "active_subcategories": active_subcategories,
            "total_types": total_types,
        }

    def reload_taxonomy(self) -> None:
        """Reload taxonomy from configuration files."""
        logger.info("Reloading canonical taxonomy")
        self._load_taxonomy()


# Global canonical taxonomy instance
canonical_taxonomy = CanonicalTaxonomy()

"""
Taxonomy management for canonical labels.

Single responsibility: Manage canonical taxonomy definitions and validation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)


class TaxonomyManager:
    """
    Manages canonical taxonomy definitions.

    Single responsibility: Taxonomy operations only.
    """

    def __init__(self, taxonomy_path: str = "config/taxonomy.yaml"):
        """
        Initialize taxonomy manager.

        Args:
            taxonomy_path: Path to taxonomy configuration file
        """
        self.taxonomy_path = Path(taxonomy_path)
        self.taxonomy: Dict[str, Dict] = {}
        self.valid_labels: Set[str] = set()
        self.load_taxonomy()

    def load_taxonomy(self) -> None:
        """Load taxonomy from configuration file."""
        try:
            if not self.taxonomy_path.exists():
                logger.warning(f"Taxonomy file not found: {self.taxonomy_path}")
                self._load_default_taxonomy()
                return

            with open(self.taxonomy_path, "r") as f:
                config = yaml.safe_load(f)

            self.taxonomy = config.get("taxonomy", {})
            self._build_valid_labels()

            logger.info(
                "Loaded taxonomy with %d categories and %d total labels",
                len(self.taxonomy),
                len(self.valid_labels),
            )

        except Exception as e:
            logger.error("Failed to load taxonomy: %s", str(e))
            self._load_default_taxonomy()

    def _load_default_taxonomy(self) -> None:
        """Load default taxonomy if file is not available."""
        self.taxonomy = {
            "PII": {
                "description": "Personally Identifiable Information",
                "subcategories": {
                    "Contact": {
                        "description": "Contact information",
                        "types": ["Email", "Phone", "Address"],
                    },
                    "Identity": {
                        "description": "Identity information",
                        "types": ["Name", "SSN", "ID"],
                    },
                    "Financial": {
                        "description": "Financial information",
                        "types": ["CreditCard", "BankAccount"],
                    },
                },
            },
            "SECURITY": {
                "description": "Security-related content",
                "subcategories": {
                    "Credentials": {
                        "description": "Authentication credentials",
                        "types": ["Password", "APIKey", "Token"],
                    },
                    "Access": {
                        "description": "Access control violations",
                        "types": ["Unauthorized", "Privilege"],
                    },
                },
            },
            "CONTENT": {
                "description": "Content moderation",
                "subcategories": {
                    "Harmful": {
                        "description": "Harmful content",
                        "types": ["Toxic", "Hate", "Violence"],
                    }
                },
            },
            "OTHER": {
                "description": "Other categories",
                "subcategories": {
                    "Unknown": {
                        "description": "Unknown or unclassified",
                        "types": ["Unknown"],
                    }
                },
            },
        }
        self._build_valid_labels()
        logger.info("Loaded default taxonomy")

    def _build_valid_labels(self) -> None:
        """Build set of valid taxonomy labels."""
        self.valid_labels.clear()

        for category, category_data in self.taxonomy.items():
            subcategories = category_data.get("subcategories", {})

            for subcategory, subcategory_data in subcategories.items():
                types = subcategory_data.get("types", [])

                for type_name in types:
                    label = f"{category}.{subcategory}.{type_name}"
                    self.valid_labels.add(label)

                # Also add subcategory-level labels
                label = f"{category}.{subcategory}"
                self.valid_labels.add(label)

    def is_valid_label(self, label: str) -> bool:
        """
        Check if a taxonomy label is valid.

        Args:
            label: Taxonomy label to validate

        Returns:
            bool: True if label is valid
        """
        return label in self.valid_labels

    def get_category(self, label: str) -> Optional[str]:
        """
        Get category from taxonomy label.

        Args:
            label: Taxonomy label

        Returns:
            Optional[str]: Category name or None
        """
        parts = label.split(".")
        if len(parts) >= 1:
            return parts[0]
        return None

    def get_subcategory(self, label: str) -> Optional[str]:
        """
        Get subcategory from taxonomy label.

        Args:
            label: Taxonomy label

        Returns:
            Optional[str]: Subcategory name or None
        """
        parts = label.split(".")
        if len(parts) >= 2:
            return parts[1]
        return None

    def get_type(self, label: str) -> Optional[str]:
        """
        Get type from taxonomy label.

        Args:
            label: Taxonomy label

        Returns:
            Optional[str]: Type name or None
        """
        parts = label.split(".")
        if len(parts) >= 3:
            return parts[2]
        return None

    def get_labels_by_category(self, category: str) -> List[str]:
        """
        Get all labels for a specific category.

        Args:
            category: Category name

        Returns:
            List[str]: List of labels in the category
        """
        return [
            label for label in self.valid_labels if label.startswith(f"{category}.")
        ]

    def get_similar_labels(self, label: str, max_results: int = 5) -> List[str]:
        """
        Get similar taxonomy labels.

        Args:
            label: Input label
            max_results: Maximum number of results

        Returns:
            List[str]: List of similar labels
        """
        # Simple similarity based on shared category/subcategory
        parts = label.split(".")
        if len(parts) < 2:
            return []

        category = parts[0]
        subcategory = parts[1] if len(parts) > 1 else ""

        similar = []

        # First, find labels in same subcategory
        if subcategory:
            subcategory_labels = [
                l
                for l in self.valid_labels
                if l.startswith(f"{category}.{subcategory}.")
            ]
            similar.extend(subcategory_labels[:max_results])

        # Then, find labels in same category
        if len(similar) < max_results:
            category_labels = [
                l
                for l in self.valid_labels
                if l.startswith(f"{category}.") and l not in similar
            ]
            similar.extend(category_labels[: max_results - len(similar)])

        return similar[:max_results]

    def get_taxonomy_stats(self) -> Dict[str, int]:
        """
        Get statistics about the taxonomy.

        Returns:
            Dict[str, int]: Taxonomy statistics
        """
        categories = set()
        subcategories = set()
        types = set()

        for label in self.valid_labels:
            parts = label.split(".")
            if len(parts) >= 1:
                categories.add(parts[0])
            if len(parts) >= 2:
                subcategories.add(f"{parts[0]}.{parts[1]}")
            if len(parts) >= 3:
                types.add(label)

        return {
            "total_labels": len(self.valid_labels),
            "categories": len(categories),
            "subcategories": len(subcategories),
            "types": len(types),
        }

    def reload_taxonomy(self) -> None:
        """Reload taxonomy from file."""
        logger.info("Reloading taxonomy")
        self.load_taxonomy()

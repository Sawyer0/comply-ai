"""
Taxonomy loader and management for the Llama Mapper system.

Handles loading, validation, and management of the canonical taxonomy
structure with version tracking and label validation.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import yaml
from pydantic import BaseModel, Field, field_validator


logger = logging.getLogger(__name__)


@dataclass
class TaxonomyLabel:
    """Represents a single taxonomy label with metadata."""
    
    name: str
    description: str
    aliases: List[str] = field(default_factory=list)
    category: str = ""
    subcategory: Optional[str] = None
    
    def __post_init__(self):
        """Extract category and subcategory from name."""
        parts = self.name.split('.')
        if len(parts) >= 2:
            self.category = parts[0]
            if len(parts) >= 3:
                self.subcategory = parts[1]
    
    def matches_alias(self, alias: str) -> bool:
        """Check if the given alias matches this label."""
        return alias.lower() in [a.lower() for a in self.aliases]
    
    def get_full_path(self) -> str:
        """Get the full hierarchical path of this label."""
        return self.name


@dataclass
class TaxonomyCategory:
    """Represents a taxonomy category with subcategories and labels."""
    
    name: str
    description: str
    subcategories: Dict[str, 'TaxonomyCategory'] = field(default_factory=dict)
    labels: List[TaxonomyLabel] = field(default_factory=list)
    
    def get_all_labels(self) -> List[TaxonomyLabel]:
        """Get all labels in this category and its subcategories."""
        all_labels = self.labels.copy()
        for subcategory in self.subcategories.values():
            all_labels.extend(subcategory.get_all_labels())
        return all_labels
    
    def find_label_by_name(self, name: str) -> Optional[TaxonomyLabel]:
        """Find a label by its exact name."""
        # Check direct labels
        for label in self.labels:
            if label.name == name:
                return label
        
        # Check subcategories
        for subcategory in self.subcategories.values():
            label = subcategory.find_label_by_name(name)
            if label:
                return label
        
        return None
    
    def find_labels_by_alias(self, alias: str) -> List[TaxonomyLabel]:
        """Find labels that match the given alias."""
        matching_labels = []
        
        # Check direct labels
        for label in self.labels:
            if label.matches_alias(alias):
                matching_labels.append(label)
        
        # Check subcategories
        for subcategory in self.subcategories.values():
            matching_labels.extend(subcategory.find_labels_by_alias(alias))
        
        return matching_labels


class TaxonomyConfig(BaseModel):
    """Pydantic model for taxonomy configuration validation."""
    
    version: str = Field(..., description="Taxonomy version")
    namespace: str = Field(..., description="Taxonomy namespace")
    last_updated: str = Field(..., description="Last update date")
    notes: Optional[str] = Field(None, description="Additional notes")
    categories: Dict[str, dict] = Field(..., description="Category definitions")
    
    @field_validator('version')
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate version format (YYYY.MM)."""
        if not v or len(v.split('.')) != 2:
            raise ValueError("Version must be in format YYYY.MM")
        try:
            year, month = v.split('.')
            int(year)
            int(month)
        except ValueError:
            raise ValueError("Version must contain valid year and month")
        return v


class Taxonomy:
    """
    Main taxonomy class that holds the complete taxonomy structure.
    
    Provides methods for label lookup, validation, and category management.
    """
    
    def __init__(self, version: str, namespace: str, last_updated: str, notes: Optional[str] = None):
        """Initialize taxonomy with metadata."""
        self.version = version
        self.namespace = namespace
        self.last_updated = last_updated
        self.notes = notes
        self.categories: Dict[str, TaxonomyCategory] = {}
        self._label_index: Dict[str, TaxonomyLabel] = {}
        self._alias_index: Dict[str, List[TaxonomyLabel]] = {}
    
    def add_category(self, category: TaxonomyCategory) -> None:
        """Add a category to the taxonomy."""
        self.categories[category.name] = category
        self._rebuild_indexes()
    
    def get_label_by_name(self, name: str) -> Optional[TaxonomyLabel]:
        """Get a label by its exact name."""
        return self._label_index.get(name)
    
    def get_labels_by_category(self, category_name: str) -> List[TaxonomyLabel]:
        """Get all labels in a specific category."""
        category = self.categories.get(category_name)
        if not category:
            return []
        return category.get_all_labels()
    
    def get_labels_by_alias(self, alias: str) -> List[TaxonomyLabel]:
        """Get labels that match the given alias."""
        return self._alias_index.get(alias.lower(), [])
    
    def get_all_labels(self) -> List[TaxonomyLabel]:
        """Get all labels in the taxonomy."""
        return list(self._label_index.values())
    
    def get_all_label_names(self) -> Set[str]:
        """Get all label names in the taxonomy."""
        return set(self._label_index.keys())
    
    def validate_label_name(self, name: str) -> bool:
        """Check if a label name exists in the taxonomy."""
        return name in self._label_index
    
    def get_category_names(self) -> List[str]:
        """Get all category names."""
        return list(self.categories.keys())
    
    def get_statistics(self) -> Dict[str, int]:
        """Get taxonomy statistics."""
        total_labels = len(self._label_index)
        total_aliases = sum(len(label.aliases) for label in self._label_index.values())
        
        category_counts = {}
        for category_name, category in self.categories.items():
            category_counts[category_name] = len(category.get_all_labels())
        
        return {
            "total_labels": total_labels,
            "total_aliases": total_aliases,
            "total_categories": len(self.categories),
            "category_counts": category_counts
        }
    
    def _rebuild_indexes(self) -> None:
        """Rebuild internal indexes for fast lookup."""
        self._label_index.clear()
        self._alias_index.clear()
        
        for category in self.categories.values():
            for label in category.get_all_labels():
                # Index by name
                self._label_index[label.name] = label
                
                # Index by aliases
                for alias in label.aliases:
                    alias_lower = alias.lower()
                    if alias_lower not in self._alias_index:
                        self._alias_index[alias_lower] = []
                    self._alias_index[alias_lower].append(label)
    
    def __repr__(self) -> str:
        """String representation of taxonomy."""
        stats = self.get_statistics()
        return (f"Taxonomy(version={self.version}, namespace={self.namespace}, "
                f"labels={stats['total_labels']}, categories={stats['total_categories']})")


class TaxonomyLoader:
    """
    Loads and validates taxonomy structure from YAML files.
    
    Supports version tracking, label validation, and category grouping
    as specified in requirements 7.1 and 7.4.
    """
    
    def __init__(self, taxonomy_path: Optional[Union[str, Path]] = None):
        """
        Initialize TaxonomyLoader.
        
        Args:
            taxonomy_path: Path to taxonomy YAML file. Defaults to pillars-detectors/taxonomy.yaml
        """
        self.taxonomy_path = Path(taxonomy_path) if taxonomy_path else Path("pillars-detectors/taxonomy.yaml")
        self._taxonomy: Optional[Taxonomy] = None
        self._last_modified: Optional[float] = None
    
    def load_taxonomy(self, force_reload: bool = False) -> Taxonomy:
        """
        Load taxonomy from YAML file.
        
        Args:
            force_reload: Force reload even if file hasn't changed
            
        Returns:
            Loaded Taxonomy object
            
        Raises:
            FileNotFoundError: If taxonomy file doesn't exist
            ValueError: If taxonomy file is invalid
        """
        if not self.taxonomy_path.exists():
            raise FileNotFoundError(f"Taxonomy file not found: {self.taxonomy_path}")
        
        # Check if reload is needed
        current_modified = self.taxonomy_path.stat().st_mtime
        if not force_reload and self._taxonomy and self._last_modified == current_modified:
            logger.debug(f"Taxonomy already loaded and up-to-date: {self.taxonomy_path}")
            return self._taxonomy
        
        logger.info(f"Loading taxonomy from: {self.taxonomy_path}")
        
        try:
            with open(self.taxonomy_path, 'r', encoding='utf-8') as f:
                raw_data = yaml.safe_load(f)
            
            if not raw_data:
                raise ValueError("Taxonomy file is empty")
            
            # Validate configuration structure
            config = TaxonomyConfig(**raw_data)
            
            # Create taxonomy object
            taxonomy = Taxonomy(
                version=config.version,
                namespace=config.namespace,
                last_updated=config.last_updated,
                notes=config.notes
            )
            
            # Parse categories and labels
            self._parse_categories(taxonomy, config.categories)
            
            # Validate taxonomy structure
            self._validate_taxonomy(taxonomy)
            
            self._taxonomy = taxonomy
            self._last_modified = current_modified
            
            logger.info(f"Successfully loaded taxonomy: {taxonomy}")
            return taxonomy
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in taxonomy file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load taxonomy: {e}")
    
    def get_taxonomy(self) -> Optional[Taxonomy]:
        """Get the currently loaded taxonomy without reloading."""
        return self._taxonomy
    
    def reload_taxonomy(self) -> Taxonomy:
        """Force reload the taxonomy from file."""
        return self.load_taxonomy(force_reload=True)
    
    def validate_taxonomy_file(self) -> bool:
        """
        Validate taxonomy file without loading it into memory.
        
        Returns:
            True if valid, False otherwise
        """
        try:
            self.load_taxonomy(force_reload=True)
            return True
        except Exception as e:
            logger.error(f"Taxonomy validation failed: {e}")
            return False
    
    def _parse_categories(self, taxonomy: Taxonomy, categories_data: Dict[str, dict]) -> None:
        """Parse categories from YAML data."""
        for category_name, category_data in categories_data.items():
            category = self._parse_category(category_name, category_data)
            taxonomy.add_category(category)
    
    def _parse_category(self, name: str, data: dict) -> TaxonomyCategory:
        """Parse a single category from YAML data."""
        category = TaxonomyCategory(
            name=name,
            description=data.get('description', '')
        )
        
        # Parse subcategories
        subcategories = data.get('subcategories', {})
        for subcat_name, subcat_data in subcategories.items():
            subcategory = self._parse_category(subcat_name, subcat_data)
            category.subcategories[subcat_name] = subcategory
        
        # Parse direct labels
        labels = data.get('labels', [])
        for label_data in labels:
            label = self._parse_label(label_data, name)
            category.labels.append(label)
        
        return category
    
    def _parse_label(self, data: dict, category: str) -> TaxonomyLabel:
        """Parse a single label from YAML data."""
        return TaxonomyLabel(
            name=data['name'],
            description=data.get('description', ''),
            aliases=data.get('aliases', []),
            category=category
        )
    
    def _validate_taxonomy(self, taxonomy: Taxonomy) -> None:
        """Validate the loaded taxonomy structure."""
        # Check for duplicate label names
        label_names = [label.name for label in taxonomy.get_all_labels()]
        duplicates = set([name for name in label_names if label_names.count(name) > 1])
        if duplicates:
            raise ValueError(f"Duplicate label names found: {duplicates}")
        
        # Check for empty categories
        for category_name, category in taxonomy.categories.items():
            if not category.get_all_labels():
                logger.warning(f"Category '{category_name}' has no labels")
        
        # Validate label name format
        for label in taxonomy.get_all_labels():
            if not self._is_valid_label_name(label.name):
                raise ValueError(f"Invalid label name format: {label.name}")
        
        # Check for conflicting aliases
        alias_conflicts = self._find_alias_conflicts(taxonomy)
        if alias_conflicts:
            logger.warning(f"Alias conflicts found: {alias_conflicts}")
    
    def _is_valid_label_name(self, name: str) -> bool:
        """Check if label name follows the expected format."""
        # Label names should follow pattern: CATEGORY.SUBCATEGORY.Label or CATEGORY.Label
        parts = name.split('.')
        if len(parts) < 2:
            return False
        
        # All parts should be non-empty and contain only alphanumeric characters and underscores
        for part in parts:
            if not part or not part.replace('_', '').replace('-', '').isalnum():
                return False
        
        return True
    
    def _find_alias_conflicts(self, taxonomy: Taxonomy) -> Dict[str, List[str]]:
        """Find aliases that map to multiple labels."""
        conflicts = {}
        
        for alias, labels in taxonomy._alias_index.items():
            if len(labels) > 1:
                conflicts[alias] = [label.name for label in labels]
        
        return conflicts
    
    def get_version_info(self) -> Optional[Dict[str, str]]:
        """Get version information from the loaded taxonomy."""
        if not self._taxonomy:
            return None
        
        return {
            "version": self._taxonomy.version,
            "namespace": self._taxonomy.namespace,
            "last_updated": self._taxonomy.last_updated,
            "file_path": str(self.taxonomy_path)
        }
    
    def __repr__(self) -> str:
        """String representation of TaxonomyLoader."""
        return f"TaxonomyLoader(path={self.taxonomy_path}, loaded={self._taxonomy is not None})"
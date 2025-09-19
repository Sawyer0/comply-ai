"""
Detector configuration loader and management for the Llama Mapper system.

Handles loading, validation, and management of detector mapping configurations
with version support and taxonomy validation.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import yaml
from pydantic import BaseModel, Field, field_validator

from .taxonomy import Taxonomy, TaxonomyLoader


logger = logging.getLogger(__name__)


class DetectorConfig(BaseModel):
    """Pydantic model for detector configuration validation."""
    
    detector: str = Field(..., description="Detector name/identifier")
    version: str = Field(..., description="Detector configuration version")
    notes: Optional[str] = Field(None, description="Additional notes about the detector")
    maps: Dict[str, str] = Field(..., description="Mapping from detector labels to taxonomy labels")
    
    @field_validator('detector')
    @classmethod
    def validate_detector_name(cls, v: str) -> str:
        """Validate detector name format."""
        if not v or not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError("Detector name must contain only alphanumeric characters, hyphens, and underscores")
        return v
    
    @field_validator('version')
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate version format."""
        if not v or not v.startswith('v'):
            raise ValueError("Version must start with 'v' (e.g., 'v1', 'v2.1')")
        return v
    
    @field_validator('maps')
    @classmethod
    def validate_maps_not_empty(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate that maps is not empty."""
        if not v:
            raise ValueError("Detector must have at least one mapping")
        return v


@dataclass
class DetectorMapping:
    """
    Represents a detector mapping configuration with validation capabilities.
    
    Provides methods for mapping detector outputs to canonical taxonomy labels
    and validating mappings against the taxonomy.
    """
    
    detector: str
    version: str
    notes: Optional[str] = None
    maps: Dict[str, str] = field(default_factory=dict)
    file_path: Optional[Path] = None
    last_modified: Optional[float] = None
    
    def get_canonical_label(self, detector_label: str) -> Optional[str]:
        """
        Get the canonical taxonomy label for a detector label.
        
        Args:
            detector_label: The original detector label
            
        Returns:
            Canonical taxonomy label or None if not found
        """
        return self.maps.get(detector_label)
    
    def get_all_detector_labels(self) -> Set[str]:
        """Get all detector labels that have mappings."""
        return set(self.maps.keys())
    
    def get_all_canonical_labels(self) -> Set[str]:
        """Get all canonical labels used in mappings."""
        return set(self.maps.values())
    
    def validate_against_taxonomy(self, taxonomy: Taxonomy) -> Dict[str, List[str]]:
        """
        Validate all mappings against the provided taxonomy.
        
        Args:
            taxonomy: Taxonomy to validate against
            
        Returns:
            Dictionary with validation results:
            - 'valid': List of valid mappings
            - 'invalid': List of invalid canonical labels
            - 'missing': List of canonical labels not found in taxonomy
        """
        valid_mappings = []
        invalid_labels = []
        missing_labels = []
        
        for detector_label, canonical_label in self.maps.items():
            if taxonomy.validate_label_name(canonical_label):
                valid_mappings.append(f"{detector_label} -> {canonical_label}")
            else:
                invalid_labels.append(canonical_label)
                missing_labels.append(f"{detector_label} -> {canonical_label}")
        
        return {
            'valid': valid_mappings,
            'invalid': invalid_labels,
            'missing': missing_labels
        }
    
    def get_mapping_statistics(self) -> Dict[str, int]:
        """Get statistics about this detector mapping."""
        canonical_labels = self.get_all_canonical_labels()
        
        # Count by category
        category_counts = {}
        for label in canonical_labels:
            category = label.split('.')[0] if '.' in label else 'OTHER'
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            'total_mappings': len(self.maps),
            'unique_canonical_labels': len(canonical_labels),
            'category_counts': category_counts
        }
    
    def has_mapping_for(self, detector_label: str) -> bool:
        """Check if a mapping exists for the given detector label."""
        return detector_label in self.maps
    
    def add_mapping(self, detector_label: str, canonical_label: str) -> None:
        """Add a new mapping."""
        self.maps[detector_label] = canonical_label
    
    def remove_mapping(self, detector_label: str) -> bool:
        """Remove a mapping. Returns True if mapping existed."""
        return self.maps.pop(detector_label, None) is not None
    
    def __repr__(self) -> str:
        """String representation of DetectorMapping."""
        return (f"DetectorMapping(detector={self.detector}, version={self.version}, "
                f"mappings={len(self.maps)})")


class DetectorConfigLoader:
    """
    Loads and validates detector mapping configurations from YAML files.
    
    Supports loading individual detector configs or all configs from a directory,
    with version support and taxonomy validation as specified in requirements 5.1, 5.3, and 7.2.
    """
    
    def __init__(self, 
                 detectors_path: Optional[Union[str, Path]] = None,
                 taxonomy_loader: Optional[TaxonomyLoader] = None):
        """
        Initialize DetectorConfigLoader.
        
        Args:
            detectors_path: Path to directory containing detector YAML files
            taxonomy_loader: TaxonomyLoader instance for validation
        """
        self.detectors_path = Path(detectors_path) if detectors_path else Path("pillars-detectors")
        self.taxonomy_loader = taxonomy_loader or TaxonomyLoader()
        self._detector_mappings: Dict[str, DetectorMapping] = {}
        self._last_scan_time: Optional[float] = None
    
    def load_detector_config(self, config_path: Union[str, Path]) -> DetectorMapping:
        """
        Load a single detector configuration from a YAML file.
        
        Args:
            config_path: Path to the detector YAML file
            
        Returns:
            DetectorMapping object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Detector config file not found: {config_path}")
        
        logger.debug(f"Loading detector config from: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw_data = yaml.safe_load(f)
            
            if not raw_data:
                raise ValueError("Detector config file is empty")
            
            # Validate configuration structure
            config = DetectorConfig(**raw_data)
            
            # Create detector mapping
            mapping = DetectorMapping(
                detector=config.detector,
                version=config.version,
                notes=config.notes,
                maps=config.maps,
                file_path=config_path,
                last_modified=config_path.stat().st_mtime
            )
            
            logger.debug(f"Successfully loaded detector config: {mapping}")
            return mapping
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in detector config file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load detector config from {config_path}: {e}")
    
    def load_all_detector_configs(self, force_reload: bool = False) -> Dict[str, DetectorMapping]:
        """
        Load all detector configurations from the detectors directory.
        
        Args:
            force_reload: Force reload even if files haven't changed
            
        Returns:
            Dictionary mapping detector names to DetectorMapping objects
            
        Raises:
            FileNotFoundError: If detectors directory doesn't exist
        """
        if not self.detectors_path.exists():
            raise FileNotFoundError(f"Detectors directory not found: {self.detectors_path}")
        
        # Check if reload is needed
        current_scan_time = max(
            (f.stat().st_mtime for f in self.detectors_path.glob("*.yaml") if f.is_file()),
            default=0
        )
        
        if not force_reload and self._detector_mappings and self._last_scan_time == current_scan_time:
            logger.debug("Detector configs already loaded and up-to-date")
            return self._detector_mappings
        
        logger.info(f"Loading detector configs from: {self.detectors_path}")
        
        detector_mappings = {}
        yaml_files = list(self.detectors_path.glob("*.yaml"))
        
        # Filter out non-detector files
        excluded_files = {"taxonomy.yaml", "frameworks.yaml", "schema.json"}
        yaml_files = [f for f in yaml_files if f.name not in excluded_files]
        
        if not yaml_files:
            logger.warning(f"No detector YAML files found in {self.detectors_path}")
            return detector_mappings
        
        for yaml_file in yaml_files:
            try:
                mapping = self.load_detector_config(yaml_file)
                detector_mappings[mapping.detector] = mapping
                logger.debug(f"Loaded detector: {mapping.detector}")
            except Exception as e:
                logger.error(f"Failed to load detector config {yaml_file}: {e}")
                # Continue loading other configs
        
        self._detector_mappings = detector_mappings
        self._last_scan_time = current_scan_time
        
        logger.info(f"Successfully loaded {len(detector_mappings)} detector configs")
        return detector_mappings
    
    def get_detector_mapping(self, detector_name: str) -> Optional[DetectorMapping]:
        """Get a specific detector mapping by name."""
        return self._detector_mappings.get(detector_name)
    
    def get_all_detector_mappings(self) -> Dict[str, DetectorMapping]:
        """Get all loaded detector mappings."""
        return self._detector_mappings.copy()
    
    def validate_all_mappings(self, taxonomy: Optional[Taxonomy] = None) -> Dict[str, Dict[str, List[str]]]:
        """
        Validate all detector mappings against the taxonomy.
        
        Args:
            taxonomy: Taxonomy to validate against. If None, loads from taxonomy_loader
            
        Returns:
            Dictionary mapping detector names to validation results
        """
        if taxonomy is None:
            taxonomy = self.taxonomy_loader.load_taxonomy()
        
        validation_results = {}
        
        for detector_name, mapping in self._detector_mappings.items():
            validation_results[detector_name] = mapping.validate_against_taxonomy(taxonomy)
        
        return validation_results
    
    def get_detector_by_canonical_label(self, canonical_label: str) -> List[str]:
        """
        Find all detectors that map to a specific canonical label.
        
        Args:
            canonical_label: The canonical taxonomy label
            
        Returns:
            List of detector names that map to this label
        """
        matching_detectors = []
        
        for detector_name, mapping in self._detector_mappings.items():
            if canonical_label in mapping.get_all_canonical_labels():
                matching_detectors.append(detector_name)
        
        return matching_detectors
    
    def get_mapping_coverage_report(self, taxonomy: Optional[Taxonomy] = None) -> Dict[str, any]:
        """
        Generate a coverage report showing which taxonomy labels are covered by detectors.
        
        Args:
            taxonomy: Taxonomy to check coverage against
            
        Returns:
            Coverage report with statistics and details
        """
        if taxonomy is None:
            taxonomy = self.taxonomy_loader.load_taxonomy()
        
        all_taxonomy_labels = taxonomy.get_all_label_names()
        covered_labels = set()
        
        # Collect all canonical labels used by detectors
        for mapping in self._detector_mappings.values():
            covered_labels.update(mapping.get_all_canonical_labels())
        
        uncovered_labels = all_taxonomy_labels - covered_labels
        
        # Count mappings per category
        category_coverage = {}
        for label in covered_labels:
            category = label.split('.')[0] if '.' in label else 'OTHER'
            if category not in category_coverage:
                category_coverage[category] = {'covered': 0, 'total': 0}
            category_coverage[category]['covered'] += 1
        
        for label in all_taxonomy_labels:
            category = label.split('.')[0] if '.' in label else 'OTHER'
            if category not in category_coverage:
                category_coverage[category] = {'covered': 0, 'total': 0}
            category_coverage[category]['total'] += 1
        
        # Calculate coverage percentages
        for category in category_coverage:
            total = category_coverage[category]['total']
            covered = category_coverage[category]['covered']
            category_coverage[category]['percentage'] = (covered / total * 100) if total > 0 else 0
        
        return {
            'total_taxonomy_labels': len(all_taxonomy_labels),
            'covered_labels': len(covered_labels),
            'uncovered_labels': len(uncovered_labels),
            'coverage_percentage': (len(covered_labels) / len(all_taxonomy_labels) * 100) if all_taxonomy_labels else 0,
            'category_coverage': category_coverage,
            'uncovered_label_list': sorted(uncovered_labels),
            'detector_count': len(self._detector_mappings)
        }
    
    def find_mapping_conflicts(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Find conflicts where different detectors map the same label to different canonical labels.
        
        Returns:
            Dictionary mapping detector labels to conflicting mappings
        """
        label_mappings = {}  # detector_label -> [(detector_name, canonical_label)]
        
        # Collect all mappings
        for detector_name, mapping in self._detector_mappings.items():
            for detector_label, canonical_label in mapping.maps.items():
                if detector_label not in label_mappings:
                    label_mappings[detector_label] = []
                label_mappings[detector_label].append({
                    'detector': detector_name,
                    'canonical_label': canonical_label
                })
        
        # Find conflicts
        conflicts = {}
        for detector_label, mappings in label_mappings.items():
            if len(mappings) > 1:
                # Check if all mappings agree
                canonical_labels = set(m['canonical_label'] for m in mappings)
                if len(canonical_labels) > 1:
                    conflicts[detector_label] = mappings
        
        return conflicts
    
    def get_version_info(self) -> Dict[str, Dict[str, str]]:
        """Get version information for all loaded detectors."""
        version_info = {}
        
        for detector_name, mapping in self._detector_mappings.items():
            version_info[detector_name] = {
                'version': mapping.version,
                'file_path': str(mapping.file_path) if mapping.file_path else '',
                'last_modified': str(mapping.last_modified) if mapping.last_modified else ''
            }
        
        return version_info
    
    def reload_detector_configs(self) -> Dict[str, DetectorMapping]:
        """Force reload all detector configurations."""
        return self.load_all_detector_configs(force_reload=True)
    
    def __repr__(self) -> str:
        """String representation of DetectorConfigLoader."""
        return (f"DetectorConfigLoader(path={self.detectors_path}, "
                f"loaded_detectors={len(self._detector_mappings)})")
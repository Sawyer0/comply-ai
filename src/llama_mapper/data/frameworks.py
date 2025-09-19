"""
Framework mapper for compliance framework mappings.

Handles loading and management of compliance framework mappings (SOC2, ISO27001, HIPAA)
with support for framework expansion and version tracking.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml
from pydantic import BaseModel, Field, field_validator

from .taxonomy import Taxonomy, TaxonomyLoader

logger = logging.getLogger(__name__)


class FrameworkConfig(BaseModel):
    """Pydantic model for framework configuration validation."""

    version: str = Field(..., description="Framework mapping version")
    last_updated: str = Field(..., description="Last update date")
    notes: Optional[str] = Field(None, description="Additional notes")
    frameworks: Dict[str, Dict[str, str]] = Field(
        ..., description="Framework definitions"
    )
    mappings: Dict[str, List[str]] = Field(
        ..., description="Taxonomy to framework control mappings"
    )

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate version format."""
        if not v or not v.startswith("v"):
            raise ValueError("Version must start with 'v' (e.g., 'v1.0', 'v2.1')")
        return v

    @field_validator("frameworks")
    @classmethod
    def validate_frameworks_not_empty(
        cls, v: Dict[str, Dict[str, str]]
    ) -> Dict[str, Dict[str, str]]:
        """Validate that frameworks is not empty."""
        if not v:
            raise ValueError("At least one framework must be defined")
        return v

    @field_validator("mappings")
    @classmethod
    def validate_mappings_format(cls, v: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Validate mapping format."""
        for taxonomy_label, controls in v.items():
            if not isinstance(controls, list):
                raise ValueError(f"Controls for {taxonomy_label} must be a list")
            for control in controls:
                if ":" not in control:
                    raise ValueError(
                        f"Control reference '{control}' must be in format 'Framework:ControlId'"
                    )
        return v


@dataclass
class ComplianceFramework:
    """Represents a compliance framework with its controls."""

    name: str
    controls: Dict[str, str] = field(default_factory=dict)  # control_id -> description

    def has_control(self, control_id: str) -> bool:
        """Check if the framework has a specific control."""
        return control_id in self.controls

    def get_control_description(self, control_id: str) -> Optional[str]:
        """Get the description for a control."""
        return self.controls.get(control_id)

    def get_all_control_ids(self) -> Set[str]:
        """Get all control IDs in this framework."""
        return set(self.controls.keys())

    def add_control(self, control_id: str, description: str) -> None:
        """Add a new control to the framework."""
        self.controls[control_id] = description

    def remove_control(self, control_id: str) -> bool:
        """Remove a control from the framework. Returns True if control existed."""
        return self.controls.pop(control_id, None) is not None

    def __repr__(self) -> str:
        """String representation of ComplianceFramework."""
        return f"ComplianceFramework(name={self.name}, controls={len(self.controls)})"


@dataclass
class FrameworkMapping:
    """
    Represents the complete framework mapping configuration.

    Maps taxonomy labels to compliance framework controls and provides
    methods for framework expansion and version tracking.
    """

    version: str
    last_updated: str
    notes: Optional[str] = None
    frameworks: Dict[str, ComplianceFramework] = field(default_factory=dict)
    mappings: Dict[str, List[str]] = field(
        default_factory=dict
    )  # taxonomy_label -> [framework:control_id]
    file_path: Optional[Path] = None
    last_modified: Optional[float] = None

    def get_framework_controls_for_label(
        self, taxonomy_label: str
    ) -> Dict[str, List[str]]:
        """
        Get all framework controls mapped to a taxonomy label.

        Args:
            taxonomy_label: The canonical taxonomy label

        Returns:
            Dictionary mapping framework names to lists of control IDs
        """
        controls: List[str] = self.mappings.get(taxonomy_label, [])
        framework_controls: Dict[str, List[str]] = {}

        for control_ref in controls:
            if ":" in control_ref:
                framework_name, control_id = control_ref.split(":", 1)
                if framework_name not in framework_controls:
                    framework_controls[framework_name] = []
                framework_controls[framework_name].append(control_id)

        return framework_controls

    def get_labels_for_framework_control(
        self, framework: str, control_id: str
    ) -> List[str]:
        """
        Get all taxonomy labels mapped to a specific framework control.

        Args:
            framework: Framework name (e.g., 'SOC2')
            control_id: Control ID (e.g., 'CC7.2')

        Returns:
            List of taxonomy labels mapped to this control
        """
        control_ref = f"{framework}:{control_id}"
        matching_labels = []

        for taxonomy_label, controls in self.mappings.items():
            if control_ref in controls:
                matching_labels.append(taxonomy_label)

        return matching_labels

    def get_all_frameworks(self) -> List[str]:
        """Get all framework names."""
        return list(self.frameworks.keys())

    def get_framework(self, name: str) -> Optional[ComplianceFramework]:
        """Get a specific framework by name."""
        return self.frameworks.get(name)

    def has_framework(self, name: str) -> bool:
        """Check if a framework exists."""
        return name in self.frameworks

    def add_framework(self, framework: ComplianceFramework) -> None:
        """Add a new framework."""
        self.frameworks[framework.name] = framework

    def remove_framework(self, name: str) -> bool:
        """Remove a framework. Returns True if framework existed."""
        if name in self.frameworks:
            # Also remove all mappings referencing this framework
            self._remove_framework_from_mappings(name)
            del self.frameworks[name]
            return True
        return False

    def add_mapping(self, taxonomy_label: str, framework: str, control_id: str) -> None:
        """Add a new mapping between taxonomy label and framework control."""
        control_ref = f"{framework}:{control_id}"

        if taxonomy_label not in self.mappings:
            self.mappings[taxonomy_label] = []

        if control_ref not in self.mappings[taxonomy_label]:
            self.mappings[taxonomy_label].append(control_ref)

    def remove_mapping(
        self, taxonomy_label: str, framework: str, control_id: str
    ) -> bool:
        """Remove a mapping. Returns True if mapping existed."""
        control_ref = f"{framework}:{control_id}"

        if taxonomy_label in self.mappings:
            try:
                self.mappings[taxonomy_label].remove(control_ref)
                # Clean up empty mapping lists
                if not self.mappings[taxonomy_label]:
                    del self.mappings[taxonomy_label]
                return True
            except ValueError:
                pass

        return False

    def validate_against_taxonomy(self, taxonomy: Taxonomy) -> Dict[str, List[str]]:
        """
        Validate all mappings against the provided taxonomy.

        Args:
            taxonomy: Taxonomy to validate against

        Returns:
            Dictionary with validation results
        """
        valid_labels = []
        invalid_labels = []

        for taxonomy_label in self.mappings.keys():
            if taxonomy.validate_label_name(taxonomy_label):
                valid_labels.append(taxonomy_label)
            else:
                invalid_labels.append(taxonomy_label)

        return {"valid": valid_labels, "invalid": invalid_labels}

    def validate_framework_references(self) -> Dict[str, List[str]]:
        """
        Validate that all framework references in mappings exist.

        Returns:
            Dictionary with validation results
        """
        valid_refs = []
        invalid_refs = []

        for controls in self.mappings.values():
            for control_ref in controls:
                if ":" in control_ref:
                    framework_name, control_id = control_ref.split(":", 1)
                    framework = self.frameworks.get(framework_name)

                    if framework and framework.has_control(control_id):
                        valid_refs.append(control_ref)
                    else:
                        invalid_refs.append(control_ref)
                else:
                    invalid_refs.append(control_ref)

        return {"valid": valid_refs, "invalid": invalid_refs}

    def get_coverage_report(self) -> Dict[str, Any]:
        """Generate a coverage report for framework mappings."""
        total_labels = len(self.mappings)

        # Count mappings per framework
        framework_counts: Dict[str, int] = {}
        for controls in self.mappings.values():
            for control_ref in controls:
                if ":" in control_ref:
                    framework_name = control_ref.split(":", 1)[0]
                    framework_counts[framework_name] = (
                        framework_counts.get(framework_name, 0) + 1
                    )

        # Count unique controls per framework
        framework_control_counts: Dict[str, int] = {}
        for framework_name in self.frameworks.keys():
            unique_controls = set()
            for controls in self.mappings.values():
                for control_ref in controls:
                    if control_ref.startswith(f"{framework_name}:"):
                        unique_controls.add(control_ref)
            framework_control_counts[framework_name] = len(unique_controls)

        return {
            "total_mapped_labels": total_labels,
            "total_frameworks": len(self.frameworks),
            "framework_mapping_counts": framework_counts,
            "framework_unique_controls": framework_control_counts,
            "version": self.version,
            "last_updated": self.last_updated,
        }

    def _remove_framework_from_mappings(self, framework_name: str) -> None:
        """Remove all mappings referencing a specific framework."""
        labels_to_update = []

        for taxonomy_label, controls in self.mappings.items():
            updated_controls = [
                c for c in controls if not c.startswith(f"{framework_name}:")
            ]
            if len(updated_controls) != len(controls):
                labels_to_update.append((taxonomy_label, updated_controls))

        for taxonomy_label, updated_controls in labels_to_update:
            if updated_controls:
                self.mappings[taxonomy_label] = updated_controls
            else:
                del self.mappings[taxonomy_label]

    def __repr__(self) -> str:
        """String representation of FrameworkMapping."""
        return (
            f"FrameworkMapping(version={self.version}, frameworks={len(self.frameworks)}, "
            f"mappings={len(self.mappings)})"
        )


class FrameworkMapper:
    """
    Loads and manages compliance framework mappings.

    Supports loading framework mappings from YAML, framework expansion with approval workflow,
    and version tracking for framework mapping changes as specified in requirements 9.1 and 9.4.
    """

    def __init__(
        self,
        frameworks_path: Optional[Union[str, Path]] = None,
        taxonomy_loader: Optional[TaxonomyLoader] = None,
    ):
        """
        Initialize FrameworkMapper.

        Args:
            frameworks_path: Path to frameworks YAML file
            taxonomy_loader: TaxonomyLoader instance for validation
        """
        self.frameworks_path = (
            Path(frameworks_path)
            if frameworks_path
            else Path("pillars-detectors/frameworks.yaml")
        )
        self.taxonomy_loader = taxonomy_loader or TaxonomyLoader()
        self._framework_mapping: Optional[FrameworkMapping] = None
        self._last_modified: Optional[float] = None

    def load_framework_mapping(self, force_reload: bool = False) -> FrameworkMapping:
        """
        Load framework mapping from YAML file.

        Args:
            force_reload: Force reload even if file hasn't changed

        Returns:
            FrameworkMapping object

        Raises:
            FileNotFoundError: If frameworks file doesn't exist
            ValueError: If frameworks file is invalid
        """
        if not self.frameworks_path.exists():
            raise FileNotFoundError(
                f"Frameworks file not found: {self.frameworks_path}"
            )

        # Check if reload is needed
        current_modified = self.frameworks_path.stat().st_mtime
        if (
            not force_reload
            and self._framework_mapping
            and self._last_modified == current_modified
        ):
            logger.debug(
                f"Framework mapping already loaded and up-to-date: {self.frameworks_path}"
            )
            return self._framework_mapping

        logger.info(f"Loading framework mapping from: {self.frameworks_path}")

        try:
            with open(self.frameworks_path, "r", encoding="utf-8") as f:
                raw_data = yaml.safe_load(f)

            if not raw_data:
                raise ValueError("Frameworks file is empty")

            # Validate configuration structure
            config = FrameworkConfig(**raw_data)

            # Create framework mapping
            mapping = FrameworkMapping(
                version=config.version,
                last_updated=config.last_updated,
                notes=config.notes,
                mappings=config.mappings,
                file_path=self.frameworks_path,
                last_modified=current_modified,
            )

            # Parse frameworks
            for framework_name, controls_data in config.frameworks.items():
                framework = ComplianceFramework(
                    name=framework_name, controls=controls_data
                )
                mapping.add_framework(framework)

            # Validate framework mapping
            self._validate_framework_mapping(mapping)

            self._framework_mapping = mapping
            self._last_modified = current_modified

            logger.info(f"Successfully loaded framework mapping: {mapping}")
            return mapping

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in frameworks file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load framework mapping: {e}")

    def get_framework_mapping(self) -> Optional[FrameworkMapping]:
        """Get the currently loaded framework mapping without reloading."""
        return self._framework_mapping

    def reload_framework_mapping(self) -> FrameworkMapping:
        """Force reload the framework mapping from file."""
        return self.load_framework_mapping(force_reload=True)

    def validate_framework_mapping_file(self) -> bool:
        """
        Validate framework mapping file without loading it into memory.

        Returns:
            True if valid, False otherwise
        """
        try:
            self.load_framework_mapping(force_reload=True)
            return True
        except Exception as e:
            logger.error(f"Framework mapping validation failed: {e}")
            return False

    def get_compliance_controls_for_label(
        self, taxonomy_label: str
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Get compliance controls for a taxonomy label with descriptions.

        Args:
            taxonomy_label: The canonical taxonomy label

        Returns:
            Dictionary mapping framework names to lists of control info
        """
        if not self._framework_mapping:
            self.load_framework_mapping()
        assert self._framework_mapping is not None

        framework_controls = self._framework_mapping.get_framework_controls_for_label(
            taxonomy_label
        )
        result: Dict[str, List[Dict[str, str]]] = {}

        for framework_name, control_ids in framework_controls.items():
            framework = self._framework_mapping.get_framework(framework_name)
            if framework is not None:
                result[framework_name] = []
                for control_id in control_ids:
                    description = framework.get_control_description(control_id)
                    result[framework_name].append(
                        {
                            "control_id": control_id,
                            "description": description or "No description available",
                        }
                    )

        return result

    def generate_compliance_report(
        self, taxonomy_labels: List[str], include_descriptions: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a compliance report for a list of taxonomy labels.

        Args:
            taxonomy_labels: List of taxonomy labels to include in report
            include_descriptions: Whether to include control descriptions

        Returns:
            Compliance report with framework mappings
        """
        if not self._framework_mapping:
            self.load_framework_mapping()
        assert self._framework_mapping is not None

        report: Dict[str, Any] = {
            "version": self._framework_mapping.version,
            "last_updated": self._framework_mapping.last_updated,
            "generated_for_labels": taxonomy_labels,
        }

        # Collect all framework mappings
        framework_mappings: Dict[str, Any] = {}
        all_controls: Dict[str, Set[str]] = {}

        for label in taxonomy_labels:
            controls = self.get_compliance_controls_for_label(label)

            if include_descriptions:
                framework_mappings[label] = controls
            else:
                # Simplified format without descriptions
                simplified: Dict[str, List[str]] = {}
                for framework, control_list in controls.items():
                    simplified[framework] = [c["control_id"] for c in control_list]
                framework_mappings[label] = simplified

            # Track unique controls per framework
            for framework, control_list in controls.items():
                if framework not in all_controls:
                    all_controls[framework] = set()
                for control_info in control_list:
                    all_controls[framework].add(control_info["control_id"])

        # Generate coverage summary
        coverage_summary: Dict[str, Dict[str, Union[int, List[str]]]] = {}
        for framework, control_set in all_controls.items():
            coverage_summary[framework] = {
                "unique_controls": len(control_set),
                "control_list": sorted(list(control_set)),
            }

        report["framework_mappings"] = framework_mappings
        report["coverage_summary"] = coverage_summary
        return report

    def add_framework_expansion(
        self,
        framework_name: str,
        controls: Dict[str, str],
        approval_required: bool = True,
    ) -> bool:
        """
        Add a new framework or expand an existing one.

        Args:
            framework_name: Name of the framework
            controls: Dictionary of control_id -> description
            approval_required: Whether approval workflow is required

        Returns:
            True if expansion was successful
        """
        if not self._framework_mapping:
            self.load_framework_mapping()
        assert self._framework_mapping is not None

        if approval_required:
            logger.info(
                f"Framework expansion for '{framework_name}' requires approval workflow"
            )
            # In a real implementation, this would trigger an approval workflow
            # For now, we'll log the requirement
            return False

        # Add or update framework
        if self._framework_mapping.has_framework(framework_name):
            framework = self._framework_mapping.get_framework(framework_name)
            assert framework is not None
            for control_id, description in controls.items():
                framework.add_control(control_id, description)
            logger.info(
                f"Expanded existing framework '{framework_name}' with {len(controls)} controls"
            )
        else:
            new_framework = ComplianceFramework(framework_name, controls)
            self._framework_mapping.add_framework(new_framework)
            logger.info(
                f"Added new framework '{framework_name}' with {len(controls)} controls"
            )

        return True

    def validate_against_taxonomy(
        self, taxonomy: Optional[Taxonomy] = None
    ) -> Dict[str, List[str]]:
        """
        Validate framework mappings against taxonomy.

        Args:
            taxonomy: Taxonomy to validate against. If None, loads from taxonomy_loader

        Returns:
            Validation results
        """
        if not self._framework_mapping:
            self.load_framework_mapping()
        assert self._framework_mapping is not None

        if taxonomy is None:
            taxonomy = self.taxonomy_loader.load_taxonomy()

        return self._framework_mapping.validate_against_taxonomy(taxonomy)

    def _validate_framework_mapping(self, mapping: FrameworkMapping) -> None:
        """Validate the loaded framework mapping structure."""
        # Validate framework references
        validation_results = mapping.validate_framework_references()

        if validation_results["invalid"]:
            logger.warning(
                f"Invalid framework references found: {validation_results['invalid']}"
            )

        # Check for empty frameworks
        for framework_name, framework in mapping.frameworks.items():
            if not framework.controls:
                logger.warning(f"Framework '{framework_name}' has no controls defined")

        # Check for unmapped frameworks
        used_frameworks = set()
        for controls in mapping.mappings.values():
            for control_ref in controls:
                if ":" in control_ref:
                    framework_name = control_ref.split(":", 1)[0]
                    used_frameworks.add(framework_name)

        defined_frameworks = set(mapping.frameworks.keys())
        unused_frameworks = defined_frameworks - used_frameworks

        if unused_frameworks:
            logger.warning(
                f"Defined frameworks not used in mappings: {unused_frameworks}"
            )

    def get_version_info(self) -> Optional[Dict[str, str]]:
        """Get version information from the loaded framework mapping."""
        if not self._framework_mapping:
            return None

        return {
            "version": self._framework_mapping.version,
            "last_updated": self._framework_mapping.last_updated,
            "file_path": str(self.frameworks_path),
        }

    def __repr__(self) -> str:
        """String representation of FrameworkMapper."""
        return f"FrameworkMapper(path={self.frameworks_path}, loaded={self._framework_mapping is not None})"

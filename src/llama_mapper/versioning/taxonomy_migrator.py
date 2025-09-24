"""
Taxonomy migration utilities for Llama Mapper.

Creates migration plans between two taxonomy versions and provides helpers to
apply those plans to detector mapping dictionaries (in-memory), with validation
and rollback support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..data.taxonomy import Taxonomy, TaxonomyLoader


@dataclass
class MigrationPlan:
    """A plan describing how labels should be migrated from A -> B."""

    from_version: str
    to_version: str
    created_at: str
    label_map: Dict[str, str]  # old_label -> new_label
    unmapped_old_labels: List[str] = field(default_factory=list)
    new_labels_without_source: List[str] = field(default_factory=list)

    def invert(self) -> "MigrationPlan":
        return MigrationPlan(
            from_version=self.to_version,
            to_version=self.from_version,
            created_at=datetime.utcnow().isoformat() + "Z",
            label_map={v: k for k, v in self.label_map.items()},
            unmapped_old_labels=self.new_labels_without_source.copy(),
            new_labels_without_source=self.unmapped_old_labels.copy(),
        )


@dataclass
class MigrationReport:
    """Validation report after applying a migration plan to mappings."""

    total_mappings: int
    remapped: int
    unchanged: int
    unknown_after_migration: int
    details: List[Dict[str, Any]] = field(default_factory=list)


class TaxonomyMigrator:
    """
    Computes migration plans between taxonomy versions and applies them to
    detector mappings. Designed to be deterministic and schema-preserving.
    """

    def __init__(self, old_taxonomy_path: Path, new_taxonomy_path: Path):
        self.old_loader = TaxonomyLoader(old_taxonomy_path)
        self.new_loader = TaxonomyLoader(new_taxonomy_path)
        self.old_taxonomy: Optional[Taxonomy] = None
        self.new_taxonomy: Optional[Taxonomy] = None

    def load(self) -> Tuple[Taxonomy, Taxonomy]:
        self.old_taxonomy = self.old_loader.load_taxonomy()
        self.new_taxonomy = self.new_loader.load_taxonomy()
        return self.old_taxonomy, self.new_taxonomy

    def compute_plan(self) -> MigrationPlan:
        if not self.old_taxonomy or not self.new_taxonomy:
            self.load()
        assert self.old_taxonomy is not None
        assert self.new_taxonomy is not None

        old_labels = {lbl.name: lbl for lbl in self.old_taxonomy.get_all_labels()}
        new_label_names = {lbl.name for lbl in self.new_taxonomy.get_all_labels()}

        label_map: Dict[str, str] = {}
        unmapped: List[str] = []

        # 1) Exact name preservation
        for old_name in old_labels:
            if old_name in new_label_names:
                label_map[old_name] = old_name

        # 2) Alias-based remapping for labels that changed name
        for old_name, old_label in old_labels.items():
            if old_name in label_map:
                continue
            candidates = set()
            # Use old aliases and the leaf token of the old name as candidates
            leaf = old_name.split(".")[-1]
            alias_candidates = {a.lower() for a in old_label.aliases} | {leaf.lower()}
            for alias in alias_candidates:
                assert self.new_taxonomy is not None
                for lbl in self.new_taxonomy.get_labels_by_alias(alias):
                    candidates.add(lbl.name)
            if len(candidates) == 1:
                label_map[old_name] = next(iter(candidates))
            else:
                unmapped.append(old_name)

        new_names_mapped = set(label_map.values())
        new_without_source = sorted(list(new_label_names - new_names_mapped))

        assert self.old_taxonomy is not None
        assert self.new_taxonomy is not None
        return MigrationPlan(
            from_version=self.old_taxonomy.version,
            to_version=self.new_taxonomy.version,
            created_at=datetime.utcnow().isoformat() + "Z",
            label_map=label_map,
            unmapped_old_labels=sorted(unmapped),
            new_labels_without_source=new_without_source,
        )

    def apply_to_mapping(
        self, mapping: Dict[str, str], plan: MigrationPlan
    ) -> Tuple[Dict[str, str], int, int]:
        """
        Apply a migration plan to a single detector mapping dictionary.
        Returns (new_mapping, remapped_count, unknown_count_after).
        """
        new_maps: Dict[str, str] = {}
        remapped = 0
        unknown_after = 0
        for det_label, canonical in mapping.items():
            if canonical in plan.label_map:
                new_maps[det_label] = plan.label_map[canonical]
                if new_maps[det_label] != canonical:
                    remapped += 1
            else:
                # Keep as is; if label no longer exists, route to OTHER.Unknown
                if self.new_taxonomy and not any(
                    label.name == canonical
                    for label in self.new_taxonomy.get_all_labels()
                ):
                    new_maps[det_label] = "OTHER.Unknown"
                    unknown_after += 1
                else:
                    new_maps[det_label] = canonical
        return new_maps, remapped, unknown_after

    def apply_to_detector_mappings(
        self, detector_mappings: Dict[str, Dict[str, str]], plan: MigrationPlan
    ) -> MigrationReport:
        total = 0
        remapped_total = 0
        unknown_total = 0
        details: List[Dict[str, Any]] = []
        for detector, maps in detector_mappings.items():
            total += len(maps)
            _new_map, remapped, unknown = self.apply_to_mapping(maps, plan)
            remapped_total += remapped
            unknown_total += unknown
            details.append(
                {
                    "detector": detector,
                    "mappings": len(maps),
                    "remapped": remapped,
                    "unknown_after": unknown,
                }
            )
        return MigrationReport(
            total_mappings=total,
            remapped=remapped_total,
            unchanged=total - remapped_total,
            unknown_after_migration=unknown_total,
            details=details,
        )

    def validate_plan_completeness(self, plan: MigrationPlan) -> Dict[str, Any]:
        """Return basic completeness metrics for the plan."""
        if not self.old_taxonomy or not self.new_taxonomy:
            self.load()
        assert self.old_taxonomy is not None
        assert self.new_taxonomy is not None
        old_count = len(self.old_taxonomy.get_all_labels())
        coverage = len(plan.label_map)
        return {
            "from_version": plan.from_version,
            "to_version": plan.to_version,
            "coverage_labels": coverage,
            "total_old_labels": old_count,
            "coverage_pct": round(coverage / old_count * 100, 2) if old_count else 0.0,
            "unmapped_old_labels": plan.unmapped_old_labels,
            "new_labels_without_source": plan.new_labels_without_source,
        }

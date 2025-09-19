"""
Version management utilities for Llama Mapper.

Provides a single place to retrieve and propagate version information for:
- Taxonomy (pillars-detectors/taxonomy.yaml)
- Framework mappings (pillars-detectors/frameworks.yaml)
- Detector configs (pillars-detectors/*.yaml)
- Model checkpoints (model_checkpoints/versions.json) — optional

The implementation is lightweight and does not import heavy training/serving
modules so it can be used in API/runtime contexts without extra deps.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..data.detectors import DetectorConfigLoader
from ..data.frameworks import FrameworkMapper
from ..data.taxonomy import TaxonomyLoader


def _first_existing_path(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


@dataclass
class VersionSnapshot:
    """Aggregate snapshot of component versions."""

    created_at: str
    taxonomy: Dict[str, Any]
    frameworks: Dict[str, Any]
    detectors: Dict[str, Any]
    model_version: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at,
            "taxonomy": self.taxonomy,
            "frameworks": self.frameworks,
            "detectors": self.detectors,
            "model_version": self.model_version,
        }


class VersionManager:
    """
    Collects versions from the config data directory and optional model registry,
    and exposes helpers to embed version tags in responses/reports.
    """

    def __init__(
        self,
        data_dir: Optional[os.PathLike] = None,
        versions_registry: Optional[os.PathLike] = None,
        env_prefix: str = "LLAMA_MAPPER",
    ) -> None:
        # Resolve data directory where taxonomy/frameworks/detectors live
        env_dir = os.getenv(f"{env_prefix}_DATA_DIR")
        from typing import Optional as _Optional, List as _List
        default_candidates_raw: _List[_Optional[Path]] = [
            Path(str(data_dir)) if data_dir else None,
            Path("pillars-detectors"),
            Path("./pillars-detectors"),
            Path(".kiro/pillars-detectors"),
        ]
        default_candidates: _List[Path] = [p for p in default_candidates_raw if p is not None]
        self.data_dir = (
            Path(env_dir)
            if env_dir
            else (_first_existing_path(default_candidates) or Path("pillars-detectors"))
        )

        # Registry of model versions; falls back to env var if not found
        env_versions = os.getenv(f"{env_prefix}_MODEL_VERSIONS")
        self.versions_registry = (
            Path(env_versions)
            if env_versions
            else (
                Path(str(versions_registry))
                if versions_registry
                else Path("model_checkpoints/versions.json")
            )
        )

        self._taxonomy_loader = TaxonomyLoader(self.data_dir / "taxonomy.yaml")
        self._framework_mapper = FrameworkMapper(self.data_dir / "frameworks.yaml")
        self._detector_loader = DetectorConfigLoader(
            self.data_dir, taxonomy_loader=self._taxonomy_loader
        )

    def snapshot(self) -> VersionSnapshot:
        """Create a fresh version snapshot."""
        taxonomy_info = self._safe_taxonomy_info()
        frameworks_info = self._safe_frameworks_info()
        detectors_info = self._safe_detectors_info()
        model_version = self._discover_model_version()

        return VersionSnapshot(
            created_at=datetime.now(timezone.utc).isoformat(),
            taxonomy=taxonomy_info or {},
            frameworks=frameworks_info or {},
            detectors=detectors_info or {},
            model_version=model_version,
        )

    # ---- Helpers to read individual components ----
    def _safe_taxonomy_info(self) -> Optional[Dict[str, Any]]:
        try:
            self._taxonomy_loader.load_taxonomy()
            return self._taxonomy_loader.get_version_info()
        except Exception:
            return None

    def _safe_frameworks_info(self) -> Optional[Dict[str, Any]]:
        try:
            self._framework_mapper.load_framework_mapping()
            return self._framework_mapper.get_version_info()
        except Exception:
            return None

    def _safe_detectors_info(self) -> Dict[str, Dict[str, str]]:
        try:
            self._detector_loader.load_all_detector_configs()
            return self._detector_loader.get_version_info()
        except Exception:
            return {}

    def _discover_model_version(self) -> str:
        """
        Determine the model version in priority order:
        1) {env_prefix}_MODEL_VERSION environment variable
        2) Latest key in versions.json registry (if exists)
        3) Derive from config component versions (taxonomy/frameworks) if available
        4) "unknown"
        """
        env_version = os.getenv("LLAMA_MAPPER_MODEL_VERSION")
        if env_version:
            return env_version

        try:
            if self.versions_registry.exists():
                with open(self.versions_registry, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and data:
                    # Choose the newest by created_at field if present; else lexical max
                    def _key(item: tuple[str, Any]) -> datetime:
                        v = item[1]
                        ts = v.get("created_at")
                        try:
                            return (
                                datetime.fromisoformat(ts.replace("Z", "+00:00"))
                                if ts
                                else datetime.fromtimestamp(0)
                            )
                        except Exception:
                            return datetime.fromtimestamp(0)

                    latest = sorted(data.items(), key=_key, reverse=True)[0]
                    return str(latest[0])
        except Exception:
            pass

        # Attempt to derive a non-"unknown" tag from available component versions
        try:
            t_info = self._safe_taxonomy_info() or {}
            f_info = self._safe_frameworks_info() or {}
            fragments: list[str] = []
            t_ver = t_info.get("version")
            f_ver = f_info.get("version")
            if isinstance(t_ver, str) and t_ver:
                fragments.append(f"taxonomy@{t_ver}")
            if isinstance(f_ver, str) and f_ver:
                fragments.append(f"frameworks@{f_ver}")
            if fragments:
                return "|".join(fragments)
        except Exception:
            pass

        return "unknown"

    # ---- Embedding helpers ----
    def annotate_notes_with_versions(self, notes: Optional[str]) -> str:
        snap = self.snapshot()
        parts = [
            f"taxonomy={snap.taxonomy.get('version', 'n/a')}",
            f"frameworks={snap.frameworks.get('version', 'n/a')}",
            f"model={snap.model_version}",
        ]
        tag = "versions: " + ", ".join(parts)
        return f"{notes}\n{tag}" if notes else tag

    def apply_to_provenance(self, provenance_obj: Any) -> None:
        """
        Mutate a MappingResponse.provenance-like object by setting the model
        field to the discovered model version. Safe no-op if attribute missing.
        """
        snap = self.snapshot()
        try:
            if hasattr(provenance_obj, "model"):
                setattr(provenance_obj, "model", snap.model_version)
        except Exception:
            pass

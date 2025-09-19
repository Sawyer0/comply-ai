"""
Configuration validator for taxonomy, frameworks, and detector mappings.

Validates configuration files under the pillars-detectors data directory
and returns a structured result suitable for CLI reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

from difflib import SequenceMatcher, get_close_matches
import yaml as _yaml

from ..data.taxonomy import TaxonomyLoader, Taxonomy
from ..data.frameworks import FrameworkMapper
from ..data.detectors import DetectorConfigLoader


@dataclass
class SectionResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, object] = field(default_factory=dict)


@dataclass
class ValidationResult:
    ok: bool
    data_dir: Path
    taxonomy: SectionResult
    frameworks: SectionResult
    detectors: SectionResult


def _resolve_data_dir(explicit: Optional[Path] = None) -> Path:
    """Resolve the data directory that contains taxonomy.yaml, frameworks.yaml, and detector YAMLs.

    Priority:
    1) Explicit path if provided
    2) Environment variable LLAMA_MAPPER_DATA_DIR or MAPPER_DATA_DIR
    3) ./pillars-detectors
    4) ./.kiro/pillars-detectors
    5) Create ./pillars-detectors if none exist
    """
    import os

    if explicit:
        return explicit

    env_dir = os.getenv("LLAMA_MAPPER_DATA_DIR") or os.getenv("MAPPER_DATA_DIR")
    if env_dir:
        return Path(env_dir)

    p1 = Path("pillars-detectors")
    if p1.exists():
        return p1

    p2 = Path(".kiro") / "pillars-detectors"
    if p2.exists():
        return p2

    # Fallback: create default path so users can start populating configs
    p1.mkdir(parents=True, exist_ok=True)
    return p1


def validate_configuration(data_dir: Optional[Path] = None) -> ValidationResult:
    """Validate taxonomy, frameworks, and detector mapping configuration under data_dir."""
    base_dir = _resolve_data_dir(data_dir)

    # Taxonomy validation
    tax_path = base_dir / "taxonomy.yaml"
    tax_loader = TaxonomyLoader(taxonomy_path=tax_path)
    tax_errors: List[str] = []
    tax_warnings: List[str] = []
    taxonomy_ok = False
    tax_details: Dict[str, object] = {}
    try:
        taxonomy = tax_loader.load_taxonomy(force_reload=True)
        taxonomy_ok = True
        stats = taxonomy.get_statistics()
        tax_details.update(
            {
                "version": taxonomy.version,
                "namespace": taxonomy.namespace,
                "total_labels": stats.get("total_labels"),
                "total_categories": stats.get("total_categories"),
            }
        )
    except Exception as e:  # noqa: BLE001
        tax_errors.append(f"Taxonomy error: {e}")
        taxonomy = None  # type: ignore[assignment]

    taxonomy_result = SectionResult(
        ok=taxonomy_ok, errors=tax_errors, warnings=tax_warnings, details=tax_details
    )

    # Frameworks validation (depends on taxonomy)
    fw_path = base_dir / "frameworks.yaml"
    fw_mapper = FrameworkMapper(frameworks_path=fw_path)
    fw_errors: List[str] = []
    fw_warnings: List[str] = []
    frameworks_ok = False
    fw_details: Dict[str, object] = {}
    try:
        fw = fw_mapper.load_framework_mapping(force_reload=True)
        if taxonomy_ok and taxonomy is not None:
            label_check = fw.validate_against_taxonomy(taxonomy)
            invalid_labels = label_check.get("invalid", [])
            if invalid_labels:
                fw_errors.append(
                    f"Framework mappings reference unknown taxonomy labels: {sorted(set(invalid_labels))}"
                )
            ref_check = fw.validate_framework_references()
            invalid_refs = ref_check.get("invalid", [])
            if invalid_refs:
                fw_errors.append(
                    f"Framework controls not found or malformed: {sorted(set(invalid_refs))}"
                )
        fw_details.update(
            {
                "version": fw.version,
                "frameworks": fw.get_all_frameworks(),
            }
        )
        frameworks_ok = len(fw_errors) == 0
    except Exception as e:  # noqa: BLE001
        fw_errors.append(f"Frameworks error: {e}")

    frameworks_result = SectionResult(
        ok=frameworks_ok, errors=fw_errors, warnings=fw_warnings, details=fw_details
    )

    # Detectors validation (depends on taxonomy)
    det_loader = DetectorConfigLoader(detectors_path=base_dir)
    det_errors: List[str] = []
    det_warnings: List[str] = []
    detectors_ok = False
    det_details: Dict[str, object] = {}
    try:
        dets = det_loader.load_all_detector_configs(force_reload=True)
        if taxonomy_ok and taxonomy is not None:
            results = det_loader.validate_all_mappings(taxonomy)
            invalid_by_detector: Dict[str, List[str]] = {}
            for det_name, res in results.items():
                invalid = res.get("invalid", [])
                if invalid:
                    invalid_by_detector[det_name] = invalid
            if invalid_by_detector:
                for det, labels in invalid_by_detector.items():
                    det_errors.append(
                        f"Detector '{det}' references unknown taxonomy labels: {sorted(set(labels))}"
                    )
        det_details.update({"detectors_found": sorted(list(dets.keys()))})
        detectors_ok = len(det_errors) == 0
    except Exception as e:  # noqa: BLE001
        det_errors.append(f"Detectors error: {e}")

    detectors_result = SectionResult(
        ok=detectors_ok, errors=det_errors, warnings=det_warnings, details=det_details
    )

    overall_ok = taxonomy_result.ok and frameworks_result.ok and detectors_result.ok
    return ValidationResult(
        ok=overall_ok,
        data_dir=base_dir,
        taxonomy=taxonomy_result,
        frameworks=frameworks_result,
        detectors=detectors_result,
    )


# --------- Detector auto-fix helpers ---------

def _all_taxonomy_labels(taxonomy: Taxonomy) -> List[str]:
    return sorted(list(taxonomy.get_all_label_names()))


def _suggest_labels_for(bad_label: str, taxonomy: Taxonomy, max_suggestions: int = 5) -> List[Tuple[str, float]]:
    """Suggest taxonomy labels for an invalid label using aliases and fuzzy matching."""
    suggestions: List[Tuple[str, float]] = []

    # 1) Exact alias matches (score 1.0)
    alias_matches = taxonomy.get_labels_by_alias(bad_label)
    for lab in alias_matches:
        suggestions.append((lab.name, 1.0))

    # 2) Fuzzy matching against all label names
    label_names = _all_taxonomy_labels(taxonomy)
    close = get_close_matches(bad_label, label_names, n=max_suggestions, cutoff=0.6)
    for cand in close:
        score = SequenceMatcher(None, bad_label, cand).ratio()
        suggestions.append((cand, score))

    # Deduplicate by best score
    best: Dict[str, float] = {}
    for cand, score in suggestions:
        if cand not in best or score > best[cand]:
            best[cand] = score

    # Sort by score desc and limit
    return sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:max_suggestions]


def build_detector_fix_plan(data_dir: Optional[Path] = None) -> Dict[str, object]:
    """Build a suggested fix plan for invalid detector mappings.

    Returns a dict with keys: data_dir, items (list per detector), total_invalid
    Each item contains: detector, file, invalid (list), suggestions (dict of bad-> [{label, score}])
    and proposed (dict of bad->best_label).
    """
    base_dir = _resolve_data_dir(data_dir)
    tax = TaxonomyLoader(base_dir / "taxonomy.yaml").load_taxonomy(force_reload=True)
    det_loader = DetectorConfigLoader(detectors_path=base_dir)
    dets = det_loader.load_all_detector_configs(force_reload=True)

    # Compute invalids
    results = det_loader.validate_all_mappings(tax)
    items = []
    total_invalid = 0
    for det_name, mapping in dets.items():
        invalid_labels = results.get(det_name, {}).get("invalid", [])
        if not invalid_labels:
            continue
        total_invalid += len(invalid_labels)
        sugg_map: Dict[str, List[Dict[str, object]]] = {}
        proposed: Dict[str, str] = {}
        for bad in invalid_labels:
            ranked = _suggest_labels_for(bad, tax)
            sugg_map[bad] = [{"label": lab, "score": round(score, 4)} for lab, score in ranked]
            if ranked:
                proposed[bad] = ranked[0][0]
        items.append(
            {
                "detector": det_name,
                "file": str(mapping.file_path) if mapping.file_path else None,
                "invalid": invalid_labels,
                "suggestions": sugg_map,
                "proposed": proposed,
            }
        )

    return {"data_dir": str(base_dir), "items": items, "total_invalid": total_invalid}


def apply_detector_fix_plan(plan: Dict[str, object], apply_threshold: float = 0.86) -> Dict[str, object]:
    """Apply detector fix plan to YAML files where the best suggestion is confident enough.

    Only applies when a suggestion has score >= apply_threshold.
    Returns a summary with applied and skipped counts.
    """
    applied = 0
    skipped = 0
    updated_files: Dict[str, int] = {}

    items = cast(List[Dict], plan.get("items", [])) if isinstance(plan, dict) else []
    for item in items:
        file_path = item.get("file") if isinstance(item, dict) else None
        suggestions = item.get("suggestions", {}) if isinstance(item, dict) else {}
        proposed = item.get("proposed", {}) if isinstance(item, dict) else {}
        if not file_path or not suggestions:
            continue
        path = Path(file_path)
        if not path.exists():
            skipped += 1
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = _yaml.safe_load(f) or {}
            maps = doc.get("maps") or {}
            changed = False
            for bad_label, proposal in proposed.items():
                # find score for proposal
                cand_list = suggestions.get(bad_label, [])
                score = 0.0
                for c in cand_list:
                    if c.get("label") == proposal:
                        score = float(c.get("score", 0.0))
                        break
                if score >= apply_threshold:
                    # update all entries in maps whose value equals bad_label
                    for k, v in list(maps.items()):
                        if v == bad_label:
                            maps[k] = proposal
                            changed = True
                            applied += 1
                else:
                    skipped += 1
            if changed:
                doc["maps"] = maps
                with open(path, "w", encoding="utf-8") as f:
                    _yaml.safe_dump(doc, f, sort_keys=False)
                updated_files[str(path)] = updated_files.get(str(path), 0) + 1
        except Exception:
            skipped += 1
            continue

    return {"applied": applied, "skipped": skipped, "updated_files": updated_files}


def scaffold_detector_yaml(
    name: str,
    version: str = "v1",
    notes: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[Path, str]:
    """Create a detector YAML skeleton in the appropriate data directory.

    Returns (path, message) tuple.
    """
    import yaml as _yaml

    base_dir = _resolve_data_dir(output_dir)
    # Prefer top-level pillars-detectors directory if both exist
    if base_dir.name == "pillars-detectors":
        target_dir = base_dir
    else:
        # If .kiro/pillars-detectors exists, use it
        target_dir = base_dir

    target_dir.mkdir(parents=True, exist_ok=True)

    # File name: sanitize name to kebab-case
    file_name = f"{name.lower().replace(' ', '-').replace('_', '-')}.yaml"
    file_path = target_dir / file_name

    if file_path.exists():
        raise FileExistsError(f"Detector config already exists: {file_path}")

    doc: Dict[str, object] = {
        "detector": name,
        "version": version,
        "notes": notes or "Add a brief description of this detector and any caveats.",
        "maps": {
            # Example mapping entry; replace with your detector labels -> taxonomy labels
            "REPLACE_ME_DETECTOR_LABEL": "CATEGORY.Subcategory.Label",
        },
    }

    with open(file_path, "w", encoding="utf-8") as f:
        _yaml.safe_dump(doc, f, sort_keys=False)

    guidance = (
        "Detector YAML created. Next steps:\n"
        f"  • File: {file_path}\n"
        "  • Replace REPLACE_ME_DETECTOR_LABEL with the detector's output label(s)\n"
        "  • Replace CATEGORY.Subcategory.Label with a canonical taxonomy label from taxonomy.yaml\n"
        "  • Run: mapper validate-config  (to lint mappings against taxonomy)\n"
    )

    return file_path, guidance

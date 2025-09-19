from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, Set

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DetectorHints:
    categories: Set[str]


def load_detector_taxonomy_hints(base_dir: str = ".kiro/pillars-detectors") -> Dict[str, DetectorHints]:
    """Load detector→taxonomy category hints from mapping YAML files.

    Derives categories from canonical labels in each detector's `maps` section.
    """
    out: Dict[str, DetectorHints] = {}
    pattern = os.path.join(base_dir, "*.yaml")
    for path in glob.glob(pattern):
        try:
            if yaml is None:
                # YAML parser not available
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            name = data.get("detector")
            maps = data.get("maps", {}) or {}
            cats: Set[str] = set()
            for _, canon in maps.items():
                if isinstance(canon, str) and "." in canon:
                    cats.add(canon.split(".")[0])
            if name and cats:
                out[name] = DetectorHints(categories=cats)
        except Exception:
            # Best-effort; skip broken files
            continue
    # Fallback hints for common built-in detectors when YAML is unavailable
    if not out:
        defaults: Dict[str, Set[str]] = {
            "regex-pii": {"PII"},
            "deberta-toxicity": {"HARM"},
            "openai-moderation": {"HARM"},
            "llama-guard": {"HARM", "PII"},
        }
        out = {k: DetectorHints(categories=v) for k, v in defaults.items()}
    return out

"""Data structures shared across training data generation modules."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrainingExample:
    """Represents a single training example for instruction-following fine-tuning."""

    instruction: str
    response: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for JSONL output."""
        return {
            "instruction": self.instruction,
            "response": self.response,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class MapperCanonicalEvent:
    """Canonical output format matching pillars-detectors/schema.json."""

    taxonomy: List[str]
    scores: Dict[str, float]
    confidence: float
    notes: Optional[str] = None
    provenance: Optional[Dict[str, Any]] = None
    policy_context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "taxonomy": self.taxonomy,
            "scores": self.scores,
            "confidence": self.confidence,
        }

        if self.notes:
            result["notes"] = self.notes
        if self.provenance:
            result["provenance"] = self.provenance
        if self.policy_context:
            result["policy_context"] = self.policy_context

        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


__all__ = ["TrainingExample", "MapperCanonicalEvent"]

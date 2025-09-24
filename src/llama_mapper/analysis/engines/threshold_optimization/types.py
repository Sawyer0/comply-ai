"""
Type definitions for threshold optimization components.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any


@dataclass
class ROCPoint:
    """Single point on ROC curve."""

    threshold: float
    true_positive_rate: float
    false_positive_rate: float
    precision: float
    recall: float
    f1_score: float


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for threshold analysis."""

    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    specificity: float
    false_positive_rate: float
    false_negative_rate: float


class OptimizationObjective(Enum):
    """Optimization objectives for threshold tuning."""

    MINIMIZE_FALSE_POSITIVES = "minimize_fp"
    MAXIMIZE_F1_SCORE = "maximize_f1"
    MAXIMIZE_PRECISION = "maximize_precision"
    MAXIMIZE_RECALL = "maximize_recall"
    BALANCED_PRECISION_RECALL = "balanced_pr"
    MINIMIZE_COST = "minimize_cost"


@dataclass
class ThresholdRecommendation:
    """Threshold recommendation with analysis."""

    detector_id: str
    current_threshold: float
    recommended_threshold: float
    expected_improvement: Dict[str, float]
    confidence: float
    rationale: str
    impact_analysis: Dict[str, Any]

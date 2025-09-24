"""
Pattern Interaction Matrix for visualizing pattern relationships.

This module creates and manages interaction matrices that visualize
relationships and dependencies between security patterns.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
import numpy as np
from enum import Enum

from ...domain import (
    Pattern,
    PatternType,
    PatternStrength,
    BusinessRelevance,
    SecurityData,
)

logger = logging.getLogger(__name__)


class InteractionVisualizationType(Enum):
    """Types of interaction visualizations."""

    STRENGTH_MATRIX = "strength_matrix"
    TEMPORAL_MATRIX = "temporal_matrix"
    DETECTOR_MATRIX = "detector_matrix"
    BUSINESS_IMPACT_MATRIX = "business_impact_matrix"
    CAUSAL_MATRIX = "causal_matrix"


class PatternInteractionMatrix:
    """
    Creates and manages interaction matrices for pattern visualization.

    Generates various types of matrices to visualize different aspects
    of pattern relationships and interactions.
    """

    def __init__(self, config: Dict[str, any] = None):
        self.config = config or {}
        self.matrix_size_limit = self.config.get("matrix_size_limit", 50)
        self.visualization_threshold = self.config.get("visualization_threshold", 0.1)

    async def create_interaction_matrices(
        self,
        patterns: List[Pattern],
        relationships: List[Dict[str, any]],
        context_data: SecurityData,
    ) -> Dict[str, any]:
        """
        Create comprehensive interaction matrices for pattern visualization.

        Args:
            patterns: List of patterns to analyze
            relationships: Pattern relationships
            context_data: Security data for context

        Returns:
            Dictionary containing various interaction matrices
        """
        if len(patterns) > self.matrix_size_limit:
            logger.warning(
                f"Pattern count ({len(patterns)}) exceeds matrix size limit ({self.matrix_size_limit})"
            )
            patterns = patterns[: self.matrix_size_limit]

        try:
            matrices = {}

            # Create strength-based interaction matrix
            matrices["strength_matrix"] = await self._create_strength_matrix(
                patterns, relationships
            )

            # Create temporal interaction matrix
            matrices["temporal_matrix"] = await self._create_temporal_matrix(patterns)

            # Create detector overlap matrix
            matrices["detector_matrix"] = await self._create_detector_matrix(patterns)

            # Create business impact matrix
            matrices["business_impact_matrix"] = (
                await self._create_business_impact_matrix(patterns, relationships)
            )

            # Create causal relationship matrix
            matrices["causal_matrix"] = await self._create_causal_matrix(
                patterns, relationships
            )

            # Create composite interaction matrix
            matrices["composite_matrix"] = await self._create_composite_matrix(
                patterns, relationships
            )

            # Generate matrix statistics
            matrices["statistics"] = self._calculate_matrix_statistics(matrices)

            # Create visualization metadata
            matrices["visualization_metadata"] = self._create_visualization_metadata(
                patterns, matrices
            )

            logger.info(
                "Interaction matrices created",
                total_patterns=len(patterns),
                matrix_types=len(matrices) - 2,  # Exclude statistics and metadata
            )

            return matrices

        except Exception as e:
            logger.error("Interaction matrices creation failed", error=str(e))
            return {}

    async def _create_strength_matrix(
        self, patterns: List[Pattern], relationships: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """Create matrix based on relationship strengths."""
        try:
            pattern_ids = [p.pattern_id for p in patterns]
            matrix_size = len(pattern_ids)

            # Initialize matrix
            strength_matrix = np.zeros((matrix_size, matrix_size))

            # Create ID to index mapping
            id_to_index = {pid: i for i, pid in enumerate(pattern_ids)}

            # Fill matrix with relationship strengths
            for rel in relationships:
                i = id_to_index.get(rel["pattern1_id"])
                j = id_to_index.get(rel["pattern2_id"])

                if i is not None and j is not None:
                    strength = rel["strength"]
                    strength_matrix[i][j] = strength
                    strength_matrix[j][i] = strength  # Symmetric matrix

            # Add self-relationships (diagonal)
            np.fill_diagonal(strength_matrix, 1.0)

            return {
                "matrix": strength_matrix.tolist(),
                "pattern_ids": pattern_ids,
                "pattern_labels": [self._create_pattern_label(p) for p in patterns],
                "matrix_type": "strength_based",
                "size": matrix_size,
                "max_value": float(np.max(strength_matrix)),
                "min_value": (
                    float(np.min(strength_matrix[strength_matrix > 0]))
                    if np.any(strength_matrix > 0)
                    else 0.0
                ),
                "avg_value": (
                    float(np.mean(strength_matrix[strength_matrix > 0]))
                    if np.any(strength_matrix > 0)
                    else 0.0
                ),
                "density": float(
                    np.count_nonzero(strength_matrix) / (matrix_size * matrix_size)
                ),
            }

        except Exception as e:
            logger.error("Strength matrix creation failed", error=str(e))
            return {}

    async def _create_temporal_matrix(self, patterns: List[Pattern]) -> Dict[str, any]:
        """Create matrix based on temporal relationships."""
        try:
            pattern_ids = [p.pattern_id for p in patterns]
            matrix_size = len(pattern_ids)

            # Initialize matrix
            temporal_matrix = np.zeros((matrix_size, matrix_size))

            # Calculate temporal overlaps
            for i, pattern1 in enumerate(patterns):
                for j, pattern2 in enumerate(patterns):
                    if i != j:
                        overlap = self._calculate_temporal_overlap(pattern1, pattern2)
                        temporal_matrix[i][j] = overlap
                    else:
                        temporal_matrix[i][j] = 1.0  # Self-overlap

            return {
                "matrix": temporal_matrix.tolist(),
                "pattern_ids": pattern_ids,
                "pattern_labels": [self._create_pattern_label(p) for p in patterns],
                "matrix_type": "temporal_overlap",
                "size": matrix_size,
                "max_value": float(np.max(temporal_matrix)),
                "min_value": (
                    float(np.min(temporal_matrix[temporal_matrix > 0]))
                    if np.any(temporal_matrix > 0)
                    else 0.0
                ),
                "avg_value": (
                    float(np.mean(temporal_matrix[temporal_matrix > 0]))
                    if np.any(temporal_matrix > 0)
                    else 0.0
                ),
                "temporal_clusters": self._identify_temporal_clusters(
                    temporal_matrix, pattern_ids
                ),
            }

        except Exception as e:
            logger.error("Temporal matrix creation failed", error=str(e))
            return {}

    async def _create_detector_matrix(self, patterns: List[Pattern]) -> Dict[str, any]:
        """Create matrix based on detector overlaps."""
        try:
            pattern_ids = [p.pattern_id for p in patterns]
            matrix_size = len(pattern_ids)

            # Initialize matrix
            detector_matrix = np.zeros((matrix_size, matrix_size))

            # Calculate detector overlaps
            for i, pattern1 in enumerate(patterns):
                for j, pattern2 in enumerate(patterns):
                    if i != j:
                        overlap = self._calculate_detector_overlap(pattern1, pattern2)
                        detector_matrix[i][j] = overlap
                    else:
                        detector_matrix[i][j] = 1.0  # Self-overlap

            # Identify detector groups
            detector_groups = self._identify_detector_groups(patterns)

            return {
                "matrix": detector_matrix.tolist(),
                "pattern_ids": pattern_ids,
                "pattern_labels": [self._create_pattern_label(p) for p in patterns],
                "matrix_type": "detector_overlap",
                "size": matrix_size,
                "max_value": float(np.max(detector_matrix)),
                "min_value": (
                    float(np.min(detector_matrix[detector_matrix > 0]))
                    if np.any(detector_matrix > 0)
                    else 0.0
                ),
                "avg_value": (
                    float(np.mean(detector_matrix[detector_matrix > 0]))
                    if np.any(detector_matrix > 0)
                    else 0.0
                ),
                "detector_groups": detector_groups,
            }

        except Exception as e:
            logger.error("Detector matrix creation failed", error=str(e))
            return {}

    async def _create_business_impact_matrix(
        self, patterns: List[Pattern], relationships: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """Create matrix based on business impact relationships."""
        try:
            pattern_ids = [p.pattern_id for p in patterns]
            matrix_size = len(pattern_ids)

            # Initialize matrix
            impact_matrix = np.zeros((matrix_size, matrix_size))

            # Create ID to index mapping
            id_to_index = {pid: i for i, pid in enumerate(pattern_ids)}

            # Calculate business impact relationships
            for i, pattern1 in enumerate(patterns):
                for j, pattern2 in enumerate(patterns):
                    if i != j:
                        impact_relationship = (
                            self._calculate_business_impact_relationship(
                                pattern1, pattern2, relationships
                            )
                        )
                        impact_matrix[i][j] = impact_relationship
                    else:
                        # Self-impact based on business relevance
                        impact_matrix[i][j] = self._get_business_relevance_score(
                            pattern1.business_relevance
                        )

            # Identify high-impact clusters
            high_impact_clusters = self._identify_high_impact_clusters(
                impact_matrix, patterns
            )

            return {
                "matrix": impact_matrix.tolist(),
                "pattern_ids": pattern_ids,
                "pattern_labels": [self._create_pattern_label(p) for p in patterns],
                "matrix_type": "business_impact",
                "size": matrix_size,
                "max_value": float(np.max(impact_matrix)),
                "min_value": (
                    float(np.min(impact_matrix[impact_matrix > 0]))
                    if np.any(impact_matrix > 0)
                    else 0.0
                ),
                "avg_value": (
                    float(np.mean(impact_matrix[impact_matrix > 0]))
                    if np.any(impact_matrix > 0)
                    else 0.0
                ),
                "high_impact_clusters": high_impact_clusters,
            }

        except Exception as e:
            logger.error("Business impact matrix creation failed", error=str(e))
            return {}

    async def _create_causal_matrix(
        self, patterns: List[Pattern], relationships: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """Create matrix based on causal relationships."""
        try:
            pattern_ids = [p.pattern_id for p in patterns]
            matrix_size = len(pattern_ids)

            # Initialize matrix
            causal_matrix = np.zeros((matrix_size, matrix_size))

            # Create ID to index mapping
            id_to_index = {pid: i for i, pid in enumerate(pattern_ids)}

            # Analyze causal relationships
            for i, pattern1 in enumerate(patterns):
                for j, pattern2 in enumerate(patterns):
                    if i != j:
                        causal_strength = self._calculate_causal_relationship(
                            pattern1, pattern2, relationships
                        )
                        causal_matrix[i][j] = causal_strength

            # Identify causal chains
            causal_chains = self._identify_causal_chains(causal_matrix, pattern_ids)

            return {
                "matrix": causal_matrix.tolist(),
                "pattern_ids": pattern_ids,
                "pattern_labels": [self._create_pattern_label(p) for p in patterns],
                "matrix_type": "causal_relationships",
                "size": matrix_size,
                "max_value": float(np.max(causal_matrix)),
                "min_value": (
                    float(np.min(causal_matrix[causal_matrix > 0]))
                    if np.any(causal_matrix > 0)
                    else 0.0
                ),
                "avg_value": (
                    float(np.mean(causal_matrix[causal_matrix > 0]))
                    if np.any(causal_matrix > 0)
                    else 0.0
                ),
                "causal_chains": causal_chains,
                "is_directed": True,  # Causal matrix is directional
            }

        except Exception as e:
            logger.error("Causal matrix creation failed", error=str(e))
            return {}

    async def _create_composite_matrix(
        self, patterns: List[Pattern], relationships: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """Create composite matrix combining multiple relationship types."""
        try:
            pattern_ids = [p.pattern_id for p in patterns]
            matrix_size = len(pattern_ids)

            # Get individual matrices
            strength_matrix_data = await self._create_strength_matrix(
                patterns, relationships
            )
            temporal_matrix_data = await self._create_temporal_matrix(patterns)
            detector_matrix_data = await self._create_detector_matrix(patterns)

            # Convert to numpy arrays
            strength_matrix = np.array(strength_matrix_data.get("matrix", []))
            temporal_matrix = np.array(temporal_matrix_data.get("matrix", []))
            detector_matrix = np.array(detector_matrix_data.get("matrix", []))

            # Weights for different components
            weights = {"strength": 0.4, "temporal": 0.3, "detector": 0.3}

            # Create composite matrix
            composite_matrix = (
                weights["strength"] * strength_matrix
                + weights["temporal"] * temporal_matrix
                + weights["detector"] * detector_matrix
            )

            # Identify key relationships
            key_relationships = self._identify_key_relationships(
                composite_matrix, pattern_ids
            )

            return {
                "matrix": composite_matrix.tolist(),
                "pattern_ids": pattern_ids,
                "pattern_labels": [self._create_pattern_label(p) for p in patterns],
                "matrix_type": "composite_relationships",
                "size": matrix_size,
                "max_value": float(np.max(composite_matrix)),
                "min_value": (
                    float(np.min(composite_matrix[composite_matrix > 0]))
                    if np.any(composite_matrix > 0)
                    else 0.0
                ),
                "avg_value": (
                    float(np.mean(composite_matrix[composite_matrix > 0]))
                    if np.any(composite_matrix > 0)
                    else 0.0
                ),
                "weights_used": weights,
                "key_relationships": key_relationships,
            }

        except Exception as e:
            logger.error("Composite matrix creation failed", error=str(e))
            return {}

    def _create_pattern_label(self, pattern: Pattern) -> str:
        """Create a descriptive label for a pattern."""
        try:
            # Truncate pattern ID for readability
            short_id = (
                pattern.pattern_id[:8]
                if len(pattern.pattern_id) > 8
                else pattern.pattern_id
            )

            # Create label with type and strength
            label = f"{short_id}_{pattern.pattern_type.value}_{pattern.strength.value}"

            return label

        except Exception:
            return pattern.pattern_id[:8]

    def _calculate_temporal_overlap(
        self, pattern1: Pattern, pattern2: Pattern
    ) -> float:
        """Calculate temporal overlap between two patterns."""
        try:
            # Calculate overlap duration
            overlap_start = max(pattern1.time_range.start, pattern2.time_range.start)
            overlap_end = min(pattern1.time_range.end, pattern2.time_range.end)

            if overlap_start >= overlap_end:
                return 0.0

            overlap_duration = (overlap_end - overlap_start).total_seconds()

            # Calculate union duration
            union_start = min(pattern1.time_range.start, pattern2.time_range.start)
            union_end = max(pattern1.time_range.end, pattern2.time_range.end)
            union_duration = (union_end - union_start).total_seconds()

            if union_duration == 0:
                return 0.0

            return overlap_duration / union_duration

        except Exception:
            return 0.0

    def _calculate_detector_overlap(
        self, pattern1: Pattern, pattern2: Pattern
    ) -> float:
        """Calculate detector overlap between two patterns."""
        try:
            set1 = set(pattern1.affected_detectors)
            set2 = set(pattern2.affected_detectors)

            intersection = len(set1 & set2)
            union = len(set1 | set2)

            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.0

    def _calculate_business_impact_relationship(
        self, pattern1: Pattern, pattern2: Pattern, relationships: List[Dict[str, any]]
    ) -> float:
        """Calculate business impact relationship between patterns."""
        try:
            # Base relationship from business relevance similarity
            relevance1 = self._get_business_relevance_score(pattern1.business_relevance)
            relevance2 = self._get_business_relevance_score(pattern2.business_relevance)

            relevance_similarity = 1.0 - abs(relevance1 - relevance2)

            # Check for direct relationship
            direct_relationship = 0.0
            for rel in relationships:
                if (
                    rel["pattern1_id"] == pattern1.pattern_id
                    and rel["pattern2_id"] == pattern2.pattern_id
                ) or (
                    rel["pattern1_id"] == pattern2.pattern_id
                    and rel["pattern2_id"] == pattern1.pattern_id
                ):
                    direct_relationship = rel["strength"]
                    break

            # Combine relevance similarity and direct relationship
            return relevance_similarity * 0.6 + direct_relationship * 0.4

        except Exception:
            return 0.0

    def _get_business_relevance_score(self, relevance: BusinessRelevance) -> float:
        """Convert business relevance to numeric score."""
        scores = {
            BusinessRelevance.LOW: 0.25,
            BusinessRelevance.MEDIUM: 0.5,
            BusinessRelevance.HIGH: 0.75,
            BusinessRelevance.CRITICAL: 1.0,
        }
        return scores.get(relevance, 0.5)

    def _calculate_causal_relationship(
        self, pattern1: Pattern, pattern2: Pattern, relationships: List[Dict[str, any]]
    ) -> float:
        """Calculate causal relationship strength."""
        try:
            # Pattern1 must precede pattern2 temporally for causality
            if pattern1.time_range.start >= pattern2.time_range.start:
                return 0.0

            # Check for detector causality
            detector_causality = self._check_detector_causality(
                pattern1.affected_detectors, pattern2.affected_detectors
            )

            # Calculate temporal proximity (closer in time = stronger causality)
            time_gap = (
                pattern2.time_range.start - pattern1.time_range.end
            ).total_seconds()
            temporal_strength = max(
                0, 1.0 - (time_gap / (24 * 3600))
            )  # Decay over 24 hours

            # Combine detector causality and temporal strength
            causal_strength = detector_causality * 0.7 + temporal_strength * 0.3

            return causal_strength

        except Exception:
            return 0.0

    def _check_detector_causality(
        self, detectors1: List[str], detectors2: List[str]
    ) -> float:
        """Check for causal relationships between detector sets."""
        try:
            # Define known causal chains
            causal_chains = {
                "presidio": ["pii-detector", "gdpr-scanner"],
                "pii-detector": ["gdpr-scanner", "hipaa-detector"],
                "network-scanner": ["vulnerability-detector", "threat-detector"],
                "file-scanner": ["malware-detector", "content-detector"],
            }

            # Check for causal relationships
            for detector1 in detectors1:
                downstream = causal_chains.get(detector1.lower(), [])
                for detector2 in detectors2:
                    if detector2.lower() in downstream:
                        return 0.8  # Strong causal relationship

            # Check for detector overlap (weaker causality)
            overlap = set(d.lower() for d in detectors1) & set(
                d.lower() for d in detectors2
            )
            if overlap:
                return 0.4  # Moderate causal relationship

            return 0.0

        except Exception:
            return 0.0

    def _identify_temporal_clusters(
        self, temporal_matrix: np.ndarray, pattern_ids: List[str]
    ) -> List[Dict[str, any]]:
        """Identify temporal clusters in the matrix."""
        clusters = []

        try:
            # Find patterns with high temporal overlap
            threshold = 0.7

            for i in range(len(pattern_ids)):
                cluster_members = [pattern_ids[i]]

                for j in range(len(pattern_ids)):
                    if i != j and temporal_matrix[i][j] > threshold:
                        cluster_members.append(pattern_ids[j])

                if len(cluster_members) > 1:
                    clusters.append(
                        {
                            "cluster_id": f"temporal_cluster_{i}",
                            "members": cluster_members,
                            "avg_overlap": float(
                                np.mean(
                                    [
                                        temporal_matrix[i][j]
                                        for j in range(len(pattern_ids))
                                        if i != j
                                    ]
                                )
                            ),
                        }
                    )

        except Exception as e:
            logger.error("Temporal cluster identification failed", error=str(e))

        return clusters

    def _identify_detector_groups(
        self, patterns: List[Pattern]
    ) -> List[Dict[str, any]]:
        """Identify groups of patterns by detector similarity."""
        groups = []

        try:
            # Group patterns by common detectors
            detector_to_patterns = {}

            for pattern in patterns:
                for detector in pattern.affected_detectors:
                    if detector not in detector_to_patterns:
                        detector_to_patterns[detector] = []
                    detector_to_patterns[detector].append(pattern.pattern_id)

            # Create groups for detectors with multiple patterns
            for detector, pattern_ids in detector_to_patterns.items():
                if len(pattern_ids) > 1:
                    groups.append(
                        {
                            "detector": detector,
                            "pattern_count": len(pattern_ids),
                            "pattern_ids": pattern_ids,
                        }
                    )

        except Exception as e:
            logger.error("Detector group identification failed", error=str(e))

        return groups

    def _identify_high_impact_clusters(
        self, impact_matrix: np.ndarray, patterns: List[Pattern]
    ) -> List[Dict[str, any]]:
        """Identify clusters with high business impact."""
        clusters = []

        try:
            # Find patterns with high business impact relationships
            threshold = 0.8

            high_impact_indices = []
            for i, pattern in enumerate(patterns):
                if pattern.business_relevance in [
                    BusinessRelevance.HIGH,
                    BusinessRelevance.CRITICAL,
                ]:
                    high_impact_indices.append(i)

            if len(high_impact_indices) > 1:
                cluster_members = [patterns[i].pattern_id for i in high_impact_indices]
                avg_impact = float(
                    np.mean(
                        [
                            impact_matrix[i][j]
                            for i in high_impact_indices
                            for j in high_impact_indices
                            if i != j
                        ]
                    )
                )

                clusters.append(
                    {
                        "cluster_type": "high_business_impact",
                        "members": cluster_members,
                        "avg_impact_score": avg_impact,
                        "member_count": len(cluster_members),
                    }
                )

        except Exception as e:
            logger.error("High impact cluster identification failed", error=str(e))

        return clusters

    def _identify_causal_chains(
        self, causal_matrix: np.ndarray, pattern_ids: List[str]
    ) -> List[Dict[str, any]]:
        """Identify causal chains in the matrix."""
        chains = []

        try:
            threshold = 0.5

            # Find causal relationships
            for i in range(len(pattern_ids)):
                for j in range(len(pattern_ids)):
                    if i != j and causal_matrix[i][j] > threshold:
                        # Look for chains (i -> j -> k)
                        for k in range(len(pattern_ids)):
                            if k != i and k != j and causal_matrix[j][k] > threshold:
                                chains.append(
                                    {
                                        "chain": [
                                            pattern_ids[i],
                                            pattern_ids[j],
                                            pattern_ids[k],
                                        ],
                                        "strengths": [
                                            float(causal_matrix[i][j]),
                                            float(causal_matrix[j][k]),
                                        ],
                                        "total_strength": float(
                                            causal_matrix[i][j] * causal_matrix[j][k]
                                        ),
                                    }
                                )

        except Exception as e:
            logger.error("Causal chain identification failed", error=str(e))

        return chains

    def _identify_key_relationships(
        self, composite_matrix: np.ndarray, pattern_ids: List[str]
    ) -> List[Dict[str, any]]:
        """Identify key relationships in the composite matrix."""
        relationships = []

        try:
            # Find top relationships
            threshold = 0.7

            for i in range(len(pattern_ids)):
                for j in range(i + 1, len(pattern_ids)):
                    if composite_matrix[i][j] > threshold:
                        relationships.append(
                            {
                                "pattern1_id": pattern_ids[i],
                                "pattern2_id": pattern_ids[j],
                                "composite_strength": float(composite_matrix[i][j]),
                                "relationship_rank": "key",
                            }
                        )

            # Sort by strength
            relationships.sort(key=lambda x: x["composite_strength"], reverse=True)

            # Return top 10 relationships
            return relationships[:10]

        except Exception as e:
            logger.error("Key relationship identification failed", error=str(e))
            return []

    def _calculate_matrix_statistics(self, matrices: Dict[str, any]) -> Dict[str, any]:
        """Calculate statistics across all matrices."""
        try:
            stats = {
                "total_matrices": len(
                    [
                        k
                        for k in matrices.keys()
                        if k not in ["statistics", "visualization_metadata"]
                    ]
                ),
                "matrix_densities": {},
                "average_values": {},
                "max_values": {},
                "relationship_counts": {},
            }

            for matrix_name, matrix_data in matrices.items():
                if isinstance(matrix_data, dict) and "matrix" in matrix_data:
                    stats["matrix_densities"][matrix_name] = matrix_data.get(
                        "density", 0.0
                    )
                    stats["average_values"][matrix_name] = matrix_data.get(
                        "avg_value", 0.0
                    )
                    stats["max_values"][matrix_name] = matrix_data.get("max_value", 0.0)

                    # Count significant relationships
                    matrix = np.array(matrix_data["matrix"])
                    significant_relationships = np.count_nonzero(
                        matrix > self.visualization_threshold
                    )
                    stats["relationship_counts"][matrix_name] = int(
                        significant_relationships
                    )

            return stats

        except Exception as e:
            logger.error("Matrix statistics calculation failed", error=str(e))
            return {}

    def _create_visualization_metadata(
        self, patterns: List[Pattern], matrices: Dict[str, any]
    ) -> Dict[str, any]:
        """Create metadata for visualization purposes."""
        try:
            metadata = {
                "pattern_count": len(patterns),
                "pattern_types": list(set(p.pattern_type.value for p in patterns)),
                "business_relevance_distribution": self._calculate_relevance_distribution(
                    patterns
                ),
                "strength_distribution": self._calculate_strength_distribution(
                    patterns
                ),
                "recommended_visualizations": self._recommend_visualizations(matrices),
                "color_schemes": self._suggest_color_schemes(patterns),
                "layout_suggestions": self._suggest_layout_options(len(patterns)),
            }

            return metadata

        except Exception as e:
            logger.error("Visualization metadata creation failed", error=str(e))
            return {}

    def _calculate_relevance_distribution(
        self, patterns: List[Pattern]
    ) -> Dict[str, int]:
        """Calculate distribution of business relevance levels."""
        distribution = {relevance.value: 0 for relevance in BusinessRelevance}

        for pattern in patterns:
            distribution[pattern.business_relevance.value] += 1

        return distribution

    def _calculate_strength_distribution(
        self, patterns: List[Pattern]
    ) -> Dict[str, int]:
        """Calculate distribution of pattern strengths."""
        distribution = {strength.value: 0 for strength in PatternStrength}

        for pattern in patterns:
            distribution[pattern.strength.value] += 1

        return distribution

    def _recommend_visualizations(self, matrices: Dict[str, any]) -> List[str]:
        """Recommend visualization types based on matrix characteristics."""
        recommendations = []

        try:
            # Always recommend composite matrix
            recommendations.append("composite_matrix")

            # Recommend based on matrix density
            for matrix_name, matrix_data in matrices.items():
                if isinstance(matrix_data, dict) and "density" in matrix_data:
                    density = matrix_data["density"]

                    if density > 0.3:  # Dense matrix
                        recommendations.append(f"{matrix_name}_heatmap")
                    elif density > 0.1:  # Moderate density
                        recommendations.append(f"{matrix_name}_network")
                    else:  # Sparse matrix
                        recommendations.append(f"{matrix_name}_graph")

        except Exception as e:
            logger.error("Visualization recommendation failed", error=str(e))

        return list(set(recommendations))  # Remove duplicates

    def _suggest_color_schemes(self, patterns: List[Pattern]) -> Dict[str, List[str]]:
        """Suggest color schemes for different visualizations."""
        return {
            "strength_based": ["#ffffff", "#ffcccc", "#ff6666", "#cc0000"],
            "temporal_based": ["#ffffff", "#cceeff", "#66ccff", "#0099cc"],
            "business_impact": ["#ffffff", "#ffffcc", "#ffcc66", "#ff9900"],
            "pattern_type": ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#feca57"],
        }

    def _suggest_layout_options(self, pattern_count: int) -> List[str]:
        """Suggest layout options based on pattern count."""
        if pattern_count <= 10:
            return ["circular", "grid", "force_directed"]
        elif pattern_count <= 25:
            return ["force_directed", "hierarchical", "grid"]
        else:
            return ["force_directed", "clustered", "hierarchical"]

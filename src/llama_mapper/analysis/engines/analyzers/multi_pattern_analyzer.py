"""
Multi-Pattern Analyzer for detecting pattern relationships.

This module implements sophisticated analysis to detect relationships
between multiple security patterns and their interactions.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timezone, timedelta
import numpy as np
from collections import defaultdict

from ...domain import (
    Pattern,
    PatternType,
    PatternStrength,
    PatternCorrelation,
    BusinessRelevance,
    SecurityData,
    TimeRange,
)

logger = logging.getLogger(__name__)


class MultiPatternAnalyzer:
    """
    Analyzes relationships and interactions between multiple security patterns.

    Detects pattern clusters, hierarchies, dependencies, and compound effects
    using sophisticated statistical and graph-based analysis methods.
    """

    def __init__(self, config: Dict[str, any] = None):
        self.config = config or {}
        self.correlation_threshold = self.config.get("correlation_threshold", 0.6)
        self.temporal_window = self.config.get("temporal_window_hours", 24)
        self.min_patterns = self.config.get("min_patterns", 2)

    async def analyze_pattern_relationships(
        self, patterns: List[Pattern], context_data: SecurityData
    ) -> Dict[str, any]:
        """
        Analyze relationships between multiple patterns.

        Args:
            patterns: List of patterns to analyze
            context_data: Security data for context

        Returns:
            Dictionary containing relationship analysis results
        """
        if len(patterns) < self.min_patterns:
            return {"relationships": [], "clusters": [], "hierarchies": []}

        try:
            # Detect direct pattern relationships
            relationships = await self._detect_pattern_relationships(
                patterns, context_data
            )

            # Identify pattern clusters
            clusters = await self._identify_pattern_clusters(patterns, relationships)

            # Detect pattern hierarchies
            hierarchies = await self._detect_pattern_hierarchies(
                patterns, relationships
            )

            # Analyze temporal relationships
            temporal_relationships = await self._analyze_temporal_relationships(
                patterns
            )

            # Detect causal relationships
            causal_relationships = await self._detect_causal_relationships(
                patterns, context_data
            )

            # Calculate relationship strength matrix
            relationship_matrix = self._build_relationship_matrix(
                patterns, relationships
            )

            analysis_result = {
                "relationships": relationships,
                "clusters": clusters,
                "hierarchies": hierarchies,
                "temporal_relationships": temporal_relationships,
                "causal_relationships": causal_relationships,
                "relationship_matrix": relationship_matrix,
                "analysis_metadata": {
                    "total_patterns": len(patterns),
                    "total_relationships": len(relationships),
                    "analysis_timestamp": datetime.utcnow().isoformat(),
                    "correlation_threshold": self.correlation_threshold,
                },
            }

            logger.info(
                "Multi-pattern analysis completed",
                total_patterns=len(patterns),
                relationships_found=len(relationships),
                clusters_found=len(clusters),
            )

            return analysis_result

        except Exception as e:
            logger.error("Multi-pattern analysis failed", error=str(e))
            return {"relationships": [], "clusters": [], "hierarchies": []}

    async def _detect_pattern_relationships(
        self, patterns: List[Pattern], context_data: SecurityData
    ) -> List[Dict[str, any]]:
        """Detect direct relationships between patterns."""
        relationships = []

        try:
            # Compare each pattern with every other pattern
            for i, pattern1 in enumerate(patterns):
                for j, pattern2 in enumerate(patterns[i + 1 :], i + 1):
                    relationship = await self._analyze_pattern_pair(
                        pattern1, pattern2, context_data
                    )
                    if (
                        relationship
                        and relationship["strength"] >= self.correlation_threshold
                    ):
                        relationships.append(relationship)

        except Exception as e:
            logger.error("Pattern relationship detection failed", error=str(e))

        return relationships

    async def _analyze_pattern_pair(
        self, pattern1: Pattern, pattern2: Pattern, context_data: SecurityData
    ) -> Optional[Dict[str, any]]:
        """Analyze relationship between two patterns."""
        try:
            # Calculate temporal overlap
            temporal_overlap = self._calculate_temporal_overlap(
                pattern1.time_range, pattern2.time_range
            )

            # Calculate detector overlap
            detector_overlap = self._calculate_detector_overlap(
                pattern1.affected_detectors, pattern2.affected_detectors
            )

            # Calculate confidence correlation
            confidence_correlation = self._calculate_confidence_correlation(
                pattern1, pattern2
            )

            # Calculate business relevance alignment
            relevance_alignment = self._calculate_relevance_alignment(
                pattern1, pattern2
            )

            # Calculate pattern type compatibility
            type_compatibility = self._calculate_type_compatibility(pattern1, pattern2)

            # Calculate overall relationship strength
            relationship_strength = self._calculate_relationship_strength(
                temporal_overlap,
                detector_overlap,
                confidence_correlation,
                relevance_alignment,
                type_compatibility,
            )

            if relationship_strength >= self.correlation_threshold:
                return {
                    "pattern1_id": pattern1.pattern_id,
                    "pattern2_id": pattern2.pattern_id,
                    "strength": relationship_strength,
                    "relationship_type": self._determine_relationship_type(
                        pattern1, pattern2, temporal_overlap, detector_overlap
                    ),
                    "components": {
                        "temporal_overlap": temporal_overlap,
                        "detector_overlap": detector_overlap,
                        "confidence_correlation": confidence_correlation,
                        "relevance_alignment": relevance_alignment,
                        "type_compatibility": type_compatibility,
                    },
                    "metadata": {
                        "pattern1_type": pattern1.pattern_type.value,
                        "pattern2_type": pattern2.pattern_type.value,
                        "pattern1_strength": pattern1.strength.value,
                        "pattern2_strength": pattern2.strength.value,
                    },
                }

        except Exception as e:
            logger.error("Pattern pair analysis failed", error=str(e))

        return None

    def _calculate_temporal_overlap(
        self, range1: TimeRange, range2: TimeRange
    ) -> float:
        """Calculate temporal overlap between two time ranges."""
        try:
            # Calculate overlap duration
            overlap_start = max(range1.start, range2.start)
            overlap_end = min(range1.end, range2.end)

            if overlap_start >= overlap_end:
                return 0.0

            overlap_duration = (overlap_end - overlap_start).total_seconds()

            # Calculate union duration
            union_start = min(range1.start, range2.start)
            union_end = max(range1.end, range2.end)
            union_duration = (union_end - union_start).total_seconds()

            if union_duration == 0:
                return 0.0

            return overlap_duration / union_duration

        except Exception:
            return 0.0

    def _calculate_detector_overlap(
        self, detectors1: List[str], detectors2: List[str]
    ) -> float:
        """Calculate detector overlap between two patterns."""
        try:
            set1 = set(detectors1)
            set2 = set(detectors2)

            intersection = len(set1 & set2)
            union = len(set1 | set2)

            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.0

    def _calculate_confidence_correlation(
        self, pattern1: Pattern, pattern2: Pattern
    ) -> float:
        """Calculate correlation between pattern confidences."""
        try:
            # Simple correlation based on confidence similarity
            conf_diff = abs(pattern1.confidence - pattern2.confidence)
            return 1.0 - conf_diff  # Higher similarity = higher correlation

        except Exception:
            return 0.0

    def _calculate_relevance_alignment(
        self, pattern1: Pattern, pattern2: Pattern
    ) -> float:
        """Calculate business relevance alignment."""
        try:
            relevance_scores = {
                BusinessRelevance.LOW: 1,
                BusinessRelevance.MEDIUM: 2,
                BusinessRelevance.HIGH: 3,
                BusinessRelevance.CRITICAL: 4,
            }

            score1 = relevance_scores.get(pattern1.business_relevance, 1)
            score2 = relevance_scores.get(pattern2.business_relevance, 1)

            # Calculate alignment based on score similarity
            max_diff = 3  # Maximum possible difference
            actual_diff = abs(score1 - score2)

            return 1.0 - (actual_diff / max_diff)

        except Exception:
            return 0.0

    def _calculate_type_compatibility(
        self, pattern1: Pattern, pattern2: Pattern
    ) -> float:
        """Calculate pattern type compatibility."""
        try:
            # Define compatibility matrix for pattern types
            compatibility_matrix = {
                (PatternType.TEMPORAL, PatternType.FREQUENCY): 0.8,
                (PatternType.TEMPORAL, PatternType.CORRELATION): 0.6,
                (PatternType.TEMPORAL, PatternType.ANOMALY): 0.4,
                (PatternType.FREQUENCY, PatternType.CORRELATION): 0.7,
                (PatternType.FREQUENCY, PatternType.ANOMALY): 0.5,
                (PatternType.CORRELATION, PatternType.ANOMALY): 0.6,
            }

            type_pair = (pattern1.pattern_type, pattern2.pattern_type)
            reverse_pair = (pattern2.pattern_type, pattern1.pattern_type)

            # Same type patterns are highly compatible
            if pattern1.pattern_type == pattern2.pattern_type:
                return 1.0

            # Look up compatibility
            return compatibility_matrix.get(
                type_pair, compatibility_matrix.get(reverse_pair, 0.3)
            )

        except Exception:
            return 0.0

    def _calculate_relationship_strength(
        self,
        temporal_overlap: float,
        detector_overlap: float,
        confidence_correlation: float,
        relevance_alignment: float,
        type_compatibility: float,
    ) -> float:
        """Calculate overall relationship strength."""
        try:
            # Weight different components
            weights = {
                "temporal": 0.25,
                "detector": 0.25,
                "confidence": 0.2,
                "relevance": 0.15,
                "type": 0.15,
            }

            weighted_score = (
                weights["temporal"] * temporal_overlap
                + weights["detector"] * detector_overlap
                + weights["confidence"] * confidence_correlation
                + weights["relevance"] * relevance_alignment
                + weights["type"] * type_compatibility
            )

            return min(1.0, max(0.0, weighted_score))

        except Exception:
            return 0.0

    def _determine_relationship_type(
        self,
        pattern1: Pattern,
        pattern2: Pattern,
        temporal_overlap: float,
        detector_overlap: float,
    ) -> str:
        """Determine the type of relationship between patterns."""
        try:
            if detector_overlap > 0.8:
                return "detector_based"
            elif temporal_overlap > 0.8:
                return "temporal_based"
            elif detector_overlap > 0.5 and temporal_overlap > 0.5:
                return "compound"
            elif pattern1.pattern_type == pattern2.pattern_type:
                return "type_based"
            else:
                return "weak_correlation"

        except Exception:
            return "unknown"

    async def _identify_pattern_clusters(
        self, patterns: List[Pattern], relationships: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """Identify clusters of related patterns."""
        clusters = []

        try:
            # Build adjacency list from relationships
            adjacency = defaultdict(set)
            for rel in relationships:
                pattern1_id = rel["pattern1_id"]
                pattern2_id = rel["pattern2_id"]
                adjacency[pattern1_id].add(pattern2_id)
                adjacency[pattern2_id].add(pattern1_id)

            # Find connected components (clusters)
            visited = set()
            pattern_id_map = {p.pattern_id: p for p in patterns}

            for pattern in patterns:
                if pattern.pattern_id not in visited:
                    cluster = self._find_connected_component(
                        pattern.pattern_id, adjacency, visited
                    )
                    if len(cluster) >= 2:  # Only clusters with 2+ patterns
                        cluster_patterns = [
                            pattern_id_map[pid]
                            for pid in cluster
                            if pid in pattern_id_map
                        ]
                        cluster_info = self._analyze_cluster(
                            cluster_patterns, relationships
                        )
                        clusters.append(cluster_info)

        except Exception as e:
            logger.error("Pattern clustering failed", error=str(e))

        return clusters

    def _find_connected_component(
        self, start_id: str, adjacency: Dict[str, Set[str]], visited: Set[str]
    ) -> Set[str]:
        """Find connected component using DFS."""
        component = set()
        stack = [start_id]

        while stack:
            current_id = stack.pop()
            if current_id not in visited:
                visited.add(current_id)
                component.add(current_id)

                # Add unvisited neighbors to stack
                for neighbor_id in adjacency.get(current_id, set()):
                    if neighbor_id not in visited:
                        stack.append(neighbor_id)

        return component

    def _analyze_cluster(
        self, cluster_patterns: List[Pattern], relationships: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """Analyze characteristics of a pattern cluster."""
        try:
            cluster_ids = {p.pattern_id for p in cluster_patterns}

            # Find relationships within cluster
            internal_relationships = [
                rel
                for rel in relationships
                if rel["pattern1_id"] in cluster_ids
                and rel["pattern2_id"] in cluster_ids
            ]

            # Calculate cluster metrics
            avg_confidence = sum(p.confidence for p in cluster_patterns) / len(
                cluster_patterns
            )
            dominant_type = self._find_dominant_pattern_type(cluster_patterns)
            avg_strength = sum(
                self._strength_to_numeric(p.strength) for p in cluster_patterns
            ) / len(cluster_patterns)

            # Determine cluster characteristics
            cluster_characteristics = self._determine_cluster_characteristics(
                cluster_patterns
            )

            return {
                "cluster_id": f"cluster_{hash(tuple(sorted(cluster_ids))) % 10000}",
                "pattern_ids": list(cluster_ids),
                "size": len(cluster_patterns),
                "internal_relationships": len(internal_relationships),
                "avg_confidence": avg_confidence,
                "dominant_type": dominant_type.value if dominant_type else "mixed",
                "avg_strength": avg_strength,
                "characteristics": cluster_characteristics,
                "business_impact": self._assess_cluster_business_impact(
                    cluster_patterns
                ),
            }

        except Exception as e:
            logger.error("Cluster analysis failed", error=str(e))
            return {}

    def _find_dominant_pattern_type(
        self, patterns: List[Pattern]
    ) -> Optional[PatternType]:
        """Find the dominant pattern type in a cluster."""
        type_counts = defaultdict(int)
        for pattern in patterns:
            type_counts[pattern.pattern_type] += 1

        if not type_counts:
            return None

        return max(type_counts.items(), key=lambda x: x[1])[0]

    def _strength_to_numeric(self, strength: PatternStrength) -> float:
        """Convert pattern strength to numeric value."""
        strength_values = {
            PatternStrength.WEAK: 1.0,
            PatternStrength.MODERATE: 2.0,
            PatternStrength.STRONG: 3.0,
        }
        return strength_values.get(strength, 1.0)

    def _determine_cluster_characteristics(self, patterns: List[Pattern]) -> List[str]:
        """Determine characteristics of a pattern cluster."""
        characteristics = []

        # Check for high confidence cluster
        avg_confidence = sum(p.confidence for p in patterns) / len(patterns)
        if avg_confidence > 0.8:
            characteristics.append("high_confidence")

        # Check for multi-detector cluster
        all_detectors = set()
        for pattern in patterns:
            all_detectors.update(pattern.affected_detectors)
        if len(all_detectors) >= 3:
            characteristics.append("multi_detector")

        # Check for high business relevance
        high_relevance_count = sum(
            1
            for p in patterns
            if p.business_relevance
            in [BusinessRelevance.HIGH, BusinessRelevance.CRITICAL]
        )
        if high_relevance_count >= len(patterns) * 0.5:
            characteristics.append("high_business_impact")

        # Check for temporal clustering
        time_ranges = [p.time_range for p in patterns]
        if self._has_temporal_clustering(time_ranges):
            characteristics.append("temporally_clustered")

        return characteristics

    def _has_temporal_clustering(self, time_ranges: List[TimeRange]) -> bool:
        """Check if time ranges show temporal clustering."""
        try:
            if len(time_ranges) < 2:
                return False

            # Calculate average overlap
            total_overlap = 0
            comparisons = 0

            for i, range1 in enumerate(time_ranges):
                for range2 in time_ranges[i + 1 :]:
                    overlap = self._calculate_temporal_overlap(range1, range2)
                    total_overlap += overlap
                    comparisons += 1

            avg_overlap = total_overlap / comparisons if comparisons > 0 else 0
            return avg_overlap > 0.5

        except Exception:
            return False

    def _assess_cluster_business_impact(self, patterns: List[Pattern]) -> str:
        """Assess business impact of a pattern cluster."""
        try:
            relevance_scores = {
                BusinessRelevance.LOW: 1,
                BusinessRelevance.MEDIUM: 2,
                BusinessRelevance.HIGH: 3,
                BusinessRelevance.CRITICAL: 4,
            }

            total_score = sum(
                relevance_scores.get(p.business_relevance, 1) for p in patterns
            )
            avg_score = total_score / len(patterns)

            if avg_score >= 3.5:
                return "critical"
            elif avg_score >= 2.5:
                return "high"
            elif avg_score >= 1.5:
                return "medium"
            else:
                return "low"

        except Exception:
            return "unknown"

    async def _detect_pattern_hierarchies(
        self, patterns: List[Pattern], relationships: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """Detect hierarchical relationships between patterns."""
        hierarchies = []

        try:
            # Look for parent-child relationships based on scope and strength
            for pattern in patterns:
                children = self._find_child_patterns(pattern, patterns, relationships)
                if children:
                    hierarchy = {
                        "parent_id": pattern.pattern_id,
                        "parent_type": pattern.pattern_type.value,
                        "children": [
                            {
                                "child_id": child.pattern_id,
                                "child_type": child.pattern_type.value,
                                "relationship_strength": self._get_relationship_strength(
                                    pattern.pattern_id, child.pattern_id, relationships
                                ),
                            }
                            for child in children
                        ],
                        "hierarchy_type": self._determine_hierarchy_type(
                            pattern, children
                        ),
                    }
                    hierarchies.append(hierarchy)

        except Exception as e:
            logger.error("Pattern hierarchy detection failed", error=str(e))

        return hierarchies

    def _find_child_patterns(
        self,
        parent: Pattern,
        all_patterns: List[Pattern],
        relationships: List[Dict[str, any]],
    ) -> List[Pattern]:
        """Find child patterns for a given parent pattern."""
        children = []

        try:
            for pattern in all_patterns:
                if pattern.pattern_id == parent.pattern_id:
                    continue

                # Check if this pattern could be a child
                if self._is_child_pattern(parent, pattern, relationships):
                    children.append(pattern)

        except Exception as e:
            logger.error("Child pattern detection failed", error=str(e))

        return children

    def _is_child_pattern(
        self, parent: Pattern, candidate: Pattern, relationships: List[Dict[str, any]]
    ) -> bool:
        """Determine if candidate is a child of parent pattern."""
        try:
            # Check if there's a relationship between them
            relationship_strength = self._get_relationship_strength(
                parent.pattern_id, candidate.pattern_id, relationships
            )

            if relationship_strength < 0.5:
                return False

            # Parent should have broader scope (more detectors)
            if len(parent.affected_detectors) <= len(candidate.affected_detectors):
                return False

            # Parent should have higher or equal confidence
            if parent.confidence < candidate.confidence - 0.1:
                return False

            # Check detector subset relationship
            parent_detectors = set(parent.affected_detectors)
            candidate_detectors = set(candidate.affected_detectors)

            # Candidate detectors should be subset of parent detectors
            return candidate_detectors.issubset(parent_detectors)

        except Exception:
            return False

    def _get_relationship_strength(
        self, pattern1_id: str, pattern2_id: str, relationships: List[Dict[str, any]]
    ) -> float:
        """Get relationship strength between two patterns."""
        for rel in relationships:
            if (
                rel["pattern1_id"] == pattern1_id and rel["pattern2_id"] == pattern2_id
            ) or (
                rel["pattern1_id"] == pattern2_id and rel["pattern2_id"] == pattern1_id
            ):
                return rel["strength"]
        return 0.0

    def _determine_hierarchy_type(
        self, parent: Pattern, children: List[Pattern]
    ) -> str:
        """Determine the type of hierarchy."""
        try:
            if parent.pattern_type == PatternType.CORRELATION:
                return "correlation_hierarchy"
            elif parent.pattern_type == PatternType.TEMPORAL:
                return "temporal_hierarchy"
            elif len(set(child.pattern_type for child in children)) == 1:
                return "homogeneous_hierarchy"
            else:
                return "heterogeneous_hierarchy"

        except Exception:
            return "unknown_hierarchy"

    async def _analyze_temporal_relationships(
        self, patterns: List[Pattern]
    ) -> List[Dict[str, any]]:
        """Analyze temporal relationships between patterns."""
        temporal_relationships = []

        try:
            # Sort patterns by start time
            sorted_patterns = sorted(patterns, key=lambda p: p.time_range.start)

            for i, pattern1 in enumerate(sorted_patterns):
                for pattern2 in sorted_patterns[i + 1 :]:
                    temporal_rel = self._analyze_temporal_relationship(
                        pattern1, pattern2
                    )
                    if temporal_rel:
                        temporal_relationships.append(temporal_rel)

        except Exception as e:
            logger.error("Temporal relationship analysis failed", error=str(e))

        return temporal_relationships

    def _analyze_temporal_relationship(
        self, pattern1: Pattern, pattern2: Pattern
    ) -> Optional[Dict[str, any]]:
        """Analyze temporal relationship between two patterns."""
        try:
            # Calculate time gap
            gap = (pattern2.time_range.start - pattern1.time_range.end).total_seconds()

            # Determine relationship type
            if gap < 0:  # Overlapping
                overlap_duration = min(
                    pattern1.time_range.end, pattern2.time_range.end
                ) - max(pattern1.time_range.start, pattern2.time_range.start)
                relationship_type = "overlapping"
                strength = min(
                    1.0, overlap_duration.total_seconds() / 3600
                )  # Normalize by hour
            elif gap <= 3600:  # Within 1 hour
                relationship_type = "sequential_immediate"
                strength = 1.0 - (gap / 3600)
            elif gap <= 24 * 3600:  # Within 24 hours
                relationship_type = "sequential_delayed"
                strength = 1.0 - (gap / (24 * 3600))
            else:
                return None  # Too far apart

            if strength > 0.3:  # Only return significant relationships
                return {
                    "pattern1_id": pattern1.pattern_id,
                    "pattern2_id": pattern2.pattern_id,
                    "relationship_type": relationship_type,
                    "strength": strength,
                    "time_gap_seconds": gap,
                    "temporal_order": "pattern1_first",
                }

        except Exception as e:
            logger.error("Temporal relationship analysis failed", error=str(e))

        return None

    async def _detect_causal_relationships(
        self, patterns: List[Pattern], context_data: SecurityData
    ) -> List[Dict[str, any]]:
        """Detect potential causal relationships between patterns."""
        causal_relationships = []

        try:
            # Simple causal detection based on temporal precedence and detector relationships
            for i, pattern1 in enumerate(patterns):
                for j, pattern2 in enumerate(patterns):
                    if i != j:
                        causal_rel = self._analyze_causal_relationship(
                            pattern1, pattern2, context_data
                        )
                        if causal_rel:
                            causal_relationships.append(causal_rel)

        except Exception as e:
            logger.error("Causal relationship detection failed", error=str(e))

        return causal_relationships

    def _analyze_causal_relationship(
        self, pattern1: Pattern, pattern2: Pattern, context_data: SecurityData
    ) -> Optional[Dict[str, any]]:
        """Analyze potential causal relationship between two patterns."""
        try:
            # Pattern1 must precede pattern2 temporally
            if pattern1.time_range.start >= pattern2.time_range.start:
                return None

            # Check for detector relationship (upstream/downstream)
            detector_relationship = self._analyze_detector_causality(
                pattern1.affected_detectors, pattern2.affected_detectors
            )

            if not detector_relationship:
                return None

            # Calculate causal strength based on temporal proximity and detector relationship
            time_gap = (
                pattern2.time_range.start - pattern1.time_range.end
            ).total_seconds()
            temporal_strength = max(
                0, 1.0 - (time_gap / (24 * 3600))
            )  # Decay over 24 hours

            causal_strength = (
                temporal_strength + detector_relationship["strength"]
            ) / 2

            if causal_strength > 0.5:
                return {
                    "cause_pattern_id": pattern1.pattern_id,
                    "effect_pattern_id": pattern2.pattern_id,
                    "causal_strength": causal_strength,
                    "causal_type": detector_relationship["type"],
                    "time_gap_seconds": time_gap,
                    "confidence": min(pattern1.confidence, pattern2.confidence),
                }

        except Exception as e:
            logger.error("Causal relationship analysis failed", error=str(e))

        return None

    def _analyze_detector_causality(
        self, detectors1: List[str], detectors2: List[str]
    ) -> Optional[Dict[str, any]]:
        """Analyze potential causality between detector sets."""
        try:
            # Define known causal relationships between detectors
            causal_chains = {
                "presidio": ["pii-detector", "gdpr-scanner"],
                "pii-detector": ["gdpr-scanner", "hipaa-detector"],
                "network-scanner": ["vulnerability-detector", "threat-detector"],
                "file-scanner": ["malware-detector", "content-detector"],
            }

            # Check for causal chains
            for detector1 in detectors1:
                downstream_detectors = causal_chains.get(detector1.lower(), [])
                for detector2 in detectors2:
                    if detector2.lower() in downstream_detectors:
                        return {
                            "type": "detector_chain",
                            "strength": 0.8,
                            "upstream": detector1,
                            "downstream": detector2,
                        }

            # Check for detector overlap (weaker causality)
            overlap = set(d.lower() for d in detectors1) & set(
                d.lower() for d in detectors2
            )
            if overlap:
                return {
                    "type": "detector_overlap",
                    "strength": 0.6,
                    "overlapping_detectors": list(overlap),
                }

        except Exception:
            pass

        return None

    def _build_relationship_matrix(
        self, patterns: List[Pattern], relationships: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """Build relationship strength matrix for patterns."""
        try:
            pattern_ids = [p.pattern_id for p in patterns]
            matrix_size = len(pattern_ids)

            # Initialize matrix
            matrix = [[0.0 for _ in range(matrix_size)] for _ in range(matrix_size)]

            # Fill matrix with relationship strengths
            id_to_index = {pid: i for i, pid in enumerate(pattern_ids)}

            for rel in relationships:
                i = id_to_index.get(rel["pattern1_id"])
                j = id_to_index.get(rel["pattern2_id"])

                if i is not None and j is not None:
                    strength = rel["strength"]
                    matrix[i][j] = strength
                    matrix[j][i] = strength  # Symmetric matrix

            return {
                "pattern_ids": pattern_ids,
                "matrix": matrix,
                "size": matrix_size,
                "max_strength": max(max(row) for row in matrix) if matrix else 0.0,
                "avg_strength": (
                    sum(sum(row) for row in matrix) / (matrix_size * matrix_size)
                    if matrix_size > 0
                    else 0.0
                ),
            }

        except Exception as e:
            logger.error("Relationship matrix building failed", error=str(e))
            return {"pattern_ids": [], "matrix": [], "size": 0}

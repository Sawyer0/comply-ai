"""
Risk factor analysis and optimization engine.

This module provides analysis of risk factors to identify
optimization opportunities and risk mitigation strategies.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RiskFactorAnalysis:
    """Risk factor analysis result."""

    factor_name: str
    importance_score: float
    optimization_potential: float
    recommendations: List[str]


class RiskFactorAnalyzer:
    """
    Risk factor analysis and optimization engine.

    Analyzes risk factors to identify the most impactful factors
    and provides optimization recommendations.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.analysis_methods = self._initialize_analysis_methods()

    def _initialize_analysis_methods(self) -> Dict[str, Any]:
        """Initialize analysis methods and configurations."""
        return {
            "importance_analysis": {
                "method": "weighted_contribution",
                "threshold": 0.1,
            },
            "correlation_analysis": {
                "method": "pearson_correlation",
                "threshold": 0.7,
            },
            "optimization_analysis": {
                "method": "impact_effort_matrix",
                "threshold": 0.5,
            },
        }

    async def analyze_risk_factors(
        self,
        risk_data: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]] = None,
    ) -> List[RiskFactorAnalysis]:
        """Analyze risk factors for optimization opportunities."""
        try:
            analyses = []

            # Extract risk factors from data
            risk_factors = self._extract_risk_factors(risk_data)

            for factor_name, factor_data in risk_factors.items():
                analysis = await self._analyze_single_factor(
                    factor_name, factor_data, historical_data
                )
                analyses.append(analysis)

            # Sort by importance score
            analyses.sort(key=lambda x: x.importance_score, reverse=True)

            return analyses

        except Exception as e:
            logger.error("Risk factor analysis failed", error=str(e))
            return []

    async def _analyze_single_factor(
        self,
        factor_name: str,
        factor_data: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]] = None,
    ) -> RiskFactorAnalysis:
        """Analyze a single risk factor."""

        # Calculate importance score
        importance_score = self._calculate_importance_score(factor_data)

        # Calculate optimization potential
        optimization_potential = self._calculate_optimization_potential(
            factor_data, historical_data
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            factor_name, factor_data, importance_score, optimization_potential
        )

        return RiskFactorAnalysis(
            factor_name=factor_name,
            importance_score=importance_score,
            optimization_potential=optimization_potential,
            recommendations=recommendations,
        )

    def _extract_risk_factors(
        self, risk_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Extract individual risk factors from risk data."""
        factors = {}

        # Extract from different risk dimensions
        for dimension in ["technical", "business", "regulatory", "temporal"]:
            dimension_data = risk_data.get(dimension, {})
            if isinstance(dimension_data, dict):
                components = dimension_data.get("components", {})
                for component_name, component_value in components.items():
                    factors[f"{dimension}_{component_name}"] = {
                        "value": component_value,
                        "dimension": dimension,
                        "component": component_name,
                    }

        return factors

    def _calculate_importance_score(self, factor_data: Dict[str, Any]) -> float:
        """Calculate importance score for a risk factor."""
        base_value = factor_data.get("value", 0.0)

        # Adjust based on dimension importance
        dimension_weights = {
            "technical": 1.2,
            "regulatory": 1.1,
            "business": 1.0,
            "temporal": 0.9,
        }

        dimension = factor_data.get("dimension", "technical")
        weight = dimension_weights.get(dimension, 1.0)

        importance_score = min(1.0, base_value * weight)
        return importance_score

    def _calculate_optimization_potential(
        self,
        factor_data: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """Calculate optimization potential for a risk factor."""
        current_value = factor_data.get("value", 0.0)

        # High current value means high optimization potential
        base_potential = current_value

        # Adjust based on historical trends if available
        if historical_data:
            # Simple trend analysis - would be more sophisticated in production
            recent_values = [
                data.get(factor_data.get("component", ""), 0.0)
                for data in historical_data[-5:]  # Last 5 data points
            ]

            if recent_values:
                trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
                if trend > 0:  # Increasing trend means higher optimization potential
                    base_potential *= 1.2
                else:  # Decreasing trend means lower optimization potential
                    base_potential *= 0.8

        return min(1.0, base_potential)

    def _generate_recommendations(
        self,
        factor_name: str,
        factor_data: Dict[str, Any],
        importance_score: float,
        optimization_potential: float,
    ) -> List[str]:
        """Generate optimization recommendations for a risk factor."""
        recommendations = []

        dimension = factor_data.get("dimension", "")
        component = factor_data.get("component", "")

        # High importance, high potential - priority recommendations
        if importance_score > 0.7 and optimization_potential > 0.7:
            recommendations.append(
                f"High priority: Optimize {component} in {dimension} risk assessment"
            )
            recommendations.append("Consider immediate mitigation strategies")

        # High importance, low potential - monitoring recommendations
        elif importance_score > 0.7 and optimization_potential <= 0.3:
            recommendations.append(f"Monitor {component} closely for changes")
            recommendations.append("Establish baseline metrics and thresholds")

        # Low importance, high potential - efficiency recommendations
        elif importance_score <= 0.3 and optimization_potential > 0.7:
            recommendations.append(
                f"Consider optimizing {component} for efficiency gains"
            )
            recommendations.append("Low priority but high potential impact")

        # General recommendations based on component type
        component_recommendations = {
            "severity_risk": [
                "Review severity classification criteria",
                "Implement automated severity detection",
            ],
            "error_risk": [
                "Improve error handling and recovery",
                "Add comprehensive error monitoring",
            ],
            "coverage_risk": [
                "Expand detection coverage",
                "Implement coverage gap analysis",
            ],
            "data_sensitivity": [
                "Enhance data classification",
                "Implement data loss prevention",
            ],
            "process_impact": [
                "Streamline affected processes",
                "Add process resilience measures",
            ],
        }

        if component in component_recommendations:
            recommendations.extend(component_recommendations[component])

        return recommendations[:3]  # Limit to top 3 recommendations

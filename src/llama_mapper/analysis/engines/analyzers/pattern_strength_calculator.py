"""
Pattern Strength Calculator using statistical significance testing.

This module implements sophisticated statistical methods to calculate
pattern strength and significance using various statistical tests.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from ...domain import (
    Pattern,
    PatternType,
    PatternStrength,
    SecurityData,
)

logger = logging.getLogger(__name__)


class PatternStrengthCalculator:
    """
    Calculates pattern strength using statistical significance testing.

    Uses various statistical tests and methods to determine the strength
    and significance of detected patterns in security data.
    """

    def __init__(self, config: Dict[str, any] = None):
        self.config = config or {}
        self.significance_level = self.config.get("significance_level", 0.05)
        self.min_sample_size = self.config.get("min_sample_size", 10)
        self.strength_thresholds = self._load_strength_thresholds()

    def calculate_pattern_strength(
        self, pattern: Pattern, context_data: SecurityData
    ) -> PatternStrength:
        """
        Calculate the statistical strength of a pattern.

        Args:
            pattern: The pattern to analyze
            context_data: Additional security data for context

        Returns:
            PatternStrength classification based on statistical analysis
        """
        try:
            # Extract relevant data for statistical analysis
            pattern_data = self._extract_pattern_data(pattern, context_data)

            if not pattern_data or len(pattern_data) < self.min_sample_size:
                logger.warning(
                    "Insufficient data for strength calculation",
                    pattern_id=pattern.pattern_id,
                    data_points=len(pattern_data) if pattern_data else 0,
                )
                return PatternStrength.WEAK

            # Calculate statistical metrics based on pattern type
            statistical_metrics = self._calculate_statistical_metrics(
                pattern, pattern_data
            )

            # Perform significance tests
            significance_results = self._perform_significance_tests(
                pattern, pattern_data, statistical_metrics
            )

            # Calculate composite strength score
            strength_score = self._calculate_composite_strength_score(
                pattern, statistical_metrics, significance_results
            )

            # Classify strength based on score
            strength_classification = self._classify_strength(strength_score)

            # Update pattern with detailed strength analysis
            self._add_strength_evidence(
                pattern, statistical_metrics, significance_results, strength_score
            )

            logger.info(
                "Pattern strength calculated",
                pattern_id=pattern.pattern_id,
                strength=strength_classification.value,
                strength_score=strength_score,
            )

            return strength_classification

        except Exception as e:
            logger.error(
                "Pattern strength calculation failed",
                error=str(e),
                pattern_id=pattern.pattern_id,
            )
            return PatternStrength.WEAK

    def calculate_multiple_pattern_strengths(
        self, patterns: List[Pattern], context_data: SecurityData
    ) -> Dict[str, PatternStrength]:
        """
        Calculate strengths for multiple patterns with comparative analysis.

        Args:
            patterns: List of patterns to analyze
            context_data: Security data for context

        Returns:
            Dictionary mapping pattern IDs to their strength classifications
        """
        strength_results = {}
        pattern_scores = {}

        # Calculate individual strengths
        for pattern in patterns:
            strength = self.calculate_pattern_strength(pattern, context_data)
            strength_results[pattern.pattern_id] = strength

            # Extract strength score for comparative analysis
            for evidence in pattern.supporting_evidence:
                if isinstance(evidence, dict) and "strength_analysis" in evidence:
                    pattern_scores[pattern.pattern_id] = evidence["strength_analysis"][
                        "composite_score"
                    ]
                    break

        # Perform comparative analysis
        self._perform_comparative_strength_analysis(
            patterns, pattern_scores, strength_results
        )

        return strength_results

    def _extract_pattern_data(
        self, pattern: Pattern, context_data: SecurityData
    ) -> List[float]:
        """Extract numerical data relevant to the pattern for analysis."""
        pattern_data = []

        try:
            if pattern.pattern_type == PatternType.TEMPORAL:
                # Extract time-series values
                for item in context_data.time_series:
                    if item.get("detector") in pattern.affected_detectors:
                        value = item.get("value")
                        if isinstance(value, (int, float)):
                            pattern_data.append(float(value))

            elif pattern.pattern_type == PatternType.FREQUENCY:
                # Extract event frequencies or intervals
                for evidence in pattern.supporting_evidence:
                    if isinstance(evidence, dict):
                        if "average_interval" in evidence:
                            pattern_data.append(evidence["average_interval"])
                        elif "event_count" in evidence:
                            pattern_data.append(evidence["event_count"])

            elif pattern.pattern_type == PatternType.CORRELATION:
                # Extract correlation coefficients or related values
                for evidence in pattern.supporting_evidence:
                    if (
                        isinstance(evidence, dict)
                        and "correlation_coefficient" in evidence
                    ):
                        pattern_data.append(abs(evidence["correlation_coefficient"]))

            elif pattern.pattern_type == PatternType.ANOMALY:
                # Extract anomaly scores or outlier values
                for item in context_data.metrics:
                    if item.get("type") in [
                        det.lower().replace("-", "_")
                        for det in pattern.affected_detectors
                    ]:
                        value = item.get("value")
                        if isinstance(value, (int, float)):
                            pattern_data.append(float(value))

        except Exception as e:
            logger.error(
                "Failed to extract pattern data",
                error=str(e),
                pattern_id=pattern.pattern_id,
            )

        return pattern_data

    def _calculate_statistical_metrics(
        self, pattern: Pattern, data: List[float]
    ) -> Dict[str, float]:
        """Calculate statistical metrics for the pattern data."""
        if not data:
            return {}

        try:
            data_array = np.array(data)

            metrics = {
                "mean": float(np.mean(data_array)),
                "median": float(np.median(data_array)),
                "std_dev": float(np.std(data_array)),
                "variance": float(np.var(data_array)),
                "skewness": float(self._calculate_skewness(data_array)),
                "kurtosis": float(self._calculate_kurtosis(data_array)),
                "coefficient_of_variation": (
                    float(np.std(data_array) / np.mean(data_array))
                    if np.mean(data_array) != 0
                    else 0
                ),
                "sample_size": len(data),
                "min_value": float(np.min(data_array)),
                "max_value": float(np.max(data_array)),
                "range": float(np.max(data_array) - np.min(data_array)),
                "q1": float(np.percentile(data_array, 25)),
                "q3": float(np.percentile(data_array, 75)),
                "iqr": float(
                    np.percentile(data_array, 75) - np.percentile(data_array, 25)
                ),
            }

            return metrics

        except Exception as e:
            logger.error("Failed to calculate statistical metrics", error=str(e))
            return {}

    def _perform_significance_tests(
        self, pattern: Pattern, data: List[float], metrics: Dict[str, float]
    ) -> Dict[str, any]:
        """Perform statistical significance tests based on pattern type."""
        if not data or len(data) < self.min_sample_size:
            return {"insufficient_data": True}

        results = {}

        try:
            data_array = np.array(data)

            # Normality test
            if len(data) >= 8:  # Minimum for Shapiro-Wilk test
                shapiro_stat, shapiro_p = stats.shapiro(data_array)
                results["normality_test"] = {
                    "statistic": float(shapiro_stat),
                    "p_value": float(shapiro_p),
                    "is_normal": shapiro_p > self.significance_level,
                }

            # Pattern-specific tests
            if pattern.pattern_type == PatternType.TEMPORAL:
                results.update(self._temporal_significance_tests(data_array, metrics))

            elif pattern.pattern_type == PatternType.FREQUENCY:
                results.update(self._frequency_significance_tests(data_array, metrics))

            elif pattern.pattern_type == PatternType.CORRELATION:
                results.update(
                    self._correlation_significance_tests(data_array, metrics)
                )

            elif pattern.pattern_type == PatternType.ANOMALY:
                results.update(self._anomaly_significance_tests(data_array, metrics))

            # General statistical tests
            results.update(self._general_significance_tests(data_array, metrics))

        except Exception as e:
            logger.error("Significance tests failed", error=str(e))
            results["error"] = str(e)

        return results

    def _temporal_significance_tests(
        self, data: np.ndarray, metrics: Dict[str, float]
    ) -> Dict[str, any]:
        """Perform significance tests specific to temporal patterns."""
        results = {}

        try:
            # Trend test using Mann-Kendall test
            if len(data) >= 4:
                # Simple trend test: correlation with time indices
                time_indices = np.arange(len(data))
                correlation, p_value = stats.pearsonr(time_indices, data)

                results["trend_test"] = {
                    "correlation": float(correlation),
                    "p_value": float(p_value),
                    "significant_trend": p_value < self.significance_level,
                    "trend_direction": (
                        "increasing" if correlation > 0 else "decreasing"
                    ),
                }

            # Stationarity test (simplified)
            if len(data) >= 10:
                # Split data into two halves and compare means
                mid_point = len(data) // 2
                first_half = data[:mid_point]
                second_half = data[mid_point:]

                t_stat, p_value = stats.ttest_ind(first_half, second_half)

                results["stationarity_test"] = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "is_stationary": p_value > self.significance_level,
                }

        except Exception as e:
            logger.error("Temporal significance tests failed", error=str(e))

        return results

    def _frequency_significance_tests(
        self, data: np.ndarray, metrics: Dict[str, float]
    ) -> Dict[str, any]:
        """Perform significance tests specific to frequency patterns."""
        results = {}

        try:
            # Test for regularity in frequency data
            if len(data) >= 5:
                # Calculate coefficient of variation as regularity measure
                cv = metrics.get("coefficient_of_variation", 0)

                results["regularity_test"] = {
                    "coefficient_of_variation": cv,
                    "is_regular": cv < 0.3,  # Low CV indicates regularity
                    "regularity_strength": max(0, 1 - cv),
                }

            # Chi-square goodness of fit test for uniform distribution
            if len(data) >= 5:
                # Test if intervals follow expected distribution
                expected_freq = len(data) / 5  # Assume 5 bins
                observed_freq, _ = np.histogram(data, bins=5)

                if expected_freq > 0:
                    chi2_stat, p_value = stats.chisquare(observed_freq)

                    results["uniformity_test"] = {
                        "chi2_statistic": float(chi2_stat),
                        "p_value": float(p_value),
                        "is_uniform": p_value > self.significance_level,
                    }

        except Exception as e:
            logger.error("Frequency significance tests failed", error=str(e))

        return results

    def _correlation_significance_tests(
        self, data: np.ndarray, metrics: Dict[str, float]
    ) -> Dict[str, any]:
        """Perform significance tests specific to correlation patterns."""
        results = {}

        try:
            # Test correlation strength significance
            if len(data) >= 3:
                # Use the correlation values themselves
                mean_correlation = np.mean(data)

                # Test if correlation is significantly different from zero
                # Using t-test for correlation coefficient
                n = len(data)
                t_stat = mean_correlation * math.sqrt(
                    (n - 2) / (1 - mean_correlation**2)
                )
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

                results["correlation_significance_test"] = {
                    "mean_correlation": float(mean_correlation),
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant_correlation": p_value < self.significance_level,
                }

        except Exception as e:
            logger.error("Correlation significance tests failed", error=str(e))

        return results

    def _anomaly_significance_tests(
        self, data: np.ndarray, metrics: Dict[str, float]
    ) -> Dict[str, any]:
        """Perform significance tests specific to anomaly patterns."""
        results = {}

        try:
            # Outlier detection using IQR method
            q1 = metrics.get("q1", 0)
            q3 = metrics.get("q3", 0)
            iqr = metrics.get("iqr", 0)

            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outliers = data[(data < lower_bound) | (data > upper_bound)]
                outlier_ratio = len(outliers) / len(data)

                results["outlier_test"] = {
                    "outlier_count": len(outliers),
                    "outlier_ratio": float(outlier_ratio),
                    "significant_anomalies": outlier_ratio
                    > 0.05,  # More than 5% outliers
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                }

            # Grubbs test for outliers (if sample size is appropriate)
            if 7 <= len(data) <= 25:
                mean_val = np.mean(data)
                std_val = np.std(data, ddof=1)

                if std_val > 0:
                    # Calculate Grubbs statistic for maximum deviation
                    max_deviation = np.max(np.abs(data - mean_val))
                    grubbs_stat = max_deviation / std_val

                    results["grubbs_test"] = {
                        "grubbs_statistic": float(grubbs_stat),
                        "max_deviation": float(max_deviation),
                        "has_outlier": grubbs_stat > 2.0,  # Simplified threshold
                    }

        except Exception as e:
            logger.error("Anomaly significance tests failed", error=str(e))

        return results

    def _general_significance_tests(
        self, data: np.ndarray, metrics: Dict[str, float]
    ) -> Dict[str, any]:
        """Perform general statistical significance tests."""
        results = {}

        try:
            # One-sample t-test against zero
            if len(data) >= 2:
                t_stat, p_value = stats.ttest_1samp(data, 0)

                results["one_sample_t_test"] = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significantly_different_from_zero": p_value
                    < self.significance_level,
                }

            # Test for randomness (runs test approximation)
            if len(data) >= 10:
                median_val = np.median(data)
                runs = self._count_runs(data, median_val)
                expected_runs = (2 * len(data) - 1) / 3

                results["randomness_test"] = {
                    "runs_count": runs,
                    "expected_runs": float(expected_runs),
                    "is_random": abs(runs - expected_runs) < expected_runs * 0.5,
                }

        except Exception as e:
            logger.error("General significance tests failed", error=str(e))

        return results

    def _count_runs(self, data: np.ndarray, threshold: float) -> int:
        """Count runs above and below threshold."""
        above_threshold = data > threshold
        runs = 1

        for i in range(1, len(above_threshold)):
            if above_threshold[i] != above_threshold[i - 1]:
                runs += 1

        return runs

    def _calculate_composite_strength_score(
        self,
        pattern: Pattern,
        metrics: Dict[str, float],
        significance_results: Dict[str, any],
    ) -> float:
        """Calculate composite strength score from all analyses."""
        if not metrics or "insufficient_data" in significance_results:
            return 0.0

        try:
            # Base score from pattern confidence and statistical significance
            base_score = (pattern.confidence + pattern.statistical_significance) / 2

            # Adjust based on sample size
            sample_size = metrics.get("sample_size", 0)
            sample_size_factor = min(
                1.0, sample_size / 30
            )  # Full weight at 30+ samples

            # Adjust based on statistical tests
            significance_factor = self._calculate_significance_factor(
                significance_results
            )

            # Adjust based on pattern type specific metrics
            pattern_specific_factor = self._calculate_pattern_specific_factor(
                pattern, metrics, significance_results
            )

            # Combine factors
            composite_score = (
                base_score
                * sample_size_factor
                * significance_factor
                * pattern_specific_factor
            )

            return min(1.0, max(0.0, composite_score))

        except Exception as e:
            logger.error("Composite strength score calculation failed", error=str(e))
            return 0.0

    def _calculate_significance_factor(
        self, significance_results: Dict[str, any]
    ) -> float:
        """Calculate factor based on significance test results."""
        if "insufficient_data" in significance_results:
            return 0.5

        factor = 1.0
        significant_tests = 0
        total_tests = 0

        # Count significant results
        for test_name, test_result in significance_results.items():
            if isinstance(test_result, dict) and "p_value" in test_result:
                total_tests += 1
                if test_result["p_value"] < self.significance_level:
                    significant_tests += 1

        if total_tests > 0:
            significance_ratio = significant_tests / total_tests
            factor = 0.5 + (significance_ratio * 0.5)  # Range: 0.5 to 1.0

        return factor

    def _calculate_pattern_specific_factor(
        self,
        pattern: Pattern,
        metrics: Dict[str, float],
        significance_results: Dict[str, any],
    ) -> float:
        """Calculate pattern-type specific strength factor."""
        if pattern.pattern_type == PatternType.TEMPORAL:
            # Strong temporal patterns have significant trends
            trend_test = significance_results.get("trend_test", {})
            if trend_test.get("significant_trend", False):
                return 1.2  # Boost for significant trend

        elif pattern.pattern_type == PatternType.FREQUENCY:
            # Strong frequency patterns are regular
            regularity_test = significance_results.get("regularity_test", {})
            if regularity_test.get("is_regular", False):
                regularity_strength = regularity_test.get("regularity_strength", 0.5)
                return 0.8 + (regularity_strength * 0.4)  # Range: 0.8 to 1.2

        elif pattern.pattern_type == PatternType.CORRELATION:
            # Strong correlations have high significance
            corr_test = significance_results.get("correlation_significance_test", {})
            if corr_test.get("significant_correlation", False):
                mean_corr = abs(corr_test.get("mean_correlation", 0))
                return 0.8 + (mean_corr * 0.4)  # Range: 0.8 to 1.2

        elif pattern.pattern_type == PatternType.ANOMALY:
            # Strong anomalies have significant outliers
            outlier_test = significance_results.get("outlier_test", {})
            if outlier_test.get("significant_anomalies", False):
                outlier_ratio = outlier_test.get("outlier_ratio", 0)
                return 0.8 + min(0.4, outlier_ratio * 2)  # Range: 0.8 to 1.2

        return 1.0  # Default factor

    def _classify_strength(self, strength_score: float) -> PatternStrength:
        """Classify strength based on composite score."""
        thresholds = self.strength_thresholds

        if strength_score >= thresholds["strong"]:
            return PatternStrength.STRONG
        elif strength_score >= thresholds["moderate"]:
            return PatternStrength.MODERATE
        else:
            return PatternStrength.WEAK

    def _add_strength_evidence(
        self,
        pattern: Pattern,
        metrics: Dict[str, float],
        significance_results: Dict[str, any],
        strength_score: float,
    ):
        """Add detailed strength analysis to pattern evidence."""
        strength_evidence = {
            "strength_analysis": {
                "composite_score": strength_score,
                "statistical_metrics": metrics,
                "significance_tests": significance_results,
                "calculation_method": "composite_statistical_analysis",
                "confidence_level": 1 - self.significance_level,
            }
        }

        pattern.supporting_evidence.append(strength_evidence)

    def _perform_comparative_strength_analysis(
        self,
        patterns: List[Pattern],
        pattern_scores: Dict[str, float],
        strength_results: Dict[str, PatternStrength],
    ):
        """Perform comparative analysis across multiple patterns."""
        if not pattern_scores:
            return

        # Calculate relative strength rankings
        sorted_patterns = sorted(
            pattern_scores.items(), key=lambda x: x[1], reverse=True
        )

        for i, (pattern_id, score) in enumerate(sorted_patterns):
            # Find the pattern object
            pattern = next((p for p in patterns if p.pattern_id == pattern_id), None)
            if pattern:
                # Add comparative ranking to evidence
                comparative_evidence = {
                    "comparative_strength_analysis": {
                        "rank": i + 1,
                        "total_patterns": len(sorted_patterns),
                        "percentile": (len(sorted_patterns) - i) / len(sorted_patterns),
                        "relative_strength": (
                            "top_tier"
                            if i < len(sorted_patterns) * 0.3
                            else (
                                "middle_tier"
                                if i < len(sorted_patterns) * 0.7
                                else "lower_tier"
                            )
                        ),
                    }
                }
                pattern.supporting_evidence.append(comparative_evidence)

    def _load_strength_thresholds(self) -> Dict[str, float]:
        """Load strength classification thresholds."""
        return self.config.get(
            "strength_thresholds", {"weak": 0.0, "moderate": 0.6, "strong": 0.8}
        )

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness using numpy."""
        try:
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val == 0:
                return 0.0
            normalized = (data - mean_val) / std_val
            return np.mean(normalized**3)
        except Exception:
            return 0.0

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis using numpy."""
        try:
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val == 0:
                return 0.0
            normalized = (data - mean_val) / std_val
            return np.mean(normalized**4) - 3  # Excess kurtosis
        except Exception:
            return 0.0

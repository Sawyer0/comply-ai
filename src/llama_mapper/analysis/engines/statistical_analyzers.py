"""
Statistical analyzers for advanced pattern detection.

This module implements the core statistical analysis algorithms for detecting
temporal, frequency, correlation, and anomaly patterns in security data.
"""

import logging
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from ..domain.analysis_models import (
    Pattern,
    PatternType,
    PatternStrength,
    BusinessRelevance,
    TimeRange,
    SecurityData
)

logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """
    Temporal analyzer for time-series pattern detection using statistical methods.
    
    Detects trends, seasonality, and temporal anomalies in time-series security data.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_data_points = self.config.get('min_data_points', 5)
        self.trend_significance_threshold = self.config.get('trend_significance', 0.05)
        self.seasonality_window = self.config.get('seasonality_window', 24)  # hours
    
    async def analyze(self, time_series: List[Dict[str, Any]]) -> List[Pattern]:
        """
        Analyze time series data for temporal patterns.
        
        Args:
            time_series: List of time-series data points
            
        Returns:
            List of detected temporal patterns
        """
        patterns = []
        
        if len(time_series) < self.min_data_points:
            logger.warning(f"Insufficient data points for temporal analysis: {len(time_series)}")
            return patterns
        
        try:
            # Convert to time-value pairs
            time_values = self._extract_time_values(time_series)
            
            if len(time_values) < self.min_data_points:
                return patterns
            
            # Detect trend patterns
            trend_patterns = await self._detect_trend_patterns(time_values, time_series)
            patterns.extend(trend_patterns)
            
            # Detect seasonal patterns
            seasonal_patterns = await self._detect_seasonal_patterns(time_values, time_series)
            patterns.extend(seasonal_patterns)
            
            # Detect cyclical patterns
            cyclical_patterns = await self._detect_cyclical_patterns(time_values, time_series)
            patterns.extend(cyclical_patterns)
            
        except Exception as e:
            logger.error(f"Temporal analysis failed: {e}")
        
        return patterns
    
    def _extract_time_values(self, time_series: List[Dict[str, Any]]) -> List[Tuple[datetime, float]]:
        """Extract and validate time-value pairs from time series data."""
        time_values = []
        
        for item in time_series:
            timestamp = item.get('timestamp')
            value = item.get('value', 0)
            
            if timestamp and isinstance(timestamp, (str, datetime)):
                try:
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_values.append((timestamp, float(value)))
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid timestamp or value: {e}")
                    continue
        
        # Sort by timestamp
        time_values.sort(key=lambda x: x[0])
        return time_values
    
    async def _detect_trend_patterns(self, time_values: List[Tuple[datetime, float]], 
                                   original_data: List[Dict[str, Any]]) -> List[Pattern]:
        """Detect trend patterns using statistical methods."""
        patterns = []
        
        if len(time_values) < 3:
            return patterns
        
        try:
            # Convert to numerical arrays for analysis
            timestamps = [tv[0] for tv in time_values]
            values = np.array([tv[1] for tv in time_values])
            
            # Convert timestamps to numerical values (seconds since first timestamp)
            time_numeric = np.array([(ts - timestamps[0]).total_seconds() for ts in timestamps])
            
            # Perform linear regression to detect trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, values)
            
            # Check if trend is statistically significant
            if p_value < self.trend_significance_threshold and abs(r_value) > 0.3:
                trend_direction = "increasing" if slope > 0 else "decreasing"
                confidence = abs(r_value)  # Use correlation coefficient as confidence
                
                # Determine pattern strength based on correlation and significance
                if abs(r_value) > 0.7 and p_value < 0.01:
                    strength = PatternStrength.STRONG
                elif abs(r_value) > 0.5 and p_value < 0.05:
                    strength = PatternStrength.MODERATE
                else:
                    strength = PatternStrength.WEAK
                
                # Assess business relevance
                business_relevance = self._assess_trend_business_relevance(
                    trend_direction, abs(slope), confidence
                )
                
                pattern = Pattern(
                    pattern_type=PatternType.TEMPORAL,
                    strength=strength,
                    confidence=confidence,
                    description=f"Temporal {trend_direction} trend detected (slope: {slope:.4f})",
                    affected_detectors=list(set(item.get('detector', 'unknown') for item in original_data)),
                    time_range=TimeRange(start=timestamps[0], end=timestamps[-1]),
                    statistical_significance=1 - p_value,  # Convert p-value to significance
                    business_relevance=business_relevance,
                    supporting_evidence=[
                        {
                            "trend_direction": trend_direction,
                            "slope": slope,
                            "correlation_coefficient": r_value,
                            "p_value": p_value,
                            "sample_size": len(time_values),
                            "standard_error": std_err
                        }
                    ]
                )
                patterns.append(pattern)
                
        except Exception as e:
            logger.error(f"Trend detection failed: {e}")
        
        return patterns
    
    async def _detect_seasonal_patterns(self, time_values: List[Tuple[datetime, float]], 
                                      original_data: List[Dict[str, Any]]) -> List[Pattern]:
        """Detect seasonal patterns using autocorrelation analysis."""
        patterns = []
        
        if len(time_values) < self.seasonality_window:
            return patterns
        
        try:
            values = np.array([tv[1] for tv in time_values])
            
            # Calculate autocorrelation for different lags
            max_lag = min(len(values) // 2, self.seasonality_window)
            autocorrelations = []
            
            for lag in range(1, max_lag + 1):
                if len(values) > lag:
                    correlation = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                    if not np.isnan(correlation):
                        autocorrelations.append((lag, correlation))
            
            # Find significant autocorrelations (potential seasonal patterns)
            significant_lags = [
                (lag, corr) for lag, corr in autocorrelations 
                if abs(corr) > 0.3  # Threshold for significant correlation
            ]
            
            if significant_lags:
                # Find the most significant seasonal pattern
                best_lag, best_correlation = max(significant_lags, key=lambda x: abs(x[1]))
                
                confidence = abs(best_correlation)
                strength = (PatternStrength.STRONG if confidence > 0.7 
                          else PatternStrength.MODERATE if confidence > 0.5 
                          else PatternStrength.WEAK)
                
                timestamps = [tv[0] for tv in time_values]
                
                pattern = Pattern(
                    pattern_type=PatternType.TEMPORAL,
                    strength=strength,
                    confidence=confidence,
                    description=f"Seasonal pattern detected with {best_lag}-period cycle",
                    affected_detectors=list(set(item.get('detector', 'unknown') for item in original_data)),
                    time_range=TimeRange(start=timestamps[0], end=timestamps[-1]),
                    statistical_significance=confidence,
                    business_relevance=self._assess_seasonal_business_relevance(best_lag, confidence),
                    supporting_evidence=[
                        {
                            "seasonal_period": best_lag,
                            "autocorrelation": best_correlation,
                            "significant_lags": significant_lags[:5],  # Top 5 for brevity
                            "analysis_window": max_lag
                        }
                    ]
                )
                patterns.append(pattern)
                
        except Exception as e:
            logger.error(f"Seasonal pattern detection failed: {e}")
        
        return patterns
    
    async def _detect_cyclical_patterns(self, time_values: List[Tuple[datetime, float]], 
                                      original_data: List[Dict[str, Any]]) -> List[Pattern]:
        """Detect cyclical patterns using peak detection."""
        patterns = []
        
        if len(time_values) < 10:
            return patterns
        
        try:
            values = np.array([tv[1] for tv in time_values])
            
            # Smooth the data to reduce noise
            if len(values) > 5:
                window_size = min(5, len(values) // 3)
                smoothed_values = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            else:
                smoothed_values = values
            
            # Find peaks and troughs
            peaks, peak_properties = find_peaks(smoothed_values, height=np.mean(smoothed_values))
            troughs, trough_properties = find_peaks(-smoothed_values, height=-np.mean(smoothed_values))
            
            # Analyze cyclical behavior
            if len(peaks) >= 2 and len(troughs) >= 2:
                # Calculate average cycle length
                peak_intervals = np.diff(peaks)
                trough_intervals = np.diff(troughs)
                
                if len(peak_intervals) > 0 and len(trough_intervals) > 0:
                    avg_cycle_length = np.mean(np.concatenate([peak_intervals, trough_intervals]))
                    cycle_regularity = 1 - (np.std(peak_intervals) / np.mean(peak_intervals) 
                                          if np.mean(peak_intervals) > 0 else 1)
                    
                    if cycle_regularity > 0.3:  # Threshold for regular cycles
                        confidence = cycle_regularity
                        strength = (PatternStrength.STRONG if confidence > 0.7 
                                  else PatternStrength.MODERATE if confidence > 0.5 
                                  else PatternStrength.WEAK)
                        
                        timestamps = [tv[0] for tv in time_values]
                        
                        pattern = Pattern(
                            pattern_type=PatternType.TEMPORAL,
                            strength=strength,
                            confidence=confidence,
                            description=f"Cyclical pattern with average cycle length {avg_cycle_length:.1f}",
                            affected_detectors=list(set(item.get('detector', 'unknown') for item in original_data)),
                            time_range=TimeRange(start=timestamps[0], end=timestamps[-1]),
                            statistical_significance=confidence,
                            business_relevance=self._assess_cyclical_business_relevance(avg_cycle_length, confidence),
                            supporting_evidence=[
                                {
                                    "peak_count": len(peaks),
                                    "trough_count": len(troughs),
                                    "average_cycle_length": avg_cycle_length,
                                    "cycle_regularity": cycle_regularity,
                                    "peak_positions": peaks.tolist()[:10]  # Limit for brevity
                                }
                            ]
                        )
                        patterns.append(pattern)
                        
        except Exception as e:
            logger.error(f"Cyclical pattern detection failed: {e}")
        
        return patterns
    
    def _assess_trend_business_relevance(self, trend_direction: str, slope: float, confidence: float) -> BusinessRelevance:
        """Assess business relevance of trend patterns."""
        if trend_direction == "increasing" and confidence > 0.8 and abs(slope) > 0.1:
            return BusinessRelevance.HIGH
        elif trend_direction == "increasing" and confidence > 0.6:
            return BusinessRelevance.MEDIUM
        elif trend_direction == "decreasing" and confidence > 0.7:
            return BusinessRelevance.MEDIUM
        else:
            return BusinessRelevance.LOW
    
    def _assess_seasonal_business_relevance(self, period: int, confidence: float) -> BusinessRelevance:
        """Assess business relevance of seasonal patterns."""
        # Business hours (8-hour), daily (24-hour), weekly (168-hour) patterns are more relevant
        business_periods = [8, 24, 168]
        
        if any(abs(period - bp) <= 2 for bp in business_periods) and confidence > 0.7:
            return BusinessRelevance.HIGH
        elif confidence > 0.6:
            return BusinessRelevance.MEDIUM
        else:
            return BusinessRelevance.LOW
    
    def _assess_cyclical_business_relevance(self, cycle_length: float, confidence: float) -> BusinessRelevance:
        """Assess business relevance of cyclical patterns."""
        if confidence > 0.8 and cycle_length > 5:
            return BusinessRelevance.HIGH
        elif confidence > 0.6:
            return BusinessRelevance.MEDIUM
        else:
            return BusinessRelevance.LOW


class FrequencyAnalyzer:
    """
    Frequency analyzer for recurring event pattern identification.
    
    Detects patterns in event frequencies, intervals, and distributions.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_events = self.config.get('min_events', 5)
        self.frequency_threshold = self.config.get('frequency_threshold', 0.1)
        self.regularity_threshold = self.config.get('regularity_threshold', 0.3)
    
    async def analyze(self, events: List[Dict[str, Any]]) -> List[Pattern]:
        """
        Analyze events for frequency patterns.
        
        Args:
            events: List of security events
            
        Returns:
            List of detected frequency patterns
        """
        patterns = []
        
        if len(events) < self.min_events:
            logger.warning(f"Insufficient events for frequency analysis: {len(events)}")
            return patterns
        
        try:
            # Group events by detector and type
            event_groups = self._group_events(events)
            
            for group_key, group_events in event_groups.items():
                if len(group_events) >= self.min_events:
                    group_patterns = await self._analyze_event_group(group_key, group_events)
                    patterns.extend(group_patterns)
            
        except Exception as e:
            logger.error(f"Frequency analysis failed: {e}")
        
        return patterns
    
    def _group_events(self, events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group events by detector and type."""
        groups = {}
        
        for event in events:
            detector = event.get('detector', 'unknown')
            event_type = event.get('type', 'unknown')
            severity = event.get('severity', 'unknown')
            
            # Create hierarchical grouping
            group_key = f"{detector}:{event_type}:{severity}"
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(event)
        
        return groups
    
    async def _analyze_event_group(self, group_key: str, events: List[Dict[str, Any]]) -> List[Pattern]:
        """Analyze a specific group of events for frequency patterns."""
        patterns = []
        
        try:
            # Extract timestamps
            timestamps = self._extract_timestamps(events)
            
            if len(timestamps) < self.min_events:
                return patterns
            
            # Analyze event frequency
            frequency_patterns = await self._detect_frequency_patterns(group_key, timestamps, events)
            patterns.extend(frequency_patterns)
            
            # Analyze event intervals
            interval_patterns = await self._detect_interval_patterns(group_key, timestamps, events)
            patterns.extend(interval_patterns)
            
            # Analyze burst patterns
            burst_patterns = await self._detect_burst_patterns(group_key, timestamps, events)
            patterns.extend(burst_patterns)
            
        except Exception as e:
            logger.error(f"Event group analysis failed for {group_key}: {e}")
        
        return patterns
    
    def _extract_timestamps(self, events: List[Dict[str, Any]]) -> List[datetime]:
        """Extract and sort timestamps from events."""
        timestamps = []
        
        for event in events:
            timestamp = event.get('timestamp')
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    timestamps.append(timestamp)
                except (ValueError, TypeError):
                    continue
        
        return sorted(timestamps)
    
    async def _detect_frequency_patterns(self, group_key: str, timestamps: List[datetime], 
                                       events: List[Dict[str, Any]]) -> List[Pattern]:
        """Detect patterns in event frequency."""
        patterns = []
        
        if len(timestamps) < 3:
            return patterns
        
        try:
            # Calculate time span and frequency
            time_span = (timestamps[-1] - timestamps[0]).total_seconds()
            if time_span <= 0:
                return patterns
            
            frequency = len(timestamps) / (time_span / 3600)  # Events per hour
            
            # Analyze frequency distribution over time windows
            window_size = max(1, int(time_span / 3600 / 10))  # Divide into ~10 windows
            if window_size > 0:
                window_frequencies = self._calculate_window_frequencies(timestamps, window_size)
                
                if len(window_frequencies) >= 3:
                    # Check for consistent frequency pattern
                    freq_mean = np.mean(window_frequencies)
                    freq_std = np.std(window_frequencies)
                    
                    if freq_mean > 0:
                        consistency = 1 - (freq_std / freq_mean)  # Coefficient of variation
                        
                        if consistency > self.regularity_threshold and frequency > self.frequency_threshold:
                            confidence = min(1.0, consistency)
                            strength = (PatternStrength.STRONG if consistency > 0.7 
                                      else PatternStrength.MODERATE if consistency > 0.5 
                                      else PatternStrength.WEAK)
                            
                            detector, event_type, severity = group_key.split(':', 2)
                            
                            pattern = Pattern(
                                pattern_type=PatternType.FREQUENCY,
                                strength=strength,
                                confidence=confidence,
                                description=f"Regular frequency pattern: {frequency:.2f} events/hour in {detector} {event_type}",
                                affected_detectors=[detector],
                                time_range=TimeRange(start=timestamps[0], end=timestamps[-1]),
                                statistical_significance=consistency,
                                business_relevance=self._assess_frequency_business_relevance(detector, frequency, consistency),
                                supporting_evidence=[
                                    {
                                        "event_count": len(events),
                                        "frequency_per_hour": frequency,
                                        "consistency_score": consistency,
                                        "time_span_hours": time_span / 3600,
                                        "window_frequencies": window_frequencies
                                    }
                                ]
                            )
                            patterns.append(pattern)
                            
        except Exception as e:
            logger.error(f"Frequency pattern detection failed: {e}")
        
        return patterns
    
    def _calculate_window_frequencies(self, timestamps: List[datetime], window_hours: int) -> List[float]:
        """Calculate event frequencies in time windows."""
        if not timestamps or window_hours <= 0:
            return []
        
        start_time = timestamps[0]
        end_time = timestamps[-1]
        window_delta = timedelta(hours=window_hours)
        
        frequencies = []
        current_time = start_time
        
        while current_time < end_time:
            window_end = min(current_time + window_delta, end_time)
            
            # Count events in this window
            events_in_window = sum(
                1 for ts in timestamps 
                if current_time <= ts < window_end
            )
            
            window_duration_hours = (window_end - current_time).total_seconds() / 3600
            if window_duration_hours > 0:
                frequencies.append(events_in_window / window_duration_hours)
            
            current_time = window_end
        
        return frequencies
    
    async def _detect_interval_patterns(self, group_key: str, timestamps: List[datetime], 
                                      events: List[Dict[str, Any]]) -> List[Pattern]:
        """Detect patterns in intervals between events."""
        patterns = []
        
        if len(timestamps) < 3:
            return patterns
        
        try:
            # Calculate intervals between consecutive events
            intervals = []
            for i in range(1, len(timestamps)):
                interval = (timestamps[i] - timestamps[i-1]).total_seconds()
                intervals.append(interval)
            
            if len(intervals) < 2:
                return patterns
            
            # Analyze interval regularity
            interval_mean = np.mean(intervals)
            interval_std = np.std(intervals)
            
            if interval_mean > 0:
                regularity = 1 - (interval_std / interval_mean)  # Coefficient of variation
                
                if regularity > self.regularity_threshold:
                    confidence = min(1.0, regularity)
                    strength = (PatternStrength.STRONG if regularity > 0.7 
                              else PatternStrength.MODERATE if regularity > 0.5 
                              else PatternStrength.WEAK)
                    
                    detector, event_type, severity = group_key.split(':', 2)
                    
                    pattern = Pattern(
                        pattern_type=PatternType.FREQUENCY,
                        strength=strength,
                        confidence=confidence,
                        description=f"Regular interval pattern: {interval_mean:.1f}s average interval in {detector} {event_type}",
                        affected_detectors=[detector],
                        time_range=TimeRange(start=timestamps[0], end=timestamps[-1]),
                        statistical_significance=regularity,
                        business_relevance=self._assess_interval_business_relevance(interval_mean, regularity),
                        supporting_evidence=[
                            {
                                "average_interval_seconds": interval_mean,
                                "interval_std": interval_std,
                                "regularity_score": regularity,
                                "interval_count": len(intervals),
                                "min_interval": min(intervals),
                                "max_interval": max(intervals)
                            }
                        ]
                    )
                    patterns.append(pattern)
                    
        except Exception as e:
            logger.error(f"Interval pattern detection failed: {e}")
        
        return patterns
    
    async def _detect_burst_patterns(self, group_key: str, timestamps: List[datetime], 
                                   events: List[Dict[str, Any]]) -> List[Pattern]:
        """Detect burst patterns in event occurrences."""
        patterns = []
        
        if len(timestamps) < 5:
            return patterns
        
        try:
            # Use DBSCAN clustering to identify bursts
            # Convert timestamps to numerical values (seconds since first timestamp)
            time_numeric = np.array([(ts - timestamps[0]).total_seconds() for ts in timestamps])
            time_numeric = time_numeric.reshape(-1, 1)
            
            # Apply DBSCAN clustering
            eps = np.std(time_numeric) * 0.5  # Adaptive epsilon based on data spread
            min_samples = max(2, len(timestamps) // 10)  # At least 2, or 10% of data
            
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(time_numeric)
            labels = clustering.labels_
            
            # Identify clusters (bursts)
            unique_labels = set(labels)
            clusters = [labels == label for label in unique_labels if label != -1]
            
            if len(clusters) >= 2:  # At least 2 bursts detected
                # Analyze burst characteristics
                burst_sizes = [np.sum(cluster) for cluster in clusters]
                avg_burst_size = np.mean(burst_sizes)
                
                # Calculate burst intensity (events per unit time within bursts)
                burst_intensities = []
                for cluster in clusters:
                    cluster_times = time_numeric[cluster].flatten()
                    if len(cluster_times) > 1:
                        burst_duration = np.max(cluster_times) - np.min(cluster_times)
                        if burst_duration > 0:
                            intensity = len(cluster_times) / (burst_duration / 3600)  # Events per hour
                            burst_intensities.append(intensity)
                
                if burst_intensities:
                    avg_intensity = np.mean(burst_intensities)
                    
                    # Calculate confidence based on burst consistency
                    burst_consistency = 1 - (np.std(burst_sizes) / avg_burst_size if avg_burst_size > 0 else 1)
                    confidence = min(1.0, burst_consistency)
                    
                    if confidence > 0.3:  # Threshold for significant burst pattern
                        strength = (PatternStrength.STRONG if confidence > 0.7 
                                  else PatternStrength.MODERATE if confidence > 0.5 
                                  else PatternStrength.WEAK)
                        
                        detector, event_type, severity = group_key.split(':', 2)
                        
                        pattern = Pattern(
                            pattern_type=PatternType.FREQUENCY,
                            strength=strength,
                            confidence=confidence,
                            description=f"Burst pattern detected: {len(clusters)} bursts with avg {avg_burst_size:.1f} events each",
                            affected_detectors=[detector],
                            time_range=TimeRange(start=timestamps[0], end=timestamps[-1]),
                            statistical_significance=confidence,
                            business_relevance=self._assess_burst_business_relevance(len(clusters), avg_intensity),
                            supporting_evidence=[
                                {
                                    "burst_count": len(clusters),
                                    "average_burst_size": avg_burst_size,
                                    "average_intensity_per_hour": avg_intensity,
                                    "burst_consistency": burst_consistency,
                                    "clustering_eps": eps,
                                    "clustering_min_samples": min_samples
                                }
                            ]
                        )
                        patterns.append(pattern)
                        
        except Exception as e:
            logger.error(f"Burst pattern detection failed: {e}")
        
        return patterns
    
    def _assess_frequency_business_relevance(self, detector: str, frequency: float, consistency: float) -> BusinessRelevance:
        """Assess business relevance of frequency patterns."""
        high_impact_detectors = ['presidio', 'pii-detector', 'gdpr-scanner', 'hipaa-detector']
        
        if detector.lower() in [d.lower() for d in high_impact_detectors]:
            if frequency > 10 and consistency > 0.7:
                return BusinessRelevance.CRITICAL
            elif frequency > 5 and consistency > 0.5:
                return BusinessRelevance.HIGH
            else:
                return BusinessRelevance.MEDIUM
        else:
            if frequency > 20 and consistency > 0.8:
                return BusinessRelevance.HIGH
            elif frequency > 10 and consistency > 0.6:
                return BusinessRelevance.MEDIUM
            else:
                return BusinessRelevance.LOW
    
    def _assess_interval_business_relevance(self, interval: float, regularity: float) -> BusinessRelevance:
        """Assess business relevance of interval patterns."""
        # Very regular intervals might indicate automated attacks or system issues
        if regularity > 0.8 and interval < 60:  # Very regular, short intervals
            return BusinessRelevance.HIGH
        elif regularity > 0.6 and interval < 300:  # Regular, moderate intervals
            return BusinessRelevance.MEDIUM
        else:
            return BusinessRelevance.LOW
    
    def _assess_burst_business_relevance(self, burst_count: int, avg_intensity: float) -> BusinessRelevance:
        """Assess business relevance of burst patterns."""
        if burst_count >= 5 and avg_intensity > 50:  # Many bursts with high intensity
            return BusinessRelevance.CRITICAL
        elif burst_count >= 3 and avg_intensity > 20:  # Multiple bursts with moderate intensity
            return BusinessRelevance.HIGH
        elif burst_count >= 2:  # At least some burst activity
            return BusinessRelevance.MEDIUM
        else:
            return BusinessRelevance.LOW


class CorrelationAnalyzer:
    """
    Correlation analyzer for cross-detector relationship detection.
    
    Detects correlations between different detectors, metrics, and security events.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.correlation_threshold = self.config.get('correlation_threshold', 0.3)
        self.min_data_points = self.config.get('min_data_points', 10)
        self.significance_level = self.config.get('significance_level', 0.05)
    
    async def analyze(self, multi_dimensional: List[Dict[str, Any]]) -> List[Pattern]:
        """
        Analyze multi-dimensional data for correlation patterns.
        
        Args:
            multi_dimensional: List of multi-dimensional data points
            
        Returns:
            List of detected correlation patterns
        """
        patterns = []
        
        if len(multi_dimensional) < self.min_data_points:
            logger.warning(f"Insufficient data points for correlation analysis: {len(multi_dimensional)}")
            return patterns
        
        try:
            # Extract numerical features
            feature_data = self._extract_features(multi_dimensional)
            
            if len(feature_data) < 2:
                logger.warning("Insufficient features for correlation analysis")
                return patterns
            
            # Detect pairwise correlations
            correlation_patterns = await self._detect_pairwise_correlations(feature_data, multi_dimensional)
            patterns.extend(correlation_patterns)
            
            # Detect multi-variate correlations
            multivariate_patterns = await self._detect_multivariate_correlations(feature_data, multi_dimensional)
            patterns.extend(multivariate_patterns)
            
            # Detect lagged correlations
            lagged_patterns = await self._detect_lagged_correlations(feature_data, multi_dimensional)
            patterns.extend(lagged_patterns)
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
        
        return patterns
    
    def _extract_features(self, multi_dimensional: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Extract numerical features from multi-dimensional data."""
        feature_data = {}
        
        for item in multi_dimensional:
            for key, value in item.items():
                if isinstance(value, (int, float)) and key not in ['timestamp', 'id']:
                    if key not in feature_data:
                        feature_data[key] = []
                    feature_data[key].append(float(value))
        
        # Filter features with sufficient data points
        filtered_features = {
            key: values for key, values in feature_data.items()
            if len(values) >= self.min_data_points
        }
        
        return filtered_features
    
    async def _detect_pairwise_correlations(self, feature_data: Dict[str, List[float]], 
                                          original_data: List[Dict[str, Any]]) -> List[Pattern]:
        """Detect pairwise correlations between features."""
        patterns = []
        
        feature_names = list(feature_data.keys())
        
        for i, feature1 in enumerate(feature_names):
            for feature2 in feature_names[i + 1:]:
                try:
                    # Ensure equal length arrays
                    min_length = min(len(feature_data[feature1]), len(feature_data[feature2]))
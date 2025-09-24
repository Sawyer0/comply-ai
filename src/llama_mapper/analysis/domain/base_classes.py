"""
Abstract base classes for common analysis patterns.

This module provides abstract base classes that implement common
functionality shared across different analysis engines.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .analysis_interfaces import IAnalysisEngine
from .analysis_models import (
    AnalysisConfiguration,
    AnalysisResult,
    AnalysisStrategy,
    SecurityData,
    TimeRange,
)
from .entities import AnalysisRequest

logger = logging.getLogger(__name__)


class BaseAnalysisEngine(IAnalysisEngine):
    """
    Abstract base class for all analysis engines.
    
    Provides common functionality and patterns that all analysis engines
    can inherit and customize as needed.
    """
    
    def __init__(self, config: AnalysisConfiguration):
        """
        Initialize the base analysis engine.
        
        Args:
            config: Configuration for this analysis engine
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.get_engine_name()}")
        self._initialized = False
        self._health_status = "unknown"
        
    async def initialize(self) -> None:
        """
        Initialize the analysis engine.
        
        Subclasses should override this method to perform any necessary
        initialization steps.
        """
        self.logger.info(f"Initializing {self.get_engine_name()} analysis engine")
        self._initialized = True
        self._health_status = "healthy"
        
    async def shutdown(self) -> None:
        """
        Shutdown the analysis engine and clean up resources.
        
        Subclasses should override this method to perform cleanup.
        """
        self.logger.info(f"Shutting down {self.get_engine_name()} analysis engine")
        self._initialized = False
        self._health_status = "shutdown"
    
    def is_initialized(self) -> bool:
        """Check if the engine is initialized."""
        return self._initialized
    
    def get_health_status(self) -> str:
        """Get the current health status of the engine."""
        return self._health_status
    
    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Perform analysis on the given request.
        
        This method implements the common analysis workflow and delegates
        to abstract methods for engine-specific logic.
        """
        if not self._initialized:
            raise RuntimeError(f"{self.get_engine_name()} engine not initialized")
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Validate input
            if not await self.validate_input(request):
                raise ValueError(f"Invalid input for {self.get_engine_name()} engine")
            
            # Perform engine-specific analysis
            result = await self._perform_analysis(request)
            
            # Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            result.processing_time_ms = int(processing_time)
            result.timestamp = start_time
            result.engine_name = self.get_engine_name()
            
            # Validate result quality
            await self._validate_result_quality(result)
            
            self.logger.info(
                f"Analysis completed",
                engine=self.get_engine_name(),
                processing_time_ms=result.processing_time_ms,
                confidence=result.confidence
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                f"Analysis failed",
                engine=self.get_engine_name(),
                error=str(e),
                request_id=getattr(request, 'request_id', 'unknown')
            )
            raise
    
    @abstractmethod
    async def _perform_analysis(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Perform the engine-specific analysis logic.
        
        Args:
            request: The analysis request to process
            
        Returns:
            AnalysisResult with engine-specific findings
        """
        pass
    
    async def validate_input(self, request: AnalysisRequest) -> bool:
        """
        Validate that the input request is suitable for this engine.
        
        Default implementation performs basic validation. Subclasses
        can override for engine-specific validation.
        """
        if not request:
            return False
        
        if not hasattr(request, 'request_id') or not request.request_id:
            return False
        
        return True
    
    def get_confidence(self, result: AnalysisResult) -> float:
        """
        Calculate confidence score for the analysis result.
        
        Default implementation returns the result's confidence.
        Subclasses can override for custom confidence calculation.
        """
        return result.confidence
    
    async def _validate_result_quality(self, result: AnalysisResult) -> None:
        """
        Validate the quality of analysis results.
        
        Args:
            result: The analysis result to validate
            
        Raises:
            ValueError: If result quality is insufficient
        """
        if result.confidence < self.config.confidence_threshold:
            self.logger.warning(
                f"Low confidence result",
                engine=self.get_engine_name(),
                confidence=result.confidence,
                threshold=self.config.confidence_threshold
            )
        
        if not result.evidence and result.confidence > 0.5:
            self.logger.warning(
                f"High confidence result without evidence",
                engine=self.get_engine_name(),
                confidence=result.confidence
            )


class StatisticalAnalysisEngine(BaseAnalysisEngine):
    """
    Base class for analysis engines that use statistical methods.
    
    Provides common statistical analysis patterns and utilities.
    """
    
    def __init__(self, config: AnalysisConfiguration):
        super().__init__(config)
        self.statistical_config = config.parameters.get('statistical', {})
        self.significance_threshold = self.statistical_config.get('significance_threshold', 0.05)
        self.min_sample_size = self.statistical_config.get('min_sample_size', 30)
    
    async def _calculate_statistical_significance(self, data: List[float]) -> float:
        """
        Calculate statistical significance of data.
        
        Args:
            data: List of numerical data points
            
        Returns:
            Statistical significance score (0.0 to 1.0)
        """
        if len(data) < self.min_sample_size:
            return 0.0
        
        # Simple implementation - can be enhanced with proper statistical tests
        import statistics
        
        try:
            mean = statistics.mean(data)
            stdev = statistics.stdev(data) if len(data) > 1 else 0.0
            
            # Calculate coefficient of variation as a proxy for significance
            if mean != 0:
                cv = stdev / abs(mean)
                significance = max(0.0, min(1.0, 1.0 - cv))
            else:
                significance = 0.0
            
            return significance
            
        except Exception as e:
            self.logger.error(f"Statistical calculation failed: {e}")
            return 0.0
    
    async def _detect_outliers(self, data: List[float], method: str = "iqr") -> List[int]:
        """
        Detect outliers in numerical data.
        
        Args:
            data: List of numerical data points
            method: Outlier detection method ("iqr" or "zscore")
            
        Returns:
            List of indices of outlier data points
        """
        if len(data) < 4:
            return []
        
        outliers = []
        
        try:
            if method == "iqr":
                # Interquartile Range method
                sorted_data = sorted(data)
                n = len(sorted_data)
                q1 = sorted_data[n // 4]
                q3 = sorted_data[3 * n // 4]
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                for i, value in enumerate(data):
                    if value < lower_bound or value > upper_bound:
                        outliers.append(i)
            
            elif method == "zscore":
                # Z-score method
                import statistics
                mean = statistics.mean(data)
                stdev = statistics.stdev(data) if len(data) > 1 else 0.0
                
                if stdev > 0:
                    for i, value in enumerate(data):
                        z_score = abs(value - mean) / stdev
                        if z_score > 2.0:  # 2 standard deviations
                            outliers.append(i)
            
        except Exception as e:
            self.logger.error(f"Outlier detection failed: {e}")
        
        return outliers
    
    async def _calculate_trend(self, time_series: List[tuple[datetime, float]]) -> Dict[str, Any]:
        """
        Calculate trend information for time series data.
        
        Args:
            time_series: List of (timestamp, value) tuples
            
        Returns:
            Dictionary with trend information
        """
        if len(time_series) < 2:
            return {"trend": "insufficient_data", "slope": 0.0, "confidence": 0.0}
        
        try:
            # Simple linear regression for trend calculation
            n = len(time_series)
            
            # Convert timestamps to numerical values (seconds since first timestamp)
            base_time = time_series[0][0]
            x_values = [(ts - base_time).total_seconds() for ts, _ in time_series]
            y_values = [value for _, value in time_series]
            
            # Calculate slope using least squares
            x_mean = sum(x_values) / n
            y_mean = sum(y_values) / n
            
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
            denominator = sum((x - x_mean) ** 2 for x in x_values)
            
            if denominator == 0:
                slope = 0.0
            else:
                slope = numerator / denominator
            
            # Determine trend direction
            if abs(slope) < 1e-6:
                trend = "stable"
            elif slope > 0:
                trend = "increasing"
            else:
                trend = "decreasing"
            
            # Calculate R-squared as confidence measure
            if denominator > 0:
                y_pred = [y_mean + slope * (x - x_mean) for x in x_values]
                ss_res = sum((y - y_pred) ** 2 for y, y_pred in zip(y_values, y_pred))
                ss_tot = sum((y - y_mean) ** 2 for y in y_values)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                confidence = max(0.0, min(1.0, r_squared))
            else:
                confidence = 0.0
            
            return {
                "trend": trend,
                "slope": slope,
                "confidence": confidence,
                "sample_size": n
            }
            
        except Exception as e:
            self.logger.error(f"Trend calculation failed: {e}")
            return {"trend": "error", "slope": 0.0, "confidence": 0.0}


class BusinessContextEngine(BaseAnalysisEngine):
    """
    Base class for analysis engines that incorporate business context.
    
    Provides utilities for business impact assessment and context-aware analysis.
    """
    
    def __init__(self, config: AnalysisConfiguration):
        super().__init__(config)
        self.business_config = config.parameters.get('business', {})
        self.impact_weights = self.business_config.get('impact_weights', {})
        self.business_hours = self.business_config.get('business_hours', {})
    
    async def _assess_business_criticality(self, detector_id: str, timestamp: datetime) -> float:
        """
        Assess business criticality based on detector and timing.
        
        Args:
            detector_id: ID of the detector
            timestamp: When the finding occurred
            
        Returns:
            Business criticality score (0.0 to 1.0)
        """
        base_criticality = self.impact_weights.get(detector_id, 0.5)
        
        # Adjust for business hours
        time_multiplier = self._get_time_criticality_multiplier(timestamp)
        
        return min(1.0, base_criticality * time_multiplier)
    
    def _get_time_criticality_multiplier(self, timestamp: datetime) -> float:
        """
        Get criticality multiplier based on time of occurrence.
        
        Args:
            timestamp: When the event occurred
            
        Returns:
            Time-based criticality multiplier
        """
        # Default business hours: Monday-Friday, 9 AM - 5 PM
        business_start = self.business_hours.get('start_hour', 9)
        business_end = self.business_hours.get('end_hour', 17)
        business_days = self.business_hours.get('days', [0, 1, 2, 3, 4])  # Mon-Fri
        
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        if weekday in business_days and business_start <= hour < business_end:
            return 1.5  # Higher criticality during business hours
        else:
            return 1.0  # Normal criticality outside business hours
    
    async def _calculate_financial_impact(self, severity: str, affected_systems: List[str]) -> Dict[str, float]:
        """
        Calculate estimated financial impact.
        
        Args:
            severity: Severity level of the finding
            affected_systems: List of affected systems
            
        Returns:
            Dictionary with financial impact estimates
        """
        # Base impact values (can be configured)
        severity_multipliers = {
            'low': 1.0,
            'medium': 2.5,
            'high': 5.0,
            'critical': 10.0
        }
        
        system_values = self.business_config.get('system_values', {})
        
        base_impact = severity_multipliers.get(severity.lower(), 1.0)
        
        # Calculate system impact
        system_impact = 0.0
        for system in affected_systems:
            system_value = system_values.get(system, 1000.0)  # Default $1000
            system_impact += system_value
        
        total_impact = base_impact * system_impact
        
        return {
            'immediate_cost': total_impact * 0.1,  # 10% immediate
            'short_term_cost': total_impact * 0.3,  # 30% short term
            'long_term_cost': total_impact * 0.6,   # 60% long term
            'total_estimated_cost': total_impact
        }


class ConfigurableEngine(BaseAnalysisEngine):
    """
    Base class for engines that support dynamic configuration.
    
    Provides configuration management and hot-reloading capabilities.
    """
    
    def __init__(self, config: AnalysisConfiguration):
        super().__init__(config)
        self._config_version = 1
        self._last_config_update = datetime.now(timezone.utc)
    
    async def update_configuration(self, new_config: AnalysisConfiguration) -> bool:
        """
        Update engine configuration dynamically.
        
        Args:
            new_config: New configuration to apply
            
        Returns:
            True if configuration was updated successfully
        """
        try:
            # Validate new configuration
            if not await self._validate_configuration(new_config):
                self.logger.error("Configuration validation failed")
                return False
            
            # Apply configuration
            old_config = self.config
            self.config = new_config
            self._config_version += 1
            self._last_config_update = datetime.now(timezone.utc)
            
            # Notify of configuration change
            await self._on_configuration_changed(old_config, new_config)
            
            self.logger.info(
                f"Configuration updated",
                engine=self.get_engine_name(),
                version=self._config_version
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration update failed: {e}")
            return False
    
    async def _validate_configuration(self, config: AnalysisConfiguration) -> bool:
        """
        Validate configuration before applying.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
        """
        # Basic validation - subclasses can override for specific validation
        if not config.engine_name:
            return False
        
        if config.confidence_threshold < 0.0 or config.confidence_threshold > 1.0:
            return False
        
        return True
    
    async def _on_configuration_changed(self, old_config: AnalysisConfiguration, new_config: AnalysisConfiguration) -> None:
        """
        Handle configuration change event.
        
        Subclasses can override this method to perform specific actions
        when configuration changes.
        
        Args:
            old_config: Previous configuration
            new_config: New configuration
        """
        pass
    
    def get_configuration_info(self) -> Dict[str, Any]:
        """
        Get information about current configuration.
        
        Returns:
            Dictionary with configuration metadata
        """
        return {
            'engine_name': self.get_engine_name(),
            'config_version': self._config_version,
            'last_update': self._last_config_update.isoformat(),
            'enabled': self.config.enabled,
            'confidence_threshold': self.config.confidence_threshold
        }
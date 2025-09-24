"""
Core analysis interfaces for the enhanced analysis system.

This module defines the fundamental interfaces that all analysis engines
must implement to provide sophisticated rule-based analysis capabilities.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from .entities import AnalysisRequest, AnalysisResponse


class IAnalysisEngine(ABC):
    """
    Base interface for all analysis engines.
    
    This interface defines the core contract that all analysis engines
    must implement to provide consistent analysis capabilities.
    """
    
    @abstractmethod
    async def analyze(self, request: AnalysisRequest) -> 'AnalysisResult':
        """
        Perform analysis on the given request.
        
        Args:
            request: The analysis request containing input data
            
        Returns:
            AnalysisResult containing the analysis findings
        """
        pass
    
    @abstractmethod
    def get_confidence(self, result: 'AnalysisResult') -> float:
        """
        Calculate confidence score for the analysis result.
        
        Args:
            result: The analysis result to evaluate
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        pass
    
    @abstractmethod
    def get_engine_name(self) -> str:
        """
        Get the name of this analysis engine.
        
        Returns:
            String identifier for this engine
        """
        pass
    
    @abstractmethod
    async def validate_input(self, request: AnalysisRequest) -> bool:
        """
        Validate that the input request is suitable for this engine.
        
        Args:
            request: The analysis request to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        pass


class IPatternRecognitionEngine(IAnalysisEngine):
    """
    Interface for pattern recognition analysis engines.
    
    Extends the base analysis engine to provide pattern detection
    and classification capabilities using statistical methods.
    """
    
    @abstractmethod
    async def detect_patterns(self, data: 'SecurityData') -> List['Pattern']:
        """
        Detect patterns in security data using statistical analysis.
        
        Args:
            data: Security data to analyze for patterns
            
        Returns:
            List of detected patterns with confidence scores
        """
        pass
    
    @abstractmethod
    async def classify_pattern_strength(self, pattern: 'Pattern') -> 'PatternStrength':
        """
        Classify the strength and significance of a detected pattern.
        
        Args:
            pattern: The pattern to classify
            
        Returns:
            PatternStrength classification with statistical significance
        """
        pass
    
    @abstractmethod
    async def correlate_patterns(self, patterns: List['Pattern']) -> List['PatternCorrelation']:
        """
        Identify correlations between multiple patterns.
        
        Args:
            patterns: List of patterns to correlate
            
        Returns:
            List of pattern correlations with strength indicators
        """
        pass


class IRiskScoringEngine(IAnalysisEngine):
    """
    Interface for risk scoring analysis engines.
    
    Provides intelligent risk assessment capabilities that consider
    business context, regulatory requirements, and temporal factors.
    """
    
    @abstractmethod
    async def calculate_risk_score(self, findings: List['SecurityFinding']) -> 'RiskScore':
        """
        Calculate comprehensive risk score for security findings.
        
        Args:
            findings: List of security findings to assess
            
        Returns:
            RiskScore with detailed breakdown and contributing factors
        """
        pass
    
    @abstractmethod
    async def get_risk_breakdown(self, score: 'RiskScore') -> 'RiskBreakdown':
        """
        Get detailed breakdown of risk score components.
        
        Args:
            score: The risk score to break down
            
        Returns:
            RiskBreakdown showing contributing factors and weights
        """
        pass
    
    @abstractmethod
    async def assess_business_impact(self, findings: List['SecurityFinding']) -> 'BusinessImpact':
        """
        Assess potential business impact of security findings.
        
        Args:
            findings: Security findings to assess
            
        Returns:
            BusinessImpact assessment with financial and operational metrics
        """
        pass


class IComplianceIntelligenceEngine(IAnalysisEngine):
    """
    Interface for compliance intelligence analysis engines.
    
    Provides automated mapping of security findings to regulatory
    frameworks and compliance requirements.
    """
    
    @abstractmethod
    async def map_to_frameworks(self, findings: List['SecurityFinding']) -> List['ComplianceMapping']:
        """
        Map security findings to compliance framework requirements.
        
        Args:
            findings: Security findings to map
            
        Returns:
            List of compliance mappings for different frameworks
        """
        pass
    
    @abstractmethod
    async def identify_compliance_gaps(self, current_state: 'ComplianceState') -> List['ComplianceGap']:
        """
        Identify gaps in compliance coverage.
        
        Args:
            current_state: Current compliance state to analyze
            
        Returns:
            List of identified compliance gaps with priorities
        """
        pass
    
    @abstractmethod
    async def generate_remediation_plan(self, gaps: List['ComplianceGap']) -> 'RemediationPlan':
        """
        Generate actionable remediation plan for compliance gaps.
        
        Args:
            gaps: Compliance gaps to address
            
        Returns:
            RemediationPlan with prioritized actions and timelines
        """
        pass


class IThresholdOptimizationEngine(IAnalysisEngine):
    """
    Interface for threshold optimization analysis engines.
    
    Provides statistical analysis and optimization of detection
    thresholds to minimize false positives while maintaining coverage.
    """
    
    @abstractmethod
    async def analyze_threshold_performance(self, detector_id: str) -> 'ThresholdPerformance':
        """
        Analyze current threshold performance for a detector.
        
        Args:
            detector_id: ID of the detector to analyze
            
        Returns:
            ThresholdPerformance metrics and analysis
        """
        pass
    
    @abstractmethod
    async def recommend_thresholds(self, performance: 'ThresholdPerformance') -> 'ThresholdRecommendations':
        """
        Recommend optimal thresholds based on performance analysis.
        
        Args:
            performance: Current threshold performance data
            
        Returns:
            ThresholdRecommendations with statistical justification
        """
        pass
    
    @abstractmethod
    async def simulate_threshold_impact(self, recommendations: 'ThresholdRecommendations') -> 'ImpactSimulation':
        """
        Simulate the impact of proposed threshold changes.
        
        Args:
            recommendations: Proposed threshold changes
            
        Returns:
            ImpactSimulation showing expected outcomes
        """
        pass


class IIncidentCorrelationEngine(IAnalysisEngine):
    """
    Interface for incident correlation analysis engines.
    
    Provides advanced correlation analysis to identify related
    security incidents and potential attack patterns.
    """
    
    @abstractmethod
    async def correlate_incidents(self, incidents: List['SecurityIncident']) -> List['IncidentCorrelation']:
        """
        Correlate security incidents to identify relationships.
        
        Args:
            incidents: List of security incidents to correlate
            
        Returns:
            List of incident correlations with relationship types
        """
        pass
    
    @abstractmethod
    async def detect_attack_patterns(self, correlations: List['IncidentCorrelation']) -> List['AttackPattern']:
        """
        Detect potential attack patterns from incident correlations.
        
        Args:
            correlations: Incident correlations to analyze
            
        Returns:
            List of detected attack patterns with confidence scores
        """
        pass
    
    @abstractmethod
    async def predict_next_steps(self, pattern: 'AttackPattern') -> List['ThreatPrediction']:
        """
        Predict likely next steps in an attack pattern.
        
        Args:
            pattern: Attack pattern to analyze
            
        Returns:
            List of threat predictions with probabilities
        """
        pass


class IPredictiveAnalyticsEngine(IAnalysisEngine):
    """
    Interface for predictive analytics analysis engines.
    
    Provides trend analysis and forecasting capabilities for
    proactive security management.
    """
    
    @abstractmethod
    async def analyze_trends(self, historical_data: 'HistoricalSecurityData') -> List['SecurityTrend']:
        """
        Analyze historical data to identify security trends.
        
        Args:
            historical_data: Historical security data to analyze
            
        Returns:
            List of identified security trends with projections
        """
        pass
    
    @abstractmethod
    async def forecast_metrics(self, trends: List['SecurityTrend']) -> List['MetricForecast']:
        """
        Forecast future security metrics based on trends.
        
        Args:
            trends: Security trends to use for forecasting
            
        Returns:
            List of metric forecasts with confidence intervals
        """
        pass
    
    @abstractmethod
    async def recommend_proactive_measures(self, forecasts: List['MetricForecast']) -> List['ProactiveMeasure']:
        """
        Recommend proactive measures based on forecasts.
        
        Args:
            forecasts: Metric forecasts to analyze
            
        Returns:
            List of recommended proactive measures
        """
        pass


class IReportGenerationEngine(IAnalysisEngine):
    """
    Interface for report generation analysis engines.
    
    Provides automated generation of role-specific reports
    with dynamic visualizations and insights.
    """
    
    @abstractmethod
    async def generate_executive_report(self, analysis_data: 'AnalysisData') -> 'ExecutiveReport':
        """
        Generate executive-level report with high-level insights.
        
        Args:
            analysis_data: Analysis data to include in report
            
        Returns:
            ExecutiveReport with business-focused insights
        """
        pass
    
    @abstractmethod
    async def generate_technical_report(self, analysis_data: 'AnalysisData') -> 'TechnicalReport':
        """
        Generate technical report with detailed analysis.
        
        Args:
            analysis_data: Analysis data to include in report
            
        Returns:
            TechnicalReport with technical details and recommendations
        """
        pass
    
    @abstractmethod
    async def generate_compliance_report(self, compliance_data: 'ComplianceData') -> 'ComplianceReport':
        """
        Generate compliance-focused report for audit purposes.
        
        Args:
            compliance_data: Compliance data to include in report
            
        Returns:
            ComplianceReport formatted for regulatory requirements
        """
        pass


class ITemplateOrchestrator(ABC):
    """
    Interface for template orchestration.
    
    Coordinates multiple analysis engines and manages the integration
    between rule-based analysis and optional AI enhancement.
    """
    
    @abstractmethod
    async def orchestrate_analysis(self, request: AnalysisRequest) -> AnalysisResponse:
        """
        Orchestrate comprehensive analysis using multiple engines.
        
        Args:
            request: Analysis request to process
            
        Returns:
            AnalysisResponse with integrated results from multiple engines
        """
        pass
    
    @abstractmethod
    async def select_analysis_strategy(self, request: AnalysisRequest) -> 'AnalysisStrategy':
        """
        Select optimal analysis strategy based on request characteristics.
        
        Args:
            request: Analysis request to evaluate
            
        Returns:
            AnalysisStrategy defining which engines to use and how
        """
        pass
    
    @abstractmethod
    async def coordinate_engines(self, strategy: 'AnalysisStrategy', request: AnalysisRequest) -> Dict[str, Any]:
        """
        Coordinate execution of multiple analysis engines.
        
        Args:
            strategy: Analysis strategy to execute
            request: Analysis request to process
            
        Returns:
            Dictionary of results from coordinated engines
        """
        pass
    
    @abstractmethod
    async def integrate_results(self, engine_results: Dict[str, Any]) -> 'IntegratedAnalysisResult':
        """
        Integrate results from multiple analysis engines.
        
        Args:
            engine_results: Results from individual engines
            
        Returns:
            IntegratedAnalysisResult combining all engine outputs
        """
        pass
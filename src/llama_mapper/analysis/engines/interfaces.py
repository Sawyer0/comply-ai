"""
Interfaces for specialized analysis engines.

This module defines the contracts for the refactored analysis engines
that were extracted from the monolithic template provider.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from ..domain.entities import AnalysisRequest, AnalysisType


class IAnalysisEngine(ABC):
    """Base interface for all analysis engines."""
    
    @abstractmethod
    async def analyze(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Perform analysis on the request.
        
        Args:
            request: Analysis request containing security data
            
        Returns:
            Analysis result dictionary
        """
        pass
    
    @abstractmethod
    def get_confidence(self, result: Dict[str, Any]) -> float:
        """
        Get confidence score for analysis result.
        
        Args:
            result: Analysis result
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        pass


class IPatternRecognitionEngine(IAnalysisEngine):
    """Interface for pattern recognition engine."""
    
    @abstractmethod
    async def detect_patterns(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Detect patterns in security data.
        
        Args:
            request: Analysis request
            
        Returns:
            Detected patterns with metadata
        """
        pass
    
    @abstractmethod
    async def classify_pattern_strength(self, patterns: Dict[str, Any]) -> str:
        """
        Classify the strength of detected patterns.
        
        Args:
            patterns: Detected patterns
            
        Returns:
            Pattern strength classification (weak, moderate, strong)
        """
        pass


class IRiskScoringEngine(IAnalysisEngine):
    """Interface for risk scoring engine."""
    
    @abstractmethod
    async def calculate_risk_score(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Calculate comprehensive risk score.
        
        Args:
            request: Analysis request
            
        Returns:
            Risk score with breakdown and contributing factors
        """
        pass
    
    @abstractmethod
    async def get_risk_breakdown(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed risk breakdown.
        
        Args:
            risk_data: Risk calculation data
            
        Returns:
            Detailed risk breakdown
        """
        pass


class IComplianceIntelligence(IAnalysisEngine):
    """Interface for compliance intelligence engine."""
    
    @abstractmethod
    async def map_to_frameworks(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Map findings to regulatory frameworks.
        
        Args:
            request: Analysis request
            
        Returns:
            Framework mappings and compliance status
        """
        pass
    
    @abstractmethod
    async def generate_compliance_policy(self, mappings: Dict[str, Any]) -> str:
        """
        Generate OPA policy for compliance enforcement.
        
        Args:
            mappings: Compliance mappings
            
        Returns:
            OPA policy string
        """
        pass


class ITemplateOrchestrator(ABC):
    """Interface for template orchestrator that coordinates analysis engines."""
    
    @abstractmethod
    async def orchestrate_analysis(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Orchestrate analysis across multiple engines.
        
        Args:
            request: Analysis request
            
        Returns:
            Comprehensive analysis result
        """
        pass
    
    @abstractmethod
    async def select_analysis_strategy(self, request: AnalysisRequest) -> AnalysisType:
        """
        Select the most appropriate analysis strategy.
        
        Args:
            request: Analysis request
            
        Returns:
            Selected analysis type
        """
        pass
    
    @abstractmethod
    async def get_template_response(
        self,
        request: AnalysisRequest,
        analysis_type: AnalysisType,
        fallback_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get template response using coordinated engines.
        
        Args:
            request: Analysis request
            analysis_type: Type of analysis to perform
            fallback_reason: Reason for using template fallback
            
        Returns:
            Template response dictionary
        """
        pass
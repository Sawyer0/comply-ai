"""
Plugin interfaces for the Analysis Service.

This module defines the plugin interfaces and extension points for the Analysis Service,
allowing custom analysis engines, ML models, and quality evaluation plugins.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel
from datetime import datetime


class PluginType(Enum):
    """Plugin type enumeration."""

    ANALYSIS_ENGINE = "analysis_engine"
    ML_MODEL = "ml_model"
    QUALITY_EVALUATOR = "quality_evaluator"
    PATTERN_DETECTOR = "pattern_detector"
    RISK_SCORER = "risk_scorer"
    COMPLIANCE_MAPPER = "compliance_mapper"


class PluginStatus(Enum):
    """Plugin status enumeration."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    LOADING = "loading"


class PluginCapability(Enum):
    """Plugin capability enumeration."""

    PATTERN_RECOGNITION = "pattern_recognition"
    RISK_SCORING = "risk_scoring"
    COMPLIANCE_MAPPING = "compliance_mapping"
    QUALITY_EVALUATION = "quality_evaluation"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    ML_INFERENCE = "ml_inference"
    BATCH_PROCESSING = "batch_processing"
    REAL_TIME_ANALYSIS = "real_time_analysis"


class PluginMetadata(BaseModel):
    """Plugin metadata model."""

    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    capabilities: List[PluginCapability]
    dependencies: List[str] = []
    config_schema: Optional[Dict[str, Any]] = None
    supported_frameworks: List[str] = []
    min_confidence_threshold: float = 0.0
    max_batch_size: Optional[int] = None


class AnalysisRequest(BaseModel):
    """Analysis request model for plugins."""

    request_id: str
    tenant_id: str
    content_hash: str  # Hash of content, not raw content
    metadata: Dict[str, Any]
    analysis_type: str
    confidence_threshold: float = 0.8
    framework: Optional[str] = None
    custom_config: Optional[Dict[str, Any]] = None


class AnalysisResult(BaseModel):
    """Analysis result model from plugins."""

    request_id: str
    plugin_name: str
    plugin_version: str
    confidence: float
    result_data: Dict[str, Any]
    processing_time_ms: float
    metadata: Dict[str, Any] = {}
    errors: List[str] = []
    warnings: List[str] = []


class QualityMetrics(BaseModel):
    """Quality metrics model for evaluation plugins."""

    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    confidence_distribution: Optional[Dict[str, float]] = None
    error_rate: Optional[float] = None
    processing_time_stats: Optional[Dict[str, float]] = None


# Base Plugin Interface


class IPlugin(ABC):
    """Base interface for all Analysis Service plugins."""

    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """
        Get plugin metadata.

        Returns:
            Plugin metadata
        """
        pass

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the plugin with configuration.

        Args:
            config: Plugin configuration

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform plugin health check.

        Returns:
            Health status information
        """
        pass

    @abstractmethod
    async def cleanup(self) -> bool:
        """
        Cleanup plugin resources.

        Returns:
            True if cleanup successful, False otherwise
        """
        pass


# Analysis Engine Plugin Interface


class IAnalysisEnginePlugin(IPlugin):
    """Interface for analysis engine plugins."""

    @abstractmethod
    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Perform analysis on the request.

        Args:
            request: Analysis request

        Returns:
            Analysis result
        """
        pass

    @abstractmethod
    async def batch_analyze(
        self, requests: List[AnalysisRequest]
    ) -> List[AnalysisResult]:
        """
        Perform batch analysis on multiple requests.

        Args:
            requests: List of analysis requests

        Returns:
            List of analysis results
        """
        pass

    @abstractmethod
    def get_supported_analysis_types(self) -> List[str]:
        """
        Get list of supported analysis types.

        Returns:
            List of analysis type names
        """
        pass

    @abstractmethod
    async def validate_request(self, request: AnalysisRequest) -> bool:
        """
        Validate if the plugin can handle the request.

        Args:
            request: Analysis request to validate

        Returns:
            True if request is valid for this plugin
        """
        pass


# ML Model Plugin Interface


class IMLModelPlugin(IPlugin):
    """Interface for ML model plugins."""

    @abstractmethod
    async def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction using the ML model.

        Args:
            input_data: Input data for prediction

        Returns:
            Prediction results
        """
        pass

    @abstractmethod
    async def batch_predict(
        self, input_batch: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Make batch predictions.

        Args:
            input_batch: Batch of input data

        Returns:
            List of prediction results
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Model information including version, architecture, etc.
        """
        pass

    @abstractmethod
    async def update_model(self, model_path: str) -> bool:
        """
        Update the model with a new version.

        Args:
            model_path: Path to new model

        Returns:
            True if update successful
        """
        pass


# Quality Evaluator Plugin Interface


class IQualityEvaluatorPlugin(IPlugin):
    """Interface for quality evaluation plugins."""

    @abstractmethod
    async def evaluate_quality(
        self,
        results: List[AnalysisResult],
        ground_truth: Optional[List[Dict[str, Any]]] = None,
    ) -> QualityMetrics:
        """
        Evaluate quality of analysis results.

        Args:
            results: Analysis results to evaluate
            ground_truth: Optional ground truth data for comparison

        Returns:
            Quality metrics
        """
        pass

    @abstractmethod
    async def detect_drift(
        self,
        current_results: List[AnalysisResult],
        baseline_results: List[AnalysisResult],
    ) -> Dict[str, Any]:
        """
        Detect model drift by comparing current results with baseline.

        Args:
            current_results: Current analysis results
            baseline_results: Baseline analysis results

        Returns:
            Drift detection results
        """
        pass

    @abstractmethod
    async def generate_quality_report(
        self, metrics: QualityMetrics, period_start: datetime, period_end: datetime
    ) -> Dict[str, Any]:
        """
        Generate quality report for a time period.

        Args:
            metrics: Quality metrics
            period_start: Report period start
            period_end: Report period end

        Returns:
            Quality report
        """
        pass

    @abstractmethod
    def get_quality_thresholds(self) -> Dict[str, float]:
        """
        Get quality thresholds for alerting.

        Returns:
            Dictionary of metric names to threshold values
        """
        pass


# Pattern Detector Plugin Interface


class IPatternDetectorPlugin(IAnalysisEnginePlugin):
    """Interface for pattern detection plugins."""

    @abstractmethod
    async def detect_patterns(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Detect patterns in the analysis request.

        Args:
            request: Analysis request

        Returns:
            Detected patterns
        """
        pass

    @abstractmethod
    async def classify_pattern_strength(self, patterns: Dict[str, Any]) -> str:
        """
        Classify the strength of detected patterns.

        Args:
            patterns: Detected patterns

        Returns:
            Pattern strength classification
        """
        pass

    @abstractmethod
    def get_pattern_types(self) -> List[str]:
        """
        Get supported pattern types.

        Returns:
            List of pattern type names
        """
        pass


# Risk Scorer Plugin Interface


class IRiskScorerPlugin(IAnalysisEnginePlugin):
    """Interface for risk scoring plugins."""

    @abstractmethod
    async def calculate_risk_score(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Calculate risk score for the analysis request.

        Args:
            request: Analysis request

        Returns:
            Risk score and breakdown
        """
        pass

    @abstractmethod
    async def get_risk_factors(self, request: AnalysisRequest) -> List[Dict[str, Any]]:
        """
        Get risk factors contributing to the score.

        Args:
            request: Analysis request

        Returns:
            List of risk factors
        """
        pass

    @abstractmethod
    def get_risk_categories(self) -> List[str]:
        """
        Get supported risk categories.

        Returns:
            List of risk category names
        """
        pass


# Compliance Mapper Plugin Interface


class IComplianceMapperPlugin(IAnalysisEnginePlugin):
    """Interface for compliance mapping plugins."""

    @abstractmethod
    async def map_to_framework(
        self, request: AnalysisRequest, framework: str
    ) -> Dict[str, Any]:
        """
        Map analysis results to compliance framework.

        Args:
            request: Analysis request
            framework: Target compliance framework

        Returns:
            Framework mapping results
        """
        pass

    @abstractmethod
    async def generate_compliance_report(
        self, mappings: List[Dict[str, Any]], framework: str
    ) -> Dict[str, Any]:
        """
        Generate compliance report for framework mappings.

        Args:
            mappings: Framework mappings
            framework: Compliance framework

        Returns:
            Compliance report
        """
        pass

    @abstractmethod
    def get_supported_frameworks(self) -> List[str]:
        """
        Get supported compliance frameworks.

        Returns:
            List of framework names
        """
        pass

    @abstractmethod
    async def validate_compliance(
        self, mappings: Dict[str, Any], framework: str
    ) -> Dict[str, Any]:
        """
        Validate compliance mappings against framework requirements.

        Args:
            mappings: Compliance mappings
            framework: Compliance framework

        Returns:
            Validation results
        """
        pass

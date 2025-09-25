"""
Plugin system for Analysis Service.

This module provides a comprehensive plugin system for extending the Analysis Service
with custom analysis engines, ML models, and quality evaluation plugins.
"""

from .interfaces import (
    IPlugin,
    IAnalysisEnginePlugin,
    IMLModelPlugin,
    IQualityEvaluatorPlugin,
    IPatternDetectorPlugin,
    IRiskScorerPlugin,
    IComplianceMapperPlugin,
    PluginType,
    PluginStatus,
    PluginCapability,
    PluginMetadata,
    AnalysisRequest,
    AnalysisResult,
    QualityMetrics,
)
from .manager import PluginManager, PluginRegistry
from .database import PluginDatabaseManager

__all__ = [
    "IPlugin",
    "IAnalysisEnginePlugin",
    "IMLModelPlugin",
    "IQualityEvaluatorPlugin",
    "IPatternDetectorPlugin",
    "IRiskScorerPlugin",
    "IComplianceMapperPlugin",
    "PluginType",
    "PluginStatus",
    "PluginCapability",
    "PluginMetadata",
    "AnalysisRequest",
    "AnalysisResult",
    "QualityMetrics",
    "PluginManager",
    "PluginRegistry",
    "PluginDatabaseManager",
]

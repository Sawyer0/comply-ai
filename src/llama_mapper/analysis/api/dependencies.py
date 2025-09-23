"""
Dependency injection for the Analysis Module API.

This module provides FastAPI dependency functions for injecting
analysis services and components into API endpoints.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from ..application.services import (
    AnalysisApplicationService,
    BatchAnalysisApplicationService,
)
from ..config.factory import AnalysisModuleFactory
from ..config.settings import AnalysisConfig


@lru_cache()
def get_analysis_config() -> AnalysisConfig:
    """
    Get analysis configuration (cached).
    
    Returns:
        Analysis configuration instance
    """
    return AnalysisConfig.from_env()


@lru_cache()
def get_analysis_factory() -> AnalysisModuleFactory:
    """
    Get analysis module factory (cached).
    
    Returns:
        Analysis module factory instance
    """
    config = get_analysis_config()
    return AnalysisModuleFactory.create_from_config(config)


def get_analysis_service() -> AnalysisApplicationService:
    """
    Get analysis application service.
    
    Returns:
        Analysis application service instance
    """
    factory = get_analysis_factory()
    return factory.create_analysis_application_service()


def get_batch_analysis_service() -> BatchAnalysisApplicationService:
    """
    Get batch analysis application service.
    
    Returns:
        Batch analysis application service instance
    """
    factory = get_analysis_factory()
    return factory.create_batch_analysis_application_service()


# Type aliases for dependency injection
AnalysisServiceDep = Annotated[AnalysisApplicationService, Depends(get_analysis_service)]
BatchAnalysisServiceDep = Annotated[BatchAnalysisApplicationService, Depends(get_batch_analysis_service)]

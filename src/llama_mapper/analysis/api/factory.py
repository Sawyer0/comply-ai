"""
Factory for creating analysis module FastAPI application.

This module provides the create_analysis_app factory function that wires
together all analysis module components with proper dependency injection.
"""

import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config.factory import AnalysisModuleFactory
from ..config.settings import AnalysisSettings
from ..monitoring.metrics_collector import AnalysisMetricsCollector
from ..infrastructure.auth import APIKeyManager
from .endpoints import AnalysisEndpoints
from .auth_endpoints import AuthEndpoints
from .middleware import AnalysisMiddleware, RequestContextMiddleware, RateLimitMiddleware
from .auth_middleware import create_auth_middleware
from .dependencies import AnalysisServiceDep, BatchAnalysisServiceDep

logger = logging.getLogger(__name__)


def create_analysis_app(
    settings: Optional[AnalysisSettings] = None,
    title: str = "Analysis Module API",
    description: str = "Automated analysis of structured security metrics",
    version: str = "1.0.0",
    disable_auth: bool = False
) -> FastAPI:
    """
    Create and configure the analysis module FastAPI application.
    
    Args:
        settings: Analysis settings (if None, will load from environment)
        title: API title
        description: API description
        version: API version
        
    Returns:
        Configured FastAPI application
    """
    logger.info("Creating analysis module FastAPI application")
    
    # Create FastAPI app
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Initialize settings and factory
    if settings is None:
        settings = AnalysisSettings()
    
    factory = AnalysisModuleFactory(settings)
    
    # Initialize API key manager
    api_key_manager = APIKeyManager()
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    
    # Add custom middleware
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        RateLimitMiddleware, 
        requests_per_minute=settings.rate_limit_requests_per_minute
    )
    
    # Add authentication middleware (unless disabled for testing)
    if not disable_auth:
        auth_middleware = create_auth_middleware(
            api_key_manager=api_key_manager,
            rate_limit_requests_per_minute=settings.rate_limit_requests_per_minute
        )
        app.add_middleware(auth_middleware)
    
    # Initialize metrics collector
    metrics_collector = AnalysisMetricsCollector("analysis-module")
    
    # Create endpoints
    endpoints = AnalysisEndpoints(api_key_manager)
    auth_endpoints = AuthEndpoints(api_key_manager)
    
    # Register endpoints with the app
    app.include_router(endpoints.router)
    app.include_router(auth_endpoints.router)
    
    # Add startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        logger.info("Starting analysis module")
        
        # Initialize the factory (loads models, etc.)
        await factory.initialize()
        
        # Store factory and metrics collector in app state
        app.state.factory = factory
        app.state.metrics_collector = metrics_collector
        
        logger.info("Analysis module startup complete")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        logger.info("Shutting down analysis module")
        
        # Cleanup factory resources
        if hasattr(app.state, 'factory'):
            await factory.cleanup()
        
        logger.info("Analysis module shutdown complete")
    
    # Add dependency overrides for testing
    def get_factory():
        return app.state.factory
    
    def get_metrics_collector():
        return app.state.metrics_collector
    
    # Override dependencies
    app.dependency_overrides[AnalysisServiceDep] = lambda: factory.get_analysis_service()
    app.dependency_overrides[BatchAnalysisServiceDep] = lambda: factory.get_batch_analysis_service()
    
    logger.info("Analysis module FastAPI application created successfully")
    return app


def create_analysis_app_with_config(
    config_path: Optional[str] = None,
    **kwargs
) -> FastAPI:
    """
    Create analysis app with configuration from file.
    
    Args:
        config_path: Path to configuration file
        **kwargs: Additional arguments passed to create_analysis_app
        
    Returns:
        Configured FastAPI application
    """
    settings = AnalysisSettings()
    
    if config_path:
        # Load settings from config file
        # This would be implemented based on your config loading strategy
        logger.info(f"Loading settings from config file: {config_path}")
        # settings = load_settings_from_file(config_path)
    
    return create_analysis_app(settings=settings, **kwargs)

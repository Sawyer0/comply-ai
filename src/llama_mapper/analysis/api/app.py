"""
FastAPI application factory for the Analysis Module.

This module provides the main FastAPI application factory that
configures the analysis module with proper middleware, routes, and dependencies.
"""

import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config.settings import AnalysisConfig
from ..config.factory import AnalysisModuleFactory
from .endpoints import AnalysisEndpoints
from .middleware import AnalysisMiddleware, RequestContextMiddleware, RateLimitMiddleware

logger = logging.getLogger(__name__)


def create_analysis_app(config: Optional[AnalysisConfig] = None) -> FastAPI:
    """
    Create and configure the analysis FastAPI application.
    
    Args:
        config: Analysis module configuration
        
    Returns:
        Configured FastAPI application
    """
    if config is None:
        config = AnalysisConfig.from_env()
    
    # Create FastAPI app
    app = FastAPI(
        title="Analysis Module API",
        version="0.1.0",
        description="Automated analysis of structured security metrics",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get_cors_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Add custom middleware
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
    
    # Add analysis-specific middleware
    app.middleware("http")(AnalysisMiddleware.security_headers_middleware())
    app.middleware("http")(AnalysisMiddleware.metrics_middleware())
    app.middleware("http")(AnalysisMiddleware.logging_middleware())
    app.middleware("http")(AnalysisMiddleware.error_handling_middleware())
    
    # Register endpoints
    analysis_endpoints = AnalysisEndpoints()
    app.include_router(analysis_endpoints.router)
    
    # Add root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "service": "Analysis Module",
            "version": "0.1.0",
            "description": "Automated analysis of structured security metrics",
            "docs_url": "/docs",
            "health_url": "/api/v1/analysis/health"
        }
    
    # Add startup event
    @app.on_event("startup")
    async def startup_event():
        """Application startup event."""
        logger.info("Analysis Module API starting up...")
        
        # Initialize factory and validate configuration
        factory = AnalysisModuleFactory.create_from_config(config)
        logger.info(f"Analysis module factory initialized with {len(factory.list_components())} components")
        
        # Store factory in app state for dependency injection
        app.state.factory = factory
        app.state.config = config
        
        logger.info("Analysis Module API startup complete")
    
    # Add shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        """Application shutdown event."""
        logger.info("Analysis Module API shutting down...")
        
        # Clean up resources
        if hasattr(app.state, "factory"):
            app.state.factory.clear_components()
        
        logger.info("Analysis Module API shutdown complete")
    
    return app


def create_analysis_app_from_env() -> FastAPI:
    """
    Create analysis app from environment variables.
    
    Returns:
        Configured FastAPI application
    """
    config = AnalysisConfig.from_env()
    return create_analysis_app(config)

"""
FastAPI application factory for the Analysis Module.

This module provides the main FastAPI application factory that
configures the analysis module with proper middleware, routes, and dependencies.
"""

import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config.factory import AnalysisModuleFactory
from ..config.settings import AnalysisConfig
from .endpoints import AnalysisEndpoints
from .metrics import MetricsMiddleware, get_metrics_response
from .middleware import (
    AnalysisMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
)

logger = logging.getLogger(__name__)


def create_analysis_app(
    config: Optional[AnalysisConfig] = None, disable_auth: bool = False
) -> FastAPI:
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
        openapi_url="/openapi.json",
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
    app.add_middleware(MetricsMiddleware)

    # Add analysis-specific middleware
    app.middleware("http")(AnalysisMiddleware.security_headers_middleware())
    app.middleware("http")(AnalysisMiddleware.metrics_middleware())
    app.middleware("http")(AnalysisMiddleware.logging_middleware())
    app.middleware("http")(AnalysisMiddleware.error_handling_middleware())

    # Register endpoints
    analysis_endpoints = (
        AnalysisEndpoints() if not disable_auth else AnalysisEndpoints(None)
    )
    app.include_router(analysis_endpoints.router)

    # Add metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return get_metrics_response()

    # Add root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "service": "Analysis Module",
            "version": "0.1.0",
            "description": "Automated analysis of structured security metrics",
            "docs_url": "/docs",
            "health_url": "/api/v1/analysis/health",
        }

    # Add startup event
    @app.on_event("startup")
    async def startup_event():
        """Application startup event."""
        logger.info("Analysis Module API starting up...")

        # Initialize factory and validate configuration
        factory = AnalysisModuleFactory.create_from_config(config)
        logger.info(
            f"Analysis module factory initialized with {len(factory.list_components())} components"
        )

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


def create_analysis_app_from_env(disable_auth: bool = False) -> FastAPI:
    """
    Create analysis app from environment variables.

    Args:
        disable_auth: Whether to disable authentication for demo purposes

    Returns:
        Configured FastAPI application
    """
    # Load main config and map to analysis config
    from ...config.manager import ConfigManager

    config_manager = ConfigManager("config.yaml")
    config = AnalysisConfig.from_config_manager(config_manager)
    return create_analysis_app(config, disable_auth=disable_auth)


# Create app instance for uvicorn (with auth disabled for demo)
app = create_analysis_app_from_env(disable_auth=True)

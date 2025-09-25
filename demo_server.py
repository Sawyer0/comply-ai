#!/usr/bin/env python3
"""
Quick demo server for investor demonstrations.

This script starts the Llama Mapper API with demo endpoints
that work without requiring full model deployment.
"""

import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.llama_mapper.api.demo import router as demo_router
from src.llama_mapper.config.manager import ConfigManager
from src.llama_mapper.monitoring.metrics_collector import MetricsCollector

def create_demo_app() -> FastAPI:
    """Create a minimal demo app for investor presentations."""
    
    app = FastAPI(
        title="Llama Mapper - Investor Demo",
        description="AI Compliance Mapping Platform - Live Demo",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS for web demos - SECURE CONFIGURATION
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",  # Development frontend
            "http://localhost:8080",  # Development dashboard
            "https://app.comply-ai.com",  # Production frontend
            "https://dashboard.comply-ai.com",  # Production dashboard
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=[
            "Content-Type",
            "Authorization", 
            "X-API-Key",
            "X-Tenant-ID",
            "X-Correlation-ID",
            "X-Request-ID"
        ],
    )
    
    # Add demo endpoints
    app.include_router(demo_router)
    
    @app.get("/")
    async def root():
        return {
            "message": "🚀 Llama Mapper - AI Compliance Platform",
            "status": "Demo Ready",
            "docs": "/docs",
            "demo_endpoints": {
                "health": "/demo/health",
                "mapping": "/demo/map", 
                "compliance_report": "/demo/compliance-report",
                "metrics": "/demo/metrics"
            },
            "example_requests": {
                "pii_detection": {
                    "url": "/demo/map",
                    "method": "POST",
                    "body": {
                        "detector": "presidio",
                        "output": "EMAIL_ADDRESS"
                    }
                },
                "toxicity_detection": {
                    "url": "/demo/map", 
                    "method": "POST",
                    "body": {
                        "detector": "deberta",
                        "output": "toxic"
                    }
                },
                "compliance_report": {
                    "url": "/demo/compliance-report?framework=SOC2",
                    "method": "GET"
                }
            }
        }
    
    return app

if __name__ == "__main__":
    print("🚀 Starting Llama Mapper Demo Server...")
    print("📊 Demo endpoints available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🎯 Investor Demo Ready!")
    print()
    print("Example API calls:")
    print("curl -X POST http://localhost:8000/demo/map -H 'Content-Type: application/json' -d '{\"detector\":\"presidio\",\"output\":\"EMAIL_ADDRESS\"}'")
    print("curl http://localhost:8000/demo/compliance-report?framework=SOC2")
    print("curl http://localhost:8000/demo/metrics")
    print()
    
    app = create_demo_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
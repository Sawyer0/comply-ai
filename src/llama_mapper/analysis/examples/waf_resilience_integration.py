"""
Example of WAF and Resilience integration.

This module demonstrates how to integrate WAF rules and resilience patterns
into the analysis module for production use.
"""

import asyncio
import logging
from typing import Optional

from fastapi import FastAPI
from ..security.waf import WAFFactory, WAFRuleEngine, WAFMiddleware
from ..resilience import (
    ResilienceFactory, ResilienceManager,
    RetryConfig, CircuitBreakerConfig,
    CircuitState, RetryStrategy
)
from ..monitoring.metrics_collector import AnalysisMetricsCollector
from ..config.settings import AnalysisSettings

logger = logging.getLogger(__name__)


class AnalysisModuleWithSecurity:
    """
    Analysis module with integrated WAF and resilience patterns.
    
    This class demonstrates how to properly integrate security and resilience
    features into the analysis module for production deployment.
    """
    
    def __init__(self, settings: AnalysisSettings):
        """
        Initialize analysis module with security and resilience.
        
        Args:
            settings: Analysis module settings
        """
        self.settings = settings
        self.app = FastAPI(title="Analysis Module with Security")
        
        # Initialize metrics collector
        self.metrics_collector = AnalysisMetricsCollector("analysis_module")
        
        # Initialize WAF components
        self._setup_waf()
        
        # Initialize resilience components
        self._setup_resilience()
        
        # Setup routes
        self._setup_routes()
    
    def _setup_waf(self):
        """Setup WAF components."""
        # Create WAF rule engine
        self.waf_engine = WAFFactory.create_rule_engine()
        
        # Create WAF middleware
        self.waf_middleware = WAFFactory.create_middleware(
            app=self.app,
            waf_engine=self.waf_engine,
            metrics_collector=self.metrics_collector,
            block_mode=True,  # Block malicious requests
            log_violations=True
        )
        
        # Add middleware to app
        self.app.add_middleware(type(self.waf_middleware), **{
            'waf_engine': self.waf_engine,
            'metrics_collector': self.metrics_collector,
            'block_mode': True,
            'log_violations': True
        })
        
        logger.info("WAF components initialized")
    
    def _setup_resilience(self):
        """Setup resilience components."""
        # Create resilience manager
        self.resilience_manager = ResilienceManager(
            metrics_collector=self.metrics_collector
        )
        
        # Create retry configuration from settings
        retry_config = RetryConfig.from_settings(self.settings)
        
        # Create circuit breaker configuration from settings
        circuit_breaker_config = CircuitBreakerConfig.from_settings(self.settings)
        
        # Create circuit breaker for model server
        self.model_circuit_breaker = self.resilience_manager.create_circuit_breaker(
            name="model_server",
            config=circuit_breaker_config
        )
        
        # Create retry manager for model server
        self.model_retry_manager = self.resilience_manager.create_retry_manager(
            name="model_server",
            config=retry_config,
            circuit_breaker=self.model_circuit_breaker
        )
        
        logger.info("Resilience components initialized")
    
    def _setup_routes(self):
        """Setup API routes."""
        from fastapi import HTTPException
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "waf_rules": len(self.waf_engine.rules),
                "circuit_breaker_state": self.model_circuit_breaker.state.value
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Metrics endpoint."""
            return {
                "waf_stats": self.waf_middleware.get_statistics(),
                "resilience_stats": self.resilience_manager.get_all_statistics(),
                "analysis_metrics": self.metrics_collector.get_analysis_metrics_summary()
            }
        
        @self.app.post("/analyze")
        async def analyze_with_resilience(request_data: dict):
            """Analyze endpoint with resilience patterns."""
            try:
                # Example analysis function with retry logic
                async def analyze_function():
                    # Simulate model server call
                    await asyncio.sleep(0.1)
                    return {"analysis": "success", "confidence": 0.95}
                
                # Execute with retry and circuit breaker
                result = await self.model_retry_manager.execute_with_retry(
                    analyze_function
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                raise HTTPException(status_code=500, detail="Analysis failed")
        
        @self.app.post("/admin/unblock-ip/{client_ip}")
        async def unblock_ip(client_ip: str):
            """Admin endpoint to unblock an IP."""
            success = self.waf_middleware.unblock_ip(client_ip)
            return {"unblocked": success, "ip": client_ip}
        
        @self.app.post("/admin/reset-circuit-breaker")
        async def reset_circuit_breaker():
            """Admin endpoint to reset circuit breaker."""
            self.model_circuit_breaker.reset()
            return {"reset": True, "state": self.model_circuit_breaker.state.value}
    
    async def start(self):
        """Start the analysis module."""
        logger.info("Starting Analysis Module with Security and Resilience")
        
        # Log initial statistics
        logger.info(f"WAF Rules loaded: {len(self.waf_engine.rules)}")
        logger.info(f"Circuit Breaker State: {self.model_circuit_breaker.state.value}")
        
        return self.app


# Example usage
async def main():
    """Example usage of the integrated analysis module."""
    # Create settings
    settings = AnalysisSettings()
    
    # Create analysis module with security
    analysis_module = AnalysisModuleWithSecurity(settings)
    
    # Start the module
    app = await analysis_module.start()
    
    # Example of monitoring
    logger.info("=== WAF Statistics ===")
    waf_stats = analysis_module.waf_middleware.get_statistics()
    logger.info(f"Total requests: {waf_stats['total_requests']}")
    logger.info(f"Blocked requests: {waf_stats['blocked_requests']}")
    
    logger.info("=== Resilience Statistics ===")
    resilience_stats = analysis_module.resilience_manager.get_all_statistics()
    logger.info(f"Circuit breaker state: {resilience_stats['circuit_breakers']['model_server']['state']}")
    
    return app


if __name__ == "__main__":
    asyncio.run(main())

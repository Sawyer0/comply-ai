"""
WAF factory for creating and configuring WAF components.

This module provides factory methods for creating WAF components
with proper configuration and dependency injection.
"""

import logging
from typing import Optional, Any

from .interfaces import IWAFRuleEngine, IWAFMiddleware, IWAFMetricsCollector
from .engine.rule_engine import WAFRuleEngine
from .middleware.waf_middleware import WAFMiddleware

logger = logging.getLogger(__name__)


class WAFFactory:
    """Factory for creating WAF components."""
    
    @staticmethod
    def create_rule_engine() -> IWAFRuleEngine:
        """
        Create a WAF rule engine with default configuration.
        
        Returns:
            Configured WAF rule engine
        """
        return WAFRuleEngine()
    
    @staticmethod
    def create_middleware(
        app: Any,
        waf_engine: Optional[IWAFRuleEngine] = None,
        metrics_collector: Optional[IWAFMetricsCollector] = None,
        block_mode: bool = True,
        log_violations: bool = True
    ) -> IWAFMiddleware:
        """
        Create WAF middleware with configuration.
        
        Args:
            app: FastAPI application
            waf_engine: WAF rule engine instance
            metrics_collector: Metrics collector for WAF events
            block_mode: Whether to block requests or just log violations
            log_violations: Whether to log WAF violations
            
        Returns:
            Configured WAF middleware
        """
        if waf_engine is None:
            waf_engine = WAFFactory.create_rule_engine()
        
        return WAFMiddleware(
            app=app,
            waf_engine=waf_engine,
            metrics_collector=metrics_collector,
            block_mode=block_mode,
            log_violations=log_violations
        )
    
    @staticmethod
    def create_default_waf_stack(
        app: Any,
        metrics_collector: Optional[IWAFMetricsCollector] = None,
        block_mode: bool = True
    ) -> tuple[IWAFRuleEngine, IWAFMiddleware]:
        """
        Create a complete WAF stack with default configuration.
        
        Args:
            app: FastAPI application
            metrics_collector: Metrics collector for WAF events
            block_mode: Whether to block requests or just log violations
            
        Returns:
            Tuple of (rule_engine, middleware)
        """
        rule_engine = WAFFactory.create_rule_engine()
        middleware = WAFFactory.create_middleware(
            app=app,
            waf_engine=rule_engine,
            metrics_collector=metrics_collector,
            block_mode=block_mode
        )
        
        logger.info("WAF stack created with default configuration")
        return rule_engine, middleware

"""
WAF factory for creating and configuring WAF components.

This module provides factory methods for creating WAF components
with proper configuration and dependency injection.
"""

import logging
from typing import Any, Optional

from .engine.rule_engine import WAFRuleEngine
from .interfaces import IWAFMetricsCollector, IWAFMiddleware, IWAFRuleEngine
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
        waf_engine: Optional[IWAFRuleEngine] = None,
        metrics_collector: Optional[IWAFMetricsCollector] = None,
        block_mode: bool = True,
        log_violations: bool = True,
    ) -> IWAFMiddleware:
        """
        Create WAF middleware with configuration.

        Args:
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
            waf_engine=waf_engine,
            metrics_collector=metrics_collector,
            block_mode=block_mode,
            log_violations=log_violations,
        )

    @staticmethod
    def create_configured_waf(config: dict) -> tuple[IWAFRuleEngine, IWAFMiddleware]:
        """
        Create a fully configured WAF system.

        Args:
            config: WAF configuration dictionary

        Returns:
            Tuple of (rule_engine, middleware)
        """
        # Create rule engine
        rule_engine = WAFFactory.create_rule_engine()

        # Load patterns if specified in config
        if "patterns" in config:
            WAFFactory._load_patterns(rule_engine, config["patterns"])

        # Create middleware
        middleware = WAFFactory.create_middleware(
            waf_engine=rule_engine,
            block_mode=config.get("block_mode", True),
            log_violations=config.get("log_violations", True),
        )

        logger.info("Created configured WAF system", config=config)
        return rule_engine, middleware

    @staticmethod
    def _load_patterns(rule_engine: IWAFRuleEngine, pattern_config: dict) -> None:
        """Load patterns into the rule engine."""
        try:
            # Import pattern collections
            from .patterns import (
                CommandInjectionPatterns,
                LDAPInjectionPatterns,
                MaliciousPayloadPatterns,
                NoSQLInjectionPatterns,
                PathTraversalPatterns,
                SQLInjectionPatterns,
                SSIInjectionPatterns,
                XMLInjectionPatterns,
                XSSPatterns,
            )

            # Load enabled pattern collections
            pattern_collections = {
                "sql_injection": SQLInjectionPatterns,
                "xss": XSSPatterns,
                "path_traversal": PathTraversalPatterns,
                "command_injection": CommandInjectionPatterns,
                "ldap_injection": LDAPInjectionPatterns,
                "nosql_injection": NoSQLInjectionPatterns,
                "xml_injection": XMLInjectionPatterns,
                "ssi_injection": SSIInjectionPatterns,
                "malicious_payload": MaliciousPayloadPatterns,
            }

            for pattern_name, enabled in pattern_config.items():
                if enabled and pattern_name in pattern_collections:
                    pattern_collection = pattern_collections[pattern_name]()
                    
                    # Add all patterns from the collection
                    for rule in pattern_collection.get_all_patterns():
                        rule_engine.add_rule(rule)
                    
                    logger.info("Loaded pattern collection", 
                              collection=pattern_name, 
                              pattern_count=pattern_collection.get_pattern_count())

        except Exception as e:
            logger.error("Failed to load WAF patterns: %s", e)
            raise

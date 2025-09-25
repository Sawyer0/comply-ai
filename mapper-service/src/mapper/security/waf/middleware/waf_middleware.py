"""
WAF Middleware for request filtering and attack detection.

This module provides middleware for Web Application Firewall (WAF)
functionality including request filtering and attack detection.
"""

from typing import Any, Dict, List, Optional
import structlog

from ..engine.rule_engine import WAFRuleEngine
from ..rule import WAFRule

logger = structlog.get_logger(__name__)


class WAFMiddleware:
    """
    WAF middleware for request filtering and attack detection.
    
    This class provides middleware functionality for filtering malicious
    requests and detecting various types of attacks.
    """
    
    def __init__(self, rules: Optional[List[WAFRule]] = None):
        """
        Initialize WAF middleware.
        
        Args:
            rules: List of WAF rules to apply
        """
        self.rule_engine = WAFRuleEngine()
        self.rules = rules or []
        
        # Load default rules
        self._load_default_rules()
        
        logger.info("WAFMiddleware initialized", rules_count=len(self.rules))
    
    def _load_default_rules(self) -> None:
        """Load default WAF rules."""
        # Add basic security rules
        default_rules = [
            WAFRule(
                name="sql_injection",
                pattern=r"(?i)(union|select|insert|update|delete|drop|create|alter|exec|execute)",
                action="block",
                severity="high"
            ),
            WAFRule(
                name="xss_attack",
                pattern=r"(?i)(<script|javascript:|on\w+\s*=)",
                action="block",
                severity="high"
            ),
            WAFRule(
                name="path_traversal",
                pattern=r"(\.\./|\.\.\\|%2e%2e%2f|%2e%2e%5c)",
                action="block",
                severity="medium"
            ),
            WAFRule(
                name="command_injection",
                pattern=r"(?i)(;|\||&|`|\$\(|\$\{)",
                action="block",
                severity="high"
            ),
        ]
        
        self.rules.extend(default_rules)
    
    async def check_request(self, request: Any) -> bool:
        """
        Check if a request should be blocked.
        
        Args:
            request: The request to check
            
        Returns:
            True if request should be blocked, False otherwise
        """
        try:
            # Extract request information
            request_info = self._extract_request_info(request)
            
            # Check against all rules
            for rule in self.rules:
                if await self.rule_engine.evaluate_rule(rule, request_info):
                    logger.warning(
                        "Request blocked by WAF",
                        rule_name=rule.name,
                        pattern=rule.pattern,
                        severity=rule.severity,
                        request_info=request_info
                    )
                    return True
            
            return False
            
        except Exception as e:
            logger.error("WAF check failed", error=str(e))
            return False  # Allow request if WAF check fails
    
    def _extract_request_info(self, request: Any) -> Dict[str, Any]:
        """
        Extract request information for WAF analysis.
        
        Args:
            request: The request to analyze
            
        Returns:
            Dictionary containing request information
        """
        try:
            request_info = {
                "method": getattr(request, "method", "GET"),
                "url": getattr(request, "url", ""),
                "path": getattr(request, "url", {}).get("path", "") if hasattr(request, "url") else "",
                "query_params": getattr(request, "query_params", {}),
                "headers": dict(getattr(request, "headers", {})),
                "client_ip": getattr(request, "client", {}).get("host", "") if hasattr(request, "client") else "",
            }
            
            # Extract body if available
            if hasattr(request, "body"):
                try:
                    body = request.body()
                    if isinstance(body, bytes):
                        body = body.decode("utf-8", errors="ignore")
                    request_info["body"] = body
                except Exception:
                    request_info["body"] = ""
            
            return request_info
            
        except Exception as e:
            logger.warning("Failed to extract request info", error=str(e))
            return {}
    
    async def update_rules(self, rules: List[WAFRule]) -> None:
        """
        Update WAF rules.
        
        Args:
            rules: New list of WAF rules
        """
        self.rules = rules
        logger.info("WAF rules updated", rules_count=len(rules))
    
    async def get_rules(self) -> List[WAFRule]:
        """
        Get current WAF rules.
        
        Returns:
            List of current WAF rules
        """
        return self.rules.copy()
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get WAF status information.
        
        Returns:
            Dictionary containing WAF status
        """
        return {
            "active": True,
            "rules_count": len(self.rules),
            "rule_engine": "active",
        }

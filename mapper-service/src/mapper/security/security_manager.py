"""
Security Manager for comprehensive security orchestration.

This module provides a centralized security manager that coordinates
all security components including authentication, authorization, WAF,
and rate limiting.
"""

from typing import Optional, Any, Dict
import structlog

from .authentication import AuthenticationService
from .authorization import AuthorizationService
from .api_key_manager import APIKeyManager
from .rate_limiting_service import RateLimitingService
from .waf import WAFMiddleware

logger = structlog.get_logger(__name__)


class SecurityManager:
    """
    Centralized security manager that orchestrates all security components.
    
    This class provides a unified interface for security operations including
    authentication, authorization, WAF protection, and rate limiting.
    """
    
    def __init__(
        self,
        auth_service: Optional[AuthenticationService] = None,
        authz_service: Optional[AuthorizationService] = None,
        api_key_manager: Optional[APIKeyManager] = None,
        rate_limiting_service: Optional[RateLimitingService] = None,
    ):
        """
        Initialize the security manager.
        
        Args:
            auth_service: Authentication service instance
            authz_service: Authorization service instance
            api_key_manager: API key manager instance
            rate_limiting_service: Rate limiting service instance
        """
        self.auth_service = auth_service
        self.authz_service = authz_service
        self.api_key_manager = api_key_manager
        self.rate_limiting_service = rate_limiting_service
        
        # Initialize WAF middleware
        self.waf = WAFMiddleware()
        
        logger.info("SecurityManager initialized")
    
    async def authenticate_request(self, request: Any) -> Optional[Dict[str, Any]]:
        """
        Authenticate a request using the authentication service.
        
        Args:
            request: The request to authenticate
            
        Returns:
            Authentication result or None if authentication fails
        """
        if not self.auth_service:
            logger.warning("Authentication service not available")
            return None
        
        try:
            return await self.auth_service.authenticate_request(request)
        except Exception as e:
            logger.error("Authentication failed", error=str(e))
            return None
    
    async def authorize_request(
        self, 
        request: Any, 
        required_permission: str
    ) -> bool:
        """
        Authorize a request using the authorization service.
        
        Args:
            request: The request to authorize
            required_permission: The required permission
            
        Returns:
            True if authorized, False otherwise
        """
        if not self.authz_service:
            logger.warning("Authorization service not available")
            return False
        
        try:
            return await self.authz_service.check_permission(
                request, required_permission
            )
        except Exception as e:
            logger.error("Authorization failed", error=str(e))
            return False
    
    async def check_rate_limit(
        self, 
        request: Any, 
        api_key_info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check rate limit for a request.
        
        Args:
            request: The request to check
            api_key_info: API key information
            
        Returns:
            True if within rate limit, False otherwise
        """
        if not self.rate_limiting_service:
            logger.warning("Rate limiting service not available")
            return True
        
        try:
            return await self.rate_limiting_service.check_rate_limit(
                request, api_key_info
            )
        except Exception as e:
            logger.error("Rate limit check failed", error=str(e))
            return True  # Allow request if rate limiting fails
    
    async def get_security_status(self) -> Dict[str, Any]:
        """
        Get the current security status.
        
        Returns:
            Dictionary containing security status information
        """
        status = {
            "authentication": self.auth_service is not None,
            "authorization": self.authz_service is not None,
            "api_key_management": self.api_key_manager is not None,
            "rate_limiting": self.rate_limiting_service is not None,
            "waf": self.waf is not None,
        }
        
        return status
    
    async def update_security_config(self, config: Dict[str, Any]) -> bool:
        """
        Update security configuration.
        
        Args:
            config: Security configuration dictionary
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Update WAF rules if provided
            if "waf_rules" in config and self.waf:
                await self.waf.update_rules(config["waf_rules"])
            
            # Update rate limiting config if provided
            if "rate_limits" in config and self.rate_limiting_service:
                await self.rate_limiting_service.update_config(config["rate_limits"])
            
            logger.info("Security configuration updated", config_keys=list(config.keys()))
            return True
            
        except Exception as e:
            logger.error("Failed to update security configuration", error=str(e))
            return False

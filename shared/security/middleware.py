from __future__ import annotations

"""Security middleware for FastAPI applications."""

import inspect
import logging
from typing import Any, Callable, Optional

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from .auth import SecurityManager, get_security_manager, Permission
from ..exceptions.base import RateLimitError

logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Combined security middleware for authentication and rate limiting."""
    
    def __init__(
        self,
        app,
        require_auth: bool = True,
        enable_rate_limiting: bool = True,
        excluded_paths: Optional[list] = None
    ):
        super().__init__(app)
        self.require_auth = require_auth
        self.enable_rate_limiting = enable_rate_limiting
        self.excluded_paths = excluded_paths or ["/", "/health", "/metrics", "/docs"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with security checks."""
        path = request.url.path
        
        # Skip security for excluded paths
        if path in self.excluded_paths:
            return await call_next(request)
        
        try:
            security_manager = get_security_manager()

            api_key_info = None
            if self.require_auth:
                headers = dict(request.headers)
                api_key_info = security_manager.authenticate_request(headers)
                request.state.user_info = api_key_info

            if self.enable_rate_limiting:
                await self._enforce_rate_limit(
                    security_manager=security_manager,
                    request=request,
                    subject=api_key_info,
                )
            
            response = await call_next(request)
            return response
            
        except Exception as e:
            logger.warning("Security check failed", extra={"error": str(e)})
            return Response(
                content='{"error": "Security check failed"}',
                status_code=403,
                media_type="application/json"
            )

    async def _enforce_rate_limit(
        self,
        security_manager: SecurityManager,
        request: Request,
        subject: Any,
    ) -> None:
        """Apply rate limiting if the security manager supports it."""

        rate_limit_handler = getattr(security_manager, "check_rate_limit", None)
        if not callable(rate_limit_handler):
            return

        try:
            result = rate_limit_handler(request, subject)
            if inspect.isawaitable(result):
                result = await result

            if result is False:
                raise RateLimitError()

            allowed = getattr(result, "allowed", True)
            if not allowed:
                headers: dict[str, str] = {}
                to_headers = getattr(result, "to_headers", None)
                if callable(to_headers):
                    try:
                        candidate = to_headers()
                        if isinstance(candidate, dict):
                            headers = {str(key): str(value) for key, value in candidate.items()}
                    except Exception:  # pragma: no cover - defensive
                        headers = {}
                raise HTTPException(status_code=429, detail="Rate limit exceeded", headers=headers)

        except RateLimitError as exc:
            headers = getattr(exc, "details", {}).get("headers", {})
            raise HTTPException(status_code=429, detail=str(exc), headers=headers) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Rate limit check failed", extra={"error": str(exc)})
            raise HTTPException(status_code=429, detail="Rate limit exceeded") from exc


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Authentication-only middleware."""
    
    def __init__(self, app, excluded_paths: Optional[list] = None):
        super().__init__(app)
        self.excluded_paths = excluded_paths or ["/", "/health", "/metrics", "/docs"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with authentication."""
        if request.url.path in self.excluded_paths:
            return await call_next(request)
        
        try:
            security_manager = get_security_manager()
            headers = dict(request.headers)
            api_key_info = security_manager.authenticate_request(headers)
            request.state.user_info = api_key_info
            
            response = await call_next(request)
            return response
            
        except Exception as e:
            logger.warning("Authentication failed", extra={"error": str(e)})
            return Response(
                content='{"error": "Authentication failed"}',
                status_code=401,
                media_type="application/json"
            )


class AuthorizationMiddleware(BaseHTTPMiddleware):
    """Authorization middleware."""
    
    def __init__(self, app, required_permission: Permission):
        super().__init__(app)
        self.required_permission = required_permission
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with authorization."""
        try:
            security_manager = get_security_manager()
            api_key_info = getattr(request.state, 'user_info', None)
            
            if not api_key_info:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            security_manager.authorize_request(api_key_info, self.required_permission)
            
            response = await call_next(request)
            return response
            
        except Exception as e:
            logger.warning("Authorization failed", extra={"error": str(e)})
            return Response(
                content='{"error": "Authorization failed"}',
                status_code=403,
                media_type="application/json"
            )


class WAFMiddleware(BaseHTTPMiddleware):
    """Web Application Firewall middleware for basic attack detection."""
    
    def __init__(self, app):
        super().__init__(app)
        self.suspicious_patterns = [
            "<script", "javascript:", "vbscript:", "onload=", "onerror=",
            "select.*from.*information_schema", "union.*select", "drop.*table",
            "../../", "..\\", "cmd.exe", "powershell", "/bin/sh"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with WAF checks."""
        try:
            # Check URL path
            if self._contains_suspicious_content(request.url.path):
                logger.warning("WAF blocked suspicious URL", extra={"path": request.url.path})
                return Response(
                    content='{"error": "Request blocked by WAF"}',
                    status_code=403,
                    media_type="application/json"
                )
            
            # Check query parameters
            for key, value in request.query_params.items():
                if self._contains_suspicious_content(value):
                    logger.warning("WAF blocked suspicious query", extra={"param": key})
                    return Response(
                        content='{"error": "Request blocked by WAF"}',
                        status_code=403,
                        media_type="application/json"
                    )
            
            response = await call_next(request)
            return response
            
        except Exception as e:
            logger.error("WAF check failed", extra={"error": str(e)})
            # Fail open - don't block requests if WAF fails
            return await call_next(request)
    
    def _contains_suspicious_content(self, content: str) -> bool:
        """Check if content contains suspicious patterns."""
        content_lower = content.lower()
        return any(pattern.lower() in content_lower for pattern in self.suspicious_patterns)

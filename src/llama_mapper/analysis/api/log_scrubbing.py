"""
Log scrubbing middleware for the Analysis Module.

This module provides comprehensive log scrubbing capabilities to remove
sensitive data from request/response logs while maintaining debugging capability.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Union
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LogScrubber:
    """
    Log scrubbing utility for removing sensitive data from logs.
    
    Provides comprehensive PII and sensitive data redaction for logging
    while maintaining useful debugging information.
    """
    
    def __init__(self):
        """Initialize the log scrubber with PII patterns."""
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'mac_address': r'\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b',
            'api_key': r'\b[A-Za-z0-9]{32,}\b',  # Generic API key pattern
            'token': r'\b[A-Za-z0-9+/]{40,}={0,2}\b',  # JWT/Bearer tokens
            'password': r'["\']?password["\']?\s*[:=]\s*["\']?[^"\']+["\']?',
            'secret': r'["\']?secret["\']?\s*[:=]\s*["\']?[^"\']+["\']?',
            'key': r'["\']?key["\']?\s*[:=]\s*["\']?[^"\']+["\']?',
            'auth': r'["\']?authorization["\']?\s*[:=]\s*["\']?[^"\']+["\']?',
            'bearer': r'bearer\s+[A-Za-z0-9+/=]+',
            'tenant_id': r'["\']?tenant_id["\']?\s*[:=]\s*["\']?[^"\']+["\']?',
            'user_id': r'["\']?user_id["\']?\s*[:=]\s*["\']?[^"\']+["\']?',
        }
        
        # Fields that should be completely redacted from logs
        self.sensitive_fields = {
            'password', 'secret', 'token', 'authorization', 'auth',
            'api_key', 'bearer', 'private_key', 'access_token',
            'refresh_token', 'session_id', 'csrf_token', 'x-api-key',
            'x-auth-token', 'x-access-token', 'tenant_id', 'user_id'
        }
        
        # Fields that should be partially redacted (show first/last few chars)
        self.partial_redact_fields = {
            'request_id', 'correlation_id', 'trace_id', 'span_id', 'x-request-id'
        }
    
    def scrub_text(self, text: str) -> str:
        """
        Scrub PII from text content.
        
        Args:
            text: Text to scrub
            
        Returns:
            Scrubbed text with PII redacted
        """
        if not isinstance(text, str):
            return text
        
        scrubbed_text = text
        
        # Apply PII pattern redaction (order matters - more specific patterns first)
        
        # JWT tokens first (most specific)
        jwt_pattern = r'eyJ[A-Za-z0-9+/=_-]+\.[A-Za-z0-9+/=_-]+\.[A-Za-z0-9+/=_-]+'
        scrubbed_text = re.sub(jwt_pattern, '[TOKEN_REDACTED]', scrubbed_text)
        
        for pii_type, pattern in self.pii_patterns.items():
            if pii_type == 'email':
                scrubbed_text = re.sub(pattern, '[EMAIL_REDACTED]', scrubbed_text, flags=re.IGNORECASE)
            elif pii_type == 'phone':
                # More comprehensive phone pattern - replace all phone formats
                phone_patterns = [
                    r'\(\d{3}\)\s*\d{3}-\d{4}',  # (555) 123-4567
                    r'\d{3}-\d{3}-\d{4}',        # 555-123-4567
                    r'\d{3}\.\d{3}\.\d{4}',      # 555.123.4567
                ]
                for phone_pattern in phone_patterns:
                    scrubbed_text = re.sub(phone_pattern, '[PHONE_REDACTED]', scrubbed_text)
            elif pii_type == 'ssn':
                scrubbed_text = re.sub(pattern, '[SSN_REDACTED]', scrubbed_text)
            elif pii_type == 'credit_card':
                scrubbed_text = re.sub(pattern, '[CARD_REDACTED]', scrubbed_text)
            elif pii_type == 'ip_address':
                scrubbed_text = re.sub(pattern, '[IP_REDACTED]', scrubbed_text)
            elif pii_type == 'mac_address':
                scrubbed_text = re.sub(pattern, '[MAC_REDACTED]', scrubbed_text)
            elif pii_type == 'token':
                # Skip JWT tokens (already handled above)
                if 'eyJ' not in pattern:
                    scrubbed_text = re.sub(pattern, '[TOKEN_REDACTED]', scrubbed_text)
            elif pii_type == 'api_key':
                # More specific API key pattern to avoid conflicts
                api_key_pattern = r'\b[A-Za-z0-9_]{32,}\b(?=\s|$|[^A-Za-z0-9_])'
                scrubbed_text = re.sub(api_key_pattern, '[API_KEY_REDACTED]', scrubbed_text)
            elif pii_type in ['password', 'secret', 'auth', 'bearer']:
                scrubbed_text = re.sub(pattern, f'[{pii_type.upper()}_REDACTED]', scrubbed_text, flags=re.IGNORECASE)
            elif pii_type == 'key':
                # Only match "key" pattern if not already matched by api_key
                if '[API_KEY_REDACTED]' not in scrubbed_text:
                    scrubbed_text = re.sub(pattern, '[KEY_REDACTED]', scrubbed_text, flags=re.IGNORECASE)
            elif pii_type in ['tenant_id', 'user_id']:
                scrubbed_text = re.sub(pattern, f'[{pii_type.upper()}_REDACTED]', scrubbed_text, flags=re.IGNORECASE)
        
        return scrubbed_text
    
    def scrub_dict(self, data: Dict[str, Any], path: str = "") -> Dict[str, Any]:
        """
        Recursively scrub sensitive data from dictionary.
        
        Args:
            data: Dictionary to scrub
            path: Current path in the data structure
            
        Returns:
            Scrubbed dictionary
        """
        if not isinstance(data, dict):
            return data
        
        scrubbed = {}
        
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            lower_key = key.lower()
            
            # Completely redact sensitive fields
            if lower_key in self.sensitive_fields:
                scrubbed[key] = "[REDACTED]"
            # Partially redact certain fields
            elif lower_key in self.partial_redact_fields and isinstance(value, str):
                if len(value) > 8:
                    scrubbed[key] = f"{value[:4]}...{value[-4:]}"
                else:
                    scrubbed[key] = "[REDACTED]"
            # Recursively scrub nested structures
            elif isinstance(value, dict):
                scrubbed[key] = self.scrub_dict(value, current_path)
            elif isinstance(value, list):
                scrubbed[key] = self.scrub_list(value, current_path)
            elif isinstance(value, str):
                scrubbed[key] = self.scrub_text(value)
            else:
                scrubbed[key] = value
        
        return scrubbed
    
    def scrub_list(self, data: List[Any], path: str = "") -> List[Any]:
        """
        Recursively scrub sensitive data from list.
        
        Args:
            data: List to scrub
            path: Current path in the data structure
            
        Returns:
            Scrubbed list
        """
        if not isinstance(data, list):
            return data
        
        scrubbed = []
        
        for i, item in enumerate(data):
            current_path = f"{path}[{i}]"
            
            if isinstance(item, dict):
                scrubbed.append(self.scrub_dict(item, current_path))
            elif isinstance(item, list):
                scrubbed.append(self.scrub_list(item, current_path))
            elif isinstance(item, str):
                scrubbed.append(self.scrub_text(item))
            else:
                scrubbed.append(item)
        
        return scrubbed
    
    def scrub_request_data(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scrub sensitive data from request data.
        
        Args:
            request_data: Request data to scrub
            
        Returns:
            Scrubbed request data
        """
        return self.scrub_dict(request_data)
    
    def scrub_response_data(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scrub sensitive data from response data.
        
        Args:
            response_data: Response data to scrub
            
        Returns:
            Scrubbed response data
        """
        return self.scrub_dict(response_data)
    
    def scrub_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Scrub sensitive data from HTTP headers.
        
        Args:
            headers: HTTP headers to scrub
            
        Returns:
            Scrubbed headers
        """
        scrubbed_headers = {}
        
        for key, value in headers.items():
            lower_key = key.lower()
            
            # Completely redact sensitive headers
            if lower_key in self.sensitive_fields:
                scrubbed_headers[key] = "[REDACTED]"
            # Partially redact certain headers
            elif lower_key in self.partial_redact_fields and isinstance(value, str):
                if len(value) > 8:
                    scrubbed_headers[key] = f"{value[:4]}...{value[-4:]}"
                else:
                    scrubbed_headers[key] = "[REDACTED]"
            # Scrub text content for other headers
            elif lower_key in ['user-agent', 'x-forwarded-for']:
                scrubbed_headers[key] = self.scrub_text(value)
            else:
                scrubbed_headers[key] = value
        
        return scrubbed_headers


class LogScrubbingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for scrubbing sensitive data from request/response logs.
    
    Automatically redacts PII and sensitive information from logs while
    maintaining useful debugging information.
    """
    
    def __init__(self, app, scrubber: Optional[LogScrubber] = None):
        """
        Initialize the log scrubbing middleware.
        
        Args:
            app: FastAPI application
            scrubber: Log scrubber instance (creates default if None)
        """
        super().__init__(app)
        self.scrubber = scrubber or LogScrubber()
        self.logger = logging.getLogger(f"{__name__}.LogScrubbingMiddleware")
    
    async def dispatch(self, request: Request, call_next):
        """
        Process request and response with log scrubbing.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response with scrubbed logs
        """
        # Extract request information
        request_info = await self._extract_request_info(request)
        
        # Log scrubbed request
        self.logger.info(
            f"Request: {request.method} {request.url.path}",
            extra={
                "request_id": request_info.get("request_id"),
                "method": request.method,
                "path": request.url.path,
                "query_params": request_info.get("query_params"),
                "headers": request_info.get("headers"),
                "client_ip": request_info.get("client_ip"),
                "user_agent": request_info.get("user_agent")
            }
        )
        
        # Process request
        response = await call_next(request)
        
        # Extract response information
        response_info = await self._extract_response_info(response)
        
        # Log scrubbed response
        self.logger.info(
            f"Response: {response.status_code}",
            extra={
                "request_id": request_info.get("request_id"),
                "status_code": response.status_code,
                "response_headers": response_info.get("headers"),
                "response_size": response_info.get("size")
            }
        )
        
        return response
    
    async def _extract_request_info(self, request: Request) -> Dict[str, Any]:
        """
        Extract and scrub request information.
        
        Args:
            request: Incoming request
            
        Returns:
            Scrubbed request information
        """
        # Get request ID from headers or generate one
        request_id = request.headers.get("x-request-id", "unknown")
        
        # Extract query parameters
        query_params = dict(request.query_params)
        scrubbed_query = self.scrubber.scrub_dict(query_params)
        
        # Extract headers
        headers = dict(request.headers)
        scrubbed_headers = self.scrubber.scrub_headers(headers)
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Get user agent
        user_agent = request.headers.get("user-agent", "unknown")
        scrubbed_user_agent = self.scrubber.scrub_text(user_agent)
        
        return {
            "request_id": request_id,
            "query_params": scrubbed_query,
            "headers": scrubbed_headers,
            "client_ip": client_ip,
            "user_agent": scrubbed_user_agent
        }
    
    async def _extract_response_info(self, response: Response) -> Dict[str, Any]:
        """
        Extract and scrub response information.
        
        Args:
            response: Outgoing response
            
        Returns:
            Scrubbed response information
        """
        # Extract response headers
        headers = dict(response.headers)
        scrubbed_headers = self.scrubber.scrub_headers(headers)
        
        # Get response size (if available)
        response_size = getattr(response, 'content_length', None)
        
        return {
            "headers": scrubbed_headers,
            "size": response_size
        }


class RequestResponseLogger:
    """
    Utility for logging request/response data with automatic scrubbing.
    
    Provides structured logging for API requests and responses with
    comprehensive PII redaction.
    """
    
    def __init__(self, scrubber: Optional[LogScrubber] = None):
        """
        Initialize the request/response logger.
        
        Args:
            scrubber: Log scrubber instance (creates default if None)
        """
        self.scrubber = scrubber or LogScrubber()
        self.logger = logging.getLogger(f"{__name__}.RequestResponseLogger")
    
    def log_request(
        self, 
        request_id: str, 
        method: str, 
        path: str, 
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Log request data with scrubbing.
        
        Args:
            request_id: Request identifier
            method: HTTP method
            path: Request path
            data: Request data (optional)
            headers: Request headers (optional)
        """
        log_data = {
            "request_id": request_id,
            "method": method,
            "path": path
        }
        
        if data:
            log_data["data"] = self.scrubber.scrub_request_data(data)
        
        if headers:
            log_data["headers"] = self.scrubber.scrub_headers(headers)
        
        # Include scrubbed data in the log message for testing
        message = f"Request: {method} {path}"
        if data:
            scrubbed_data = self.scrubber.scrub_request_data(data)
            message += f" | Data: {scrubbed_data}"
        if headers:
            scrubbed_headers = self.scrubber.scrub_headers(headers)
            message += f" | Headers: {scrubbed_headers}"
        
        self.logger.info(message, extra=log_data)
    
    def log_response(
        self, 
        request_id: str, 
        status_code: int, 
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Log response data with scrubbing.
        
        Args:
            request_id: Request identifier
            status_code: HTTP status code
            data: Response data (optional)
            headers: Response headers (optional)
        """
        log_data = {
            "request_id": request_id,
            "status_code": status_code
        }
        
        if data:
            log_data["data"] = self.scrubber.scrub_response_data(data)
        
        if headers:
            log_data["headers"] = self.scrubber.scrub_headers(headers)
        
        # Include scrubbed data in the log message for testing
        message = f"Response: {status_code}"
        if data:
            scrubbed_data = self.scrubber.scrub_response_data(data)
            message += f" | Data: {scrubbed_data}"
        if headers:
            scrubbed_headers = self.scrubber.scrub_headers(headers)
            message += f" | Headers: {scrubbed_headers}"
        
        self.logger.info(message, extra=log_data)
    
    def log_error(
        self, 
        request_id: str, 
        error: Exception, 
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Log error with scrubbing.
        
        Args:
            request_id: Request identifier
            error: Exception that occurred
            context: Additional context (optional)
        """
        log_data = {
            "request_id": request_id,
            "error_type": type(error).__name__,
            "error_message": str(error)
        }
        
        if context:
            log_data["context"] = self.scrubber.scrub_dict(context)
        
        # Include scrubbed context in the log message for testing
        message = f"Error: {type(error).__name__}"
        if context:
            scrubbed_context = self.scrubber.scrub_dict(context)
            message += f" | Context: {scrubbed_context}"
        
        self.logger.error(message, extra=log_data, exc_info=True)
